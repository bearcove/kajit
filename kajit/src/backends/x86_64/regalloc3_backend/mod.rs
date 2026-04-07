//! x86_64 backend (regalloc3 native types).

mod calls;
mod context;
mod control;
mod fusion;
mod inst;

use kajit_emit::x64::{self, Mem};
use kajit_lir::LinearOp;
use kajit_mir::cfg_mir;
use kajit_mir::regalloc3_result::AllocatedCfgProgramRa3;
use std::collections::HashMap;

use crate::arch::x64::EmitCtx;
use crate::harness::{
    AllocationMap, LocationMap, compute_edge_source_locations, compute_inst_source_locations,
};
use crate::ir_backend::{BackendBuf, DataRelocInfo, ExternAddrRelocInfo, LinearBackendResult};

use context::{EmitContext, emit_parallel_reg_moves};

/// Compute the base frame offset for spill slots (past callee-saved register save area).
pub fn compute_base_frame(alloc: &AllocatedCfgProgramRa3) -> u32 {
    let extra_saved_pairs = regalloc3_extra_saved_pairs(alloc);
    let is_leaf = alloc.cfg_program.funcs.iter().all(|func| {
        func.insts.iter().all(|inst| {
            !matches!(
                inst.op,
                LinearOp::CallIntrinsic { .. }
                    | LinearOp::CallPure { .. }
                    | LinearOp::CallEffect { .. }
                    | LinearOp::CallLambda { .. }
            )
        })
    });
    let base = if is_leaf {
        crate::arch::x64::LEAF_BASE_FRAME
    } else {
        crate::arch::x64::BASE_FRAME
    };
    base + extra_saved_pairs * 16
}

/// Compile CFG-MIR with regalloc3 allocations to x86_64 machine code.
pub fn compile_regalloc3(alloc: &AllocatedCfgProgramRa3) -> LinearBackendResult {
    let program = &alloc.cfg_program;

    let max_spillslots = alloc
        .functions
        .iter()
        .map(|f| f.num_spillslots)
        .max()
        .unwrap_or(0);

    let extra_saved_pairs = regalloc3_extra_saved_pairs(alloc);

    let is_leaf = program.funcs.iter().all(|func| {
        func.insts.iter().all(|inst| {
            !matches!(
                inst.op,
                LinearOp::CallIntrinsic { .. }
                    | LinearOp::CallPure { .. }
                    | LinearOp::CallEffect { .. }
                    | LinearOp::CallLambda { .. }
            )
        })
    });

    // Count actually-used slots.
    let actual_slot_count = {
        let mut max_slot: Option<u32> = None;
        for func in &program.funcs {
            for inst in &func.insts {
                match &inst.op {
                    LinearOp::WriteToSlot { slot, .. } | LinearOp::ReadFromSlot { slot, .. } => {
                        let s = slot.index() as u32;
                        max_slot = Some(max_slot.map_or(s, |m: u32| m.max(s)));
                    }
                    _ => {}
                }
            }
        }
        max_slot.map_or(0, |m| m + 1)
    };

    let max_edge_args = program
        .funcs
        .iter()
        .flat_map(|func| func.edges.iter().map(|edge| edge.args.len()))
        .max()
        .unwrap_or(0);

    let base_frame = if is_leaf {
        crate::arch::x64::LEAF_BASE_FRAME
    } else {
        crate::arch::x64::BASE_FRAME
    } + extra_saved_pairs * 16;

    let extra_stack = ((max_spillslots + actual_slot_count as usize + max_edge_args) * 8) as u32;
    let mut ectx = EmitCtx::new_regalloc(extra_stack, base_frame, is_leaf);
    let slot_base = ectx.base_frame + (max_spillslots * 8) as u32;
    let edge_tmp_base = slot_base + (actual_slot_count as u32 * 8);

    // Derive ctx_enc from RA-assigned register for data_args[1].
    // Used by ErrorExit (will be removed in 011-3).
    let ctx_enc =
        if let (Some(func), Some(alloc_func)) = (program.funcs.first(), alloc.functions.first()) {
            func.data_args
                .get(1)
                .and_then(|&v| alloc_func.preg_for_vreg(v))
                .map(|p| p.0)
                .unwrap_or(scalar_abi_arg_enc(1))
        } else {
            scalar_abi_arg_enc(1)
        };

    ectx.ctx_enc = ctx_enc;

    // Emit prologue.
    let (entry, error_exit) =
        emit_scalar_prologue(&mut ectx, alloc, program, extra_saved_pairs, is_leaf);

    // Create success exit label.
    let success_exit = ectx.new_label();

    // Compile first function.
    let mut intrinsic_call_sites = Vec::new();
    let mut data_relocs = Vec::<DataRelocInfo>::new();
    let mut extern_addr_relocs = Vec::<ExternAddrRelocInfo>::new();
    if let (Some(func), Some(alloc_func)) = (program.funcs.first(), alloc.functions.first()) {
        let mut const_values = HashMap::new();
        for inst in &func.insts {
            if let LinearOp::Const { dst, value } = &inst.op {
                const_values.insert(*dst, *value);
            }
        }

        let (line_by_op, _) = super::build_debug_line_maps(program);
        let lambda_id = func.lambda_id.index() as u32;
        let line_map: HashMap<cfg_mir::OpId, u32> = line_by_op
            .iter()
            .filter(|((lid, _), _)| *lid == lambda_id)
            .map(|((_, op_id), &line)| (*op_id, line))
            .collect();

        let fused_cmps = fusion::compute_fusable_cmps(func);
        let mut fused_skip = std::collections::HashSet::new();
        let fused_addr_offsets =
            fusion::compute_fusable_addr_offsets(func, alloc_func, &const_values, &mut fused_skip);
        let alloc_map = AllocationMap::from_regalloc3(alloc_func, ectx.base_frame);
        let location_map = LocationMap::from_alloc_map_and_cfg(&alloc_map, program, alloc);
        let edge_source_locations = compute_edge_source_locations(&location_map, program);
        let inst_source_locations = compute_inst_source_locations(&location_map, program);

        let mut ctx = EmitContext {
            ectx: &mut ectx,
            func,
            alloc_func,
            block_labels: HashMap::new(),
            success_exit,
            slot_base,
            edge_tmp_base,
            const_values,
            line_map,
            intrinsic_call_sites: Vec::new(),
            data_relocs: Vec::new(),
            extern_addr_relocs: Vec::new(),
            fused_cmps,
            fused_addr_offsets,
            fused_skip,
            ctx_enc,
            is_last_emitted_block: false,
            edge_trampoline_labels: HashMap::new(),
            edge_source_locations,
            inst_source_locations,
            current_inst: None,
        };

        ctx.emit_function();
        intrinsic_call_sites = ctx.intrinsic_call_sites.clone();
        data_relocs = ctx.data_relocs.clone();
        extern_addr_relocs = ctx.extern_addr_relocs.clone();
    }

    // Bind success exit and emit epilogue.
    ectx.bind_label(success_exit);
    emit_scalar_epilogue(&mut ectx, alloc, program, extra_saved_pairs);
    ectx.emit.bind_label(error_exit).expect("bind error_exit");
    // Error: return 0.
    ectx.emit
        .emit_with(|buf| x64::encode_xor_r32_r32(0, 0, buf))
        .expect("xor rax,rax");
    emit_scalar_epilogue_restore(&mut ectx, extra_saved_pairs);

    // Append data section.
    let mut data_blob_offsets = Vec::new();
    if !program.data_blobs.is_empty() {
        let code_end = ectx.emit.code_len();
        let padding = (8 - (code_end % 8)) % 8;
        if padding > 0 {
            ectx.emit.emit_raw_bytes(&vec![0u8; padding]);
        }
        for blob in &program.data_blobs {
            let offset = ectx.emit.code_len();
            data_blob_offsets.push(offset);
            ectx.emit.emit_raw_bytes(blob);
            let blob_padding = (8 - (blob.len() % 8)) % 8;
            if blob_padding > 0 {
                ectx.emit.emit_raw_bytes(&vec![0u8; blob_padding]);
            }
        }
    }

    let buf = ectx.finalize();

    // Patch data address relocations.
    if !data_relocs.is_empty() {
        let base = buf.exec.as_ptr() as u64;
        for reloc in &data_relocs {
            let blob_offset = data_blob_offsets[reloc.blob_id as usize];
            let addr = base + blob_offset as u64;
            unsafe {
                buf.exec.patch_u64_load(reloc.code_offset, addr);
            }
        }
    }

    // Patch extern addr relocations with in-process values.
    for reloc in &extern_addr_relocs {
        unsafe {
            buf.exec.patch_u64_load(reloc.code_offset, reloc.value);
        }
    }

    let source_map = buf.source_map.clone();
    LinearBackendResult {
        buf: BackendBuf::X86_64(buf),
        entry,
        source_map: if source_map.is_empty() {
            None
        } else {
            Some(source_map)
        },
        backend_debug_info: None,
        asm_program: None,
        intrinsic_call_sites,
        data_relocs,
        extern_addr_relocs,
    }
}

// ── Prologue / Epilogue helpers ──────────────────────────────────────────

/// Emit prologue for a scalar function (no ctx/cursor).
fn emit_scalar_prologue(
    ectx: &mut EmitCtx,
    alloc: &AllocatedCfgProgramRa3,
    program: &cfg_mir::Program,
    extra_saved_pairs: u32,
    _is_leaf: bool,
) -> (u32, x64::LabelId) {
    let entry = ectx.emit.current_offset();
    let error_exit = ectx.emit.new_label();
    let frame_size = ectx.frame_size;

    // push rbp; sub rsp, frame_size
    ectx.emit
        .emit_with(|buf| {
            x64::encode_push_r64(5, buf)?;
            x64::encode_sub_r64_imm32(4, frame_size, buf)
        })
        .expect("scalar prologue frame");

    // Save callee-saved registers used by regalloc.
    emit_save_callee_saved(ectx, extra_saved_pairs);

    // Materialize scalar data_args from ABI registers.
    if let (Some(func), Some(alloc_func)) = (program.funcs.first(), alloc.functions.first()) {
        // Store spilled args first.
        for (i, &arg) in func.data_args.iter().enumerate() {
            let abi_enc = scalar_abi_arg_enc(i);
            if let Some(slot) = alloc_func.spill_slot_for_vreg(arg) {
                let offset = (ectx.base_frame + (slot.0 * 8)) as i32;
                ectx.emit
                    .emit_with(|buf| {
                        x64::encode_mov_m_r64(
                            Mem {
                                base: 4,
                                disp: offset,
                            },
                            abi_enc,
                            buf,
                        )
                    })
                    .expect("store spilled data_arg");
            }
        }

        // Move register-resident args with parallel semantics.
        let mut arg_moves = Vec::new();
        for (i, &arg) in func.data_args.iter().enumerate() {
            let abi_enc = scalar_abi_arg_enc(i);
            if let Some(preg) = alloc_func.preg_for_vreg(arg) {
                let assigned = preg.0;
                if assigned != abi_enc {
                    arg_moves.push((assigned, abi_enc));
                }
            }
        }
        if !arg_moves.is_empty() {
            emit_parallel_reg_moves(ectx, &arg_moves, 10);
        }
    }

    ectx.error_exit = error_exit;
    (entry, error_exit)
}

/// Emit scalar epilogue (move results to return regs, restore, ret).
fn emit_scalar_epilogue(
    ectx: &mut EmitCtx,
    alloc: &AllocatedCfgProgramRa3,
    program: &cfg_mir::Program,
    extra_saved_pairs: u32,
) {
    // Move data_results to rax, rdx.
    if let (Some(func), Some(alloc_func)) = (program.funcs.first(), alloc.functions.first()) {
        let ret_encs: &[u8] = &[0, 2]; // rax, rdx
        for (i, &vreg) in func.data_results.iter().enumerate() {
            if i >= ret_encs.len() {
                break;
            }
            let target = ret_encs[i];
            if let Some(preg) = alloc_func.preg_for_vreg(vreg) {
                let src = preg.0;
                if src != target {
                    ectx.emit
                        .emit_with(|buf| x64::encode_mov_r64_r64(target, src, buf))
                        .expect("mov result");
                }
            } else if let Some(slot) = alloc_func.spill_slot_for_vreg(vreg) {
                let offset = (ectx.base_frame + (slot.0 * 8)) as i32;
                ectx.emit
                    .emit_with(|buf| {
                        x64::encode_mov_r64_m(
                            target,
                            Mem {
                                base: 4,
                                disp: offset,
                            },
                            buf,
                        )
                    })
                    .expect("mov result from spill");
            }
        }
    }

    emit_scalar_epilogue_restore(ectx, extra_saved_pairs);
}

fn emit_scalar_epilogue_restore(ectx: &mut EmitCtx, extra_saved_pairs: u32) {
    emit_restore_callee_saved(ectx, extra_saved_pairs);
    let frame_size = ectx.frame_size;
    ectx.emit
        .emit_with(|buf| {
            x64::encode_add_r64_imm32(4, frame_size, buf)?;
            x64::encode_pop_r64(5, buf)?;
            x64::encode_ret(buf)
        })
        .expect("scalar epilogue");
}

// ── Callee-saved register save/restore ──────────────────────────────────

/// x86_64 callee-saved registers that regalloc3 may assign (beyond the fixed ones):
/// pair 0: r12, r13
/// pair 1: r14, r15
/// pair 2: rbx, rbp
/// (The non-leaf prologue already saves rbp, rbx, r12-r15, so extra_saved_pairs
///  only matters for scalar functions.)
fn callee_saved_pair(pair: u32) -> (u8, u8) {
    match pair {
        0 => (12, 13), // r12, r13
        1 => (14, 15), // r14, r15
        2 => (3, 5),   // rbx, rbp
        _ => panic!("unexpected callee-saved pair index {pair}"),
    }
}

fn emit_save_callee_saved(ectx: &mut EmitCtx, pairs: u32) {
    let mut offset: i32 = 0;
    // Save rbp first (always, for frame pointer).
    ectx.emit
        .emit_with(|buf| {
            x64::encode_mov_m_r64(
                Mem {
                    base: 4,
                    disp: offset,
                },
                5,
                buf,
            )
        })
        .expect("save rbp");
    offset += 8;
    ectx.emit
        .emit_with(|buf| {
            x64::encode_mov_m_r64(
                Mem {
                    base: 4,
                    disp: offset,
                },
                3,
                buf,
            )
        })
        .expect("save rbx");
    offset += 8;

    for i in 0..pairs {
        let (a, b) = callee_saved_pair(i);
        ectx.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: offset,
                    },
                    a,
                    buf,
                )
            })
            .expect("save callee-saved a");
        offset += 8;
        ectx.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: offset,
                    },
                    b,
                    buf,
                )
            })
            .expect("save callee-saved b");
        offset += 8;
    }
}

fn emit_restore_callee_saved(ectx: &mut EmitCtx, pairs: u32) {
    let mut offset: i32 = 0;
    ectx.emit
        .emit_with(|buf| {
            x64::encode_mov_r64_m(
                5,
                Mem {
                    base: 4,
                    disp: offset,
                },
                buf,
            )
        })
        .expect("restore rbp");
    offset += 8;
    ectx.emit
        .emit_with(|buf| {
            x64::encode_mov_r64_m(
                3,
                Mem {
                    base: 4,
                    disp: offset,
                },
                buf,
            )
        })
        .expect("restore rbx");
    offset += 8;

    for i in 0..pairs {
        let (a, b) = callee_saved_pair(i);
        ectx.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    a,
                    Mem {
                        base: 4,
                        disp: offset,
                    },
                    buf,
                )
            })
            .expect("restore callee-saved a");
        offset += 8;
        ectx.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    b,
                    Mem {
                        base: 4,
                        disp: offset,
                    },
                    buf,
                )
            })
            .expect("restore callee-saved b");
        offset += 8;
    }
}

/// Scalar function ABI arg register encoding (SysV: rdi, rsi, rdx, rcx, r8, r9).
fn scalar_abi_arg_enc(index: usize) -> u8 {
    #[cfg(not(windows))]
    const ABI_ARGS: &[u8] = &[7, 6, 2, 1, 8, 9]; // rdi, rsi, rdx, rcx, r8, r9
    #[cfg(windows)]
    const ABI_ARGS: &[u8] = &[1, 2, 8, 9]; // rcx, rdx, r8, r9
    ABI_ARGS[index]
}

/// Count how many extra callee-saved register pairs are used by regalloc3.
fn regalloc3_extra_saved_pairs(alloc: &AllocatedCfgProgramRa3) -> u32 {
    use kajit_mir::regalloc3::linear_scan;

    let mut max_pair = None::<u32>;
    let observe = |a: &linear_scan::Allocation, max_pair: &mut Option<u32>| {
        if let linear_scan::Allocation::Reg(preg) = a {
            let pair = match preg.0 {
                12 | 13 => Some(0),
                14 | 15 => Some(1),
                3 | 5 => Some(2),
                _ => None,
            };
            if let Some(pair) = pair {
                *max_pair = Some(max_pair.map_or(pair, |cur| cur.max(pair)));
            }
        }
    };

    for func in &alloc.functions {
        for (_, a) in &func.allocations {
            observe(a, &mut max_pair);
        }
    }

    max_pair.map_or(0, |p| p + 1)
}
