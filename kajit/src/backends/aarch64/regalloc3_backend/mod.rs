//! aarch64 backend (regalloc3 native types).

mod calls;
mod context;
mod control;
mod fusion;
mod inst;

use kajit_emit::aarch64::{Reg, Width};
use kajit_mir::cfg_mir;
use kajit_mir::regalloc3_result::AllocatedCfgProgramRa3;

use crate::arch::aarch64::EmitCtx;
use crate::harness::{
    AllocationMap, LocationMap, compute_edge_source_locations, compute_inst_source_locations,
};
use crate::ir_backend::{BackendBuf, DataRelocInfo, LinearBackendResult};
use kajit_lir::LinearOp;
use std::collections::HashMap;

use context::{EmitContext, emit_parallel_reg_moves};

struct BfiInfo {
    byte_src: kajit_ir::VReg,
    accum: kajit_ir::VReg,
    lsb: u8,
    width: u8,
}

/// Compute the base frame offset for spill slots (past callee-saved register save area).
/// Used by the lockstep debugger to read spilled vregs from the JIT's stack.
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
        crate::arch::aarch64::LEAF_BASE_FRAME
    } else {
        crate::arch::aarch64::BASE_FRAME
    };
    base + extra_saved_pairs * 16
}

/// Compile CFG-MIR with regalloc3 allocations to aarch64 machine code.
pub fn compile_regalloc3(alloc: &AllocatedCfgProgramRa3) -> LinearBackendResult {
    compile_regalloc3_with_root_data_abi(alloc, crate::compiler::RootDecoderDataAbi::None)
}

pub fn compile_regalloc3_with_root_data_abi(
    alloc: &AllocatedCfgProgramRa3,
    _root_data_abi: crate::compiler::RootDecoderDataAbi,
) -> LinearBackendResult {
    let program = &alloc.cfg_program;

    // Calculate max spillslots and extra callee-saved pairs needed
    let max_spillslots = alloc
        .functions
        .iter()
        .map(|f| f.num_spillslots)
        .max()
        .unwrap_or(0);

    // Check which callee-saved registers are used
    let extra_saved_pairs = regalloc3_extra_saved_pairs(alloc);

    // Detect leaf functions (no bl instructions needed)
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

    // Count actually-used slots (slot_count may be stale after slot_to_reg promotion).
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

    // Create emission context with stack space for spills + actual slots
    let extra_stack = ((max_spillslots + actual_slot_count as usize + max_edge_args) * 8) as u32;
    let mut ectx = EmitCtx::new_regalloc(extra_stack, extra_saved_pairs, is_leaf);
    let slot_base = ectx.base_frame + (max_spillslots * 8) as u32;
    let edge_tmp_base = slot_base + (actual_slot_count * 8);

    // Check if regalloc actually uses x19 or x20 for anything.
    let uses_x19_x20 = alloc.functions.iter().any(|f| {
        f.allocations.values().any(|a| {
            matches!(a, kajit_mir::regalloc3::linear_scan::Allocation::Reg(p) if p.0 == 19 || p.0 == 20)
        })
    });
    let need_save_x19_x20 = if is_leaf {
        uses_x19_x20
    } else {
        // Non-leaf: always save (prologue may use x19/x20).
        true
    };

    let prologue_config = crate::arch::aarch64::PrologueConfig {
        save_x21_x22: !is_leaf,
        save_x19_x20: need_save_x19_x20,
    };

    let is_scalar_function = program.is_scalar;

    // Emit function prologue
    let (entry, error_exit) = if is_scalar_function {
        // Scalar function prologue: frame setup, callee-saved register
        // save, and data_arg moves from ABI registers to RA-assigned registers.
        let entry = ectx.emit.current_offset();
        let error_exit = ectx.emit.new_label();
        let frame_size = ectx.frame_size;

        let saved_pairs: [(Reg, Reg); 3] = [
            (Reg::X23, Reg::X24),
            (Reg::X25, Reg::X26),
            (Reg::X27, Reg::X28),
        ];
        let pairs_to_save = extra_saved_pairs as usize;

        // Allocate frame: sub sp, sp, total_size
        // Frame layout (low to high):
        //   [sp+0]:  FP/LR save (16 bytes, if non-leaf)
        //   [sp+16]: callee-saved pairs (pairs_to_save * 16 bytes)
        //   [sp+16+pairs*16]: spill slots + user slots (frame_size already accounts for these)
        if frame_size > 0 {
            ectx.emit_sub_imm_any(Reg::SP, Reg::SP, frame_size);
        }

        // Save FP/LR (needed for calls)
        let mut offset: i16 = 0;
        ectx.emit
            .emit_stp(Width::X64, Reg::X29, Reg::X30, Reg::SP, offset)
            .expect("stp fp,lr");
        offset += 16;

        // Save callee-saved pairs
        #[allow(clippy::needless_range_loop)]
        for i in 0..pairs_to_save {
            ectx.emit
                .emit_stp(
                    Width::X64,
                    saved_pairs[i].0,
                    saved_pairs[i].1,
                    Reg::SP,
                    offset,
                )
                .expect("stp callee-saved");
            offset += 16;
        }

        // Materialize scalar data_args from ABI registers into their assigned homes.
        // Spilled args must be stored before any register shuffles so later moves
        // cannot clobber their ABI source registers.
        if let Some(alloc_func) = alloc.functions.first()
            && let Some(func) = program.funcs.first()
        {
            for (i, &arg) in func.data_args.iter().enumerate() {
                let abi_reg = Reg::from_raw(i as u8);
                if let Some(slot) = alloc_func.spill_slot_for_vreg(arg) {
                    let offset = ectx.base_frame + (slot.0 * 8);
                    ectx.emit
                        .emit_str_imm(Width::X64, abi_reg, Reg::SP, offset)
                        .expect("str spilled data_arg");
                }
            }

            let mut arg_moves = Vec::new();
            for (i, &arg) in func.data_args.iter().enumerate() {
                let abi_reg = Reg::from_raw(i as u8);
                if let Some(preg) = alloc_func.preg_for_vreg(arg) {
                    let assigned = Reg::from_raw(preg.0);
                    if assigned != abi_reg {
                        arg_moves.push((assigned, abi_reg));
                    }
                }
            }
            if !arg_moves.is_empty() {
                emit_parallel_reg_moves(&mut ectx, &arg_moves, Reg::X16);
            }
        }

        ectx.error_exit = error_exit;
        (entry, error_exit)
    } else {
        ectx.begin_func_with_config(&prologue_config)
    };

    if !is_scalar_function
        && let Some(alloc_func) = alloc.functions.first()
        && let Some(func) = program.funcs.first()
    {
        for (i, &arg) in func.data_args.iter().enumerate() {
            let abi_reg = Reg::from_raw(i as u8 + 2);
            if let Some(slot) = alloc_func.spill_slot_for_vreg(arg) {
                let offset = ectx.base_frame + (slot.0 * 8);
                ectx.emit
                    .emit_str_imm(Width::X64, abi_reg, Reg::SP, offset)
                    .expect("str spilled decoder data_arg");
            }
        }

        let mut arg_moves = Vec::new();
        for (i, &arg) in func.data_args.iter().enumerate() {
            let abi_reg = Reg::from_raw(i as u8 + 2);
            if let Some(preg) = alloc_func.preg_for_vreg(arg) {
                let assigned = Reg::from_raw(preg.0);
                if assigned != abi_reg {
                    arg_moves.push((assigned, abi_reg));
                }
            }
        }
        if !arg_moves.is_empty() {
            emit_parallel_reg_moves(&mut ectx, &arg_moves, Reg::X16);
        }
    }

    // Create success exit label
    let success_exit = ectx.new_label();

    // Compile first function
    let mut intrinsic_call_sites = Vec::new();
    let mut data_relocs = Vec::<DataRelocInfo>::new();
    if let (Some(func), Some(alloc_func)) = (program.funcs.first(), alloc.functions.first()) {
        // Build constant value map for immediate folding
        let mut const_values = HashMap::new();
        for inst in &func.insts {
            if let LinearOp::Const { dst, value } = &inst.op {
                const_values.insert(*dst, *value);
            }
        }

        // Build debug line map for source location tracking
        let (line_by_op, _) = super::build_debug_line_maps(program);
        let lambda_id = func.lambda_id.index() as u32;
        let line_map: HashMap<cfg_mir::OpId, u32> = line_by_op
            .iter()
            .filter(|((lid, _), _)| *lid == lambda_id)
            .map(|((_, op_id), &line)| (*op_id, line))
            .collect();

        let fused_cmps = EmitContext::compute_fusable_cmps(func);
        let (fused_bfi, mut fused_skip) = EmitContext::compute_fusable_bfis(func, &const_values);
        EmitContext::compute_fusable_bit_tests(func, &const_values, &mut fused_skip);
        let fused_addr_offsets = EmitContext::compute_fusable_addr_offsets(
            func,
            alloc_func,
            &const_values,
            &mut fused_skip,
        );
        let alloc_map = AllocationMap::from_regalloc3(alloc_func, ectx.base_frame);
        let location_map = LocationMap::from_alloc_map_and_cfg(&alloc_map, program, alloc);
        let edge_source_locations = compute_edge_source_locations(&location_map, program);
        let inst_source_locations = compute_inst_source_locations(&location_map, program);

        // For leaf functions: keep output_ptr in x0 and ctx_ptr in x1
        // (avoids saving/restoring x21/x22 and the arg moves).
        let (output_reg, ctx_reg) = if is_leaf {
            (Reg::X0, Reg::X1)
        } else {
            (Reg::X21, Reg::X22)
        };

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
            fused_cmps,
            fused_bfi,
            fused_skip,
            fused_addr_offsets,
            output_reg,
            ctx_reg,
            is_last_emitted_block: false,
            edge_trampoline_labels: HashMap::new(),
            edge_source_locations,
            inst_source_locations,
            current_inst: None,
        };

        ctx.emit_function();
        intrinsic_call_sites = ctx.intrinsic_call_sites.clone();
        data_relocs = ctx.data_relocs.clone();
    }

    // Bind success exit and emit epilogue
    ectx.bind_label(success_exit);
    if is_scalar_function {
        // Scalar function epilogue: move data_results to x0, x1, ..., restore frame, ret.
        if let Some(func) = program.funcs.first()
            && let Some(alloc_func) = alloc.functions.first()
        {
            // Resolve each result vreg to its physical location.
            let result_regs: Vec<Option<Reg>> = func
                .data_results
                .iter()
                .map(|&vreg| {
                    if let Some(preg) = alloc_func.preg_for_vreg(vreg) {
                        Some(Reg::from_raw(preg.0))
                    } else if let Some(slot) = alloc_func.spill_slot_for_vreg(vreg) {
                        // Load spilled values into scratch first.
                        let offset = ectx.base_frame + (slot.0 * 8);
                        ectx.emit
                            .emit_ldr_imm(Width::X64, Reg::X16, Reg::SP, offset)
                            .expect("ldr result from spill");
                        Some(Reg::X16)
                    } else {
                        None
                    }
                })
                .collect();

            // Emit parallel move: check if any target is a source for a
            // later move and use x9 as scratch to break cycles.
            let n = result_regs.len();
            let mut done = vec![false; n];
            for round in 0..n + 1 {
                let mut progress = false;
                for i in 0..n {
                    if done[i] {
                        continue;
                    }
                    let target = Reg::from_raw(i as u8);
                    let Some(src) = result_regs[i] else {
                        done[i] = true;
                        continue;
                    };
                    if src == target {
                        done[i] = true;
                        progress = true;
                        continue;
                    }
                    // Check if target is needed as source by an undone move.
                    let blocked =
                        (0..n).any(|j| !done[j] && j != i && result_regs[j] == Some(target));
                    if !blocked || round == n {
                        // If blocked on last round, use scratch to break cycle.
                        if blocked {
                            // Save the blocking value through scratch.
                            let blocker = (0..n)
                                .find(|&j| !done[j] && j != i && result_regs[j] == Some(target))
                                .unwrap();
                            ectx.emit
                                .emit_mov_reg(Width::X64, Reg::X16, target)
                                .expect("mov scratch");
                            ectx.emit
                                .emit_mov_reg(Width::X64, target, src)
                                .expect("mov result");
                            ectx.emit
                                .emit_mov_reg(Width::X64, Reg::from_raw(blocker as u8), Reg::X16)
                                .expect("mov from scratch");
                            done[i] = true;
                            done[blocker] = true;
                        } else {
                            ectx.emit
                                .emit_mov_reg(Width::X64, target, src)
                                .expect("mov result to return reg");
                            done[i] = true;
                        }
                        progress = true;
                    }
                }
                if done.iter().all(|&d| d) || !progress {
                    break;
                }
            }
        }
        // Restore callee-saved registers and tear down frame.
        let saved_pairs: [(Reg, Reg); 3] = [
            (Reg::X23, Reg::X24),
            (Reg::X25, Reg::X26),
            (Reg::X27, Reg::X28),
        ];
        let pairs_to_save = extra_saved_pairs as usize;
        let frame_size = ectx.frame_size;

        let emit_scalar_epilogue = |ectx: &mut EmitCtx| {
            let mut offset: i16 = 0;
            ectx.emit
                .emit_ldp(Width::X64, Reg::X29, Reg::X30, Reg::SP, offset)
                .expect("ldp fp,lr");
            offset += 16;
            #[allow(clippy::needless_range_loop)]
            for i in 0..pairs_to_save {
                ectx.emit
                    .emit_ldp(
                        Width::X64,
                        saved_pairs[i].0,
                        saved_pairs[i].1,
                        Reg::SP,
                        offset,
                    )
                    .expect("ldp callee-saved");
                offset += 16;
            }
            if frame_size > 0 {
                ectx.emit_add_imm_any(Reg::SP, Reg::SP, frame_size);
            }
            ectx.emit.emit_ret().expect("ret");
        };

        emit_scalar_epilogue(&mut ectx);

        // Bind error exit (just returns 0 for now).
        ectx.emit.bind_label(error_exit).expect("bind error_exit");
        let zero = Reg::XZR;
        ectx.emit
            .emit_mov_reg(Width::X64, Reg::X0, zero)
            .expect("mov x0, xzr");
        emit_scalar_epilogue(&mut ectx);
    } else {
        ectx.end_func_with_config(error_exit, &prologue_config);
        // Emit shared error trampolines after the epilogue (cold, unreachable
        // from the success/error return paths — only reached via error-site branches)
        ectx.emit_error_trampolines();
    }

    // Append data section to the code buffer (before finalization so it's
    // included in the mmap'd executable buffer).
    let mut data_blob_offsets = Vec::new();
    if !program.data_blobs.is_empty() {
        // Align data section start to 8 bytes.
        let code_end = ectx.emit.code_len();
        let padding = (8 - (code_end % 8)) % 8;
        if padding > 0 {
            ectx.emit.emit_raw_bytes(&vec![0u8; padding]);
        }
        for blob in &program.data_blobs {
            let offset = ectx.emit.code_len();
            data_blob_offsets.push(offset);
            ectx.emit.emit_raw_bytes(blob);
            // Align each blob to 8 bytes.
            let blob_padding = (8 - (blob.len() % 8)) % 8;
            if blob_padding > 0 {
                ectx.emit.emit_raw_bytes(&vec![0u8; blob_padding]);
            }
        }
    }

    // Finalize (resolves branch fixups, creates executable buffer)
    let (buf, asm_program) = ectx.finalize();

    // Patch data address relocations with actual runtime addresses.
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

    let source_map = buf.source_map.clone();
    LinearBackendResult {
        buf: BackendBuf::Aarch64(buf),
        entry,
        source_map: if source_map.is_empty() {
            None
        } else {
            Some(source_map)
        },
        backend_debug_info: None,
        asm_program,
        intrinsic_call_sites,
        data_relocs,
    }
}

/// Count how many callee-saved register pairs (x23/x24, x25/x26, x27/x28) are used.
fn regalloc3_extra_saved_pairs(alloc: &AllocatedCfgProgramRa3) -> u32 {
    use kajit_mir::regalloc3::linear_scan;

    let mut max_pair = None::<u32>;
    let mut observe = |a: &linear_scan::Allocation| {
        if let linear_scan::Allocation::Reg(preg) = a {
            let pair = match preg.0 {
                23 | 24 => Some(0),
                25 | 26 => Some(1),
                27 | 28 => Some(2),
                _ => None,
            };
            if let Some(pair) = pair {
                max_pair = Some(max_pair.map_or(pair, |cur| cur.max(pair)));
            }
        }
    };

    for func in &alloc.functions {
        for a in func.allocations.values() {
            observe(a);
        }
    }

    max_pair.map_or(0, |p| p + 1)
}
