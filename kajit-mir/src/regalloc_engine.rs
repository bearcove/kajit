//! Register allocation over canonical CFG MIR (regalloc3 native SSA coloring).

use std::collections::HashMap;
use std::fmt;

use crate::cfg_mir;
use crate::regalloc3::*;
use crate::regalloc3_result::*;

/// Errors from register allocation.
#[derive(Debug, Clone)]
pub enum RegallocEngineError {
    Checker(String),
}

impl fmt::Display for RegallocEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checker(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for RegallocEngineError {}

/// Run regalloc3 (native allocator) over canonical CFG MIR.
/// Returns native regalloc3 allocation types.
pub fn allocate_cfg_program_regalloc3_native(
    program: &cfg_mir::Program,
) -> Result<AllocatedCfgProgramRa3, RegallocEngineError> {
    program
        .validate()
        .map_err(|err| RegallocEngineError::Checker(err.to_string()))?;

    // Get ABI info for current architecture
    // Reserved: x9-x11 (backend scratch), x16 (phi-copy temp), x17 (IP1), x18 (platform), x19-x22 (cursor/ctx)
    let abi = &machine_inst::AbiInfo {
        #[cfg(target_arch = "aarch64")]
        caller_saved_gpr: &[
            // x0-x8, x12-x15 (excluding x9-x11 scratch, x16 phi-copy temp,
            // x17 reserved IP1, x18 platform)
            machine_inst::PReg(0),
            machine_inst::PReg(1),
            machine_inst::PReg(2),
            machine_inst::PReg(3),
            machine_inst::PReg(4),
            machine_inst::PReg(5),
            machine_inst::PReg(6),
            machine_inst::PReg(7),
            machine_inst::PReg(8),
            machine_inst::PReg(12),
            machine_inst::PReg(13),
            machine_inst::PReg(14),
            machine_inst::PReg(15),
        ],
        #[cfg(target_arch = "aarch64")]
        callee_saved_gpr: &[
            // x19-x22 reserved for cursor/ctx, only x23-x28 allocatable
            machine_inst::PReg(23),
            machine_inst::PReg(24),
            machine_inst::PReg(25),
            machine_inst::PReg(26),
            machine_inst::PReg(27),
            machine_inst::PReg(28),
        ],
        #[cfg(target_arch = "x86_64")]
        caller_saved_gpr: &[
            machine_inst::PReg(0),
            machine_inst::PReg(1),
            machine_inst::PReg(2),
            machine_inst::PReg(6),
            machine_inst::PReg(7),
            machine_inst::PReg(8),
            machine_inst::PReg(9),
            machine_inst::PReg(10),
            machine_inst::PReg(11),
        ],
        #[cfg(target_arch = "x86_64")]
        callee_saved_gpr: &[
            machine_inst::PReg(3),
            machine_inst::PReg(5),
            machine_inst::PReg(12),
            machine_inst::PReg(13),
            machine_inst::PReg(14),
            machine_inst::PReg(15),
        ],
        arg_gprs: &[],
        ret_gprs: &[],
        red_zone_size: 0,
    };

    let scratch = &machine_inst::ScratchPolicy {
        #[cfg(target_arch = "aarch64")]
        reserved: &[machine_inst::PReg(31)],
        #[cfg(target_arch = "x86_64")]
        reserved: &[machine_inst::PReg(10), machine_inst::PReg(11)], // r10, r11 scratch
        max_simultaneous_spills: 2,
    };

    let mut functions = Vec::with_capacity(program.funcs.len());
    let mut modified_funcs = Vec::with_capacity(program.funcs.len());

    for func in &program.funcs {
        let mut func_mut = func.clone();

        // NOTE: critical edge splitting is deferred to after allocation,
        // together with phi copy insertion. The copy insertion function
        // handles edge placement correctly.

        // SSA-first allocation: do NOT insert phi copies before RA.
        // Instead, build copy hints from edge args so the allocator prefers
        // to assign the same register to both sides of each phi connection.
        // Phi copies are inserted AFTER allocation, only where needed.
        let copy_hints = linear_scan::CopyHints::build(&func_mut);
        let progpoints = progpoint::ProgPointMap::build(&func_mut);
        let liveness = liveness::compute_liveness(&func_mut, &progpoints);

        // Enrich hints: mark constants as rematerializable (cheap to spill)
        let mut hints = program.hints.clone();
        for inst in &func_mut.insts {
            if let kajit_lir::LinearOp::Const { dst, .. } = &inst.op {
                hints.entry(*dst).or_default().spill_cost = hints::SpillCost::Rematerializable;
            }
        }

        // Pre-color data_args to their ABI argument registers.
        let mut fixed_colors = std::collections::HashMap::new();
        for (i, &arg_vreg) in func.data_args.iter().enumerate() {
            let abi_preg = machine_inst::PReg(i as u8);
            if !program.extra_excluded_regs.contains(&abi_preg) {
                fixed_colors.insert(arg_vreg, abi_preg);
            }
        }

        let mut alloc_result = crate::regalloc3::ssa_coloring::allocate_with_excluded(
            &func_mut,
            &liveness,
            abi,
            scratch,
            &hints,
            &copy_hints,
            &program.extra_excluded_regs,
            &fixed_colors,
        );

        // SSA destruction: split critical edges then insert copies for phi edges.
        // temp_vreg is assigned to a reserved scratch register for cycle breaking.
        let temp_vreg = kajit_ir::VReg::new(program.vreg_count);
        #[cfg(target_arch = "aarch64")]
        let temp_scratch = crate::regalloc3::machine_inst::PReg(16); // x16 = IP0 scratch
        #[cfg(target_arch = "x86_64")]
        let temp_scratch = crate::regalloc3::machine_inst::PReg(10); // r10 scratch
        alloc_result
            .allocations
            .insert(temp_vreg, linear_scan::Allocation::Reg(temp_scratch));

        let force_all_copies = std::env::var("KAJIT_NO_COALESCE").is_ok()
            || std::env::var("KAJIT_FORCE_ALL_COPIES").is_ok();
        if force_all_copies {
            critical_edge::split_critical_edges(&mut func_mut);
            phi_resolution::insert_phi_copies(&mut func_mut, temp_vreg);
        } else {
            insert_phi_copies_with_coalescing(&mut func_mut, &alloc_result, temp_vreg);
        }

        // Assign spill slots
        let mut spill_slots = HashMap::new();
        let mut next_slot = 0u32;
        for &vreg in &alloc_result.spilled {
            spill_slots.insert(vreg, spill_rewrite::SpillSlot(next_slot));
            next_slot += 1;
        }

        // Build rematerialization map: spilled constants can be re-emitted as movz
        let mut rematerializable = HashMap::new();
        for inst in &func_mut.insts {
            if let kajit_lir::LinearOp::Const { dst, value } = &inst.op
                && alloc_result.allocations.get(dst) == Some(&linear_scan::Allocation::Spill)
            {
                rematerializable.insert(*dst, *value);
            }
        }

        functions.push(AllocatedCfgFunctionRa3 {
            lambda_id: func_mut.lambda_id,
            num_spillslots: next_slot as usize,
            allocations: alloc_result.allocations.clone(),
            spill_slots,
            rematerializable,
            edits: alloc_result.edits,
        });

        modified_funcs.push(func_mut);
    }

    // Build modified program with phi copies inserted
    let modified_program = cfg_mir::Program {
        funcs: modified_funcs,
        vreg_count: program.vreg_count,
        slot_count: program.slot_count,
        param_slot_count: program.param_slot_count,
        debug: program.debug.clone(),
        hints: program.hints.clone(),
        extra_excluded_regs: program.extra_excluded_regs.clone(),
        data_blobs: program.data_blobs.clone(),
        stack_allocs: program.stack_allocs.clone(),
        data_arg_layouts: program.data_arg_layouts.clone(),
    };

    Ok(AllocatedCfgProgramRa3 {
        cfg_program: modified_program,
        functions,
    })
}

/// Insert phi copies AFTER register allocation, skipping copies where
/// the allocator already assigned the same physical register to both sides.
///
/// This is SSA destruction with coalescing: edge args that got the same
/// register need no copy, edge args that got different registers (or
/// involve spilled values) need explicit Copy instructions.
fn insert_phi_copies_with_coalescing(
    func: &mut cfg_mir::Function,
    alloc_result: &crate::regalloc3::linear_scan::AllocationResult,
    temp_vreg: kajit_ir::VReg,
) {
    use crate::regalloc3::linear_scan::Allocation as Ra3Alloc;
    use crate::regalloc3::machine_inst::PReg;
    use crate::regalloc3::parallel_copy::Copy;

    // Callee-saved registers survive calls — any other register may be clobbered.
    let callee_saved_set: std::collections::HashSet<PReg> = [
        #[cfg(target_arch = "aarch64")]
        &[PReg(23), PReg(24), PReg(25), PReg(26), PReg(27), PReg(28)][..],
        #[cfg(target_arch = "x86_64")]
        &[PReg(3), PReg(5), PReg(12), PReg(13), PReg(14), PReg(15)][..],
    ]
    .into_iter()
    .flatten()
    .copied()
    .collect();

    // Split critical edges first so copies can be placed on specific edges
    crate::regalloc3::critical_edge::split_critical_edges(func);

    for edge_idx in 0..func.edges.len() {
        let edge_id = cfg_mir::EdgeId(edge_idx as u32);
        let edge = &func.edges[edge_idx];

        // Skip dead edges (from critical edge splitting) and edges with no args
        if edge.from.0 == u32::MAX || edge.args.is_empty() {
            continue;
        }

        // Check if the predecessor block has a clobbering instruction (call).
        // If so, caller-saved registers may have been destroyed even if source
        // and target share the same register.
        let pred_block = &func.blocks[edge.from.index()];
        let pred_has_clobber = pred_block
            .insts
            .iter()
            .any(|inst_id| func.insts[inst_id.0 as usize].clobbers.caller_saved_gpr);

        // Build parallel copies, but SKIP coalesced pairs (same register)
        // UNLESS the register was clobbered by a call in the predecessor block.
        let copies: Vec<Copy> = edge
            .args
            .iter()
            .filter(|arg| {
                if arg.target == arg.source {
                    return false; // identity
                }
                // Check if both sides got the same physical register
                let target_alloc = alloc_result.allocations.get(&arg.target);
                let source_alloc = alloc_result.allocations.get(&arg.source);
                match (target_alloc, source_alloc) {
                    (Some(Ra3Alloc::Reg(t)), Some(Ra3Alloc::Reg(s))) if t == s => {
                        // Same register — but was it clobbered by a call?
                        if pred_has_clobber && !callee_saved_set.contains(t) {
                            true // caller-saved register clobbered, need copy
                        } else {
                            false // genuinely coalesced
                        }
                    }
                    _ => true, // different registers or spilled, need copy
                }
            })
            .map(|arg| Copy {
                dst: arg.target,
                src: arg.source,
            })
            .collect();

        if copies.is_empty() {
            continue;
        }

        // Unified location-based parallel copy resolution.
        //
        // All copies — reg→reg, reg→spill, spill→reg, spill→spill — go through
        // one resolver that sees the full dependency graph on physical locations.
        // This prevents lost-copy bugs where a Reg→Spill copy's source register
        // is clobbered by a Reg→Reg copy that runs before it.
        use crate::regalloc3::parallel_copy::LocationCopy;

        let location_copies: Vec<LocationCopy> = copies
            .iter()
            .filter_map(|c| {
                let src_loc = vreg_location(c.src, alloc_result)?;
                let dst_loc = vreg_location(c.dst, alloc_result)?;
                Some(LocationCopy {
                    dst_loc,
                    src_loc,
                    dst_vreg: c.dst,
                    src_vreg: c.src,
                })
            })
            .collect();

        let resolved_moves =
            crate::regalloc3::parallel_copy::resolve_location_copies(&location_copies, temp_vreg);

        crate::regalloc3::phi_resolution::insert_moves_on_edge(func, edge_id, &resolved_moves);
    }
}

/// Map a vreg to its physical location (register or stack).
///
/// For spilled vregs, we use the vreg index as a unique stack "slot ID".
/// The resolver only needs location identity for dependency tracking — it
/// doesn't need the actual byte offset. Each spilled vreg has a unique slot,
/// so using the vreg index as the slot ID is correct.
fn vreg_location(
    vreg: kajit_ir::VReg,
    alloc: &crate::regalloc3::linear_scan::AllocationResult,
) -> Option<crate::regalloc3::parallel_copy::Location> {
    use crate::regalloc3::linear_scan::Allocation as Ra3Alloc;
    use crate::regalloc3::parallel_copy::Location;
    match alloc.allocations.get(&vreg)? {
        Ra3Alloc::Reg(preg) => Some(Location::Reg(*preg)),
        Ra3Alloc::Spill => Some(Location::Stack(vreg.index() as u32)),
    }
}
