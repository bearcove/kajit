//! End-to-end integration tests for regalloc3.
//!
//! These tests verify the entire pipeline:
//! 1. Program point construction
//! 2. Liveness analysis
//! 3. Linear scan allocation
//! 4. Spill/reload insertion
//! 5. Verification
//!
//! ## Test Scenarios
//!
//! - Simple allocation (single vreg, no pressure)
//! - Register pressure (multiple vregs, some spills)
//! - Block parameters (edge-specific liveness)
//! - Non-overlapping intervals (register reuse)

use super::*;
use kajit_ir::VReg;
use kajit_lir::LinearOp;
use kajit_reprs::mir::{
    Block, BlockId, Clobbers, Edge, EdgeArg, EdgeId, Function, Inst, InstId, Operand, OperandKind,
    RegClass, Terminator,
};

// Test ABI/scratch policy
const TEST_ABI: machine_inst::AbiInfo = machine_inst::AbiInfo {
    caller_saved_gpr: &[
        machine_inst::PReg(0),
        machine_inst::PReg(1),
        machine_inst::PReg(2),
    ],
    callee_saved_gpr: &[machine_inst::PReg(3), machine_inst::PReg(4)],
    arg_gprs: &[],
    ret_gprs: &[],
    red_zone_size: 0,
};

const TEST_SCRATCH: machine_inst::ScratchPolicy = machine_inst::ScratchPolicy {
    reserved: &[machine_inst::PReg(10)],
    max_simultaneous_spills: 1,
};

#[test]
fn test_end_to_end_simple() {
    // Simple function: v1 = const 42
    let func = Function {
        id: kajit_reprs::mir::FunctionId(0),
        lambda_id: kajit_ir::LambdaId::new(0),
        entry: BlockId(0),
        data_args: vec![],
        data_results: vec![],
        output_size: 0,
        blocks: vec![Block {
            id: BlockId(0),
            params: vec![],
            insts: vec![InstId(0)],
            term: kajit_reprs::mir::TermId(0),
            preds: vec![],
            succs: vec![],
            dead: false,
        }],
        edges: vec![],
        insts: vec![Inst {
            id: InstId(0),
            op: LinearOp::Const {
                dst: VReg::new(1),
                value: 42,
            },
            operands: vec![],
            clobbers: Clobbers::default(),
        }],
        terms: vec![Terminator::Return],
    };

    // Run full pipeline
    let progpoints = progpoint::ProgPointMap::build(&func);
    let liveness = liveness::compute_liveness(&func, &progpoints);
    let hints = hints::HintMap::new();
    let copy_hints = linear_scan::CopyHints::default();
    let alloc = linear_scan::allocate(liveness, &TEST_ABI, &TEST_SCRATCH, &hints, &copy_hints);

    // Should allocate v1 to a register
    assert!(matches!(
        alloc.allocations.get(&VReg::new(1)),
        Some(linear_scan::Allocation::Reg(_))
    ));

    // Verification should pass
    let liveness = liveness::compute_liveness(&func, &progpoints);
    let result = verify::verify(&func, &alloc, &liveness, &TEST_SCRATCH);
    assert!(result.is_ok());
}

#[test]
fn test_end_to_end_with_use() {
    // Function: v1 = const 42; v2 = copy v1
    let func = Function {
        id: kajit_reprs::mir::FunctionId(0),
        lambda_id: kajit_ir::LambdaId::new(0),
        entry: BlockId(0),
        data_args: vec![],
        data_results: vec![],
        output_size: 0,
        blocks: vec![Block {
            id: BlockId(0),
            params: vec![],
            insts: vec![InstId(0), InstId(1)],
            term: kajit_reprs::mir::TermId(0),
            preds: vec![],
            succs: vec![],
            dead: false,
        }],
        edges: vec![],
        insts: vec![
            Inst {
                id: InstId(0),
                op: LinearOp::Const {
                    dst: VReg::new(1),
                    value: 42,
                },
                operands: vec![],
                clobbers: Clobbers::default(),
            },
            Inst {
                id: InstId(1),
                op: LinearOp::Copy {
                    dst: VReg::new(2),
                    src: VReg::new(1),
                },
                operands: vec![Operand {
                    vreg: VReg::new(1),
                    kind: OperandKind::Use,
                    class: RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: Clobbers::default(),
            },
        ],
        terms: vec![Terminator::Return],
    };

    // Run full pipeline
    let progpoints = progpoint::ProgPointMap::build(&func);
    let liveness = liveness::compute_liveness(&func, &progpoints);
    let hints = hints::HintMap::new();
    let copy_hints = linear_scan::CopyHints::default();
    let alloc = linear_scan::allocate(liveness, &TEST_ABI, &TEST_SCRATCH, &hints, &copy_hints);

    // Both should be allocated
    assert!(alloc.allocations.contains_key(&VReg::new(1)));
    assert!(alloc.allocations.contains_key(&VReg::new(2)));

    // Verification should pass
    let liveness = liveness::compute_liveness(&func, &progpoints);
    let result = verify::verify(&func, &alloc, &liveness, &TEST_SCRATCH);
    assert!(result.is_ok());
}

#[test]
fn test_end_to_end_block_params() {
    // Function with block parameter:
    // b0: br b1(v1)
    // b1(v2): ...
    let func = Function {
        id: kajit_reprs::mir::FunctionId(0),
        lambda_id: kajit_ir::LambdaId::new(0),
        entry: BlockId(0),
        data_args: vec![],
        data_results: vec![],
        output_size: 0,
        blocks: vec![
            Block {
                id: BlockId(0),
                params: vec![],
                insts: vec![],
                term: kajit_reprs::mir::TermId(0),
                preds: vec![],
                succs: vec![EdgeId(0)],
                dead: false,
            },
            Block {
                id: BlockId(1),
                params: vec![VReg::new(2)],
                insts: vec![],
                term: kajit_reprs::mir::TermId(1),
                preds: vec![EdgeId(0)],
                succs: vec![],
                dead: false,
            },
        ],
        edges: vec![Edge {
            id: EdgeId(0),
            from: BlockId(0),
            to: BlockId(1),
            args: vec![EdgeArg {
                target: VReg::new(2),
                source: VReg::new(1),
            }],
        }],
        insts: vec![],
        terms: vec![Terminator::Branch { edge: EdgeId(0) }, Terminator::Return],
    };

    // Run full pipeline
    let progpoints = progpoint::ProgPointMap::build(&func);
    let liveness = liveness::compute_liveness(&func, &progpoints);
    let hints = hints::HintMap::new();
    let copy_hints = linear_scan::CopyHints::default();
    let alloc = linear_scan::allocate(liveness, &TEST_ABI, &TEST_SCRATCH, &hints, &copy_hints);

    // Both vregs should be allocated
    assert!(alloc.allocations.contains_key(&VReg::new(1)));
    assert!(alloc.allocations.contains_key(&VReg::new(2)));

    // Verification should pass
    let liveness = liveness::compute_liveness(&func, &progpoints);
    let result = verify::verify(&func, &alloc, &liveness, &TEST_SCRATCH);
    assert!(result.is_ok());
}

#[test]
fn test_end_to_end_register_pressure() {
    // Function with many vregs (more than available registers)
    let func = Function {
        id: kajit_reprs::mir::FunctionId(0),
        lambda_id: kajit_ir::LambdaId::new(0),
        entry: BlockId(0),
        data_args: vec![],
        data_results: vec![],
        output_size: 0,
        blocks: vec![Block {
            id: BlockId(0),
            params: vec![],
            insts: vec![
                InstId(0),
                InstId(1),
                InstId(2),
                InstId(3),
                InstId(4),
                InstId(5),
                InstId(6),
            ],
            term: kajit_reprs::mir::TermId(0),
            preds: vec![],
            succs: vec![],
            dead: false,
        }],
        edges: vec![],
        insts: vec![
            Inst {
                id: InstId(0),
                op: LinearOp::Const {
                    dst: VReg::new(1),
                    value: 1,
                },
                operands: vec![],
                clobbers: Clobbers::default(),
            },
            Inst {
                id: InstId(1),
                op: LinearOp::Const {
                    dst: VReg::new(2),
                    value: 2,
                },
                operands: vec![],
                clobbers: Clobbers::default(),
            },
            Inst {
                id: InstId(2),
                op: LinearOp::Const {
                    dst: VReg::new(3),
                    value: 3,
                },
                operands: vec![],
                clobbers: Clobbers::default(),
            },
            Inst {
                id: InstId(3),
                op: LinearOp::Const {
                    dst: VReg::new(4),
                    value: 4,
                },
                operands: vec![],
                clobbers: Clobbers::default(),
            },
            Inst {
                id: InstId(4),
                op: LinearOp::Const {
                    dst: VReg::new(5),
                    value: 5,
                },
                operands: vec![],
                clobbers: Clobbers::default(),
            },
            Inst {
                id: InstId(5),
                op: LinearOp::Const {
                    dst: VReg::new(6),
                    value: 6,
                },
                operands: vec![],
                clobbers: Clobbers::default(),
            },
            Inst {
                id: InstId(6),
                op: LinearOp::Const {
                    dst: VReg::new(7),
                    value: 7,
                },
                operands: vec![],
                clobbers: Clobbers::default(),
            },
        ],
        terms: vec![Terminator::Return],
    };

    // Run full pipeline
    let progpoints = progpoint::ProgPointMap::build(&func);
    let liveness = liveness::compute_liveness(&func, &progpoints);
    let hints = hints::HintMap::new();
    let copy_hints = linear_scan::CopyHints::default();
    let alloc = linear_scan::allocate(liveness, &TEST_ABI, &TEST_SCRATCH, &hints, &copy_hints);

    // All vregs should have allocations (some may be spilled)
    assert_eq!(alloc.allocations.len(), 7);

    // May have spills (7 vregs, only 5 GPRs available)
    // But this is OK - dead code doesn't cause pressure

    // Verification should pass
    let liveness = liveness::compute_liveness(&func, &progpoints);
    let result = verify::verify(&func, &alloc, &liveness, &TEST_SCRATCH);
    assert!(result.is_ok());
}

#[test]
fn test_end_to_end_with_spill_rewrite() {
    // Test the full pipeline including spill rewrite
    let mut func = Function {
        id: kajit_reprs::mir::FunctionId(0),
        lambda_id: kajit_ir::LambdaId::new(0),
        entry: BlockId(0),
        data_args: vec![],
        data_results: vec![],
        output_size: 0,
        blocks: vec![Block {
            id: BlockId(0),
            params: vec![],
            insts: vec![InstId(0), InstId(1)],
            term: kajit_reprs::mir::TermId(0),
            preds: vec![],
            succs: vec![],
            dead: false,
        }],
        edges: vec![],
        insts: vec![
            Inst {
                id: InstId(0),
                op: LinearOp::Const {
                    dst: VReg::new(1),
                    value: 42,
                },
                operands: vec![],
                clobbers: Clobbers::default(),
            },
            Inst {
                id: InstId(1),
                op: LinearOp::Copy {
                    dst: VReg::new(2),
                    src: VReg::new(1),
                },
                operands: vec![Operand {
                    vreg: VReg::new(1),
                    kind: OperandKind::Use,
                    class: RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: Clobbers::default(),
            },
        ],
        terms: vec![Terminator::Return],
    };

    // Run allocation
    let progpoints = progpoint::ProgPointMap::build(&func);
    let liveness = liveness::compute_liveness(&func, &progpoints);
    let hints = hints::HintMap::new();
    let copy_hints = linear_scan::CopyHints::default();
    let alloc = linear_scan::allocate(liveness, &TEST_ABI, &TEST_SCRATCH, &hints, &copy_hints);

    // Run spill rewrite
    let spill_result = spill_rewrite::rewrite_spills(&mut func, &alloc, &TEST_SCRATCH);

    // If any vregs were spilled, should have spill slots
    if !alloc.spilled.is_empty() {
        assert!(spill_result.num_spill_slots > 0);
    }

    // Verification should still pass (spill rewrite doesn't change liveness)
    let liveness = liveness::compute_liveness(&func, &progpoints);
    let result = verify::verify(&func, &alloc, &liveness, &TEST_SCRATCH);
    assert!(result.is_ok());
}
