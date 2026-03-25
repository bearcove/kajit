//! Regalloc3 allocation results (native types, no regalloc2 conversion).

use crate::cfg_mir::{self};
use crate::regalloc3::{linear_scan, machine_inst::PReg};
use kajit_ir::{LambdaId, VReg};
use std::collections::HashMap;

/// Allocation result for a single CFG function (regalloc3 native format).
#[derive(Debug, Clone)]
pub struct AllocatedCfgFunctionRa3 {
    pub lambda_id: LambdaId,

    /// Number of spill slots needed
    pub num_spillslots: usize,

    /// VReg → Allocation mapping
    pub allocations: HashMap<VReg, linear_scan::Allocation>,

    /// VReg → SpillSlot mapping (for spilled vregs)
    pub spill_slots: HashMap<VReg, crate::regalloc3::spill_rewrite::SpillSlot>,

    /// VReg → constant value (for rematerialization of spilled constants)
    pub rematerializable: HashMap<VReg, u64>,
}

/// Allocation result for entire CFG program (regalloc3 native format).
#[derive(Debug, Clone)]
pub struct AllocatedCfgProgramRa3 {
    /// The CFG program (potentially modified with phi copies)
    pub cfg_program: cfg_mir::Program,

    /// Per-function allocation results
    pub functions: Vec<AllocatedCfgFunctionRa3>,
}

impl AllocatedCfgFunctionRa3 {
    /// Get allocation for a vreg, or None if not allocated (dead vreg)
    pub fn alloc_for_vreg(&self, vreg: VReg) -> Option<linear_scan::Allocation> {
        self.allocations.get(&vreg).copied()
    }

    /// Get physical register for a vreg, or None if spilled or dead
    pub fn preg_for_vreg(&self, vreg: VReg) -> Option<PReg> {
        match self.alloc_for_vreg(vreg)? {
            linear_scan::Allocation::Reg(preg) => Some(preg),
            linear_scan::Allocation::Spill => None,
        }
    }

    /// Get spill slot for a vreg, or None if in register or dead
    pub fn spill_slot_for_vreg(
        &self,
        vreg: VReg,
    ) -> Option<crate::regalloc3::spill_rewrite::SpillSlot> {
        self.spill_slots.get(&vreg).copied()
    }
}
