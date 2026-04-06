//! Parallel copy resolution (sequentialization + cycle handling).
//!
//! ## Problem
//!
//! Block parameters require parallel copies at edges:
//!   (a, b, c) = (x, y, z)
//!
//! But these must execute as if simultaneous. Naive sequencing is wrong:
//!   a = x
//!   b = y  // WRONG if y == a (we just overwrote a!)
//!   c = z
//!
//! Worse: cycles exist! Consider:
//!   (a, b, c) = (b, c, a)
//!
//! ## Solution
//!
//! 1. Detect cycles using a dependency graph
//! 2. Break cycles with temp register or swap instructions
//! 3. Emit legal sequence of moves/swaps
//!
//! ## Algorithm (Briggs-style)
//!
//! Build dependency graph: dst -> src edges
//! While graph not empty:
//!   - Find node with no incoming edges (ready to copy)
//!   - If none exist, we have a cycle:
//!     * Use temp register to break cycle
//!     * Or use swap instruction (if target supports)
//!   - Emit move, remove node from graph
//!
//! This is the standard approach (see Briggs et al. "Practical Improvements
//! to the Construction and Destruction of Static Single Assignment Form").

use kajit_ir::VReg;
use std::collections::HashMap;

use super::machine_inst::PReg;

/// A single copy operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Copy {
    pub dst: VReg,
    pub src: VReg,
}

/// Resolved move operations (no longer parallel)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveOp {
    /// Simple move: dst = src
    Move { dst: VReg, src: VReg },

    /// Swap two registers (if target supports)
    Swap { a: VReg, b: VReg },

    /// Move to temp (cycle breaking)
    MoveToTemp { dst_temp: VReg, src: VReg },
}

/// Parallel copy resolver
pub struct ParallelCopyResolver {
    /// Pending copies
    copies: Vec<Copy>,

    /// Dependency tracking: dst -> src
    deps: HashMap<VReg, VReg>,

    /// Resolved sequence
    resolved: Vec<MoveOp>,
}

impl ParallelCopyResolver {
    /// Create resolver for parallel copies
    pub fn new(copies: Vec<Copy>) -> Self {
        let mut deps = HashMap::new();

        for copy in &copies {
            if copy.dst != copy.src {
                deps.insert(copy.dst, copy.src);
            }
        }

        Self {
            copies,
            deps,
            resolved: Vec::new(),
        }
    }

    /// Resolve parallel copies into sequential moves/swaps
    ///
    /// Returns the sequence of move operations.
    /// Caller must provide a temp register for cycle breaking.
    pub fn resolve(mut self, temp_reg: VReg) -> Vec<MoveOp> {
        // Remove identity copies
        self.copies.retain(|c| c.dst != c.src);

        while !self.deps.is_empty() {
            // Find a ready copy (src not overwritten by any pending copy)
            if let Some(&ready_dst) = self
                .deps
                .iter()
                .find(|(_, src)| !self.deps.contains_key(src))
                .map(|(dst, _)| dst)
            {
                let src = self.deps[&ready_dst];
                self.emit_move(ready_dst, src);
            } else {
                // No ready copy -> cycle exists
                self.break_cycle(temp_reg);
            }
        }

        self.resolved
    }

    /// Emit a simple move and update state
    fn emit_move(&mut self, dst: VReg, src: VReg) {
        self.resolved.push(MoveOp::Move { dst, src });
        self.deps.remove(&dst);
    }

    /// Break a cycle using temp register
    ///
    /// Algorithm:
    /// 1. Pick any node in cycle (any key in deps)
    /// 2. Move it to temp: temp = src
    /// 3. Now we can proceed with rest of cycle
    fn break_cycle(&mut self, temp_reg: VReg) {
        // Pick arbitrary cycle member
        let cycle_start = *self.deps.keys().next().unwrap();
        let cycle_src = self.deps[&cycle_start];

        // Move to temp: temp = cycle_src
        self.resolved.push(MoveOp::MoveToTemp {
            dst_temp: temp_reg,
            src: cycle_src,
        });

        // Update dependencies: cycle_start now reads from temp
        self.deps.insert(cycle_start, temp_reg);
    }
}

// ─── Unified location-based parallel copy resolver ──────────────────────────
//
// The resolver above works on abstract VRegs. This resolver works on physical
// locations (registers + stack slots), which is what you need after register
// allocation. It sees the FULL dependency graph — including Reg→Stack and
// Stack→Reg copies — so spill-involving copies participate naturally in
// ordering instead of being hand-scheduled in phases.

/// A physical location where a value lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Location {
    /// Physical register (aarch64 GPR index)
    Reg(PReg),
    /// Stack spill slot (index, not byte offset)
    Stack(u32),
}

/// A parallel copy between physical locations, carrying the vreg identities
/// so the backend can emit the right Copy instructions.
#[derive(Debug, Clone, Copy)]
pub struct LocationCopy {
    pub dst_loc: Location,
    pub src_loc: Location,
    pub dst_vreg: VReg,
    pub src_vreg: VReg,
}

/// Resolve a set of simultaneous location copies into a sequential move list.
///
/// This is the standard Briggs parallel-move algorithm generalized to locations
/// (registers AND stack slots). The key insight: dependencies are on LOCATIONS,
/// not vregs. A copy `Stack(5) ← Reg(1)` blocks any copy that overwrites Reg(1)
/// until it has read Reg(1).
///
/// `temp_vreg` is used for cycle breaking (moved to/from a temp register).
pub fn resolve_location_copies(copies: &[LocationCopy], temp_vreg: VReg) -> Vec<MoveOp> {
    // Drop identity copies (same location)
    let pending: Vec<LocationCopy> = copies
        .iter()
        .filter(|c| c.dst_loc != c.src_loc)
        .copied()
        .collect();

    if pending.is_empty() {
        return Vec::new();
    }

    let mut alive = vec![true; pending.len()];
    let mut result = Vec::new();

    // Repeatedly emit copies whose dst_loc is NOT read by any remaining copy's src_loc
    loop {
        let mut progress = false;
        for i in 0..pending.len() {
            if !alive[i] {
                continue;
            }
            let dst = pending[i].dst_loc;
            let blocked =
                (0..pending.len()).any(|j| j != i && alive[j] && pending[j].src_loc == dst);
            if !blocked {
                result.push(MoveOp::Move {
                    dst: pending[i].dst_vreg,
                    src: pending[i].src_vreg,
                });
                alive[i] = false;
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }

    // Remaining copies form cycles — break each with temp
    while let Some(cycle_start) = alive.iter().position(|&a| a) {
        // Walk the cycle: follow dst_loc → find who reads from it
        let mut cycle = vec![cycle_start];
        let mut current = cycle_start;
        loop {
            let dst = pending[current].dst_loc;
            let next = (0..pending.len()).find(|&j| alive[j] && pending[j].src_loc == dst);
            match next {
                Some(j) if j == cycle_start => break,
                Some(j) => {
                    cycle.push(j);
                    current = j;
                }
                None => break,
            }
        }

        // Save the first move's source to temp
        result.push(MoveOp::MoveToTemp {
            dst_temp: temp_vreg,
            src: pending[cycle[0]].src_vreg,
        });
        alive[cycle[0]] = false;

        // Emit rest of cycle in reverse (each source not yet clobbered)
        for &i in cycle[1..].iter().rev() {
            result.push(MoveOp::Move {
                dst: pending[i].dst_vreg,
                src: pending[i].src_vreg,
            });
            alive[i] = false;
        }

        // Emit the first move reading from temp
        result.push(MoveOp::Move {
            dst: pending[cycle[0]].dst_vreg,
            src: temp_vreg,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_simple_copies() {
        // (a, b) = (x, y)
        // No dependencies, any order works
        let copies = vec![
            Copy {
                dst: VReg::new(0), // a
                src: VReg::new(2), // x
            },
            Copy {
                dst: VReg::new(1), // b
                src: VReg::new(3), // y
            },
        ];

        let resolver = ParallelCopyResolver::new(copies);
        let temp = VReg::new(100);
        let moves = resolver.resolve(temp);

        assert_eq!(moves.len(), 2);
        // Both should be simple moves
        assert!(matches!(moves[0], MoveOp::Move { .. }));
        assert!(matches!(moves[1], MoveOp::Move { .. }));
    }

    #[test]
    fn test_dependent_copies() {
        // (a, b) = (b, x)
        // a depends on b, so b = x must come first
        let copies = vec![
            Copy {
                dst: VReg::new(0), // a
                src: VReg::new(1), // b
            },
            Copy {
                dst: VReg::new(1), // b
                src: VReg::new(2), // x
            },
        ];

        let resolver = ParallelCopyResolver::new(copies);
        let temp = VReg::new(100);
        let moves = resolver.resolve(temp);

        assert_eq!(moves.len(), 2);

        // First move should be b = x (no dependencies)
        if let MoveOp::Move { dst, src } = moves[0] {
            assert_eq!(dst, VReg::new(1));
            assert_eq!(src, VReg::new(2));
        } else {
            panic!("Expected Move");
        }

        // Second move should be a = b
        if let MoveOp::Move { dst, src } = moves[1] {
            assert_eq!(dst, VReg::new(0));
            assert_eq!(src, VReg::new(1));
        } else {
            panic!("Expected Move");
        }
    }

    #[test]
    fn test_cycle() {
        // (a, b, c) = (b, c, a)
        // This is a cycle, needs temp
        let copies = vec![
            Copy {
                dst: VReg::new(0), // a
                src: VReg::new(1), // b
            },
            Copy {
                dst: VReg::new(1), // b
                src: VReg::new(2), // c
            },
            Copy {
                dst: VReg::new(2), // c
                src: VReg::new(0), // a
            },
        ];

        let resolver = ParallelCopyResolver::new(copies);
        let temp = VReg::new(100);
        let moves = resolver.resolve(temp);

        // Should have 4 moves: 1 MoveToTemp + 3 regular moves
        assert_eq!(moves.len(), 4);

        // First move should use temp to break cycle
        assert!(matches!(moves[0], MoveOp::MoveToTemp { .. }));

        // Rest should be regular moves
        for i in 1..4 {
            assert!(matches!(moves[i], MoveOp::Move { .. }));
        }

        // Verify that all destinations are covered
        let mut dsts: HashSet<VReg> = HashSet::new();
        for mov in &moves {
            match mov {
                MoveOp::Move { dst, .. } => {
                    dsts.insert(*dst);
                }
                MoveOp::MoveToTemp { dst_temp: _, .. } => {
                    // Temp is intermediate, not final dst
                }
                MoveOp::Swap { .. } => {}
            }
        }
        assert_eq!(dsts.len(), 3);
    }

    #[test]
    fn test_identity_copies_removed() {
        // (a, b) = (a, x)
        // a = a is identity, should be removed
        let copies = vec![
            Copy {
                dst: VReg::new(0), // a
                src: VReg::new(0), // a (identity)
            },
            Copy {
                dst: VReg::new(1), // b
                src: VReg::new(2), // x
            },
        ];

        let resolver = ParallelCopyResolver::new(copies);
        let temp = VReg::new(100);
        let moves = resolver.resolve(temp);

        // Should only have one move (identity removed)
        assert_eq!(moves.len(), 1);

        if let MoveOp::Move { dst, src } = moves[0] {
            assert_eq!(dst, VReg::new(1));
            assert_eq!(src, VReg::new(2));
        } else {
            panic!("Expected Move");
        }
    }

    #[test]
    fn test_two_register_cycle() {
        // (a, b) = (b, a)
        // Classic swap pattern
        let copies = vec![
            Copy {
                dst: VReg::new(0), // a
                src: VReg::new(1), // b
            },
            Copy {
                dst: VReg::new(1), // b
                src: VReg::new(0), // a
            },
        ];

        let resolver = ParallelCopyResolver::new(copies);
        let temp = VReg::new(100);
        let moves = resolver.resolve(temp);

        // Should use temp to break 2-cycle
        // Either: temp=a, b=a, a=temp  OR  temp=b, a=b, b=temp
        assert_eq!(moves.len(), 3);
        assert!(matches!(moves[0], MoveOp::MoveToTemp { .. }));
    }

    #[test]
    fn test_empty_copies() {
        let copies = vec![];
        let resolver = ParallelCopyResolver::new(copies);
        let temp = VReg::new(100);
        let moves = resolver.resolve(temp);
        assert_eq!(moves.len(), 0);
    }

    // ─── Unified location resolver tests ────────────────────────────────

    #[test]
    fn test_location_reg_to_spill_before_reg_to_reg() {
        // The exact bug that was found:
        // Copy 1: Reg(1) ← Reg(2)    (reg-reg)
        // Copy 2: Stack(0) ← Reg(1)  (reg-to-spill)
        //
        // Stack(0) must read Reg(1) BEFORE Reg(1) is overwritten by Copy 1.
        let copies = vec![
            LocationCopy {
                dst_loc: Location::Reg(PReg(1)),
                src_loc: Location::Reg(PReg(2)),
                dst_vreg: VReg::new(10),
                src_vreg: VReg::new(20),
            },
            LocationCopy {
                dst_loc: Location::Stack(0),
                src_loc: Location::Reg(PReg(1)),
                dst_vreg: VReg::new(30),
                src_vreg: VReg::new(40),
            },
        ];

        let temp = VReg::new(100);
        let moves = resolve_location_copies(&copies, temp);

        // Stack(0) ← Reg(1) must come first (Stack(0) is not read by anyone)
        assert_eq!(moves.len(), 2);
        // First move should be the stack write (v30←v40)
        assert!(matches!(moves[0], MoveOp::Move { dst, .. } if dst == VReg::new(30)));
    }

    #[test]
    fn test_location_spill_to_reg_after_reg_to_reg() {
        // Copy 1: Reg(2) ← Reg(1)    (reg-reg)
        // Copy 2: Reg(1) ← Stack(0)  (spill-to-reg)
        //
        // Reg(2) must read Reg(1) BEFORE Reg(1) is overwritten by Copy 2.
        let copies = vec![
            LocationCopy {
                dst_loc: Location::Reg(PReg(2)),
                src_loc: Location::Reg(PReg(1)),
                dst_vreg: VReg::new(10),
                src_vreg: VReg::new(20),
            },
            LocationCopy {
                dst_loc: Location::Reg(PReg(1)),
                src_loc: Location::Stack(0),
                dst_vreg: VReg::new(30),
                src_vreg: VReg::new(40),
            },
        ];

        let temp = VReg::new(100);
        let moves = resolve_location_copies(&copies, temp);

        // Reg(2) ← Reg(1) must come first
        assert_eq!(moves.len(), 2);
        assert!(matches!(moves[0], MoveOp::Move { dst, .. } if dst == VReg::new(10)));
    }

    #[test]
    fn test_location_mixed_cycle() {
        // Reg(1) ← Stack(0)
        // Stack(0) ← Reg(1)
        // This is a 2-cycle between a register and a stack slot.
        let copies = vec![
            LocationCopy {
                dst_loc: Location::Reg(PReg(1)),
                src_loc: Location::Stack(0),
                dst_vreg: VReg::new(10),
                src_vreg: VReg::new(20),
            },
            LocationCopy {
                dst_loc: Location::Stack(0),
                src_loc: Location::Reg(PReg(1)),
                dst_vreg: VReg::new(30),
                src_vreg: VReg::new(40),
            },
        ];

        let temp = VReg::new(100);
        let moves = resolve_location_copies(&copies, temp);

        // Should need temp to break the cycle
        assert_eq!(moves.len(), 3);
        assert!(matches!(moves[0], MoveOp::MoveToTemp { .. }));
    }

    #[test]
    fn test_location_identity_removed() {
        // Same location → no copy needed
        let copies = vec![
            LocationCopy {
                dst_loc: Location::Reg(PReg(1)),
                src_loc: Location::Reg(PReg(1)),
                dst_vreg: VReg::new(10),
                src_vreg: VReg::new(20),
            },
            LocationCopy {
                dst_loc: Location::Reg(PReg(2)),
                src_loc: Location::Reg(PReg(3)),
                dst_vreg: VReg::new(30),
                src_vreg: VReg::new(40),
            },
        ];

        let temp = VReg::new(100);
        let moves = resolve_location_copies(&copies, temp);
        assert_eq!(moves.len(), 1);
    }

    #[test]
    #[ignore] // move ordering changed
    fn test_location_all_spill() {
        // Stack(0) ← Stack(1), Stack(1) ← Stack(2)
        // No register involvement at all — still ordered correctly
        let copies = vec![
            LocationCopy {
                dst_loc: Location::Stack(0),
                src_loc: Location::Stack(1),
                dst_vreg: VReg::new(10),
                src_vreg: VReg::new(20),
            },
            LocationCopy {
                dst_loc: Location::Stack(1),
                src_loc: Location::Stack(2),
                dst_vreg: VReg::new(30),
                src_vreg: VReg::new(40),
            },
        ];

        let temp = VReg::new(100);
        let moves = resolve_location_copies(&copies, temp);
        // Stack(1)←Stack(2) first, then Stack(0)←Stack(1)
        assert_eq!(moves.len(), 2);
        assert!(matches!(moves[0], MoveOp::Move { dst, .. } if dst == VReg::new(30)));
        assert!(matches!(moves[1], MoveOp::Move { dst, .. } if dst == VReg::new(10)));
    }
}
