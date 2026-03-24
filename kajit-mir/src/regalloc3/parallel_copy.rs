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
                MoveOp::MoveToTemp { dst_temp, .. } => {
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
}
