use regalloc2::Allocation;

pub(crate) trait MoveEmitter {
    fn flush_all_vregs(&mut self);
    fn emit_move(&mut self, from: Allocation, to: Allocation);
    fn save_move_src_to_tmp(&mut self, tmp_index: usize, from: Allocation);
    fn restore_move_tmp_to_dst(&mut self, tmp_index: usize, to: Allocation);
}

pub(crate) fn emit_parallel_moves<E: MoveEmitter>(
    emitter: &mut E,
    moves: &[(Allocation, Allocation)],
) {
    let filtered: Vec<(Allocation, Allocation)> = moves
        .iter()
        .copied()
        .filter(|(from, to)| *from != *to && !from.is_none() && !to.is_none())
        .collect();
    if filtered.is_empty() {
        return;
    }
    if filtered.len() == 1 {
        emitter.flush_all_vregs();
        let (from, to) = filtered[0];
        emitter.emit_move(from, to);
        return;
    }

    // Parallel-move semantics: read all sources first, then write all destinations.
    emitter.flush_all_vregs();
    for (index, (from, _)) in filtered.iter().copied().enumerate() {
        emitter.save_move_src_to_tmp(index, from);
    }
    for (index, (_, to)) in filtered.iter().copied().enumerate() {
        emitter.restore_move_tmp_to_dst(index, to);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regalloc2::{Allocation, PReg, RegClass, SpillSlot};
    use std::collections::BTreeMap;

    struct MockEmitter {
        values: BTreeMap<String, u64>,
        tmps: Vec<u64>,
        flushes: usize,
    }

    impl MockEmitter {
        fn new() -> Self {
            Self {
                values: BTreeMap::new(),
                tmps: Vec::new(),
                flushes: 0,
            }
        }

        fn key(a: Allocation) -> String {
            if let Some(r) = a.as_reg() {
                return format!("r{}", r.hw_enc());
            }
            if let Some(s) = a.as_stack() {
                return format!("s{}", s.index());
            }
            "none".to_string()
        }

        fn set(&mut self, a: Allocation, v: u64) {
            self.values.insert(Self::key(a), v);
        }

        fn get(&self, a: Allocation) -> u64 {
            *self.values.get(&Self::key(a)).unwrap_or(&0)
        }
    }

    impl MoveEmitter for MockEmitter {
        fn flush_all_vregs(&mut self) {
            self.flushes += 1;
        }

        fn emit_move(&mut self, from: Allocation, to: Allocation) {
            let v = self.get(from);
            self.set(to, v);
        }

        fn save_move_src_to_tmp(&mut self, tmp_index: usize, from: Allocation) {
            if self.tmps.len() <= tmp_index {
                self.tmps.resize(tmp_index + 1, 0);
            }
            self.tmps[tmp_index] = self.get(from);
        }

        fn restore_move_tmp_to_dst(&mut self, tmp_index: usize, to: Allocation) {
            let v = self.tmps[tmp_index];
            self.set(to, v);
        }
    }

    fn r(enc: usize) -> Allocation {
        Allocation::reg(PReg::new(enc, RegClass::Int))
    }

    fn s(idx: usize) -> Allocation {
        Allocation::stack(SpillSlot::new(idx))
    }

    #[test]
    fn ignores_self_copies() {
        let mut m = MockEmitter::new();
        m.set(r(0), 7);
        emit_parallel_moves(&mut m, &[(r(0), r(0))]);
        assert_eq!(m.get(r(0)), 7);
        assert_eq!(m.flushes, 0);
    }

    #[test]
    fn handles_simple_chain() {
        let mut m = MockEmitter::new();
        m.set(r(0), 10);
        m.set(r(1), 20);
        m.set(r(2), 30);
        emit_parallel_moves(&mut m, &[(r(0), r(1)), (r(1), r(2))]);
        assert_eq!(m.get(r(1)), 10);
        assert_eq!(m.get(r(2)), 20);
    }

    #[test]
    fn handles_two_cycle_reg_to_reg() {
        let mut m = MockEmitter::new();
        m.set(r(0), 1);
        m.set(r(1), 2);
        emit_parallel_moves(&mut m, &[(r(0), r(1)), (r(1), r(0))]);
        assert_eq!(m.get(r(0)), 2);
        assert_eq!(m.get(r(1)), 1);
    }

    #[test]
    fn handles_reg_stack_cycle() {
        let mut m = MockEmitter::new();
        m.set(r(0), 11);
        m.set(s(0), 22);
        emit_parallel_moves(&mut m, &[(r(0), s(0)), (s(0), r(0))]);
        assert_eq!(m.get(r(0)), 22);
        assert_eq!(m.get(s(0)), 11);
    }

    #[test]
    fn handles_stack_to_stack_via_temps() {
        let mut m = MockEmitter::new();
        m.set(s(0), 100);
        m.set(s(1), 200);
        m.set(r(0), 7);
        m.set(r(1), 8);
        emit_parallel_moves(&mut m, &[(s(0), s(1)), (r(0), r(1))]);
        assert_eq!(m.get(s(1)), 100);
        assert_eq!(m.get(r(1)), 7);
    }
}
