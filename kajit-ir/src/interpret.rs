//! RVSDG ideal interpreter.
//!
//! Executes an `IrFunc` directly on the RVSDG structure without linearization.
//! Used for differential testing: compare interpreter output against JIT output
//! to catch optimization bugs.

use std::collections::HashMap;

use crate::{ArgId, IrFunc, IrOp, NodeId, NodeKind, OutputRef, PortSource, RegionArgRef, RegionId};

/// Interpreter outcome.
#[derive(Debug)]
pub struct Outcome {
    /// Output buffer (deserialized data).
    pub output: Vec<u8>,
    /// Final cursor position.
    pub cursor: usize,
    /// Error trap, if any.
    pub trap: Option<Trap>,
}

#[derive(Debug, Clone)]
pub struct Trap {
    pub code: crate::ErrorCode,
}

/// Interpreter state.
struct State<'a> {
    /// Input bytes.
    input: &'a [u8],
    /// Current read position.
    cursor: usize,
    /// Output buffer.
    output: Vec<u8>,
    /// Current output write pointer offset.
    out_ptr: usize,
    /// Abstract stack slots.
    slots: Vec<u64>,
    /// Address-based slot memory (8-byte stride).
    slot_mem: Vec<u8>,
    /// Trap flag.
    trap: Option<Trap>,
    /// Step counter (prevent infinite loops).
    steps: u64,
    /// Max steps.
    max_steps: u64,
}

const SLOT_STRIDE: usize = 8;

impl<'a> State<'a> {
    fn new(input: &'a [u8], output_size: usize) -> Self {
        Self {
            input,
            cursor: 0,
            output: vec![0u8; output_size],
            out_ptr: 0,
            slots: Vec::new(),
            slot_mem: Vec::new(),
            trap: None,
            steps: 0,
            max_steps: 1_000_000,
        }
    }

    fn ensure_slot(&mut self, slot: usize) {
        if slot >= self.slots.len() {
            self.slots.resize(slot + 1, 0);
        }
    }

    fn ensure_slot_mem(&mut self, slot: usize, num_slots: usize) {
        let needed = (slot + num_slots) * SLOT_STRIDE;
        if needed > self.slot_mem.len() {
            self.slot_mem.resize(needed, 0);
        }
    }

    fn ensure_output(&mut self, offset: usize, width: usize) {
        let end = self.out_ptr + offset + width;
        if end > self.output.len() {
            self.output.resize(end, 0);
        }
    }

    fn trapped(&self) -> bool {
        self.trap.is_some()
    }

    fn tick(&mut self) -> bool {
        self.steps += 1;
        if self.steps > self.max_steps {
            self.trap = Some(Trap {
                code: crate::ErrorCode::InvalidVarint, // repurposed as step limit
            });
            false
        } else {
            true
        }
    }
}

/// Value environment: maps output ports and region args to u64 values.
struct Env {
    /// Node output values: OutputRef → u64.
    outputs: HashMap<OutputRef, u64>,
    /// Region arg values: ArgId → u64.
    args: HashMap<ArgId, u64>,
}

impl Env {
    fn new() -> Self {
        Self {
            outputs: HashMap::new(),
            args: HashMap::new(),
        }
    }

    fn resolve(&self, source: &PortSource) -> u64 {
        match source {
            PortSource::Node(out_ref) => self.outputs.get(out_ref).copied().unwrap_or_else(|| {
                panic!(
                    "unresolved output ref: n{}.{}",
                    out_ref.node.index(),
                    out_ref.index
                )
            }),
            PortSource::RegionArg(RegionArgRef { arg, .. }) => self
                .args
                .get(arg)
                .copied()
                .unwrap_or_else(|| panic!("unresolved region arg: a{}", arg.index())),
        }
    }

    fn set_output(&mut self, node: NodeId, index: u16, value: u64) {
        self.outputs.insert(OutputRef { node, index }, value);
    }
}

/// Execute an IrFunc on the given input bytes.
///
/// `output_size` is the expected size of the output buffer (from the shape).
pub fn interpret(func: &IrFunc, input: &[u8], output_size: usize) -> Outcome {
    let mut state = State::new(input, output_size);
    let mut env = Env::new();

    // Find the root lambda and execute its body.
    let root_node = &func.nodes[func.root];
    let NodeKind::Lambda { body, .. } = &root_node.kind else {
        panic!("root node is not a Lambda");
    };

    // Lambda body region args correspond to state domains.
    // Set up initial state domain "values" (they're just tokens, use 0).
    let body_region = &func.regions[*body];
    for &arg_id in &body_region.args {
        env.args.insert(arg_id, 0);
    }

    eval_region(func, *body, &mut state, &mut env);

    Outcome {
        output: state.output,
        cursor: state.cursor,
        trap: state.trap,
    }
}

/// Evaluate all nodes in a region in order.
fn eval_region(func: &IrFunc, region_id: RegionId, state: &mut State, env: &mut Env) {
    let region = &func.regions[region_id];

    for &node_id in &region.nodes {
        if state.trapped() {
            break;
        }
        if !state.tick() {
            break;
        }
        eval_node(func, node_id, state, env);
    }
}

/// Evaluate a single node.
fn eval_node(func: &IrFunc, node_id: NodeId, state: &mut State, env: &mut Env) {
    let node = &func.nodes[node_id];

    match &node.kind {
        NodeKind::Simple(op) => eval_simple(func, node_id, op, state, env),
        NodeKind::Gamma { regions } => eval_gamma(func, node_id, regions, state, env),
        NodeKind::Theta {
            body,
            max_iterations,
        } => eval_theta(func, node_id, *body, *max_iterations, state, env),
        NodeKind::Lambda { .. } => {
            // Nested lambda definitions: just record as a value (function pointer).
            // For now, lambdas that appear as nested nodes are not executed here.
            env.set_output(node_id, 0, node_id.index() as u64);
        }
        NodeKind::Apply { target } => {
            // Call a lambda. Find the lambda node's body and execute it.
            let target_node_id = func.lambdas[target.index()];
            let target_node = &func.nodes[target_node_id];
            let NodeKind::Lambda { body, .. } = &target_node.kind else {
                panic!("apply target n{} is not a Lambda", target.index());
            };
            let body_region = &func.regions[*body];

            // Map apply inputs to lambda body args.
            let mut child_env = Env::new();
            for (i, &arg_id) in body_region.args.iter().enumerate() {
                let val = env.resolve(&node.inputs[i].source);
                child_env.args.insert(arg_id, val);
            }

            eval_region(func, *body, state, &mut child_env);

            // Map body results to apply outputs.
            for (i, &result_id) in body_region.results.iter().enumerate() {
                let val = child_env.resolve(&func.region_results[result_id].source);
                env.set_output(node_id, i as u16, val);
            }
        }
    }
}

/// Evaluate a simple (non-structured) operation.
fn eval_simple(func: &IrFunc, node_id: NodeId, op: &IrOp, state: &mut State, env: &mut Env) {
    let node = &func.nodes[node_id];

    // Gather data inputs (skip state inputs which carry ordering tokens).
    let data_inputs: Vec<u64> = node
        .inputs
        .iter()
        .filter(|inp| inp.kind == crate::PortKind::Data)
        .map(|inp| env.resolve(&inp.source))
        .collect();

    // Set all state outputs to 0 (ordering tokens, not real values).
    for (i, out) in node.outputs.iter().enumerate() {
        if out.kind != crate::PortKind::Data {
            env.set_output(node_id, i as u16, 0);
        }
    }

    match op {
        // ─── Pure ops ───
        IrOp::Const { value } => {
            env.set_output(node_id, 0, *value);
        }
        IrOp::Add
        | IrOp::Sub
        | IrOp::Mul
        | IrOp::And
        | IrOp::Or
        | IrOp::Xor
        | IrOp::Shl
        | IrOp::Shr
        | IrOp::Sar
        | IrOp::CmpEq
        | IrOp::CmpNe
        | IrOp::CmpLt
        | IrOp::CmpLe
        | IrOp::CmpGt
        | IrOp::CmpGe
        | IrOp::ZigzagDecode { .. }
        | IrOp::SignExtend { .. } => {
            if let Some(val) = crate::const_fold::evaluate_op(op, &data_inputs) {
                env.set_output(node_id, 0, val);
            } else {
                panic!(
                    "eval_simple: pure op {:?} failed with {} inputs",
                    op,
                    data_inputs.len()
                );
            }
        }
        IrOp::Identity => {
            env.set_output(node_id, 0, data_inputs[0]);
        }
        IrOp::Nop => {}

        // ─── Cursor ops ───
        IrOp::BoundsCheck { count } => {
            let count = *count as usize;
            if state.cursor + count > state.input.len() {
                state.trap = Some(Trap {
                    code: crate::ErrorCode::UnexpectedEof,
                });
            }
        }
        IrOp::ReadBytes { count } => {
            let count = *count as usize;
            if state.cursor + count > state.input.len() {
                state.trap = Some(Trap {
                    code: crate::ErrorCode::UnexpectedEof,
                });
                return;
            }
            let mut value = 0u64;
            for i in 0..count {
                value |= (state.input[state.cursor + i] as u64) << (i * 8);
            }
            state.cursor += count;
            env.set_output(node_id, 0, value);
        }
        IrOp::PeekByte => {
            if state.cursor >= state.input.len() {
                state.trap = Some(Trap {
                    code: crate::ErrorCode::UnexpectedEof,
                });
                return;
            }
            env.set_output(node_id, 0, state.input[state.cursor] as u64);
        }
        IrOp::AdvanceCursor { count } => {
            let count = *count as usize;
            if state.cursor + count > state.input.len() {
                state.trap = Some(Trap {
                    code: crate::ErrorCode::UnexpectedEof,
                });
                return;
            }
            state.cursor += count;
        }
        IrOp::AdvanceCursorBy => {
            let count = data_inputs[0] as usize;
            if state.cursor + count > state.input.len() {
                state.trap = Some(Trap {
                    code: crate::ErrorCode::UnexpectedEof,
                });
                return;
            }
            state.cursor += count;
        }
        IrOp::SaveCursor => {
            // Save cursor as raw pointer (matches JIT/CFG-MIR semantics).
            let ptr = unsafe { state.input.as_ptr().add(state.cursor) } as u64;
            env.set_output(node_id, 0, ptr);
        }
        IrOp::SaveInputEnd => {
            let end = unsafe { state.input.as_ptr().add(state.input.len()) } as u64;
            env.set_output(node_id, 0, end);
        }
        IrOp::RestoreCursor => {
            // Restore cursor from raw pointer.
            let ptr = data_inputs[0] as usize;
            let base = state.input.as_ptr() as usize;
            let offset = ptr.wrapping_sub(base);
            if offset > state.input.len() {
                state.trap = Some(Trap {
                    code: crate::ErrorCode::UnexpectedEof,
                });
                return;
            }
            state.cursor = offset;
        }

        // ─── Output ops ───
        IrOp::WriteToField { offset, width } => {
            let value = data_inputs[0];
            let off = *offset as usize;
            let w = width.bytes() as usize;
            state.ensure_output(off, w);
            let base = state.out_ptr + off;
            let bytes = value.to_le_bytes();
            state.output[base..base + w].copy_from_slice(&bytes[..w]);
        }
        IrOp::ReadFromField { offset, width } => {
            let off = *offset as usize;
            let w = width.bytes() as usize;
            state.ensure_output(off, w);
            let base = state.out_ptr + off;
            let mut value = 0u64;
            for i in 0..w {
                value |= (state.output[base + i] as u64) << (i * 8);
            }
            env.set_output(node_id, 0, value);
        }
        IrOp::SaveOutPtr => {
            // Return raw pointer to current output position.
            let ptr = unsafe { state.output.as_ptr().add(state.out_ptr) } as u64;
            env.set_output(node_id, 0, ptr);
        }
        IrOp::SetOutPtr => {
            // Restore output pointer from raw pointer.
            let ptr = data_inputs[0] as usize;
            let base = state.output.as_ptr() as usize;
            state.out_ptr = ptr.wrapping_sub(base);
        }

        // ─── Slot ops ───
        IrOp::WriteToSlot { slot } => {
            let s = slot.index();
            state.ensure_slot(s);
            state.slots[s] = data_inputs[0];
        }
        IrOp::ReadFromSlot { slot } => {
            let s = slot.index();
            state.ensure_slot(s);
            env.set_output(node_id, 0, state.slots[s]);
        }
        IrOp::SlotAddr { slot, num_slots } => {
            let s = slot.index();
            state.ensure_slot_mem(s, *num_slots as usize);
            let addr = state.slot_mem.as_ptr() as usize + s * SLOT_STRIDE;
            env.set_output(node_id, 0, addr as u64);
        }
        IrOp::StoreToAddr { width } => {
            let addr = data_inputs[0] as *mut u8;
            let value = data_inputs[1];
            let w = width.bytes() as usize;
            let bytes = value.to_le_bytes();
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), addr, w);
            }
        }
        IrOp::LoadFromAddr { width } => {
            let addr = data_inputs[0] as *const u8;
            let w = width.bytes() as usize;
            let mut value = 0u64;
            unsafe {
                let mut buf = [0u8; 8];
                core::ptr::copy_nonoverlapping(addr, buf.as_mut_ptr(), w);
                for i in 0..w {
                    value |= (buf[i] as u64) << (i * 8);
                }
            }
            env.set_output(node_id, 0, value);
        }

        // ─── Error ops ───
        IrOp::ErrorExit { code } => {
            state.trap = Some(Trap { code: *code });
        }

        // ─── Call ops (stub) ───
        IrOp::CallIntrinsic { .. } | IrOp::CallPure { .. } => {
            // Intrinsics can't be interpreted without the function pointers.
            // Set outputs to 0.
            for i in 0..node.outputs.len() {
                if node.outputs[i].kind == crate::PortKind::Data {
                    env.set_output(node_id, i as u16, 0);
                }
            }
        }

        // ─── SIMD ops (scalar fallback) ───
        IrOp::SimdStringScan | IrOp::SimdWhitespaceSkip => {
            // These are SIMD acceleration ops; scalar fallback: return 0.
            for i in 0..node.outputs.len() {
                if node.outputs[i].kind == crate::PortKind::Data {
                    env.set_output(node_id, i as u16, 0);
                }
            }
        }
    }
}

/// Evaluate a gamma node (conditional).
fn eval_gamma(
    func: &IrFunc,
    node_id: NodeId,
    regions: &[RegionId],
    state: &mut State,
    env: &mut Env,
) {
    let node = &func.nodes[node_id];

    // First input is the predicate.
    let pred = env.resolve(&node.inputs[0].source) as usize;
    let branch = pred.min(regions.len() - 1);
    let branch_region_id = regions[branch];
    let branch_region = &func.regions[branch_region_id];

    // Map gamma inputs (skip predicate) to branch region args.
    let mut child_env = Env::new();
    for (i, &arg_id) in branch_region.args.iter().enumerate() {
        let val = env.resolve(&node.inputs[i + 1].source);
        child_env.args.insert(arg_id, val);
    }

    eval_region(func, branch_region_id, state, &mut child_env);

    // Map branch results to gamma outputs.
    for (i, &result_id) in branch_region.results.iter().enumerate() {
        let val = child_env.resolve(&func.region_results[result_id].source);
        env.set_output(node_id, i as u16, val);
    }
}

/// Evaluate a theta node (loop).
fn eval_theta(
    func: &IrFunc,
    node_id: NodeId,
    body: RegionId,
    max_iterations: Option<u32>,
    state: &mut State,
    env: &mut Env,
) {
    let node = &func.nodes[node_id];
    let body_region = &func.regions[body];
    let max_iter = max_iterations.unwrap_or(u32::MAX);

    // Initialize loop variables from theta inputs.
    let mut loop_vals: Vec<u64> = node
        .inputs
        .iter()
        .map(|inp| env.resolve(&inp.source))
        .collect();

    for iteration in 0..max_iter {
        if state.trapped() {
            break;
        }
        if !state.tick() {
            break;
        }

        // Set body args from current loop values.
        let mut child_env = Env::new();
        for (i, &arg_id) in body_region.args.iter().enumerate() {
            child_env.args.insert(arg_id, loop_vals[i]);
        }

        eval_region(func, body, state, &mut child_env);

        if state.trapped() {
            break;
        }

        // Read body results: [predicate, var0, var1, ..., state0, state1, ...]
        let results: Vec<u64> = body_region
            .results
            .iter()
            .map(|&result_id| child_env.resolve(&func.region_results[result_id].source))
            .collect();

        let predicate = results[0];

        // Update loop variables from results (skip predicate).
        loop_vals = results[1..].to_vec();

        // predicate == 0 → exit.
        if predicate == 0 {
            break;
        }

        // Prevent infinite loops even without max_iterations.
        if iteration + 1 >= max_iter {
            break;
        }
    }

    // Set theta outputs from final loop values.
    for (i, &val) in loop_vals.iter().enumerate() {
        env.set_output(node_id, i as u16, val);
    }
}
