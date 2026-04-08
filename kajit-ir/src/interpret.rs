//! RVSDG ideal interpreter (very naive)
//!
//! Executes an `IrFunc` directly on the RVSDG structure without linearization.
//! Used for differential testing: compare interpreter output against JIT output
//! to catch optimization bugs.

use std::collections::HashMap;

use kajit_types::{Arguments, SymbolTable};

use crate::{ArgId, IrFunc, IrOp, NodeId, NodeKind, OutputRef, PortSource, RegionArgRef, RegionId};

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

pub type Outcome = ();

pub type InResult = Result<Outcome, InError>;

#[derive(thiserror::Error, Debug)]
pub enum InError {
    #[error("ran out of gas (max steps reached)")]
    OutOfGas,

    #[error("something went wrong while interpreting: {0}")]
    Stringly(String),
}

/// Interpreter state.
struct State {
    /// Step counter (prevent infinite loops).
    steps: u64,

    /// Max steps.
    max_steps: u64,

    /// External symbol resolution table.
    symbol_table: SymbolTable,

    /// Stack allocations (each is a pinned Vec<u8>).
    stack_allocs: Vec<Vec<u8>>,
}

impl State {
    /// Initialize interpreter state
    fn new(symbol_table: SymbolTable) -> Self {
        Self {
            steps: 0,
            max_steps: 10_000,
            symbol_table,
            stack_allocs: Vec::new(),
        }
    }

    fn alloc_stack(&mut self, size: usize, _align: usize) -> usize {
        let alloc = vec![0u8; size];
        let ptr = alloc.as_ptr() as usize;
        self.stack_allocs.push(alloc);
        ptr
    }

    fn tick(&mut self) -> InResult {
        self.steps += 1;
        if self.steps > self.max_steps {
            return Err(InError::OutOfGas);
        }
        Ok(())
    }

    /// Execute an IrFunc on the given input bytes.
    ///
    /// `output_size` is the expected size of the output buffer (from the shape).
    pub fn interpret(func: &IrFunc, symtab: SymbolTable, args: &Arguments) -> InResult {
        let mut state = State::new(symtab);
        let mut env = Env::new();

        // Find the root lambda and execute its body.
        let root_node = &func.nodes[func.root];
        let NodeKind::Lambda { body, .. } = &root_node.kind else {
            panic!("root node is not a Lambda");
        };

        // Lambda body region args: data args first, then state domains.
        // For now, all args are GPR (U64). FPR support will come later.
        let (gprs, _fprs) = args.to_register_slots();
        let body_region = &func.regions[*body];
        for (i, &arg_id) in body_region.args.iter().enumerate() {
            let value = if i < gprs.len() {
                gprs[i]
            } else {
                0 // state domain token
            };
            env.args.insert(arg_id, value);
        }

        state.eval_region(func, *body, &mut env)?;
        Ok(())
    }

    /// Evaluate all nodes in a region in order.
    fn eval_region(&mut self, func: &IrFunc, region_id: RegionId, env: &mut Env) -> InResult {
        let region = &func.regions[region_id];

        for &node_id in &region.nodes {
            self.tick()?;
            self.eval_node(func, node_id, env)?;
        }
        Ok(())
    }

    /// Evaluate a single node.
    fn eval_node(&mut self, func: &IrFunc, node_id: NodeId, env: &mut Env) -> InResult {
        let node = &func.nodes[node_id];

        match &node.kind {
            NodeKind::Simple(op) => self.eval_simple(func, node_id, op, env)?,
            NodeKind::Gamma { regions } => self.eval_gamma(func, node_id, regions, env),
            NodeKind::Theta {
                body,
                max_iterations,
            } => self.eval_theta(func, node_id, *body, *max_iterations, env),
            NodeKind::Lambda { .. } => {
                todo!("nested lambdas are not supported r/n")
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

                self.eval_region(func, *body, &mut child_env)?;

                // Map body results to apply outputs.
                for (i, &result_id) in body_region.results.iter().enumerate() {
                    let val = child_env.resolve(&func.region_results[result_id].source);
                    env.set_output(node_id, i as u16, val);
                }
            }
        }
        Ok(())
    }

    /// Evaluate a simple (non-structured) operation.
    fn eval_simple(
        &mut self,
        func: &IrFunc,
        node_id: NodeId,
        op: &IrOp,
        env: &mut Env,
    ) -> InResult {
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
            IrOp::DataAddr { blob_id } => {
                let blob = &func.data_blobs[*blob_id as usize];
                env.set_output(node_id, 0, blob.as_ptr() as u64);
            }
            IrOp::ExternAddr { symbol } => {
                let addr = self.symbol_table.resolve(symbol).as_u64();
                env.set_output(node_id, 0, addr);
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

            // ─── Stack ops ───
            IrOp::StackAlloc { id } => {
                let info = &func.stack_allocs[id.index()];
                let alloc = self.alloc_stack(info.size as usize, info.align as usize);
                env.set_output(node_id, 0, alloc as u64);
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
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..w {
                        value |= (buf[i] as u64) << (i * 8);
                    }
                }
                env.set_output(node_id, 0, value);
            }

            // ─── Call ops ───
            IrOp::Call { .. } => {
                panic!("wow this function should return Result")
            }
        }
        Ok(())
    }

    /// Evaluate a gamma node (conditional).
    fn eval_gamma(
        &mut self,
        func: &IrFunc,
        node_id: NodeId,
        regions: &[RegionId],
        env: &mut Env,
    ) -> InResult {
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

        self.eval_region(func, branch_region_id, &mut child_env)?;

        // Map branch results to gamma outputs.
        for (i, &result_id) in branch_region.results.iter().enumerate() {
            let val = child_env.resolve(&func.region_results[result_id].source);
            env.set_output(node_id, i as u16, val);
        }
        Ok(())
    }

    /// Evaluate a theta node (loop).
    fn eval_theta(
        &mut self,
        func: &IrFunc,
        node_id: NodeId,
        body: RegionId,
        max_iterations: Option<u32>,
        env: &mut Env,
    ) -> InResult {
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
            self.tick()?;

            // Set body args from current loop values.
            let mut child_env = Env::new();
            for (i, &arg_id) in body_region.args.iter().enumerate() {
                child_env.args.insert(arg_id, loop_vals[i]);
            }

            self.eval_region(func, body, &mut child_env)?;

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
        Ok(())
    }
}
