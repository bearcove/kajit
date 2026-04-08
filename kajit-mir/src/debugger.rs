use std::collections::HashMap;

use kajit_ir::ErrorCode;
use kajit_lir::{BinOpKind, LinearOp};

use crate::InterpreterTrap;
use crate::cfg_mir;

/// Minimal runtime context matching the JIT's DeserContext layout.
/// Used to call real intrinsic functions from the interpreter.
#[repr(C)]
struct RuntimeDeserContext {
    input_ptr: *const u8,
    input_end: *const u8,
    error: RuntimeErrorSlot,
    key_scratch_ptr: *mut u8,
    key_scratch_cap: usize,
    trusted_utf8: bool,
}

#[repr(C)]
struct RuntimeSliceU8 {
    ptr: *const u8,
    len: usize,
}

#[repr(C)]
struct RuntimeCursorArg {
    bytes: RuntimeSliceU8,
    pos: u64,
}

// Safety: RuntimeDeserContext and RuntimeCursorArg contain raw pointers that
// point into DebuggerSession-owned data (input_data, etc.). They are only
// dereferenced while the session holds &mut self.
unsafe impl Send for RuntimeDeserContext {}
unsafe impl Send for RuntimeCursorArg {}

#[repr(C)]
struct RuntimeErrorSlot {
    code: u32,
    offset: u32,
}

fn error_code_from_u32(code: u32) -> ErrorCode {
    // Safety: ErrorCode is repr(u32) and all valid codes are contiguous
    unsafe { core::mem::transmute(code) }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DebuggerError {
    NoFunctions,
    UnknownBlock {
        block: cfg_mir::BlockId,
    },
    UnknownEdge {
        edge: cfg_mir::EdgeId,
    },
    UnknownInst {
        inst: cfg_mir::InstId,
    },
    UnknownTerm {
        term: cfg_mir::TermId,
    },
    EdgeArgArityMismatch {
        from: cfg_mir::BlockId,
        to: cfg_mir::BlockId,
        expected: usize,
        got: usize,
    },
    UnsupportedOp {
        block: cfg_mir::BlockId,
        op: String,
    },
    UnsupportedTerminator {
        block: cfg_mir::BlockId,
        term: String,
    },
}

impl std::fmt::Display for DebuggerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFunctions => write!(f, "CFG-MIR program has no functions"),
            Self::UnknownBlock { block } => write!(f, "unknown block b{}", block.0),
            Self::UnknownEdge { edge } => {
                write!(f, "unknown edge e{}", edge.0)
            }
            Self::UnknownInst { inst } => write!(f, "unknown inst i{}", inst.0),
            Self::UnknownTerm { term } => write!(f, "unknown term t{}", term.0),
            Self::EdgeArgArityMismatch {
                from,
                to,
                expected,
                got,
            } => write!(
                f,
                "edge arg arity mismatch on b{} -> b{}: expected {}, got {}",
                from.0, to.0, expected, got
            ),
            Self::UnsupportedOp { block, op } => {
                write!(f, "unsupported CFG-MIR op in block b{}: {}", block.0, op)
            }
            Self::UnsupportedTerminator { block, term } => {
                write!(
                    f,
                    "unsupported CFG-MIR terminator in block b{}: {}",
                    block.0, term
                )
            }
        }
    }
}

impl std::error::Error for DebuggerError {}

// --- Symbolic pointer tracking ---

/// Symbolic pointer identity. Two pointers with the same PtrId refer to the
/// same logical allocation, even if concrete addresses differ (interpreter vs JIT).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtrId(pub u32);

/// A vreg value with optional pointer provenance.
#[derive(Debug, Clone, Copy)]
pub enum TaggedValue {
    /// A plain scalar value, compared by concrete value.
    Scalar(u64),
    /// A pointer with known provenance: compared by (id, offset).
    Pointer {
        id: PtrId,
        concrete: u64,
        offset: u64,
    },
    /// A pointer whose provenance was destroyed (bitwise ops, unknown casts, etc.).
    /// Not comparable — neither match nor divergence.
    UnknownPointer(u64),
}

impl TaggedValue {
    /// Get the concrete u64 value for execution purposes.
    pub fn concrete(&self) -> u64 {
        match *self {
            TaggedValue::Scalar(v) => v,
            TaggedValue::Pointer { concrete, .. } => concrete,
            TaggedValue::UnknownPointer(v) => v,
        }
    }

    /// Is this a pointer (known or unknown provenance)?
    pub fn is_pointer(&self) -> bool {
        !matches!(self, TaggedValue::Scalar(_))
    }
}

impl Default for TaggedValue {
    fn default() -> Self {
        TaggedValue::Scalar(0)
    }
}

/// Shadow memory entry: tracks pointer provenance stored through memory.
#[derive(Debug, Clone, Copy)]
struct ShadowEntry {
    width: u8,
    value: TaggedValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramLocation {
    pub block: cfg_mir::BlockId,
    pub next_inst_index: usize,
    pub at_terminator: bool,
}

#[derive(Debug, Clone)]
pub struct DebuggerState {
    pub step_count: usize,
    pub location: ProgramLocation,
    /// Concrete vreg values (for backwards compatibility).
    pub vregs: Vec<u64>,
    /// Tagged vreg values with pointer provenance info.
    pub tagged_vregs: Vec<TaggedValue>,
    pub output: Vec<u8>,
    pub trap: Option<InterpreterTrap>,
    pub returned: bool,
    pub halted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    Instruction,
    Terminator,
    HaltedNoop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepEvent {
    pub step_index: usize,
    pub kind: StepKind,
    pub location_before: ProgramLocation,
    pub location_after: ProgramLocation,
    pub trap: Option<InterpreterTrap>,
    pub returned: bool,
    pub halted_after: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunUntilTarget {
    Block(cfg_mir::BlockId),
    Trap,
    Return,
}

#[derive(Debug, Clone)]
struct SessionSnapshot {
    slots: Vec<u8>,
    vregs: Vec<TaggedValue>,
    trap: Option<InterpreterTrap>,
    returned: bool,
    current: cfg_mir::BlockId,
    next_inst: usize,
    steps: usize,
    pointer_shadow: HashMap<(PtrId, u64), ShadowEntry>,
}

pub struct DebuggerSession {
    func: cfg_mir::Function,
    block_indices: HashMap<cfg_mir::BlockId, usize>,
    /// Input data (kept alive so cursor_arg.bytes.ptr stays valid).
    input_data: Vec<u8>,
    /// Interpreter-owned output buffer, pointed to by data_args[1].
    output_buf: Vec<u8>,
    /// Interpreter-owned RuntimeCursorArg, pointed to by data_args[0].
    cursor_arg: Box<RuntimeCursorArg>,
    /// Separate storage for slot values (not part of the output).
    slots: Vec<u8>,
    vregs: Vec<TaggedValue>,
    trap: Option<InterpreterTrap>,
    returned: bool,
    current: cfg_mir::BlockId,
    next_inst: usize,
    steps: usize,
    history: Vec<SessionSnapshot>,
    /// Embedded constant data blobs (string literals, etc.).
    data_blobs: Vec<Vec<u8>>,
    /// Runtime deserialization context, passed to intrinsics as ctx_ptr.
    ctx: RuntimeDeserContext,
    /// External symbol resolution table.
    symbol_table: kajit_types::SymbolTable,
    /// Stack allocation metadata (size/align per ID).
    stack_allocs_info: Vec<kajit_ir::StackAllocInfo>,
    /// Stack allocations (kept alive for the session).
    stack_allocs: Vec<Vec<u8>>,
    /// Next PtrId to allocate.
    next_ptr_id: u32,
    /// Shadow memory: tracks pointer provenance stored through memory.
    /// Key is (PtrId of base allocation, offset within allocation).
    pointer_shadow: HashMap<(PtrId, u64), ShadowEntry>,
    /// Maps PtrId → base concrete address (interpreter side, for offset computation in shadow memory).
    ptr_interp_bases: HashMap<PtrId, u64>,
}

impl DebuggerSession {
    pub fn new(
        program: &cfg_mir::Program,
        input: &[u8],
        _args: &kajit_types::Arguments,
    ) -> Result<Self, DebuggerError> {
        tracing::info!("DebuggerSession::new start");
        let func = program
            .funcs
            .first()
            .ok_or(DebuggerError::NoFunctions)?
            .clone();
        let block_indices = build_block_index(&func);
        let input_data = input.to_vec();
        let output_size = infer_output_size(&func);
        tracing::info!(
            output_size,
            func_output_size = func.output_size,
            "output_buf size"
        );
        let output_buf = vec![0u8; output_size];
        let cursor_arg = Box::new(RuntimeCursorArg {
            bytes: RuntimeSliceU8 {
                ptr: input_data.as_ptr(),
                len: input_data.len(),
            },
            pos: 0,
        });
        let mut session = Self {
            output_buf,
            cursor_arg,
            slots: Vec::new(),
            vregs: vec![TaggedValue::Scalar(0); program.vreg_count as usize],
            trap: None,
            returned: false,
            current: func.entry,
            next_inst: 0,
            steps: 0,
            history: Vec::new(),
            func,
            block_indices,
            input_data,
            data_blobs: program.data_blobs.clone(),
            ctx: RuntimeDeserContext {
                input_ptr: std::ptr::null(),
                input_end: std::ptr::null(),
                error: RuntimeErrorSlot { code: 0, offset: 0 },
                key_scratch_ptr: std::ptr::null_mut(),
                key_scratch_cap: 0,
                trusted_utf8: false,
            },
            symbol_table: kajit_types::SymbolTable::new(),
            stack_allocs_info: program.stack_allocs.clone(),
            stack_allocs: Vec::new(),
            next_ptr_id: 0,
            pointer_shadow: HashMap::new(),
            ptr_interp_bases: HashMap::new(),
        };
        tracing::info!("DebuggerSession::new struct created");
        // Initialize ctx with input pointers
        session.ctx.input_ptr = session.input_data.as_ptr();
        session.ctx.input_end =
            unsafe { session.input_data.as_ptr().add(session.input_data.len()) };

        // Set data_arg vregs to point to interpreter-owned allocations.
        // HIR param order: [cursor, out, ctx]
        // Each gets a stable PtrId.
        let data_args = session.func.data_args.clone();
        let data_arg_layouts = &program.data_arg_layouts;
        let fallback_names = ["cursor", "out", "ctx"];
        tracing::info!(
            n_data_args = data_args.len(),
            data_args = ?data_args.iter().map(|v| v.index()).collect::<Vec<_>>(),
            n_layouts = data_arg_layouts.len(),
            "DebuggerSession::new seeding data_args"
        );
        for (i, layout) in data_arg_layouts.iter().enumerate() {
            tracing::info!(i, name = %layout.name, n_fields = layout.pointer_fields.len(), "layout");
            for f in &layout.pointer_fields {
                tracing::info!(i, offset = f.offset, label = %f.label, "  pointer_field");
            }
        }
        if let Some(&vreg) = data_args.get(0) {
            tracing::info!("seeding data_arg[0]");
            let ptr = &*session.cursor_arg as *const RuntimeCursorArg as u64;
            let name = data_arg_layouts
                .get(0)
                .map(|l| l.name.as_str())
                .unwrap_or(fallback_names[0]);
            let id = session.alloc_ptr_id(ptr, &format!("data_arg[0] ({name})"));
            session.write_vreg_tagged(
                vreg.index(),
                TaggedValue::Pointer {
                    id,
                    concrete: ptr,
                    offset: 0,
                },
            );
            tracing::info!("seeding data_arg[0] shadow");
            // Seed shadow memory for pointer fields within this struct
            if let Some(layout) = data_arg_layouts.get(0) {
                session.seed_shadow_for_layout(id, ptr, layout);
            }
            tracing::info!("data_arg[0] done");
        }
        tracing::info!("between data_arg[0] and data_arg[1]");
        if let Some(&vreg) = data_args.get(1) {
            tracing::info!("seeding data_arg[1]");
            let ptr = session.output_buf.as_ptr() as u64;
            tracing::info!(ptr, "data_arg[1] ptr");
            let name = data_arg_layouts
                .get(1)
                .map(|l| l.name.as_str())
                .unwrap_or(fallback_names[1]);
            tracing::info!("data_arg[1] alloc_ptr_id");
            let id = session.alloc_ptr_id(ptr, &format!("data_arg[1] ({name})"));
            tracing::info!("data_arg[1] write_vreg_tagged");
            session.write_vreg_tagged(
                vreg.index(),
                TaggedValue::Pointer {
                    id,
                    concrete: ptr,
                    offset: 0,
                },
            );
            tracing::info!("data_arg[1] seed_shadow");
            if let Some(layout) = data_arg_layouts.get(1) {
                session.seed_shadow_for_layout(id, ptr, layout);
            }
            tracing::info!("data_arg[1] done");
        }
        if let Some(&vreg) = data_args.get(2) {
            let ptr = &session.ctx as *const RuntimeDeserContext as u64;
            let name = data_arg_layouts
                .get(2)
                .map(|l| l.name.as_str())
                .unwrap_or(fallback_names[2]);
            let id = session.alloc_ptr_id(ptr, &format!("data_arg[2] ({name})"));
            session.write_vreg_tagged(
                vreg.index(),
                TaggedValue::Pointer {
                    id,
                    concrete: ptr,
                    offset: 0,
                },
            );
            if let Some(layout) = data_arg_layouts.get(2) {
                session.seed_shadow_for_layout(id, ptr, layout);
            }
        }
        tracing::info!("DebuggerSession::new done");
        Ok(session)
    }

    pub fn state(&self) -> DebuggerState {
        DebuggerState {
            step_count: self.steps,
            location: self.location(),
            vregs: self.vregs.iter().map(|v| v.concrete()).collect(),
            tagged_vregs: self.vregs.clone(),
            output: self.output_buf.clone(),
            trap: self.trap,
            returned: self.returned,
            halted: self.is_halted(),
        }
    }

    pub fn inspect_vreg(&self, vreg_index: usize) -> u64 {
        self.read_vreg(vreg_index)
    }

    pub fn inspect_vreg_tagged(&self, vreg_index: usize) -> TaggedValue {
        self.read_vreg_tagged(vreg_index)
    }

    pub fn inspect_output(&self, start: usize, len: usize) -> Vec<u8> {
        if start >= self.output_buf.len() {
            return Vec::new();
        }
        let end = start.saturating_add(len).min(self.output_buf.len());
        self.output_buf[start..end].to_vec()
    }

    pub fn step_forward(&mut self) -> Result<StepEvent, DebuggerError> {
        let location_before = self.location();
        if self.is_halted() {
            return Ok(StepEvent {
                step_index: self.steps,
                kind: StepKind::HaltedNoop,
                location_before,
                location_after: location_before,
                trap: self.trap,
                returned: self.returned,
                halted_after: true,
                detail: "halted".to_owned(),
            });
        }

        let snapshot = self.snapshot();
        self.history.push(snapshot.clone());

        let step_detail;
        let step_kind;
        let (block_id, inst_len) = {
            let block = self.current_block()?;
            (block.id, block.insts.len())
        };
        if self.next_inst < inst_len {
            let inst_id = {
                let block = self.current_block()?;
                block.insts[self.next_inst]
            };
            let op = self
                .func
                .inst(inst_id)
                .ok_or(DebuggerError::UnknownInst { inst: inst_id })?
                .op
                .clone();
            self.execute_op(block_id, &op)?;
            self.next_inst += 1;
            step_detail = format!("{op:?}");
            step_kind = StepKind::Instruction;
        } else {
            let term_id = {
                let block = self.current_block()?;
                block.term
            };
            let term = self
                .func
                .term(term_id)
                .ok_or(DebuggerError::UnknownTerm { term: term_id })?
                .clone();
            self.execute_terminator(block_id, &term)?;
            step_detail = format!("{term:?}");
            step_kind = StepKind::Terminator;
        }

        self.steps += 1;
        let location_after = self.location();
        Ok(StepEvent {
            step_index: self.steps,
            kind: step_kind,
            location_before,
            location_after,
            trap: self.trap,
            returned: self.returned,
            halted_after: self.is_halted(),
            detail: step_detail,
        })
    }

    pub fn step_back(&mut self) -> bool {
        match self.history.pop() {
            Some(snapshot) => {
                self.restore(snapshot);
                true
            }
            None => false,
        }
    }

    pub fn run_until(
        &mut self,
        target: RunUntilTarget,
        max_steps: usize,
    ) -> Result<Vec<StepEvent>, DebuggerError> {
        let mut events = Vec::new();
        if self.target_reached(target) {
            return Ok(events);
        }

        for _ in 0..max_steps {
            let event = self.step_forward()?;
            events.push(event);
            if self.target_reached(target) || self.is_halted() {
                break;
            }
        }

        Ok(events)
    }

    fn target_reached(&self, target: RunUntilTarget) -> bool {
        match target {
            RunUntilTarget::Block(block) => self.current == block,
            RunUntilTarget::Trap => self.trap.is_some(),
            RunUntilTarget::Return => self.returned,
        }
    }

    fn is_halted(&self) -> bool {
        self.trap.is_some() || self.returned
    }

    fn location(&self) -> ProgramLocation {
        let at_terminator = self
            .current_block()
            .map(|block| self.next_inst >= block.insts.len())
            .unwrap_or(true);
        ProgramLocation {
            block: self.current,
            next_inst_index: self.next_inst,
            at_terminator,
        }
    }

    fn current_block(&self) -> Result<&cfg_mir::Block, DebuggerError> {
        let idx = *self
            .block_indices
            .get(&self.current)
            .ok_or(DebuggerError::UnknownBlock {
                block: self.current,
            })?;
        Ok(&self.func.blocks[idx])
    }

    fn snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            slots: self.slots.clone(),
            vregs: self.vregs.clone(),
            trap: self.trap,
            returned: self.returned,
            current: self.current,
            next_inst: self.next_inst,
            steps: self.steps,
            pointer_shadow: self.pointer_shadow.clone(),
        }
    }

    fn restore(&mut self, snapshot: SessionSnapshot) {
        self.slots = snapshot.slots;
        self.vregs = snapshot.vregs;
        self.trap = snapshot.trap;
        self.returned = snapshot.returned;
        self.current = snapshot.current;
        self.next_inst = snapshot.next_inst;
        self.steps = snapshot.steps;
        self.pointer_shadow = snapshot.pointer_shadow;
    }

    /// Read the concrete u64 value of a vreg (for execution).
    fn read_vreg(&self, idx: usize) -> u64 {
        self.vregs.get(idx).map(|v| v.concrete()).unwrap_or(0)
    }

    /// Read the tagged value of a vreg (for comparison/provenance tracking).
    fn read_vreg_tagged(&self, idx: usize) -> TaggedValue {
        self.vregs
            .get(idx)
            .copied()
            .unwrap_or(TaggedValue::Scalar(0))
    }

    /// Write a plain scalar value to a vreg.
    fn write_vreg(&mut self, idx: usize, value: u64) {
        self.write_vreg_tagged(idx, TaggedValue::Scalar(value));
    }

    /// Write a tagged value to a vreg.
    fn write_vreg_tagged(&mut self, idx: usize, value: TaggedValue) {
        if idx >= self.vregs.len() {
            self.vregs.resize(idx + 1, TaggedValue::Scalar(0));
        }
        self.vregs[idx] = value;
    }

    /// Seed shadow memory for pointer fields within a data_arg struct.
    /// Reads the actual pointer values from interpreter memory and records
    /// them with fresh PtrIds.
    fn seed_shadow_for_layout(
        &mut self,
        base_id: PtrId,
        base_addr: u64,
        layout: &kajit_types::DataArgLayout,
    ) {
        for field in &layout.pointer_fields {
            let field_addr = base_addr + field.offset;
            // Read the actual pointer value from interpreter memory
            let ptr_val = unsafe { *(field_addr as *const u64) };
            let field_ptr_id = self.alloc_ptr_id(ptr_val, &field.label);
            self.pointer_shadow.insert(
                (base_id, field.offset),
                ShadowEntry {
                    width: 8,
                    value: TaggedValue::Pointer {
                        id: field_ptr_id,
                        concrete: ptr_val,
                        offset: 0,
                    },
                },
            );
        }
    }

    /// Allocate a new PtrId and register its base address.
    fn alloc_ptr_id(&mut self, base_concrete: u64, _origin: &str) -> PtrId {
        let id = PtrId(self.next_ptr_id);
        self.next_ptr_id += 1;
        self.ptr_interp_bases.insert(id, base_concrete);
        id
    }

    fn ensure_slots_len(&mut self, len: usize) {
        if self.slots.len() < len {
            self.slots.resize(len, 0);
        }
    }

    fn trap(&mut self, code: ErrorCode) {
        if self.trap.is_none() {
            self.trap = Some(InterpreterTrap {
                code,
                offset: self.cursor_arg.pos as u32,
            });
        }
    }

    /// Invalidate shadow memory entries that overlap with a store at `[offset, offset+width)`.
    fn shadow_invalidate_range(&mut self, base_id: PtrId, offset: u64, width: u64) {
        self.pointer_shadow.retain(|&(pid, entry_offset), entry| {
            if pid != base_id {
                return true;
            }
            let entry_end = entry_offset + entry.width as u64;
            let store_end = offset + width;
            // Keep if no overlap
            entry_end <= offset || store_end <= entry_offset
        });
    }

    /// Invalidate all shadow memory (conservative, for opaque effectful calls).
    fn shadow_invalidate_all(&mut self) {
        self.pointer_shadow.clear();
    }

    /// Invalidate shadow entries for specific PtrIds (for calls that take pointer args).
    fn shadow_invalidate_for_ptrs(&mut self, ptr_ids: &[PtrId]) {
        self.pointer_shadow
            .retain(|&(pid, _), _| !ptr_ids.contains(&pid));
    }

    /// Compute the tagged result of a binary operation, preserving pointer provenance
    /// for Add/Sub with one pointer operand, destroying it for everything else.
    fn tagged_binop(
        &self,
        op: BinOpKind,
        lhs_tag: TaggedValue,
        rhs_tag: TaggedValue,
    ) -> TaggedValue {
        let lhs_val = lhs_tag.concrete();
        let rhs_val = rhs_tag.concrete();
        let result = exec_binop(op, lhs_val, rhs_val);

        match op {
            BinOpKind::Add => match (lhs_tag, rhs_tag) {
                (TaggedValue::Pointer { id, offset, .. }, TaggedValue::Scalar(s)) => {
                    TaggedValue::Pointer {
                        id,
                        concrete: result,
                        offset: offset.wrapping_add(s),
                    }
                }
                (TaggedValue::Scalar(s), TaggedValue::Pointer { id, offset, .. }) => {
                    TaggedValue::Pointer {
                        id,
                        concrete: result,
                        offset: offset.wrapping_add(s),
                    }
                }
                (TaggedValue::Pointer { .. }, TaggedValue::Pointer { .. }) => {
                    TaggedValue::UnknownPointer(result)
                }
                (_, TaggedValue::UnknownPointer(_)) | (TaggedValue::UnknownPointer(_), _) => {
                    TaggedValue::UnknownPointer(result)
                }
                _ => TaggedValue::Scalar(result),
            },
            BinOpKind::Sub => match (lhs_tag, rhs_tag) {
                (TaggedValue::Pointer { id, offset, .. }, TaggedValue::Scalar(s)) => {
                    TaggedValue::Pointer {
                        id,
                        concrete: result,
                        offset: offset.wrapping_sub(s),
                    }
                }
                (TaggedValue::Pointer { .. }, TaggedValue::Pointer { .. }) => {
                    // ptr - ptr = scalar difference
                    TaggedValue::Scalar(result)
                }
                (_, TaggedValue::UnknownPointer(_)) | (TaggedValue::UnknownPointer(_), _) => {
                    TaggedValue::UnknownPointer(result)
                }
                _ => TaggedValue::Scalar(result),
            },
            // Comparison ops always produce scalar results
            BinOpKind::CmpEq
            | BinOpKind::CmpNe
            | BinOpKind::CmpLt
            | BinOpKind::CmpLe
            | BinOpKind::CmpGt
            | BinOpKind::CmpGe => TaggedValue::Scalar(result),
            // All bitwise/mul/shift ops destroy provenance
            _ => {
                if lhs_tag.is_pointer() || rhs_tag.is_pointer() {
                    TaggedValue::UnknownPointer(result)
                } else {
                    TaggedValue::Scalar(result)
                }
            }
        }
    }

    fn execute_op(&mut self, block: cfg_mir::BlockId, op: &LinearOp) -> Result<(), DebuggerError> {
        match op {
            LinearOp::Const { dst, value } => self.write_vreg(dst.index(), *value),
            LinearOp::ExternAddr { dst, symbol } => {
                let addr = self.symbol_table.resolve(symbol).as_u64();
                let id = self.alloc_ptr_id(addr, &format!("extern {symbol:?}"));
                self.write_vreg_tagged(
                    dst.index(),
                    TaggedValue::Pointer {
                        id,
                        concrete: addr,
                        offset: 0,
                    },
                );
            }
            LinearOp::DataAddr { dst, blob_id } => {
                let blob = &self.data_blobs[*blob_id as usize];
                let addr = blob.as_ptr() as u64;
                let id = self.alloc_ptr_id(addr, &format!("data_blob[{blob_id}]"));
                self.write_vreg_tagged(
                    dst.index(),
                    TaggedValue::Pointer {
                        id,
                        concrete: addr,
                        offset: 0,
                    },
                );
            }
            LinearOp::Copy { dst, src } => {
                let tagged = self.read_vreg_tagged(src.index());
                self.write_vreg_tagged(dst.index(), tagged);
            }
            LinearOp::BinOp { op, dst, lhs, rhs } => {
                let lhs_tag = self.read_vreg_tagged(lhs.index());
                let rhs_tag = self.read_vreg_tagged(rhs.index());
                let result = self.tagged_binop(*op, lhs_tag, rhs_tag);
                self.write_vreg_tagged(dst.index(), result);
            }
            LinearOp::LoadFromAddr { dst, addr, width } => {
                let addr_tag = self.read_vreg_tagged(addr.index());
                let ptr = addr_tag.concrete() as *const u8;
                let width_bytes = width.bytes() as usize;
                let mut value = 0u64;
                for i in 0..width_bytes {
                    value |= (unsafe { ptr.add(i).read() } as u64) << (i * 8);
                }
                // Check shadow memory for pointer provenance
                let result_tag = if let TaggedValue::Pointer { id, offset, .. } = addr_tag {
                    if let Some(entry) = self.pointer_shadow.get(&(id, offset)) {
                        if entry.width as usize == width_bytes {
                            // Recover provenance, but update concrete from actual load
                            match entry.value {
                                TaggedValue::Pointer {
                                    id: stored_id,
                                    offset: stored_offset,
                                    ..
                                } => TaggedValue::Pointer {
                                    id: stored_id,
                                    concrete: value,
                                    offset: stored_offset,
                                },
                                TaggedValue::UnknownPointer(_) => {
                                    TaggedValue::UnknownPointer(value)
                                }
                                TaggedValue::Scalar(_) => TaggedValue::Scalar(value),
                            }
                        } else {
                            TaggedValue::Scalar(value)
                        }
                    } else {
                        TaggedValue::Scalar(value)
                    }
                } else {
                    TaggedValue::Scalar(value)
                };
                self.write_vreg_tagged(dst.index(), result_tag);
            }
            LinearOp::StoreToAddr { addr, src, width } => {
                let addr_tag = self.read_vreg_tagged(addr.index());
                let src_tag = self.read_vreg_tagged(src.index());
                let ptr = addr_tag.concrete() as *mut u8;
                let value = src_tag.concrete();
                let width_bytes = width.bytes() as usize;
                // Perform the actual store
                for i in 0..width_bytes {
                    unsafe { ptr.add(i).write(((value >> (i * 8)) & 0xff) as u8) };
                }
                // Update shadow memory
                if let TaggedValue::Pointer { id, offset, .. } = addr_tag {
                    // Invalidate overlapping entries
                    self.shadow_invalidate_range(id, offset, width_bytes as u64);
                    // Only shadow non-scalar provenance
                    if src_tag.is_pointer() {
                        self.pointer_shadow.insert(
                            (id, offset),
                            ShadowEntry {
                                width: width_bytes as u8,
                                value: src_tag,
                            },
                        );
                    }
                }
            }
            LinearOp::UnaryOp { op, dst, src } => {
                let src_val = self.read_vreg(src.index());
                let result = match op {
                    kajit_lir::UnaryOpKind::ZigzagDecode { wide } => {
                        if *wide {
                            let v = src_val as i64;
                            ((v >> 1) ^ -(v & 1)) as u64
                        } else {
                            let v = src_val as u32 as i32;
                            ((v >> 1) ^ -(v & 1)) as u32 as u64
                        }
                    }
                    kajit_lir::UnaryOpKind::SignExtend { from_width } => {
                        let bits = from_width.bytes() * 8;
                        let mask = 1u64 << (bits - 1);
                        (src_val ^ mask).wrapping_sub(mask)
                    }
                };
                // UnaryOps destroy pointer provenance
                self.write_vreg(dst.index(), result);
            }
            LinearOp::SlotAddr { dst, slot } => {
                let base = slot.index() * kajit_ir::SLOT_ADDR_STRIDE_BYTES;
                self.ensure_slots_len(base + kajit_ir::SLOT_ADDR_STRIDE_BYTES);
                let addr = unsafe { self.slots.as_ptr().add(base) as u64 };
                // Stable PtrId per slot index
                let id = self.alloc_ptr_id(addr, &format!("slot[{}]", slot.index()));
                self.write_vreg_tagged(
                    dst.index(),
                    TaggedValue::Pointer {
                        id,
                        concrete: addr,
                        offset: 0,
                    },
                );
            }
            LinearOp::StackAlloc { dst, id } => {
                let info = &self.stack_allocs_info[id.index()];
                let alloc = vec![0u8; info.size as usize];
                let ptr = alloc.as_ptr() as u64;
                self.stack_allocs.push(alloc);
                // Fresh PtrId per dynamic allocation
                let pid = self.alloc_ptr_id(ptr, &format!("stack_alloc[{}]", id.index()));
                self.write_vreg_tagged(
                    dst.index(),
                    TaggedValue::Pointer {
                        id: pid,
                        concrete: ptr,
                        offset: 0,
                    },
                );
            }
            LinearOp::WriteToSlot { src, slot } => {
                let value = self.read_vreg(src.index());
                let base = slot.index() * kajit_ir::SLOT_ADDR_STRIDE_BYTES;
                let width = 8; // slots are u64-sized
                self.ensure_slots_len(base + width);
                for i in 0..width {
                    self.slots[base + i] = ((value >> (i * 8)) & 0xff) as u8;
                }
            }
            LinearOp::ReadFromSlot { dst, slot } => {
                let base = slot.index() * kajit_ir::SLOT_ADDR_STRIDE_BYTES;
                let width = 8;
                self.ensure_slots_len(base + width);
                let mut value = 0u64;
                for i in 0..width {
                    value |= (self.slots[base + i] as u64) << (i * 8);
                }
                self.write_vreg(dst.index(), value);
            }
            LinearOp::CallIntrinsic { func, args, dst } => {
                // Collect PtrIds of pointer arguments for shadow invalidation
                let ptr_ids: Vec<PtrId> = args
                    .iter()
                    .filter_map(|v| match self.read_vreg_tagged(v.index()) {
                        TaggedValue::Pointer { id, .. } => Some(id),
                        _ => None,
                    })
                    .collect();
                self.execute_call_intrinsic(func.0, args, *dst)?;
                // Invalidate shadow for pointer args (effectful call)
                self.shadow_invalidate_for_ptrs(&ptr_ids);
            }
            LinearOp::CallPure { func, args, dst } => {
                // Pure calls have no side effects — no shadow invalidation needed
                self.execute_call_pure(func.0, args, *dst)?;
            }
            LinearOp::CallEffect { func, args, dst } => {
                // Effectful calls may write through pointers — invalidate conservatively
                self.execute_call_pure(func.0, args, *dst)?;
                self.shadow_invalidate_all();
            }
            op => {
                return Err(DebuggerError::UnsupportedOp {
                    block,
                    op: format!("{op:?}"),
                });
            }
        }

        Ok(())
    }

    /// Execute a call to an intrinsic function.
    ///
    /// Calls the real C function with the vreg values as arguments.
    /// The vregs already point to interpreter-owned memory (cursor, output, ctx).
    fn execute_call_intrinsic(
        &mut self,
        func_addr: usize,
        args: &[kajit_ir::VReg],
        dst: Option<kajit_ir::VReg>,
    ) -> Result<(), DebuggerError> {
        let call_args: Vec<u64> = args.iter().map(|v| self.read_vreg(v.index())).collect();

        let ret = unsafe {
            match call_args.len() {
                1 => {
                    let f: unsafe extern "C" fn(u64) -> u64 = core::mem::transmute(func_addr);
                    f(call_args[0])
                }
                2 => {
                    let f: unsafe extern "C" fn(u64, u64) -> u64 = core::mem::transmute(func_addr);
                    f(call_args[0], call_args[1])
                }
                3 => {
                    let f: unsafe extern "C" fn(u64, u64, u64) -> u64 =
                        core::mem::transmute(func_addr);
                    f(call_args[0], call_args[1], call_args[2])
                }
                4 => {
                    let f: unsafe extern "C" fn(u64, u64, u64, u64) -> u64 =
                        core::mem::transmute(func_addr);
                    f(call_args[0], call_args[1], call_args[2], call_args[3])
                }
                5 => {
                    let f: unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64 =
                        core::mem::transmute(func_addr);
                    f(
                        call_args[0],
                        call_args[1],
                        call_args[2],
                        call_args[3],
                        call_args[4],
                    )
                }
                _ => {
                    return Err(DebuggerError::UnsupportedOp {
                        block: self.current,
                        op: format!(
                            "CallIntrinsic with {} args (unsupported arity)",
                            call_args.len()
                        ),
                    });
                }
            }
        };

        // Check for errors from ctx.
        if self.ctx.error.code != 0 {
            self.trap(error_code_from_u32(self.ctx.error.code));
        }

        // Store return value.
        if let Some(dst) = dst {
            self.write_vreg(dst.index(), ret);
        }

        Ok(())
    }

    /// Execute a pure function call (no context, no side effects).
    fn execute_call_pure(
        &mut self,
        func_addr: usize,
        args: &[kajit_ir::VReg],
        dst: kajit_ir::VReg,
    ) -> Result<(), DebuggerError> {
        let arg_values: Vec<u64> = args.iter().map(|v| self.read_vreg(v.index())).collect();
        let ret = unsafe {
            match arg_values.len() {
                0 => {
                    let f: unsafe extern "C" fn() -> u64 = core::mem::transmute(func_addr);
                    f()
                }
                1 => {
                    let f: unsafe extern "C" fn(u64) -> u64 = core::mem::transmute(func_addr);
                    f(arg_values[0])
                }
                2 => {
                    let f: unsafe extern "C" fn(u64, u64) -> u64 = core::mem::transmute(func_addr);
                    f(arg_values[0], arg_values[1])
                }
                3 => {
                    let f: unsafe extern "C" fn(u64, u64, u64) -> u64 =
                        core::mem::transmute(func_addr);
                    f(arg_values[0], arg_values[1], arg_values[2])
                }
                4 => {
                    let f: unsafe extern "C" fn(u64, u64, u64, u64) -> u64 =
                        core::mem::transmute(func_addr);
                    f(arg_values[0], arg_values[1], arg_values[2], arg_values[3])
                }
                5 => {
                    let f: unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64 =
                        core::mem::transmute(func_addr);
                    f(
                        arg_values[0],
                        arg_values[1],
                        arg_values[2],
                        arg_values[3],
                        arg_values[4],
                    )
                }
                6 => {
                    let f: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 =
                        core::mem::transmute(func_addr);
                    f(
                        arg_values[0],
                        arg_values[1],
                        arg_values[2],
                        arg_values[3],
                        arg_values[4],
                        arg_values[5],
                    )
                }
                _ => {
                    return Err(DebuggerError::UnsupportedOp {
                        block: self.current,
                        op: format!(
                            "CallPure with {} args (unsupported arity)",
                            arg_values.len()
                        ),
                    });
                }
            }
        };
        self.write_vreg(dst.index(), ret);
        Ok(())
    }

    fn execute_terminator(
        &mut self,
        block_id: cfg_mir::BlockId,
        term: &cfg_mir::Terminator,
    ) -> Result<(), DebuggerError> {
        match term {
            cfg_mir::Terminator::Return => {
                self.returned = true;
            }
            cfg_mir::Terminator::Branch { edge } => {
                self.apply_edge(*edge)?;
                self.next_inst = 0;
            }
            cfg_mir::Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => {
                let cond_val = self.read_vreg(cond.index());
                let next = if cond_val != 0 { *taken } else { *fallthrough };
                if std::env::var("KAJIT_LOCKSTEP_TRACE").is_ok() {
                    let taken_to = self.func.edge(*taken).map(|e| e.to.index()).unwrap_or(999);
                    let fall_to = self
                        .func
                        .edge(*fallthrough)
                        .map(|e| e.to.index())
                        .unwrap_or(999);
                    let next_to = self.func.edge(next).map(|e| e.to.index()).unwrap_or(999);
                    eprintln!(
                        "[interp] branch_if v{}={} in b{}: taken=e{}→b{}, fall=e{}→b{}, going to e{}→b{}",
                        cond.index(),
                        cond_val,
                        block_id.index(),
                        taken.index(),
                        taken_to,
                        fallthrough.index(),
                        fall_to,
                        next.index(),
                        next_to,
                    );
                }
                self.apply_edge(next)?;
                self.next_inst = 0;
            }
            cfg_mir::Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => {
                let next = if self.read_vreg(cond.index()) == 0 {
                    *taken
                } else {
                    *fallthrough
                };
                self.apply_edge(next)?;
                self.next_inst = 0;
            }
            term => {
                return Err(DebuggerError::UnsupportedTerminator {
                    block: block_id,
                    term: format!("{term:?}"),
                });
            }
        }

        Ok(())
    }

    fn apply_edge(&mut self, edge_id: cfg_mir::EdgeId) -> Result<(), DebuggerError> {
        let edge = self
            .func
            .edge(edge_id)
            .ok_or(DebuggerError::UnknownEdge { edge: edge_id })?;
        let from = edge.from;
        let to = edge.to;
        let to_idx = *self
            .block_indices
            .get(&to)
            .ok_or(DebuggerError::UnknownBlock { block: to })?;
        let to_block = &self.func.blocks[to_idx];

        if edge.args.len() != to_block.params.len() {
            return Err(DebuggerError::EdgeArgArityMismatch {
                from,
                to,
                expected: to_block.params.len(),
                got: edge.args.len(),
            });
        }

        let transfers: Vec<(usize, TaggedValue)> = edge
            .args
            .iter()
            .map(|arg| {
                (
                    arg.target.index(),
                    self.read_vreg_tagged(arg.source.index()),
                )
            })
            .collect();
        for (target, value) in transfers {
            self.write_vreg_tagged(target, value);
        }
        self.current = to;
        Ok(())
    }
}

fn build_block_index(func: &cfg_mir::Function) -> HashMap<cfg_mir::BlockId, usize> {
    let mut out = HashMap::with_capacity(func.blocks.len());
    for (idx, block) in func.blocks.iter().enumerate() {
        out.insert(block.id, idx);
    }
    out
}

fn infer_output_size(func: &cfg_mir::Function) -> usize {
    let from_fields = func
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|_inst_id| -> Option<usize> { None })
        .max()
        .unwrap_or(0);
    from_fields.max(func.output_size)
}

fn exec_binop(op: BinOpKind, lhs: u64, rhs: u64) -> u64 {
    match op {
        BinOpKind::Add => lhs.wrapping_add(rhs),
        BinOpKind::Sub => lhs.wrapping_sub(rhs),
        BinOpKind::Mul => lhs.wrapping_mul(rhs),
        BinOpKind::And => lhs & rhs,
        BinOpKind::Or => lhs | rhs,
        BinOpKind::Xor => lhs ^ rhs,
        BinOpKind::Shl => {
            if rhs >= 64 {
                0
            } else {
                lhs.wrapping_shl(rhs as u32)
            }
        }
        BinOpKind::Shr => {
            if rhs >= 64 {
                0
            } else {
                lhs.wrapping_shr(rhs as u32)
            }
        }
        BinOpKind::Sar => {
            if rhs >= 64 {
                // Arithmetic shift by 64+ bits fills with sign bit
                ((lhs as i64) >> 63) as u64
            } else {
                ((lhs as i64).wrapping_shr(rhs as u32)) as u64
            }
        }
        BinOpKind::CmpEq => u64::from(lhs == rhs),
        BinOpKind::CmpNe => u64::from(lhs != rhs),
        BinOpKind::CmpLt => u64::from(lhs < rhs),
        BinOpKind::CmpLe => u64::from(lhs <= rhs),
        BinOpKind::CmpGt => u64::from(lhs > rhs),
        BinOpKind::CmpGe => u64::from(lhs >= rhs),
    }
}

#[cfg(test)]
mod tests {
    use kajit_ir::{LambdaId, VReg};
    use kajit_lir::LinearOp;

    use crate::{DebuggerSession, cfg_mir};

    fn v(index: u32) -> VReg {
        VReg::new(index)
    }

    fn test_inst(id: u32, op: LinearOp) -> cfg_mir::Inst {
        cfg_mir::Inst {
            id: cfg_mir::InstId(id),
            op,
            operands: Vec::new(),
            clobbers: cfg_mir::Clobbers::default(),
        }
    }

    fn make_simple_program() -> cfg_mir::Program {
        let b0 = cfg_mir::Block {
            id: cfg_mir::BlockId(0),
            params: Vec::new(),
            insts: vec![cfg_mir::InstId(0)],
            term: cfg_mir::TermId(0),
            preds: Vec::new(),
            succs: Vec::new(),
            dead: false,
        };
        cfg_mir::Program {
            funcs: vec![cfg_mir::Function {
                id: cfg_mir::FunctionId(0),
                lambda_id: LambdaId::new(0),
                entry: cfg_mir::BlockId(0),
                data_args: Vec::new(),
                data_results: Vec::new(),
                output_size: 0,
                blocks: vec![b0],
                edges: Vec::new(),
                insts: vec![test_inst(
                    0,
                    LinearOp::Const {
                        dst: v(0),
                        value: 0x2a,
                    },
                )],
                terms: vec![cfg_mir::Terminator::Return],
            }],
            vreg_count: 1,
            slot_count: 0,
            param_slot_count: 0,
            debug: Default::default(),
            hints: Default::default(),
            extra_excluded_regs: vec![],
            data_blobs: vec![],
            stack_allocs: vec![],
            data_arg_layouts: vec![],
        }
    }

    fn make_trap_program() -> cfg_mir::Program {
        let b0 = cfg_mir::Block {
            id: cfg_mir::BlockId(0),
            params: Vec::new(),
            insts: vec![cfg_mir::InstId(0)],
            term: cfg_mir::TermId(0),
            preds: Vec::new(),
            succs: Vec::new(),
            dead: false,
        };
        cfg_mir::Program {
            funcs: vec![cfg_mir::Function {
                id: cfg_mir::FunctionId(0),
                lambda_id: LambdaId::new(0),
                entry: cfg_mir::BlockId(0),
                data_args: Vec::new(),
                data_results: Vec::new(),
                output_size: 0,
                blocks: vec![b0],
                edges: Vec::new(),
                insts: vec![test_inst(
                    0,
                    LinearOp::Const {
                        dst: VReg::new(0),
                        value: 0,
                    },
                )],
                terms: vec![cfg_mir::Terminator::Return],
            }],
            vreg_count: 0,
            slot_count: 0,
            param_slot_count: 0,
            debug: Default::default(),
            hints: Default::default(),
            extra_excluded_regs: vec![],
            data_blobs: vec![],
            stack_allocs: vec![],
            data_arg_layouts: vec![],
        }
    }

    #[test]
    fn step_forward_and_back_restores_state() {
        let program = make_simple_program();
        let mut session = DebuggerSession::new(&program, &[], &kajit_types::Arguments::new())
            .expect("debugger should init");

        let first = session.step_forward().expect("step should work");
        assert_eq!(first.kind, crate::StepKind::Instruction);
        assert_eq!(session.inspect_vreg(0), 0x2a);

        assert!(session.step_back());
        assert_eq!(session.inspect_vreg(0), 0);

        assert!(!session.step_back());
    }

    #[test]
    fn simple_program_executes_and_returns() {
        let program = make_trap_program();
        let mut session = DebuggerSession::new(&program, &[], &kajit_types::Arguments::new())
            .expect("debugger should init");
        // Step through const op
        let event = session.step_forward().expect("step should work");
        assert!(!event.halted_after, "const op should not halt");
        // Step through return terminator
        let event = session.step_forward().expect("step should work");
        assert!(event.halted_after, "return should halt");
        assert!(event.trap.is_none(), "should not trap");
    }
}
