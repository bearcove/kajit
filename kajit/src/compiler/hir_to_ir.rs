#![allow(dead_code)]
//! HIR→IR lowering — converts hir::Module into RVSDG (ir::IrFunc).

use super::*;

/// Public wrapper for word_count_for_type — used by extract_data_arg_layouts.
pub(super) fn word_count_for_type(module: &hir::Module, ty: &hir::Type) -> usize {
    ScalarHirIrLowerer::word_count_for_type(module, ty)
}

// ─── Region interface analysis ─────────────────────────────────────────────
//
// Before lowering a theta/gamma, we analyze the HIR block to determine which
// locals cross the region boundary. This produces a RegionInterface that tells
// the lowerer exactly what to thread through.

/// The set of locals that cross a region boundary.
#[derive(Debug)]
struct RegionInterface {
    /// Locals read inside the region whose value comes from outside.
    /// These must be passed in as region args.
    captures: Vec<hir::LocalId>,
    /// Locals written inside the region whose post-region value matters.
    /// For gamma: these are locals whose updated values must come out.
    /// For theta: these are loop-carried variables (read + written).
    live_out: Vec<hir::LocalId>,
}

/// Collect all LocalIds read by an expression.
fn expr_reads(expr: &hir::Expr, out: &mut std::collections::HashSet<hir::LocalId>) {
    match expr {
        hir::Expr::Local(id) => {
            out.insert(*id);
        }
        hir::Expr::Literal(_) => {}
        hir::Expr::Deref(e) => expr_reads(e, out),
        hir::Expr::Load { addr, .. } => expr_reads(addr, out),
        hir::Expr::SliceData { value } | hir::Expr::SliceLen { value } => expr_reads(value, out),
        hir::Expr::Str { data, len } => {
            expr_reads(data, out);
            expr_reads(len, out);
        }
        hir::Expr::Field { base, .. } => expr_reads(base, out),
        hir::Expr::Index { base, index } => {
            expr_reads(base, out);
            expr_reads(index, out);
        }
        hir::Expr::AddrOf(place) => place_reads(place, out),
        hir::Expr::Struct { fields, .. } | hir::Expr::Variant { fields, .. } => {
            for (_, e) in fields {
                expr_reads(e, out);
            }
        }
        hir::Expr::Unary { value, .. } => expr_reads(value, out),
        hir::Expr::Binary { lhs, rhs, .. } => {
            expr_reads(lhs, out);
            expr_reads(rhs, out);
        }
        hir::Expr::Call(call) => {
            for arg in &call.args {
                expr_reads(arg, out);
            }
        }
    }
}

/// Collect all LocalIds read by a place (including nested expressions).
fn place_reads(place: &hir::Place, out: &mut std::collections::HashSet<hir::LocalId>) {
    match place {
        hir::Place::Local(id) => {
            out.insert(*id);
        }
        hir::Place::Deref { base } => expr_reads(base, out),
        hir::Place::Field { base, .. } => place_reads(base, out),
        hir::Place::Index { base, index } => {
            place_reads(base, out);
            expr_reads(index, out);
        }
    }
}

/// Collect all LocalIds written by a place (the target of Init/Assign).
fn place_writes(place: &hir::Place, out: &mut std::collections::HashSet<hir::LocalId>) {
    match place {
        hir::Place::Local(id) => {
            out.insert(*id);
        }
        hir::Place::Field { base, .. } => place_writes(base, out),
        // Deref and Index write to memory, not to a local
        hir::Place::Deref { .. } | hir::Place::Index { .. } => {}
    }
}

/// Collect reads and writes for a block of statements.
fn block_reads_writes(
    stmts: &[hir::Stmt],
    reads: &mut std::collections::HashSet<hir::LocalId>,
    writes: &mut std::collections::HashSet<hir::LocalId>,
) {
    for stmt in stmts {
        match &stmt.kind {
            hir::StmtKind::Init { place, value } | hir::StmtKind::Assign { place, value } => {
                expr_reads(value, reads);
                // Place reads (e.g., Deref base, Index base)
                place_reads(place, reads);
                place_writes(place, writes);
            }
            hir::StmtKind::Store { addr, value, .. } => {
                expr_reads(addr, reads);
                expr_reads(value, reads);
            }
            hir::StmtKind::Expr(e) => expr_reads(e, reads),
            hir::StmtKind::If {
                condition,
                then_block,
                else_block,
            } => {
                expr_reads(condition, reads);
                block_reads_writes(&then_block.statements, reads, writes);
                if let Some(eb) = else_block {
                    block_reads_writes(&eb.statements, reads, writes);
                }
            }
            hir::StmtKind::Match { scrutinee, arms } => {
                expr_reads(scrutinee, reads);
                for arm in arms {
                    block_reads_writes(&arm.body.statements, reads, writes);
                }
            }
            hir::StmtKind::Loop { body, .. } => {
                block_reads_writes(&body.statements, reads, writes);
            }
            hir::StmtKind::Return(Some(e)) => expr_reads(e, reads),
            hir::StmtKind::Return(None) | hir::StmtKind::Break | hir::StmtKind::Continue => {}
        }
    }
}

/// Compute the region interface for a block that will become a theta/gamma body.
/// `outer_locals` is the set of locals defined in the enclosing scope.
fn compute_region_interface(
    stmts: &[hir::Stmt],
    outer_locals: &std::collections::HashSet<hir::LocalId>,
) -> RegionInterface {
    let mut reads = std::collections::HashSet::new();
    let mut writes = std::collections::HashSet::new();
    block_reads_writes(stmts, &mut reads, &mut writes);

    // Captures: locals read inside that are defined outside
    let captures: Vec<hir::LocalId> = reads
        .iter()
        .copied()
        .filter(|id| outer_locals.contains(id))
        .collect();

    // Live-out: locals written inside whose updated value needs to flow back out.
    // This includes both locals already defined outside (updated) and locals
    // first defined inside (born) — both need gamma/theta outputs.
    let live_out: Vec<hir::LocalId> = writes.iter().copied().collect();

    // Deterministic ordering for stable IR generation
    let mut captures = captures;
    let mut live_out = live_out;
    captures.sort();
    live_out.sort();

    RegionInterface { captures, live_out }
}

/// Compute the set of locals that need to be threaded through a theta body.
/// This is the union of captures and live_out — a local that is both read
/// and written is loop-carried.
fn theta_loop_vars(iface: &RegionInterface) -> Vec<hir::LocalId> {
    let mut all: Vec<hir::LocalId> = iface.captures.clone();
    for id in &iface.live_out {
        if !all.contains(id) {
            all.push(*id);
        }
    }
    all.sort();
    all
}

// ─── Runtime dialect lowering ──────────────────────────────────────────────

struct RuntimeDialectLowerer;

impl RuntimeDialectLowerer {
    fn requires_memory_state(intrinsic: Option<hir::RuntimeIntrinsic>) -> bool {
        matches!(
            intrinsic,
            Some(
                hir::RuntimeIntrinsic::AllocTransient
                    | hir::RuntimeIntrinsic::Memcpy
                    | hir::RuntimeIntrinsic::FreeTransient
            )
        )
    }

    fn lower_scalar_value_call(
        rb: &mut RegionBuilder<'_>,
        callable: &hir::CallableSpec,
        args: &[crate::ir::PortSource],
    ) -> Option<crate::ir::PortSource> {
        match callable.intrinsic {
            Some(hir::RuntimeIntrinsic::AllocTransient) => Some(rb.call_effect(
                crate::ir::IntrinsicFn(intrinsics::kajit_alloc_transient as *const () as usize),
                args,
            )),
            Some(hir::RuntimeIntrinsic::Memcpy) => Some(rb.call_effect(
                crate::ir::IntrinsicFn(intrinsics::kajit_memcpy as *const () as usize),
                args,
            )),
            Some(hir::RuntimeIntrinsic::AllocPersistent) => Some(
                rb.call_intrinsic(
                    crate::ir::IntrinsicFn(
                        intrinsics::kajit_alloc_persistent as *const () as usize,
                    ),
                    args,
                    true,
                )
                .expect("alloc_persistent returns a value"),
            ),
            Some(hir::RuntimeIntrinsic::StringValidateAllocCopy) => Some(
                rb.call_intrinsic(
                    crate::ir::IntrinsicFn(
                        intrinsics::kajit_string_validate_alloc_copy as *const () as usize,
                    ),
                    args,
                    true,
                )
                .expect("string_validate_alloc_copy returns a value"),
            ),
            _ => None,
        }
    }

    fn lower_scalar_effect_call(
        rb: &mut RegionBuilder<'_>,
        callable: &hir::CallableSpec,
        args: &[crate::ir::PortSource],
    ) -> bool {
        match callable.intrinsic {
            Some(hir::RuntimeIntrinsic::FreeTransient) => {
                let _ = rb.call_effect(
                    crate::ir::IntrinsicFn(intrinsics::kajit_free_transient as *const () as usize),
                    args,
                );
                true
            }
            Some(hir::RuntimeIntrinsic::OptionInitNone) => {
                // args: [ctx, init_fn, out_addr]
                rb.call_intrinsic(
                    crate::ir::IntrinsicFn(
                        intrinsics::kajit_option_init_none_ctx as *const () as usize,
                    ),
                    args,
                    false,
                );
                true
            }
            Some(hir::RuntimeIntrinsic::OptionInitSome) => {
                // args: [ctx, init_fn, out_addr, payload_addr]
                rb.call_intrinsic(
                    crate::ir::IntrinsicFn(
                        intrinsics::kajit_option_init_some_ctx as *const () as usize,
                    ),
                    args,
                    false,
                );
                true
            }
            Some(hir::RuntimeIntrinsic::ValidateUtf8Range) => {
                rb.call_intrinsic(
                    crate::ir::IntrinsicFn(
                        intrinsics::kajit_validate_utf8_range as *const () as usize,
                    ),
                    args,
                    false,
                );
                true
            }
            Some(hir::RuntimeIntrinsic::Memcpy) => {
                let _ = rb.call_effect(
                    crate::ir::IntrinsicFn(intrinsics::kajit_memcpy as *const () as usize),
                    args,
                );
                true
            }
            _ => false,
        }
    }
}

/// Lower HIR to IR without any facet Shape. All layout info must come from
/// HIR annotations.
pub fn lower_hir_module(module: &hir::Module) -> crate::ir::IrFunc {
    let (_, function) = module
        .functions
        .iter()
        .next()
        .expect("HIR module should contain at least one function");
    build_scalar_hir_ir(module, function)
}

/// Lower a plain scalar HIR function into IR.
///
/// This is the clean lowering path for functions that take ordinary params
/// and return a value — no cursor, no destination, no decoder-specific
/// machinery. All params get slots, all locals get slots, expressions lower
/// to IR operations, and `return expr` sets the region result.
/// Check if an HIR function body uses effect callables that require the MEMORY state domain.
fn build_scalar_hir_ir(module: &hir::Module, function: &hir::Function) -> crate::ir::IrFunc {
    // Count the total number of u64-sized words across all params.
    let param_word_count: usize = function
        .params
        .iter()
        .map(|p| ScalarHirIrLowerer::word_count_for_type(module, &p.ty))
        .sum();

    let (mut builder, data_arg_sources) =
        crate::ir::IrBuilder::new_with_data_args(&function.name, 0, param_word_count);
    // TODO: When the function doesn't use effect calls, we could skip
    // threading the memory state domain. For now, it's always present as a builtin.
    {
        let mut rb = builder.root_region();
        let mut lowerer = ScalarHirIrLowerer::new(module, function, &data_arg_sources);
        let ret = lowerer.lower_block(&mut rb, &function.body.statements);
        match ret {
            Some(ret_vals) => rb.set_results(&ret_vals),
            None => rb.set_results(&[]),
        }
    }
    let mut func = builder.finish();
    func.param_slot_count = param_word_count as u32;
    func
}

struct ScalarHirIrLowerer<'a> {
    module: &'a hir::Module,
    /// Port sources for each local (params and locals). Multi-word types
    /// have multiple port sources (one per u64 word).
    local_values: std::collections::HashMap<hir::LocalId, Vec<crate::ir::PortSource>>,
    /// Type of each local, for field resolution.
    local_types: std::collections::HashMap<hir::LocalId, &'a hir::Type>,
}

impl<'a> ScalarHirIrLowerer<'a> {
    fn new(
        module: &'a hir::Module,
        function: &'a hir::Function,
        data_arg_sources: &[crate::ir::PortSource],
    ) -> Self {
        let mut local_values = std::collections::HashMap::new();
        let mut local_types = std::collections::HashMap::new();

        // Params: consume data arg port sources in order.
        let mut arg_cursor = 0;
        for param in &function.params {
            let word_count = Self::word_count_for_type(module, &param.ty);
            let sources = data_arg_sources[arg_cursor..arg_cursor + word_count].to_vec();
            arg_cursor += word_count;
            local_values.insert(param.local, sources);
            local_types.insert(param.local, &param.ty);
        }
        assert_eq!(arg_cursor, data_arg_sources.len());

        // Locals: register types. Values are populated by Let or field writes.
        for local in &function.locals {
            local_types.insert(local.local, &local.ty);
        }

        Self {
            module,
            local_values,
            local_types,
        }
    }

    /// Get the set of currently defined locals (for region interface analysis).
    fn defined_locals(&self) -> std::collections::HashSet<hir::LocalId> {
        self.local_values.keys().copied().collect()
    }

    /// Flatten a set of locals into a contiguous port source vector.
    /// Returns (local_ids_in_order, flat_port_sources).
    fn flatten_locals(&self, locals: &[hir::LocalId]) -> Vec<crate::ir::PortSource> {
        let mut flat = Vec::new();
        for id in locals {
            let values = self.local_values.get(id).unwrap_or_else(|| {
                panic!("flatten_locals: local {id:?} not defined");
            });
            flat.extend_from_slice(values);
        }
        flat
    }

    fn flatten_locals_with_zeros(
        &self,
        rb: &mut RegionBuilder<'_>,
        locals: &[hir::LocalId],
    ) -> Vec<crate::ir::PortSource> {
        let mut flat = Vec::new();
        for id in locals {
            if let Some(values) = self.local_values.get(id) {
                flat.extend_from_slice(values);
            } else {
                // Local not yet defined — provide zero placeholders so the
                // gamma/theta invariant count stays consistent.
                let word_count = Self::word_count_for_type(self.module, self.local_types[id]);
                let zero = rb.const_val(0);
                for _ in 0..word_count {
                    flat.push(zero);
                }
            }
        }
        flat
    }

    /// Remap locals from a flat port source vector (e.g. region args or theta outputs).
    fn remap_locals(&mut self, locals: &[hir::LocalId], sources: &[crate::ir::PortSource]) {
        let mut cursor = 0;
        for id in locals {
            let word_count = self
                .local_values
                .get(id)
                .map(|v| v.len())
                .unwrap_or_else(|| Self::word_count_for_type(self.module, self.local_types[id]));
            let new_values = sources[cursor..cursor + word_count].to_vec();
            cursor += word_count;
            self.local_values.insert(*id, new_values);
        }
        assert_eq!(cursor, sources.len(), "remap_locals: source count mismatch");
    }

    /// Number of u64-sized words needed to represent a type.
    fn word_count_for_type(module: &hir::Module, ty: &hir::Type) -> usize {
        match ty {
            hir::Type::Unit
            | hir::Type::Bool
            | hir::Type::Integer(_)
            | hir::Type::Ref { .. }
            | hir::Type::Address { .. }
            | hir::Type::Handle { .. } => 1,
            hir::Type::Str { .. } | hir::Type::Slice { .. } => 2,
            hir::Type::Array { element, len } => Self::word_count_for_type(module, element)
                .saturating_mul(*len)
                .max(1),
            hir::Type::Named { def, .. } => match &module.type_defs[*def].kind {
                hir::TypeDefKind::Struct { fields } => fields
                    .iter()
                    .map(|f| Self::word_count_for_type(module, &f.ty))
                    .sum::<usize>()
                    .max(1),
                hir::TypeDefKind::Enum { variants, .. } => {
                    let payload_words = variants
                        .iter()
                        .map(|v| {
                            v.fields
                                .iter()
                                .map(|f| Self::word_count_for_type(module, &f.ty))
                                .sum::<usize>()
                        })
                        .max()
                        .unwrap_or(0);
                    (1 + payload_words).max(1)
                }
            },
        }
    }

    /// Resolve a local + field path to the type and word offset within the
    /// local's port source vector.
    fn resolve_local_field(
        &self,
        local: hir::LocalId,
        fields: &[(String, &hir::Type)],
    ) -> (usize, &'a hir::Type) {
        let mut offset = 0usize;
        let mut ty = self.local_types[&local];
        for (field_name, _) in fields {
            let hir::Type::Named { def, .. } = ty else {
                panic!("field access requires a named struct type, got {:?}", ty);
            };
            let hir::TypeDefKind::Struct { fields: field_defs } = &self.module.type_defs[*def].kind
            else {
                panic!("field access requires a struct type def");
            };
            let mut running = 0usize;
            let (field_ty, field_offset) = field_defs
                .iter()
                .find_map(|candidate| {
                    let found =
                        (candidate.name == field_name.as_str()).then_some((&candidate.ty, running));
                    running += Self::word_count_for_type(self.module, &candidate.ty);
                    found
                })
                .unwrap_or_else(|| panic!("missing struct field {field_name}"));
            offset += field_offset;
            ty = field_ty;
        }
        (offset, ty)
    }

    /// Collect the chain of field accesses from an expression.
    /// Returns (base local, list of field names with types).
    fn expr_field_chain(expr: &hir::Expr) -> (hir::LocalId, Vec<(String, &hir::Type)>) {
        match expr {
            hir::Expr::Local(local) => (*local, vec![]),
            hir::Expr::Field { base, field } => {
                let (local, mut chain) = Self::expr_field_chain(base);
                chain.push((field.clone(), &hir::Type::Unit));
                (local, chain)
            }
            other => panic!("expected local or field chain, got {other:?}"),
        }
    }

    /// Non-panicking version of `expr_field_chain`. Returns `None` when the
    /// base is not a local-rooted field chain.
    fn try_expr_field_chain(expr: &hir::Expr) -> Option<(hir::LocalId, Vec<(String, &hir::Type)>)> {
        match expr {
            hir::Expr::Local(local) => Some((*local, vec![])),
            hir::Expr::Field { base, field } => {
                let (local, mut chain) = Self::try_expr_field_chain(base)?;
                chain.push((field.clone(), &hir::Type::Unit));
                Some((local, chain))
            }
            _ => None,
        }
    }

    /// Infer the type of an HIR expression. Used by the type-driven field
    /// projection fallback when the base is not a local-rooted chain.
    fn infer_expr_type(&self, expr: &hir::Expr) -> hir::Type {
        match expr {
            hir::Expr::Local(id) => self.local_types[id].clone(),
            hir::Expr::Deref(base) => {
                let base_ty = self.infer_expr_type(base);
                match base_ty {
                    hir::Type::Ref { pointee, .. } => *pointee,
                    other => panic!("cannot deref non-ref expression of type {other:?}"),
                }
            }
            hir::Expr::Literal(hir::Literal::Bool(_)) => hir::Type::Bool,
            hir::Expr::Literal(hir::Literal::Integer(_)) => hir::Type::u(64),
            hir::Expr::Literal(hir::Literal::ExternAddr { .. }) => hir::Type::u(64),
            hir::Expr::Literal(hir::Literal::String(_)) => {
                // String literals produce (ptr, len) — a built-in Str type.
                hir::Type::Str {
                    region: hir::RegionId::new(0),
                }
            }
            hir::Expr::Literal(hir::Literal::Unit) => hir::Type::Unit,
            hir::Expr::Struct { def, .. } => hir::Type::Named {
                def: *def,
                args: vec![],
            },
            hir::Expr::Field { base, field } => {
                let base_ty = self.infer_expr_type(base);
                self.resolve_field_type_in(&base_ty, field)
            }
            hir::Expr::Binary { op, .. } => {
                // Comparison ops produce bool, arithmetic ops produce u64.
                match op {
                    hir::BinaryOp::Eq
                    | hir::BinaryOp::Ne
                    | hir::BinaryOp::Lt
                    | hir::BinaryOp::Le
                    | hir::BinaryOp::Gt
                    | hir::BinaryOp::Ge => hir::Type::Bool,
                    _ => hir::Type::u(64),
                }
            }
            hir::Expr::Unary { .. } => hir::Type::u(64),
            hir::Expr::Call(call) => {
                let callable = &self.module.callables[match call.target {
                    hir::CallTarget::Callable(id) => id,
                }];
                callable
                    .signature
                    .returns
                    .first()
                    .cloned()
                    .unwrap_or(hir::Type::Unit)
            }
            hir::Expr::SliceData { .. } | hir::Expr::SliceLen { .. } => hir::Type::u(64),
            hir::Expr::Load { .. } => hir::Type::u(64),
            other => panic!("infer_expr_type: cannot infer type of expression: {other:?}"),
        }
    }

    /// Resolve the type of a field within a parent type.
    fn resolve_field_type_in(&self, ty: &hir::Type, field: &str) -> hir::Type {
        match ty {
            hir::Type::Ref { pointee, .. } => self.resolve_field_type_in(pointee, field),
            hir::Type::Named { def, .. } => {
                let hir::TypeDefKind::Struct { fields } = &self.module.type_defs[*def].kind else {
                    panic!("field access on non-struct named type: {:?}", ty);
                };
                fields
                    .iter()
                    .find(|f| f.name == field)
                    .map(|f| f.ty.clone())
                    .unwrap_or_else(|| panic!("field '{field}' not found in struct"))
            }
            hir::Type::Str { .. } => match field {
                "ptr" | "data" => hir::Type::u(64),
                "len" => hir::Type::u(64),
                _ => panic!("unknown Str field: '{field}'"),
            },
            hir::Type::Slice { .. } => match field {
                "data" | "ptr" => hir::Type::u(64),
                "len" => hir::Type::u(64),
                _ => panic!("unknown Slice field: '{field}'"),
            },
            _ => panic!("field access on type that has no fields: {:?}", ty),
        }
    }

    /// Compute the word offset and word count of a field within a type.
    /// Returns `(word_offset, field_word_count)`.
    fn field_offset_in_type(&self, ty: &hir::Type, field_name: &str) -> (usize, usize) {
        match ty {
            hir::Type::Named { def, .. } => {
                let hir::TypeDefKind::Struct { fields } = &self.module.type_defs[*def].kind else {
                    panic!("field_offset_in_type: not a struct: {:?}", ty);
                };
                let mut offset = 0;
                for f in fields {
                    let wc = Self::word_count_for_type(self.module, &f.ty);
                    if f.name == field_name {
                        return (offset, wc);
                    }
                    offset += wc;
                }
                panic!("field '{field_name}' not found in struct {:?}", def);
            }
            hir::Type::Str { .. } => match field_name {
                "ptr" | "data" => (0, 1),
                "len" => (1, 1),
                _ => panic!("unknown Str field: '{field_name}'"),
            },
            hir::Type::Slice { .. } => match field_name {
                "data" | "ptr" => (0, 1),
                "len" => (1, 1),
                _ => panic!("unknown Slice field: '{field_name}'"),
            },
            _ => panic!(
                "field_offset_in_type: type {:?} has no field '{}'",
                ty, field_name
            ),
        }
    }

    /// Collect the chain of field accesses from a place.
    fn place_field_chain(place: &hir::Place) -> (hir::LocalId, Vec<(String, &hir::Type)>) {
        match place {
            hir::Place::Local(local) => (*local, vec![]),
            hir::Place::Field { base, field } => {
                let (local, mut chain) = Self::place_field_chain(base);
                chain.push((field.clone(), &hir::Type::Unit));
                (local, chain)
            }
            other => panic!("expected local or field place, got {other:?}"),
        }
    }

    /// Get the port sources for an expression (one per word).
    fn get_local_values(&self, local: hir::LocalId) -> &[crate::ir::PortSource] {
        self.local_values
            .get(&local)
            .unwrap_or_else(|| panic!("local {local:?} not yet defined"))
    }

    fn lower_block(
        &mut self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
    ) -> Option<Vec<crate::ir::PortSource>> {
        for stmt in statements {
            if let Some(ret) = self.lower_stmt(rb, stmt) {
                return Some(ret);
            }
        }
        None
    }

    /// Lower an expression into one or more port sources (one per word).
    fn lower_expr_multi(
        &self,
        rb: &mut RegionBuilder<'_>,
        expr: &hir::Expr,
    ) -> Vec<crate::ir::PortSource> {
        match expr {
            hir::Expr::Local(local) => self.get_local_values(*local).to_vec(),
            hir::Expr::Field { base, field } => {
                // Fast path: local-rooted field chain (most common case).
                if let Some((local, chain)) = Self::try_expr_field_chain(expr) {
                    let values = self.get_local_values(local);
                    let (offset, ty) = self.resolve_local_field(local, &chain);
                    let word_count = Self::word_count_for_type(self.module, ty);
                    return values[offset..offset + word_count].to_vec();
                }
                // Memory-backed field access: compute address and load.
                if Self::expr_is_memory_backed(base) {
                    let base_ty = self.infer_expr_type(base);
                    let base_addr = self.lower_expr_to_addr(rb, base);
                    let byte_offset = self.field_byte_offset_in(&base_ty, field);
                    let field_ty = self.resolve_field_type_in(&base_ty, field);
                    let addr = if byte_offset == 0 {
                        base_addr
                    } else {
                        let off = rb.const_val(byte_offset as u64);
                        rb.binop(crate::ir::IrOp::Add, base_addr, off)
                    };
                    let word_count = Self::word_count_for_type(self.module, &field_ty);
                    if word_count == 1 {
                        let width = Self::scalar_store_width(&field_ty);
                        return vec![rb.load_from_addr(addr, width)];
                    }
                    // Multi-word field: load each word at 8-byte stride.
                    return (0..word_count)
                        .map(|i| {
                            let target = if i == 0 {
                                addr
                            } else {
                                let off = rb.const_val((i * 8) as u64);
                                rb.binop(crate::ir::IrOp::Add, addr, off)
                            };
                            rb.load_from_addr(target, crate::ir::Width::W8)
                        })
                        .collect();
                }
                // Register-backed fallback: evaluate base to word vector, project field.
                let words = self.lower_expr_multi(rb, base);
                let base_ty = self.infer_expr_type(base);
                let (offset, count) = self.field_offset_in_type(&base_ty, field);
                words[offset..offset + count].to_vec()
            }
            hir::Expr::Struct { def, fields } => {
                let hir::TypeDefKind::Struct { fields: field_defs } =
                    &self.module.type_defs[*def].kind
                else {
                    panic!("Expr::Struct requires a struct type def");
                };
                let mut result = Vec::new();
                for field_def in field_defs {
                    let (_, expr) = fields
                        .iter()
                        .find(|(name, _)| name == &field_def.name)
                        .unwrap_or_else(|| panic!("missing struct field {}", field_def.name));
                    result.extend(self.lower_expr_multi(rb, expr));
                }
                result
            }
            hir::Expr::Literal(hir::Literal::String(s)) => {
                // String literal → (ptr, len) where ptr is a DataAddr relocation.
                let blob_id = rb.add_data_blob(s.as_bytes().to_vec());
                let ptr = rb.data_addr(blob_id);
                let len = rb.const_val(s.len() as u64);
                vec![ptr, len]
            }
            _ => vec![self.lower_expr(rb, expr)],
        }
    }

    fn lower_stmt(
        &mut self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
    ) -> Option<Vec<crate::ir::PortSource>> {
        match &stmt.kind {
            hir::StmtKind::Init { place, value } | hir::StmtKind::Assign { place, value } => {
                self.lower_place_write(rb, place, value);
                None
            }
            hir::StmtKind::Store { addr, width, value } => {
                let addr = self.lower_expr(rb, addr);
                let value = self.lower_expr(rb, value);
                rb.store_to_addr(addr, value, Self::ir_width(*width));
                None
            }
            hir::StmtKind::Expr(hir::Expr::Call(call)) => {
                let args: Vec<_> = call
                    .args
                    .iter()
                    .map(|arg| self.lower_expr(rb, arg))
                    .collect();
                let callable = &self.module.callables[match call.target {
                    hir::CallTarget::Callable(id) => id,
                }];
                if !RuntimeDialectLowerer::lower_scalar_effect_call(rb, callable, &args) {
                    panic!(
                        "unsupported scalar HIR effect call: {} ({:?})",
                        callable.name, callable.intrinsic
                    );
                }
                None
            }
            hir::StmtKind::Expr(_) => None,
            hir::StmtKind::If {
                condition,
                then_block,
                else_block,
            } => {
                let predicate = self.lower_expr(rb, condition);
                let else_block = else_block
                    .as_ref()
                    .expect("scalar HIR if requires else branch");

                // Compute region interface: which locals cross into branches?
                let outer = self.defined_locals();
                let mut all_stmts = then_block.statements.clone();
                all_stmts.extend_from_slice(&else_block.statements);
                let iface = compute_region_interface(&all_stmts, &outer);

                // All locals that need to be threaded through the gamma
                let threaded: Vec<hir::LocalId> = {
                    let mut all = iface.captures.clone();
                    for id in &iface.live_out {
                        if !all.contains(id) {
                            all.push(*id);
                        }
                    }
                    all.sort();
                    all
                };

                let invariant_sources = self.flatten_locals_with_zeros(rb, &threaded);
                let invariant_count = invariant_sources.len();

                let mut branch_ret_count: Option<usize> = None;
                let gamma_outputs =
                    rb.gamma(predicate, &invariant_sources, 2, |branch_idx, branch| {
                        // Remap locals to inner region args
                        let args = branch.region_args(invariant_count);
                        self.remap_locals(&threaded, &args);

                        let block = match branch_idx {
                            0 => &else_block.statements,
                            1 => &then_block.statements,
                            _ => unreachable!(),
                        };
                        let ret = self.lower_block(branch, block);

                        // Gather updated locals for output
                        let updated = self.flatten_locals(&threaded);
                        let mut results = updated;
                        if let Some(vals) = ret {
                            if let Some(expected) = branch_ret_count {
                                assert_eq!(vals.len(), expected);
                            }
                            branch_ret_count = Some(vals.len());
                            results.extend(vals);
                        }
                        branch.set_results(&results);
                    });

                // Remap locals from gamma outputs
                let threaded_output_count = invariant_count;
                self.remap_locals(&threaded, &gamma_outputs[..threaded_output_count]);

                branch_ret_count.map(|ret_count| {
                    gamma_outputs[threaded_output_count..threaded_output_count + ret_count].to_vec()
                })
            }
            hir::StmtKind::Match { scrutinee, arms } => {
                let predicate = self.lower_expr(rb, scrutinee);
                for (expected, arm) in arms.iter().enumerate() {
                    let hir::Pattern::Integer(value) = arm.pattern else {
                        panic!("scalar HIR only supports integer match patterns");
                    };
                    assert_eq!(
                        value, expected as u64,
                        "scalar HIR requires contiguous integer match arms starting at 0"
                    );
                }

                // Compute region interface across all arms
                let outer = self.defined_locals();
                let mut all_stmts = Vec::new();
                for arm in arms {
                    all_stmts.extend_from_slice(&arm.body.statements);
                }
                let iface = compute_region_interface(&all_stmts, &outer);

                let threaded: Vec<hir::LocalId> = {
                    let mut all = iface.captures.clone();
                    for id in &iface.live_out {
                        if !all.contains(id) {
                            all.push(*id);
                        }
                    }
                    all.sort();
                    all
                };

                let invariant_sources = self.flatten_locals_with_zeros(rb, &threaded);
                let invariant_count = invariant_sources.len();

                let mut branch_ret_count: Option<usize> = None;
                let gamma_outputs = rb.gamma(
                    predicate,
                    &invariant_sources,
                    arms.len(),
                    |branch_idx, branch| {
                        let args = branch.region_args(invariant_count);
                        self.remap_locals(&threaded, &args);

                        let ret = self.lower_block(branch, &arms[branch_idx].body.statements);

                        let updated = self.flatten_locals(&threaded);
                        let mut results = updated;
                        if let Some(vals) = ret {
                            if let Some(expected) = branch_ret_count {
                                assert_eq!(vals.len(), expected);
                            }
                            branch_ret_count = Some(vals.len());
                            results.extend(vals);
                        }
                        branch.set_results(&results);
                    },
                );

                let threaded_output_count = invariant_count;
                self.remap_locals(&threaded, &gamma_outputs[..threaded_output_count]);

                branch_ret_count.map(|ret_count| {
                    gamma_outputs[threaded_output_count..threaded_output_count + ret_count].to_vec()
                })
            }
            hir::StmtKind::Loop {
                body,
                max_iterations,
                ..
            } => {
                // Compute region interface for the loop body
                let outer = self.defined_locals();
                let iface = compute_region_interface(&body.statements, &outer);
                let loop_vars = theta_loop_vars(&iface);

                let loop_var_sources = self.flatten_locals_with_zeros(rb, &loop_vars);
                let loop_var_word_count = loop_var_sources.len();

                let active_slot = rb.alloc_slot();
                let continue_slot = rb.alloc_slot();

                let build_body = |body_rb: &mut kajit_ir::RegionBuilder<'_>| {
                    // Remap locals to inner region args (loop var port sources)
                    let args = body_rb.region_args(loop_var_word_count);
                    self.remap_locals(&loop_vars, &args);

                    let one = body_rb.const_val(1);
                    body_rb.write_to_slot(active_slot, one);
                    body_rb.write_to_slot(continue_slot, one);
                    self.lower_loop_block(body_rb, &body.statements, active_slot, continue_slot);
                    let predicate = body_rb.read_from_slot(continue_slot);

                    // Results: [predicate, updated_loop_vars..., state]
                    let updated = self.flatten_locals(&loop_vars);
                    let mut results = vec![predicate];
                    results.extend(updated);
                    body_rb.set_results(&results);
                };

                let outputs = if let Some(max_iter) = max_iterations {
                    rb.theta_bounded(&loop_var_sources, *max_iter, build_body)
                } else {
                    rb.theta(&loop_var_sources, build_body)
                };

                // Remap locals from theta outputs
                self.remap_locals(&loop_vars, &outputs);
                None
            }
            hir::StmtKind::Return(Some(expr)) => Some(self.lower_expr_multi(rb, expr)),
            hir::StmtKind::Return(None) => None,
            other => panic!("unsupported scalar HIR statement: {other:?}"),
        }
    }

    /// Write a value to a place. Handles register-tracked locals (word vectors)
    /// and memory-backed places (deref, index) via store_to_addr.
    fn lower_place_write(
        &mut self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
        value: &hir::Expr,
    ) {
        match place {
            hir::Place::Local(local) => {
                let new_values = self.lower_expr_multi(rb, value);
                self.local_values.insert(*local, new_values);
            }
            hir::Place::Field { .. } => {
                // Check if this is a local-rooted field chain (register-tracked).
                if let Some((local, chain)) = Self::try_place_field_chain(place) {
                    let new_values = self.lower_expr_multi(rb, value);
                    let (offset, ty) = self.resolve_local_field(local, &chain);
                    let word_count = Self::word_count_for_type(self.module, ty);
                    assert_eq!(new_values.len(), word_count);
                    if !self.local_values.contains_key(&local) {
                        let local_ty = self.local_types[&local];
                        let total_words = Self::word_count_for_type(self.module, local_ty);
                        let zero = rb.const_val(0);
                        self.local_values.insert(local, vec![zero; total_words]);
                    }
                    let values = self.local_values.get_mut(&local).unwrap();
                    for (i, val) in new_values.into_iter().enumerate() {
                        values[offset + i] = val;
                    }
                } else {
                    // Memory-backed field write: compute byte address and store.
                    let (addr, place_ty) = self.lower_place_addr(rb, place);
                    self.store_value_to_addr(rb, addr, &place_ty, value);
                }
            }
            hir::Place::Deref { base } => {
                let base_ty = self.infer_expr_type(base);
                let hir::Type::Ref { pointee, .. } = base_ty else {
                    panic!("deref place requires a ref-typed base, got {base_ty:?}");
                };
                let addr = self.lower_expr(rb, base);
                self.store_value_to_addr(rb, addr, &pointee, value);
            }
            hir::Place::Index { base, index } => {
                let (base_addr, base_ty) = self.lower_place_addr(rb, base);
                let hir::Type::Array { element, .. } = &base_ty else {
                    panic!("index place requires array type, got {base_ty:?}");
                };
                let elem_size = hir_type_byte_size(self.module, element);
                let index_val = self.lower_expr(rb, index);
                let elem_size_val = rb.const_val(elem_size as u64);
                let byte_offset = rb.binop(crate::ir::IrOp::Mul, index_val, elem_size_val);
                let addr = rb.binop(crate::ir::IrOp::Add, base_addr, byte_offset);
                self.store_value_to_addr(rb, addr, element, value);
            }
        }
    }

    /// Resolve a place to its memory address and HIR type.
    /// Only works for memory-backed places (deref chains, not pure locals).
    fn lower_place_addr(
        &self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
    ) -> (crate::ir::PortSource, hir::Type) {
        match place {
            hir::Place::Local(local) => {
                let values = self.get_local_values(*local);
                let ty = self.local_types[local].clone();
                if matches!(ty, hir::Type::Ref { .. }) && values.len() == 1 {
                    // Ref-typed local: the value IS a pointer already.
                    (values[0], ty.clone())
                } else {
                    // Value-typed local: spill to a stack allocation and
                    // return its address. Each word is stored at an 8-byte offset.
                    let num_words = values.len().max(1) as u32;
                    let addr = rb.stack_alloc(num_words * 8, 8);
                    for (i, &val) in values.iter().enumerate() {
                        if i == 0 {
                            rb.store_to_addr(addr, val, crate::ir::Width::W8);
                        } else {
                            let off = rb.const_val((i * 8) as u64);
                            let target = rb.binop(crate::ir::IrOp::Add, addr, off);
                            rb.store_to_addr(target, val, crate::ir::Width::W8);
                        }
                    }
                    (addr, ty.clone())
                }
            }
            hir::Place::Deref { base } => {
                let base_ty = self.infer_expr_type(base);
                let hir::Type::Ref { pointee, .. } = base_ty else {
                    panic!("deref place requires a ref-typed base, got {base_ty:?}");
                };
                let addr = self.lower_expr(rb, base);
                (addr, *pointee)
            }
            hir::Place::Field { base, field } => {
                let (base_addr, base_ty) = self.lower_place_addr(rb, base);
                let byte_offset = self.field_byte_offset_in(&base_ty, field);
                let field_ty = self.resolve_field_type_in(&base_ty, field);
                if byte_offset == 0 {
                    (base_addr, field_ty)
                } else {
                    let offset_val = rb.const_val(byte_offset as u64);
                    let addr = rb.binop(crate::ir::IrOp::Add, base_addr, offset_val);
                    (addr, field_ty)
                }
            }
            hir::Place::Index { base, index } => {
                let (base_addr, base_ty) = self.lower_place_addr(rb, base);
                let hir::Type::Array { element, .. } = &base_ty else {
                    panic!("index place requires array type, got {base_ty:?}");
                };
                let elem_size = hir_type_byte_size(self.module, element);
                let index_val = self.lower_expr(rb, index);
                let elem_size_val = rb.const_val(elem_size as u64);
                let byte_offset = rb.binop(crate::ir::IrOp::Mul, index_val, elem_size_val);
                let addr = rb.binop(crate::ir::IrOp::Add, base_addr, byte_offset);
                (addr, *element.clone())
            }
        }
    }

    /// Store a value expression to a memory address. Handles multi-word types
    /// (Str, structs) by storing each word at the appropriate byte offset.
    fn store_value_to_addr(
        &self,
        rb: &mut RegionBuilder<'_>,
        addr: crate::ir::PortSource,
        ty: &hir::Type,
        value: &hir::Expr,
    ) {
        let word_count = Self::word_count_for_type(self.module, ty);
        if word_count == 1 {
            let val = self.lower_expr(rb, value);
            let width = Self::scalar_store_width(ty);
            rb.store_to_addr(addr, val, width);
        } else {
            // Multi-word: lower to word vector and store each word at 8-byte stride.
            let values = self.lower_expr_multi(rb, value);
            assert_eq!(values.len(), word_count);
            for (i, val) in values.into_iter().enumerate() {
                let offset = (i * 8) as u64;
                let target = if offset == 0 {
                    addr
                } else {
                    let off = rb.const_val(offset);
                    rb.binop(crate::ir::IrOp::Add, addr, off)
                };
                rb.store_to_addr(target, val, crate::ir::Width::W8);
            }
        }
    }

    /// Determine the IR store width for a single-word HIR type.
    fn scalar_store_width(ty: &hir::Type) -> crate::ir::Width {
        match ty {
            hir::Type::Bool => crate::ir::Width::W1,
            hir::Type::Integer(kind) => match kind.bits {
                8 => crate::ir::Width::W1,
                16 => crate::ir::Width::W2,
                32 => crate::ir::Width::W4,
                64 => crate::ir::Width::W8,
                _ => panic!("unsupported integer width: {}", kind.bits),
            },
            hir::Type::Ref { .. } | hir::Type::Address { .. } | hir::Type::Handle { .. } => {
                crate::ir::Width::W8
            }
            _ => crate::ir::Width::W8,
        }
    }

    /// Compute the byte offset of a named field within a type.
    fn field_byte_offset_in(&self, ty: &hir::Type, field_name: &str) -> usize {
        match ty {
            hir::Type::Named { def, .. } => {
                let type_def = &self.module.type_defs[*def];
                let hir::TypeDefKind::Struct { fields } = &type_def.kind else {
                    panic!("field_byte_offset_in: not a struct: {:?}", ty);
                };
                // Check for explicit offset annotation first.
                for f in fields {
                    if f.name == field_name
                        && let Some(offset) = f.offset
                    {
                        return offset as usize;
                    }
                }
                // Fallback: sum sizes of preceding fields.
                let mut running = 0usize;
                for f in fields {
                    if f.name == field_name {
                        return running;
                    }
                    running += hir_type_byte_size(self.module, &f.ty);
                }
                panic!("field '{field_name}' not found");
            }
            hir::Type::Str { .. } | hir::Type::Slice { .. } => match field_name {
                "ptr" | "data" => 0,
                "len" => 8,
                _ => panic!("unknown field '{field_name}' on Str/Slice"),
            },
            _ => panic!("field_byte_offset_in: no fields on {:?}", ty),
        }
    }

    /// Non-panicking place field chain: returns None if the base is not a local.
    fn try_place_field_chain(
        place: &hir::Place,
    ) -> Option<(hir::LocalId, Vec<(String, &hir::Type)>)> {
        match place {
            hir::Place::Local(local) => Some((*local, vec![])),
            hir::Place::Field { base, field } => {
                let (local, mut chain) = Self::try_place_field_chain(base)?;
                chain.push((field.clone(), &hir::Type::Unit));
                Some((local, chain))
            }
            _ => None,
        }
    }

    /// Check if an expression is memory-backed (requires address computation
    /// for field access) rather than register-backed (word vector projection).
    fn expr_is_memory_backed(expr: &hir::Expr) -> bool {
        match expr {
            hir::Expr::Deref(_) => true,
            hir::Expr::Field { base, .. } => Self::expr_is_memory_backed(base),
            _ => false,
        }
    }

    /// Lower an expression that represents a memory location to its address.
    fn lower_expr_to_addr(
        &self,
        rb: &mut RegionBuilder<'_>,
        expr: &hir::Expr,
    ) -> crate::ir::PortSource {
        match expr {
            hir::Expr::Deref(base) => self.lower_expr(rb, base),
            hir::Expr::Field { base, field } => {
                let base_ty = self.infer_expr_type(base);
                let base_addr = self.lower_expr_to_addr(rb, base);
                let byte_offset = self.field_byte_offset_in(&base_ty, field);
                if byte_offset == 0 {
                    base_addr
                } else {
                    let off = rb.const_val(byte_offset as u64);
                    rb.binop(crate::ir::IrOp::Add, base_addr, off)
                }
            }
            hir::Expr::Local(local) => {
                let values = self.get_local_values(*local);
                assert_eq!(
                    values.len(),
                    1,
                    "lower_expr_to_addr: local must be single-word"
                );
                values[0]
            }
            _ => self.lower_expr(rb, expr),
        }
    }

    fn lower_loop_block(
        &mut self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
        active_slot: crate::ir::SlotId,
        continue_slot: crate::ir::SlotId,
    ) {
        let mut i = 0;
        while i < statements.len() {
            let start = i;
            while i < statements.len() && !Self::is_loop_control_flow(&statements[i]) {
                i += 1;
            }
            if start < i {
                self.with_active_guard(rb, active_slot, &statements[start..i], |guard_rb, this| {
                    for stmt in &statements[start..i] {
                        this.lower_stmt(guard_rb, stmt);
                    }
                });
            }
            if i < statements.len() {
                self.lower_loop_control_flow_stmt(rb, &statements[i], active_slot, continue_slot);
                i += 1;
            }
        }
    }

    fn is_loop_control_flow(stmt: &hir::Stmt) -> bool {
        matches!(
            stmt.kind,
            hir::StmtKind::Break
                | hir::StmtKind::Continue
                | hir::StmtKind::If { .. }
                | hir::StmtKind::Match { .. }
                | hir::StmtKind::Loop { .. }
        )
    }

    fn lower_loop_control_flow_stmt(
        &mut self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
        active_slot: crate::ir::SlotId,
        continue_slot: crate::ir::SlotId,
    ) {
        self.with_active_guard(
            rb,
            active_slot,
            std::slice::from_ref(stmt),
            |guard_rb, this| match &stmt.kind {
                hir::StmtKind::Break => {
                    let zero = guard_rb.const_val(0);
                    guard_rb.write_to_slot(active_slot, zero);
                    guard_rb.write_to_slot(continue_slot, zero);
                }
                hir::StmtKind::Continue => {
                    let zero = guard_rb.const_val(0);
                    let one = guard_rb.const_val(1);
                    guard_rb.write_to_slot(active_slot, zero);
                    guard_rb.write_to_slot(continue_slot, one);
                }
                hir::StmtKind::If {
                    condition,
                    then_block,
                    else_block,
                } => {
                    let predicate = this.lower_expr(guard_rb, condition);
                    let else_block = else_block
                        .as_ref()
                        .expect("scalar HIR loop if requires else branch");

                    // Collect all branch statements for interface analysis
                    let mut all_stmts: Vec<hir::Stmt> = then_block.statements.clone();
                    all_stmts.extend(else_block.statements.iter().cloned());
                    let outer = this.defined_locals();
                    let iface = compute_region_interface(&all_stmts, &outer);
                    let threaded = theta_loop_vars(&iface);
                    let invariant_sources = this.flatten_locals_with_zeros(guard_rb, &threaded);
                    let threaded_word_count = invariant_sources.len();

                    let outputs =
                        guard_rb.gamma(predicate, &invariant_sources, 2, |branch_idx, branch| {
                            let args = branch.region_args(threaded_word_count);
                            this.remap_locals(&threaded, &args);
                            match branch_idx {
                                0 => this.lower_loop_block(
                                    branch,
                                    &else_block.statements,
                                    active_slot,
                                    continue_slot,
                                ),
                                1 => this.lower_loop_block(
                                    branch,
                                    &then_block.statements,
                                    active_slot,
                                    continue_slot,
                                ),
                                _ => unreachable!(),
                            }
                            let updated = this.flatten_locals(&threaded);
                            branch.set_results(&updated);
                        });
                    this.remap_locals(&threaded, &outputs);
                }
                hir::StmtKind::Match { scrutinee, arms } => {
                    let predicate = this.lower_expr(guard_rb, scrutinee);
                    for (expected, arm) in arms.iter().enumerate() {
                        let hir::Pattern::Integer(value) = arm.pattern else {
                            panic!("scalar HIR loop only supports integer match patterns");
                        };
                        assert_eq!(
                            value, expected as u64,
                            "scalar HIR loop requires contiguous integer match arms starting at 0"
                        );
                    }

                    let all_stmts: Vec<hir::Stmt> = arms
                        .iter()
                        .flat_map(|arm| arm.body.statements.iter().cloned())
                        .collect();
                    let outer = this.defined_locals();
                    let iface = compute_region_interface(&all_stmts, &outer);
                    let threaded = theta_loop_vars(&iface);
                    let invariant_sources = this.flatten_locals_with_zeros(guard_rb, &threaded);
                    let threaded_word_count = invariant_sources.len();

                    let outputs = guard_rb.gamma(
                        predicate,
                        &invariant_sources,
                        arms.len(),
                        |branch_idx, branch| {
                            let args = branch.region_args(threaded_word_count);
                            this.remap_locals(&threaded, &args);
                            this.lower_loop_block(
                                branch,
                                &arms[branch_idx].body.statements,
                                active_slot,
                                continue_slot,
                            );
                            let updated = this.flatten_locals(&threaded);
                            branch.set_results(&updated);
                        },
                    );
                    this.remap_locals(&threaded, &outputs);
                }
                hir::StmtKind::Loop {
                    body,
                    max_iterations,
                    ..
                } => {
                    let outer = this.defined_locals();
                    let iface = compute_region_interface(&body.statements, &outer);
                    let loop_vars = theta_loop_vars(&iface);
                    let loop_var_sources = this.flatten_locals_with_zeros(guard_rb, &loop_vars);
                    let loop_var_word_count = loop_var_sources.len();

                    let nested_active = guard_rb.alloc_slot();
                    let nested_continue = guard_rb.alloc_slot();
                    let build_body = |body_rb: &mut kajit_ir::RegionBuilder<'_>| {
                        let args = body_rb.region_args(loop_var_word_count);
                        this.remap_locals(&loop_vars, &args);
                        let one = body_rb.const_val(1);
                        body_rb.write_to_slot(nested_active, one);
                        body_rb.write_to_slot(nested_continue, one);
                        this.lower_loop_block(
                            body_rb,
                            &body.statements,
                            nested_active,
                            nested_continue,
                        );
                        let predicate = body_rb.read_from_slot(nested_continue);
                        let updated = this.flatten_locals(&loop_vars);
                        let mut results = vec![predicate];
                        results.extend(updated);
                        body_rb.set_results(&results);
                    };
                    let outputs = if let Some(max_iter) = max_iterations {
                        guard_rb.theta_bounded(&loop_var_sources, *max_iter, build_body)
                    } else {
                        guard_rb.theta(&loop_var_sources, build_body)
                    };
                    this.remap_locals(&loop_vars, &outputs);
                }
                other => {
                    panic!("is_loop_control_flow returned true for non-control-flow: {other:?}")
                }
            },
        );
    }

    fn with_active_guard(
        &mut self,
        rb: &mut RegionBuilder<'_>,
        active_slot: crate::ir::SlotId,
        stmts: &[hir::Stmt],
        f: impl FnOnce(&mut RegionBuilder<'_>, &mut Self),
    ) {
        let outer = self.defined_locals();
        let iface = compute_region_interface(stmts, &outer);
        let threaded = theta_loop_vars(&iface);
        let invariant_sources = self.flatten_locals_with_zeros(rb, &threaded);
        let threaded_word_count = invariant_sources.len();

        let active = rb.read_from_slot(active_slot);
        let mut f = Some(f);
        let outputs = rb.gamma(active, &invariant_sources, 2, |branch_idx, branch| {
            let args = branch.region_args(threaded_word_count);
            self.remap_locals(&threaded, &args);
            if branch_idx == 1 {
                f.take().expect("active branch should lower exactly once")(branch, self);
            }
            let updated = self.flatten_locals(&threaded);
            branch.set_results(&updated);
        });
        self.remap_locals(&threaded, &outputs);
    }

    fn lower_call_expr(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
    ) -> crate::ir::PortSource {
        let args: Vec<_> = call
            .args
            .iter()
            .map(|arg| self.lower_expr(rb, arg))
            .collect();
        let callable = &self.module.callables[match call.target {
            hir::CallTarget::Callable(id) => id,
        }];
        RuntimeDialectLowerer::lower_scalar_value_call(rb, callable, &args).unwrap_or_else(|| {
            panic!(
                "unsupported scalar HIR call target: {} ({:?})",
                callable.name, callable.intrinsic
            )
        })
    }

    fn lower_expr(&self, rb: &mut RegionBuilder<'_>, expr: &hir::Expr) -> crate::ir::PortSource {
        match expr {
            hir::Expr::Literal(hir::Literal::Bool(value)) => rb.const_val(u64::from(*value)),
            hir::Expr::Literal(hir::Literal::Integer(value)) => rb.const_val(*value),
            hir::Expr::Literal(hir::Literal::ExternAddr { symbol }) => {
                rb.extern_addr(symbol.clone())
            }
            hir::Expr::Local(local) => {
                let values = self.get_local_values(*local);
                assert_eq!(values.len(), 1, "lower_expr on multi-word local");
                values[0]
            }
            hir::Expr::Load { addr, width } => {
                let addr = self.lower_expr(rb, addr);
                rb.load_from_addr(addr, Self::ir_width(*width))
            }
            hir::Expr::SliceData { value } => {
                let values = self.lower_expr_multi(rb, value);
                assert!(values.len() >= 2, "SliceData requires 2-word type");
                values[0]
            }
            hir::Expr::SliceLen { value } => {
                let values = self.lower_expr_multi(rb, value);
                assert!(values.len() >= 2, "SliceLen requires 2-word type");
                values[1]
            }
            hir::Expr::Field { base, field } => {
                // Fast path: local-rooted field chain (most common case).
                if let Some((local, chain)) = Self::try_expr_field_chain(expr) {
                    let values = self.get_local_values(local);
                    let (offset, ty) = self.resolve_local_field(local, &chain);
                    assert_eq!(
                        Self::word_count_for_type(self.module, ty),
                        1,
                        "lower_expr on multi-word field"
                    );
                    return values[offset];
                }
                // Memory-backed field access: compute address and load.
                if Self::expr_is_memory_backed(base) {
                    let base_ty = self.infer_expr_type(base);
                    let base_addr = self.lower_expr_to_addr(rb, base);
                    let byte_offset = self.field_byte_offset_in(&base_ty, field);
                    let field_ty = self.resolve_field_type_in(&base_ty, field);
                    let addr = if byte_offset == 0 {
                        base_addr
                    } else {
                        let off = rb.const_val(byte_offset as u64);
                        rb.binop(crate::ir::IrOp::Add, base_addr, off)
                    };
                    let width = Self::scalar_store_width(&field_ty);
                    return rb.load_from_addr(addr, width);
                }
                // Register-backed fallback: evaluate base to word vector, project field.
                let words = self.lower_expr_multi(rb, base);
                let base_ty = self.infer_expr_type(base);
                let (offset, count) = self.field_offset_in_type(&base_ty, field);
                assert_eq!(count, 1, "lower_expr on multi-word field projection");
                words[offset]
            }
            hir::Expr::Binary { op, lhs, rhs } => {
                let lhs = self.lower_expr(rb, lhs);
                let rhs = self.lower_expr(rb, rhs);
                let ir_op = match op {
                    hir::BinaryOp::Add => crate::ir::IrOp::Add,
                    hir::BinaryOp::Sub => crate::ir::IrOp::Sub,
                    hir::BinaryOp::Mul => crate::ir::IrOp::Mul,
                    hir::BinaryOp::BitAnd => crate::ir::IrOp::And,
                    hir::BinaryOp::BitOr => crate::ir::IrOp::Or,
                    hir::BinaryOp::Xor => crate::ir::IrOp::Xor,
                    hir::BinaryOp::Shl => crate::ir::IrOp::Shl,
                    hir::BinaryOp::Shr => crate::ir::IrOp::Shr,
                    hir::BinaryOp::Sar => crate::ir::IrOp::Sar,
                    hir::BinaryOp::Eq => crate::ir::IrOp::CmpEq,
                    hir::BinaryOp::Ne => crate::ir::IrOp::CmpNe,
                    hir::BinaryOp::Lt => crate::ir::IrOp::CmpLt,
                    hir::BinaryOp::Le => crate::ir::IrOp::CmpLe,
                    hir::BinaryOp::Gt => crate::ir::IrOp::CmpGt,
                    hir::BinaryOp::Ge => crate::ir::IrOp::CmpGe,
                    hir::BinaryOp::And => crate::ir::IrOp::And,
                    hir::BinaryOp::Or => crate::ir::IrOp::Or,
                    other => panic!("unsupported scalar HIR binary op: {other:?}"),
                };
                rb.binop(ir_op, lhs, rhs)
            }
            hir::Expr::Deref(base) => {
                // Deref a pointer: load from the address.
                let base_ty = self.infer_expr_type(base);
                let hir::Type::Ref { pointee, .. } = base_ty else {
                    panic!("deref expr requires a ref-typed base, got {base_ty:?}");
                };
                let addr = self.lower_expr(rb, base);
                let width = Self::scalar_store_width(&pointee);
                rb.load_from_addr(addr, width)
            }
            hir::Expr::AddrOf(place) => {
                let (addr, _ty) = self.lower_place_addr(rb, place);
                addr
            }
            hir::Expr::Unary { op, value } => {
                let val = self.lower_expr(rb, value);
                match op {
                    hir::UnaryOp::Not => {
                        let one = rb.const_val(1);
                        rb.binop(crate::ir::IrOp::Xor, val, one)
                    }
                    hir::UnaryOp::Neg => {
                        let zero = rb.const_val(0);
                        rb.binop(crate::ir::IrOp::Sub, zero, val)
                    }
                }
            }
            hir::Expr::Call(call) => self.lower_call_expr(rb, call),
            other => panic!("unsupported HIR expression: {other:?}"),
        }
    }

    fn ir_width(width: hir::MemoryWidth) -> crate::ir::Width {
        match width {
            hir::MemoryWidth::W1 => crate::ir::Width::W1,
            hir::MemoryWidth::W2 => crate::ir::Width::W2,
            hir::MemoryWidth::W4 => crate::ir::Width::W4,
            hir::MemoryWidth::W8 => crate::ir::Width::W8,
        }
    }
}

fn hir_type_byte_size(module: &hir::Module, ty: &hir::Type) -> usize {
    match ty {
        hir::Type::Unit => 0,
        hir::Type::Bool => 1,
        hir::Type::Integer(kind) => (kind.bits as usize) / 8,
        hir::Type::Ref { .. } | hir::Type::Address { .. } | hir::Type::Handle { .. } => {
            core::mem::size_of::<usize>()
        }
        hir::Type::Str { .. } | hir::Type::Slice { .. } => core::mem::size_of::<usize>() * 2,
        hir::Type::Array { element, len } => hir_type_byte_size(module, element) * len,
        hir::Type::Named { def, .. } => {
            let type_def = &module.type_defs[*def];
            if let Some(size) = type_def.size {
                return size as usize;
            }
            hir_type_def_byte_size(module, type_def)
        }
    }
}

fn hir_type_def_byte_size(module: &hir::Module, type_def: &hir::TypeDef) -> usize {
    match &type_def.kind {
        hir::TypeDefKind::Struct { fields } => {
            let mut max_end = 0usize;
            for field in fields {
                let field_size = hir_type_byte_size(module, &field.ty);
                if let Some(offset) = field.offset {
                    max_end = max_end.max(offset as usize + field_size);
                } else {
                    max_end += field_size;
                }
            }
            max_end
        }
        hir::TypeDefKind::Enum {
            variants,
            discriminant_width,
        } => {
            let disc_size = discriminant_width.unwrap_or(1) as usize;
            let max_payload = variants
                .iter()
                .map(|variant| {
                    variant
                        .fields
                        .iter()
                        .map(|field| {
                            let field_size = hir_type_byte_size(module, &field.ty);
                            if let Some(offset) = field.offset {
                                offset as usize + field_size
                            } else {
                                field_size
                            }
                        })
                        .max()
                        .unwrap_or(0)
                })
                .max()
                .unwrap_or(0);
            disc_size + max_payload
        }
    }
}
