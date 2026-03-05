use std::collections::HashMap;
use std::path::{Path, PathBuf};

use facet::{
    Def, EnumRepr, KnownPointer, OptionDef, PointerDef, ScalarType, Shape, StructKind, Type,
    UserType,
};
use regalloc2::RegClass;

use crate::arch::EmitCtx;
use crate::format::{
    Decoder, FieldEmitInfo, FieldLowerInfo, SkippedFieldInfo, VariantEmitInfo, VariantKind,
    VariantLowerInfo,
};
use crate::ir::{LambdaId, RegionBuilder, Width as IrWidth};
use crate::pipeline_opts::PipelineOptions;

/// A compiled deserializer. Owns the executable buffer containing JIT'd machine code.
pub struct CompiledDecoder {
    #[cfg(target_arch = "x86_64")]
    buf: kajit_emit::x64::FinalizedEmission,
    #[cfg(target_arch = "aarch64")]
    buf: kajit_emit::aarch64::FinalizedEmission,
    entry: usize,
    func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext),
    trusted_utf8_input: bool,
    _jit_registration: Option<crate::jit_debug::JitRegistration>,
}

impl CompiledDecoder {
    pub(crate) fn func(&self) -> unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) {
        self.func
    }

    /// The raw executable code buffer.
    pub fn code(&self) -> &[u8] {
        #[cfg(target_arch = "x86_64")]
        {
            return self.buf.exec.as_ref();
        }

        #[cfg(target_arch = "aarch64")]
        {
            &self.buf.code
        }
    }

    /// Byte offset of the entry point within the code buffer.
    pub fn entry_offset(&self) -> usize {
        self.entry
    }

    /// Whether `from_str` can safely enable trusted UTF-8 mode for this format.
    pub fn supports_trusted_utf8_input(&self) -> bool {
        self.trusted_utf8_input
    }
}

pub(crate) const DEFAULT_PRE_LINEARIZATION_PASSES_ENABLED: bool = true;

#[cfg(target_arch = "aarch64")]
fn materialize_backend_result(
    result: crate::ir_backend::LinearBackendResult,
) -> (
    kajit_emit::aarch64::FinalizedEmission,
    usize,
    Option<kajit_emit::SourceMap>,
) {
    let crate::ir_backend::LinearBackendResult {
        buf,
        entry,
        source_map,
    } = result;
    (buf, entry as usize, source_map)
}

#[cfg(target_arch = "x86_64")]
fn materialize_backend_result(
    result: crate::ir_backend::LinearBackendResult,
) -> (
    kajit_emit::x64::FinalizedEmission,
    usize,
    Option<kajit_emit::SourceMap>,
) {
    let crate::ir_backend::LinearBackendResult {
        buf,
        entry,
        source_map,
    } = result;
    (buf, entry as usize, source_map)
}

fn jit_debug_enabled() -> bool {
    let Ok(raw) = std::env::var("KAJIT_DEBUG") else {
        return false;
    };
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sanitize_debug_file_stem(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let out = out.trim_matches('_');
    if out.is_empty() {
        "jit".to_string()
    } else {
        out.to_string()
    }
}

fn format_ra_terminator(term: &crate::regalloc_mir::RaTerminator) -> String {
    match term {
        crate::regalloc_mir::RaTerminator::Return => "return".to_string(),
        crate::regalloc_mir::RaTerminator::ErrorExit { code } => format!("error_exit {code:?}"),
        crate::regalloc_mir::RaTerminator::Branch { target } => format!("branch b{}", target.0),
        crate::regalloc_mir::RaTerminator::BranchIf {
            cond,
            target,
            fallthrough,
        } => format!(
            "branch_if v{} -> b{} else b{}",
            cond.index(),
            target.0,
            fallthrough.0
        ),
        crate::regalloc_mir::RaTerminator::BranchIfZero {
            cond,
            target,
            fallthrough,
        } => format!(
            "branch_if_zero v{} -> b{} else b{}",
            cond.index(),
            target.0,
            fallthrough.0
        ),
        crate::regalloc_mir::RaTerminator::JumpTable {
            predicate,
            targets,
            default,
        } => {
            let targets = targets
                .iter()
                .map(|target| format!("b{}", target.0))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "jump_table v{} [{}] default b{}",
                predicate.index(),
                targets,
                default.0
            )
        }
    }
}

fn build_ra_mir_listing(program: &crate::regalloc_mir::RaProgram) -> (String, HashMap<usize, u32>) {
    let mut lines = Vec::new();
    let mut line_by_linear_op = HashMap::<usize, u32>::new();
    let mut next_line = 1u32;
    for func in &program.funcs {
        for block in &func.blocks {
            for inst in &block.insts {
                lines.push(format!("{:?}", inst.op));
                line_by_linear_op
                    .entry(inst.linear_op_index)
                    .or_insert(next_line);
                next_line += 1;
            }
            let linear_op_index = block.term_linear_op_index;
            lines.push(format_ra_terminator(&block.term));
            line_by_linear_op
                .entry(linear_op_index)
                .or_insert(next_line);
            next_line += 1;
        }
    }
    let mut listing = lines.join("\n");
    if !listing.is_empty() {
        listing.push('\n');
    }
    (listing, line_by_linear_op)
}

fn write_ra_mir_listing_file(type_name: &str, listing: &str) -> Option<PathBuf> {
    let stem = sanitize_debug_file_stem(type_name);
    let dir = Path::new("/tmp/kajit-debug");
    std::fs::create_dir_all(dir).ok()?;
    let path = dir.join(format!("{stem}.ra-mir"));
    std::fs::write(&path, listing).ok()?;
    Some(path)
}

fn jit_dwarf_target_arch() -> crate::jit_dwarf::DwarfTargetArch {
    if cfg!(target_arch = "x86_64") {
        crate::jit_dwarf::DwarfTargetArch::X86_64
    } else if cfg!(target_arch = "aarch64") {
        crate::jit_dwarf::DwarfTargetArch::Aarch64
    } else {
        panic!("unsupported target architecture for DWARF generation")
    }
}

fn linear_pc_ranges(
    source_map: &kajit_emit::SourceMap,
    code_len: usize,
) -> HashMap<usize, Vec<(u64, u64)>> {
    let mut out = HashMap::<usize, Vec<(u64, u64)>>::new();
    for (i, entry) in source_map.iter().enumerate() {
        if entry.location.line == 0 {
            continue;
        }
        let start = entry.offset as u64;
        let end = source_map
            .get(i + 1)
            .map_or(code_len as u64, |next| next.offset as u64);
        if end <= start {
            continue;
        }
        let linear_op_index = (entry.location.line - 1) as usize;
        out.entry(linear_op_index).or_default().push((start, end));
    }
    out
}

fn collect_named_field_vregs(
    ir: &crate::linearize::LinearIr,
    _root_shape: &'static Shape,
) -> HashMap<crate::ir::VReg, String> {
    let trace = std::env::var("KAJIT_DEBUG_DWARF_TRACE")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);
    fn field_map_for_shape(shape: &'static Shape) -> Option<HashMap<u32, String>> {
        let Type::User(UserType::Struct(_)) = &shape.ty else {
            return None;
        };
        let (fields, _) = collect_fields(shape);
        let mut by_offset = HashMap::<u32, String>::new();
        for field in fields {
            by_offset.insert(field.offset as u32, field.name.to_string());
        }
        Some(by_offset)
    }

    let mut named = HashMap::<crate::ir::VReg, String>::new();
    let mut current_lambda_shapes = Vec::<(&'static Shape, Option<HashMap<u32, String>>)>::new();
    let mut write_to_field_total = 0usize;
    let mut write_to_field_mapped = 0usize;
    for op in &ir.ops {
        match op {
            crate::linearize::LinearOp::FuncStart { shape, .. } => {
                current_lambda_shapes.push((*shape, field_map_for_shape(*shape)));
            }
            crate::linearize::LinearOp::FuncEnd => {
                let _ = current_lambda_shapes.pop();
            }
            crate::linearize::LinearOp::WriteToField { src, offset, .. } => {
                write_to_field_total += 1;
                if let Some((shape, field_map)) = current_lambda_shapes.last() {
                    if trace && write_to_field_total <= 8 {
                        let mapped_name = field_map.as_ref().and_then(|m| m.get(offset));
                        eprintln!(
                            "[kajit-dwarf] WriteToField: shape={} offset={} src=v{} mapped={}",
                            shape.type_identifier,
                            offset,
                            src.index(),
                            mapped_name.is_some()
                        );
                    }
                }
                if let Some((shape, Some(field_map))) = current_lambda_shapes.last()
                    && let Some(name) = field_map.get(offset)
                {
                    write_to_field_mapped += 1;
                    named
                        .entry(*src)
                        .or_insert_with(|| format!("{}.{}", shape.type_identifier, name));
                }
            }
            _ => {}
        }
    }
    if trace {
        eprintln!(
            "[kajit-dwarf] collect_named_field_vregs: total WriteToField ops={}, mapped WriteToField ops={}, mapped names={}",
            write_to_field_total,
            write_to_field_mapped,
            named.len()
        );
    }
    named
}

fn aarch64_extra_saved_pairs(alloc: &crate::regalloc_engine::AllocatedProgram) -> u32 {
    let mut max_pair = None::<u32>;
    let mut observe = |a: regalloc2::Allocation| {
        let Some(reg) = a.as_reg() else {
            return;
        };
        if reg.class() != RegClass::Int {
            return;
        }
        let pair = match reg.hw_enc() as u8 {
            23 | 24 => Some(0),
            25 | 26 => Some(1),
            27 | 28 => Some(2),
            _ => None,
        };
        if let Some(pair) = pair {
            max_pair = Some(max_pair.map_or(pair, |cur| cur.max(pair)));
        }
    };

    for func in &alloc.functions {
        for inst_allocs in &func.inst_allocs {
            for &a in inst_allocs {
                observe(a);
            }
        }
        for (_, edit) in &func.edits {
            let regalloc2::Edit::Move { from, to } = edit;
            observe(*from);
            observe(*to);
        }
        for edge in &func.edge_edits {
            observe(edge.from);
            observe(edge.to);
        }
        for &a in &func.return_result_allocs {
            observe(a);
        }
    }

    max_pair.map_or(0, |p| p + 1)
}

fn spill_fbreg_offset(
    target_arch: crate::jit_dwarf::DwarfTargetArch,
    program: &crate::regalloc_mir::RaProgram,
    alloc: &crate::regalloc_engine::AllocatedProgram,
    spillslot_index: usize,
) -> i64 {
    let slot_count_bytes = (program.slot_count as i64) * 8;
    let spill_bytes = (spillslot_index as i64) * 8;
    match target_arch {
        crate::jit_dwarf::DwarfTargetArch::X86_64 => {
            let slot_base = crate::arch::BASE_FRAME as i64;
            slot_base + slot_count_bytes + spill_bytes
        }
        crate::jit_dwarf::DwarfTargetArch::Aarch64 => {
            let extra_saved_pairs = aarch64_extra_saved_pairs(alloc) as i64;
            let slot_base = crate::arch::BASE_FRAME as i64 + extra_saved_pairs * 16;
            slot_base + slot_count_bytes + spill_bytes
        }
    }
}

fn build_dwarf_variables_from_regalloc(
    ir: &crate::linearize::LinearIr,
    ra_mir: &crate::regalloc_mir::RaProgram,
    alloc: &crate::regalloc_engine::AllocatedProgram,
    source_map: &kajit_emit::SourceMap,
    code_len: usize,
    root_shape: &'static Shape,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
) -> Vec<crate::jit_dwarf::DwarfVariable> {
    let trace = std::env::var("KAJIT_DEBUG_DWARF_TRACE")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);
    let named_vregs = collect_named_field_vregs(ir, root_shape);
    if trace {
        eprintln!(
            "[kajit-dwarf] named field vregs discovered: {}",
            named_vregs.len()
        );
    }
    if named_vregs.is_empty() {
        return Vec::new();
    }

    let ranges_by_linear_op = linear_pc_ranges(source_map, code_len);
    let mut ranges_by_name = HashMap::<String, Vec<crate::jit_dwarf::DwarfLocationRange>>::new();

    for func in &alloc.functions {
        for inst_idx in 0..func.inst_allocs.len() {
            let Some(Some(linear_op_index)) = func.inst_linear_op_indices.get(inst_idx) else {
                continue;
            };
            let Some(pc_ranges) = ranges_by_linear_op.get(linear_op_index) else {
                continue;
            };
            let Some(operand_specs) = func.inst_operands.get(inst_idx) else {
                continue;
            };
            let Some(operand_allocs) = func.inst_allocs.get(inst_idx) else {
                continue;
            };
            if operand_specs.len() != operand_allocs.len() {
                continue;
            }

            for ((vreg, _kind), location_alloc) in operand_specs.iter().zip(operand_allocs.iter()) {
                let Some(name) = named_vregs.get(vreg) else {
                    continue;
                };
                let expr = if let Some(reg) = location_alloc.as_reg() {
                    let Some(dwarf_reg) = crate::jit_dwarf::dwarf_register_from_hw_encoding(
                        target_arch,
                        reg.hw_enc() as u8,
                    ) else {
                        continue;
                    };
                    crate::jit_dwarf::expr_reg(dwarf_reg)
                } else if let Some(slot) = location_alloc.as_stack() {
                    let off = spill_fbreg_offset(target_arch, ra_mir, alloc, slot.index());
                    crate::jit_dwarf::expr_fbreg(off)
                } else {
                    continue;
                };
                for (start, end) in pc_ranges {
                    ranges_by_name.entry(name.clone()).or_default().push(
                        crate::jit_dwarf::DwarfLocationRange {
                            start: *start,
                            end: *end,
                            expression: expr.clone(),
                        },
                    );
                }
            }
        }
    }

    let mut vars = ranges_by_name
        .into_iter()
        .map(|(name, mut ranges)| {
            ranges.sort_by_key(|r| (r.start, r.end));
            let mut merged = Vec::<crate::jit_dwarf::DwarfLocationRange>::new();
            for r in ranges {
                if let Some(last) = merged.last_mut()
                    && last.end == r.start
                    && last.expression == r.expression
                {
                    last.end = r.end;
                    continue;
                }
                if r.end > r.start {
                    merged.push(r);
                }
            }
            crate::jit_dwarf::DwarfVariable {
                name,
                locations: merged,
            }
        })
        .collect::<Vec<_>>();
    vars.sort_by(|a, b| a.name.cmp(&b.name));

    if trace {
        eprintln!("[kajit-dwarf] variables emitted: {}", vars.len());
        for var in vars.iter().take(8) {
            eprintln!(
                "[kajit-dwarf] var {} ranges={} first={:?}",
                var.name,
                var.locations.len(),
                var.locations.first().map(|r| (r.start, r.end))
            );
        }
    }
    vars
}

fn build_dwarf_from_source_map(
    code_ptr: *const u8,
    code_len: usize,
    source_map: Option<&kajit_emit::SourceMap>,
    listing_path: &Path,
    line_by_linear_op: &HashMap<usize, u32>,
    subprogram_name: &str,
    variables: &[crate::jit_dwarf::DwarfVariable],
    target_arch: crate::jit_dwarf::DwarfTargetArch,
) -> Option<crate::jit_dwarf::JitDwarfSections> {
    let source_map = source_map?;
    let mut dwarf_map = Vec::<(u32, u32)>::new();
    for entry in source_map {
        if entry.location.line == 0 {
            continue;
        }
        let linear_op_index = (entry.location.line - 1) as usize;
        let Some(&listing_line) = line_by_linear_op.get(&linear_op_index) else {
            continue;
        };
        dwarf_map.push((entry.offset, listing_line.saturating_sub(1)));
    }

    let file_name = listing_path.file_name()?.to_str()?;
    let directory = listing_path.parent().and_then(Path::to_str);
    crate::jit_dwarf::build_jit_dwarf_sections_with_variables(
        target_arch,
        code_ptr as u64,
        code_len as u64,
        &dwarf_map,
        file_name,
        directory,
        subprogram_name,
        variables,
    )
    .ok()
}

// r[impl compiler.walk]
// r[impl compiler.recursive]
// r[impl compiler.recursive.one-func-per-shape]

fn collect_fields(shape: &'static Shape) -> (Vec<FieldEmitInfo>, Vec<SkippedFieldInfo>) {
    let mut out = Vec::new();
    let mut skipped = Vec::new();
    let container_has_default = shape.has_default_attr();
    collect_fields_recursive(shape, 0, container_has_default, &mut out, &mut skipped);
    check_field_name_collisions(&out);
    (out, skipped)
}

// r[impl deser.default]
// r[impl deser.default.fn-ptr]
// r[impl deser.skip]
// r[impl deser.skip.filter]
fn collect_fields_recursive(
    shape: &'static Shape,
    base_offset: usize,
    container_has_default: bool,
    out: &mut Vec<FieldEmitInfo>,
    skipped: &mut Vec<SkippedFieldInfo>,
) {
    use crate::format::DefaultInfo;
    use facet::DefaultSource;

    let st = match &shape.ty {
        Type::User(UserType::Struct(st)) => st,
        _ => panic!("unsupported shape: {}", shape.type_identifier),
    };
    for f in st.fields {
        if f.is_flattened() {
            collect_fields_recursive(
                f.shape(),
                base_offset + f.offset,
                container_has_default,
                out,
                skipped,
            );
            continue;
        }

        // Resolve default information for this field.
        let default = match f.default {
            Some(DefaultSource::Custom(custom_fn)) => {
                // Custom default expression: #[facet(default = expr)]
                Some(DefaultInfo {
                    trampoline: crate::intrinsics::kajit_field_default_custom as *const u8,
                    fn_ptr: custom_fn as *const u8,
                    shape: None,
                })
            }
            Some(DefaultSource::FromTrait) => {
                // Field-level #[facet(default)] — use the field type's Default impl.
                resolve_trait_default(f.shape())
            }
            None if container_has_default => {
                // Container-level #[facet(default)] — all fields get Default.
                resolve_trait_default(f.shape())
            }
            None => None,
        };

        // r[impl deser.skip.default-required]
        if f.should_skip_deserializing() {
            // Skipped fields are excluded from the dispatch list but need default init.
            let default = default.unwrap_or_else(|| {
                panic!(
                    "field \"{}\" on {} is skipped but has no default — \
                     add #[facet(default)] or impl Default",
                    f.effective_name(),
                    shape.type_identifier,
                )
            });
            skipped.push(SkippedFieldInfo {
                offset: base_offset + f.offset,
                default,
            });
            continue;
        }

        out.push(FieldEmitInfo {
            offset: base_offset + f.offset,
            shape: f.shape(),
            name: f.effective_name(),
            required_index: out.len(),
            default,
        });
    }
}

/// Resolve a trait-based default for a field type.
/// Returns the DefaultInfo if the type has a Default impl via its shape vtable.
fn resolve_trait_default(shape: &'static Shape) -> Option<crate::format::DefaultInfo> {
    use crate::format::DefaultInfo;

    // Get the default_in_place function from the shape's TypeOps.
    let type_ops = shape.type_ops?;
    match type_ops {
        facet::TypeOps::Direct(ops) => {
            let default_fn = ops.default_in_place?;
            Some(DefaultInfo {
                trampoline: crate::intrinsics::kajit_field_default_trait as *const u8,
                fn_ptr: default_fn as *const u8,
                shape: None,
            })
        }
        facet::TypeOps::Indirect(ops) => {
            let default_fn = ops.default_in_place?;
            Some(DefaultInfo {
                trampoline: crate::intrinsics::kajit_field_default_indirect as *const u8,
                fn_ptr: default_fn as *const u8,
                shape: Some(shape),
            })
        }
    }
}

// r[impl deser.flatten.conflict]
fn check_field_name_collisions(fields: &[FieldEmitInfo]) {
    let mut seen = std::collections::HashSet::new();
    for f in fields {
        if !seen.insert(f.name) {
            panic!(
                "field name collision: \"{}\" (possibly from #[facet(flatten)])",
                f.name
            );
        }
    }
}

// r[impl deser.enum.variant-kinds]

fn collect_variants(enum_type: &'static facet::EnumType) -> Vec<VariantEmitInfo> {
    enum_type
        .variants
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let kind = VariantKind::from_struct_type(&v.data);
            let mut fields = Vec::new();
            let mut skipped = Vec::new();
            for f in v.data.fields {
                if f.is_flattened() {
                    collect_fields_recursive(f.shape(), f.offset, false, &mut fields, &mut skipped);
                } else {
                    fields.push(FieldEmitInfo {
                        offset: f.offset,
                        shape: f.shape(),
                        name: f.effective_name(),
                        required_index: fields.len(),
                        default: None,
                    });
                }
            }
            // Note: skipped fields in enum variants are not yet supported.
            // If needed, we'd emit default calls in the variant body.
            VariantEmitInfo {
                index: i,
                name: v.effective_name(),
                rust_discriminant: v.discriminant.expect(
                    "enum variant must have a known discriminant (use #[repr(u8)] or similar)",
                ),
                fields,
                kind,
            }
        })
        .collect()
}

/// Get the discriminant storage size in bytes from an EnumRepr.
fn discriminant_size(repr: EnumRepr) -> u32 {
    match repr {
        EnumRepr::U8 | EnumRepr::I8 => 1,
        EnumRepr::U16 | EnumRepr::I16 => 2,
        EnumRepr::U32 | EnumRepr::I32 => 4,
        EnumRepr::U64 | EnumRepr::I64 | EnumRepr::USize | EnumRepr::ISize => 8,
        EnumRepr::Rust | EnumRepr::RustNPO => {
            panic!("cannot JIT-compile enums with #[repr(Rust)] — use #[repr(u8)] or similar")
        }
    }
}

/// Returns the OptionDef if this shape is an Option type.
fn get_option_def(shape: &'static Shape) -> Option<&'static OptionDef> {
    match &shape.def {
        Def::Option(opt_def) => Some(opt_def),
        _ => None,
    }
}

// r[impl deser.pointer]

/// Returns the PointerDef if this shape is a supported smart pointer (Box, Arc, Rc).
fn get_pointer_def(shape: &'static Shape) -> Option<&'static PointerDef> {
    match &shape.def {
        Def::Pointer(ptr_def)
            if matches!(
                ptr_def.known,
                Some(KnownPointer::Box | KnownPointer::Arc | KnownPointer::Rc)
            ) =>
        {
            Some(ptr_def)
        }
        _ => None,
    }
}

// r[impl deser.pointer.nesting]

/// Returns true if the shape is a struct type.
/// Emit a default initialization call for a field.
///
/// Direct types use a 2-arg call (trampoline, fn_ptr, offset).
/// Indirect types (generic containers) use a 3-arg call that also passes the shape.
pub fn emit_default_init(ectx: &mut EmitCtx, default: &crate::format::DefaultInfo, offset: u32) {
    if let Some(shape) = default.shape {
        ectx.emit_call_trampoline_3(
            default.trampoline,
            default.fn_ptr,
            offset,
            shape as *const _ as *const u8,
        );
    } else {
        ectx.emit_call_option_init_none(default.trampoline, default.fn_ptr, offset);
    }
}

fn is_unit(shape: &'static Shape) -> bool {
    shape.scalar_type() == Some(ScalarType::Unit)
}

fn is_string_like_scalar(scalar_type: ScalarType) -> bool {
    matches!(
        scalar_type,
        ScalarType::String | ScalarType::Str | ScalarType::CowStr
    )
}

/// Emit code for a single field, dispatching to nested struct calls, inline
/// expansion, scalar intrinsics, or Option handling.
fn ir_width_from_disc_size(size: u32) -> IrWidth {
    match size {
        1 => IrWidth::W1,
        2 => IrWidth::W2,
        4 => IrWidth::W4,
        8 => IrWidth::W8,
        _ => panic!("unsupported discriminant size: {size}"),
    }
}

fn lower_fields_for_ir(
    shape: &'static Shape,
    base_offset: usize,
) -> (Vec<FieldLowerInfo>, Vec<SkippedFieldInfo>) {
    let (fields, skipped) = collect_fields(shape);
    let lowered = fields
        .into_iter()
        .map(|f| FieldLowerInfo {
            offset: base_offset + f.offset,
            shape: f.shape,
            name: f.name,
            required_index: f.required_index,
            has_default: f.default.is_some(),
        })
        .collect();
    (lowered, skipped)
}

fn lower_variants_for_ir(
    enum_type: &'static facet::EnumType,
    base_offset: usize,
) -> Vec<VariantLowerInfo> {
    collect_variants(enum_type)
        .into_iter()
        .map(|v| VariantLowerInfo {
            index: v.index,
            name: v.name,
            rust_discriminant: v.rust_discriminant,
            kind: v.kind,
            fields: v
                .fields
                .into_iter()
                .map(|f| FieldLowerInfo {
                    offset: base_offset + f.offset,
                    shape: f.shape,
                    name: f.name,
                    required_index: f.required_index,
                    has_default: f.default.is_some(),
                })
                .collect(),
        })
        .collect()
}

fn ir_shape_needs_lambda(shape: &'static Shape) -> bool {
    if is_unit(shape) {
        return false;
    }

    if shape.is_transparent() || get_option_def(shape).is_some() {
        return true;
    }

    if matches!(&shape.def, Def::List(_) | Def::Map(_) | Def::Array(_)) {
        return true;
    }

    if get_pointer_def(shape).is_some() {
        return true;
    }

    matches!(
        &shape.ty,
        Type::User(UserType::Struct(_) | UserType::Enum(_))
    )
}

fn collect_ir_lambda_shapes(
    shape: &'static Shape,
    seen: &mut std::collections::HashSet<*const Shape>,
    out: &mut Vec<&'static Shape>,
) {
    if !ir_shape_needs_lambda(shape) {
        return;
    }

    let key = shape as *const Shape;
    if !seen.insert(key) {
        return;
    }
    out.push(shape);

    if shape.is_transparent() {
        let (fields, _) = collect_fields(shape);
        for field in &fields {
            collect_ir_lambda_shapes(field.shape, seen, out);
        }
        return;
    }

    if let Some(opt_def) = get_option_def(shape) {
        collect_ir_lambda_shapes(opt_def.t, seen, out);
        return;
    }

    if let Some(ptr_def) = get_pointer_def(shape) {
        if let Some(pointee) = ptr_def.pointee {
            collect_ir_lambda_shapes(pointee, seen, out);
        }
        return;
    }

    match &shape.def {
        Def::List(list_def) => {
            collect_ir_lambda_shapes(list_def.t, seen, out);
            return;
        }
        Def::Map(map_def) => {
            collect_ir_lambda_shapes(map_def.k, seen, out);
            collect_ir_lambda_shapes(map_def.v, seen, out);
            return;
        }
        Def::Array(array_def) => {
            collect_ir_lambda_shapes(array_def.t, seen, out);
            return;
        }
        _ => {}
    }

    match &shape.ty {
        Type::User(UserType::Struct(_)) => {
            let (fields, _) = collect_fields(shape);
            for field in &fields {
                collect_ir_lambda_shapes(field.shape, seen, out);
            }
        }
        Type::User(UserType::Enum(enum_type)) => {
            let variants = collect_variants(enum_type);
            for variant in &variants {
                for field in &variant.fields {
                    collect_ir_lambda_shapes(field.shape, seen, out);
                }
            }
        }
        _ => {}
    }
}

struct IrShapeLowerer<'a> {
    decoder: &'a dyn Decoder,
    lambda_by_shape: HashMap<*const Shape, LambdaId>,
}

impl<'a> IrShapeLowerer<'a> {
    fn new(decoder: &'a dyn Decoder, lambda_by_shape: HashMap<*const Shape, LambdaId>) -> Self {
        Self {
            decoder,
            lambda_by_shape,
        }
    }

    fn lambda_for_shape(&self, shape: &'static Shape) -> LambdaId {
        *self
            .lambda_by_shape
            .get(&(shape as *const Shape))
            .unwrap_or_else(|| {
                panic!(
                    "missing lambda for composite shape in IR lowering: {}",
                    shape.type_identifier
                )
            })
    }

    fn emit_apply_with_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        shape: &'static Shape,
        offset: usize,
    ) {
        let lambda = self.lambda_for_shape(shape);
        if offset == 0 {
            let _ = rb.apply(lambda, &[], 0);
            return;
        }

        let saved_out = rb.save_out_ptr();
        let off = rb.const_val(offset as u64);
        let adjusted_out = rb.binop(crate::ir::IrOp::Add, saved_out, off);
        rb.set_out_ptr(adjusted_out);
        let _ = rb.apply(lambda, &[], 0);
        rb.set_out_ptr(saved_out);
    }

    fn lower_value(&self, rb: &mut RegionBuilder<'_>, shape: &'static Shape, offset: usize) {
        if is_unit(shape) {
            return;
        }

        if ir_shape_needs_lambda(shape) {
            self.emit_apply_with_offset(rb, shape, offset);
            return;
        }

        match shape.scalar_type() {
            Some(st) if is_string_like_scalar(st) => {
                self.decoder.lower_read_string(rb, offset, st);
            }
            Some(st) => {
                self.decoder.lower_read_scalar(rb, offset, st);
            }
            None => panic!("unsupported IR-lowered shape: {}", shape.type_identifier),
        }
    }

    fn lower_shape_body(
        &self,
        rb: &mut RegionBuilder<'_>,
        shape: &'static Shape,
        base_offset: usize,
    ) {
        if is_unit(shape) {
            return;
        }

        if shape.is_transparent() {
            let (fields, skipped) = lower_fields_for_ir(shape, base_offset);
            if !skipped.is_empty() {
                panic!(
                    "IR path does not support transparent wrappers with skipped/defaulted fields: {}",
                    shape.type_identifier
                );
            }
            if fields.len() != 1 {
                panic!(
                    "transparent wrapper must lower to exactly one field: {}",
                    shape.type_identifier
                );
            }
            self.lower_value(rb, fields[0].shape, fields[0].offset);
            return;
        }

        if let Some(opt_def) = get_option_def(shape) {
            let init_none_fn = opt_def.vtable.init_none as *const u8;
            let init_some_fn = opt_def.vtable.init_some as *const u8;
            let inner_layout = opt_def
                .t
                .layout
                .sized_layout()
                .expect("Option inner type must be Sized");
            let bytes = inner_layout.size().max(1);
            let slots = bytes.div_ceil(8);
            let scratch_slot = rb.alloc_slot();
            for _ in 1..slots {
                let _ = rb.alloc_slot();
            }
            self.decoder.lower_option(
                rb,
                base_offset,
                init_none_fn,
                init_some_fn,
                scratch_slot,
                &mut |inner_rb| {
                    self.lower_value(inner_rb, opt_def.t, 0);
                },
            );
            return;
        }

        if get_pointer_def(shape).is_some() {
            panic!(
                "IR path does not support pointer lowering yet: {}",
                shape.type_identifier
            );
        }

        if let Def::List(list_def) = &shape.def {
            let vec_offsets = crate::malum::discover_vec_offsets(list_def, shape);
            self.decoder
                .lower_vec(rb, base_offset, list_def.t, &vec_offsets, &mut |inner_rb| {
                    self.lower_value(inner_rb, list_def.t, 0);
                });
            return;
        }

        if let Def::Map(map_def) = &shape.def {
            let _ = (rb, map_def, base_offset);
            panic!(
                "IR path does not support Map lowering yet: {}",
                shape.type_identifier
            );
        }

        if let Def::Array(array_def) = &shape.def {
            let elem_layout = array_def
                .t
                .layout
                .sized_layout()
                .expect("array element must be Sized");
            let stride = elem_layout.size();
            for i in 0..array_def.n {
                self.lower_value(rb, array_def.t, base_offset + i * stride);
            }
            return;
        }

        match &shape.ty {
            Type::User(UserType::Struct(st)) => {
                let (fields, skipped) = lower_fields_for_ir(shape, base_offset);
                if !skipped.is_empty() {
                    panic!(
                        "IR path does not support skipped/defaulted fields yet: {}",
                        shape.type_identifier
                    );
                }
                let deny_unknown_fields = shape.has_deny_unknown_fields_attr();
                let is_positional = matches!(st.kind, StructKind::Tuple | StructKind::TupleStruct);
                if is_positional {
                    self.decoder
                        .lower_positional_fields(rb, &fields, &mut |inner_rb, field| {
                            self.lower_value(inner_rb, field.shape, field.offset);
                        });
                } else {
                    self.decoder.lower_struct_fields(
                        rb,
                        &fields,
                        deny_unknown_fields,
                        &mut |inner_rb, field| {
                            self.lower_value(inner_rb, field.shape, field.offset);
                        },
                    );
                }
            }
            Type::User(UserType::Enum(enum_type)) => {
                let disc_size = discriminant_size(enum_type.enum_repr);
                let disc_width = ir_width_from_disc_size(disc_size);
                let variants = lower_variants_for_ir(enum_type, base_offset);
                self.decoder
                    .lower_enum(rb, &variants, &mut |inner_rb, variant| {
                        let disc = inner_rb.const_val(variant.rust_discriminant as u64);
                        inner_rb.write_to_field(disc, base_offset as u32, disc_width);
                        for field in &variant.fields {
                            self.lower_value(inner_rb, field.shape, field.offset);
                        }
                    });
            }
            _ => match shape.scalar_type() {
                Some(st) if is_string_like_scalar(st) => {
                    self.decoder.lower_read_string(rb, base_offset, st);
                }
                Some(st) => {
                    self.decoder.lower_read_scalar(rb, base_offset, st);
                }
                None => panic!("unsupported IR-lowered shape: {}", shape.type_identifier),
            },
        }
    }
}

/// Compile a deserializer through RVSDG + linearization + backend adapter.
pub fn compile_decoder(shape: &'static Shape, decoder: &dyn Decoder) -> CompiledDecoder {
    let pipeline_opts = PipelineOptions::from_env();
    compile_decoder_with_options(shape, decoder, &pipeline_opts)
}

// r[impl compiler.opts.api]
pub fn compile_decoder_with_options(
    shape: &'static Shape,
    decoder: &dyn Decoder,
    pipeline_opts: &PipelineOptions,
) -> CompiledDecoder {
    let mut func = build_decoder_ir(shape, decoder);
    run_configured_default_passes(&mut func, pipeline_opts);
    let linear = crate::linearize::linearize(&mut func);
    compile_linear_ir_decoder_with_options(
        &linear,
        decoder.supports_trusted_utf8_input(),
        pipeline_opts.clone(),
    )
}

// r[impl ir.regalloc.regressions]
/// Build IR + linear form and run regalloc over it, returning total edit count.
///
/// This is a full-pipeline diagnostic helper, not a lightweight metric.
pub fn regalloc_edit_count(shape: &'static Shape, ir_decoder: &dyn Decoder) -> usize {
    let pipeline_opts = PipelineOptions::from_env();
    regalloc_edit_count_with_options(shape, ir_decoder, &pipeline_opts)
}

/// Build IR + linear form and run regalloc, returning a detailed edits dump.
pub fn regalloc_edits_text(shape: &'static Shape, ir_decoder: &dyn Decoder) -> String {
    let pipeline_opts = PipelineOptions::from_env();
    regalloc_edits_text_with_options(shape, ir_decoder, &pipeline_opts)
}

// r[impl compiler.opts.api]
pub fn regalloc_edit_count_with_options(
    shape: &'static Shape,
    ir_decoder: &dyn Decoder,
    pipeline_opts: &PipelineOptions,
) -> usize {
    if !pipeline_opts.resolve_regalloc(true) {
        return 0;
    }
    let mut func = build_decoder_ir(shape, ir_decoder);
    run_configured_default_passes(&mut func, pipeline_opts);
    let linear = crate::linearize::linearize(&mut func);
    let alloc = crate::regalloc_engine::allocate_linear_ir(&linear)
        .unwrap_or_else(|err| panic!("regalloc2 allocation failed while counting edits: {err}"));
    alloc.functions.iter().map(|f| f.edits.len()).sum()
}

/// Same as [`regalloc_edits_text`], but with explicit pipeline options.
pub fn regalloc_edits_text_with_options(
    shape: &'static Shape,
    ir_decoder: &dyn Decoder,
    pipeline_opts: &PipelineOptions,
) -> String {
    let mut func = build_decoder_ir(shape, ir_decoder);
    run_configured_default_passes(&mut func, pipeline_opts);
    let linear = crate::linearize::linearize(&mut func);
    let ra_mir = crate::regalloc_mir::lower_linear_ir(&linear);
    let alloc = if pipeline_opts.resolve_regalloc(true) {
        let mut alloc = crate::regalloc_engine::allocate_program(&ra_mir).unwrap_or_else(|err| {
            panic!("regalloc2 allocation failed while formatting edits: {err}")
        });
        maybe_disable_regalloc_edits(&mut alloc, pipeline_opts);
        alloc
    } else {
        no_regalloc_alloc_for_program(&ra_mir)
    };
    format_allocated_regalloc_edits(&alloc)
}

fn format_allocated_regalloc_edits(alloc: &crate::regalloc_engine::AllocatedProgram) -> String {
    let mut out = String::new();
    let total_pp_edits: usize = alloc.functions.iter().map(|f| f.edits.len()).sum();
    let total_edge_edits: usize = alloc.functions.iter().map(|f| f.edge_edits.len()).sum();
    let _ = std::fmt::Write::write_fmt(
        &mut out,
        format_args!(
            "total_progpoint_edits: {total_pp_edits}\ntotal_edge_edits: {total_edge_edits}\n"
        ),
    );

    for func in &alloc.functions {
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!(
                "\nlambda @{}:\n  num_spillslots: {}\n  progpoint_edits ({}):\n",
                func.lambda_id.index(),
                func.num_spillslots,
                func.edits.len()
            ),
        );
        for (prog_point, edit) in &func.edits {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!("    - {:?}: {:?}\n", prog_point, edit),
            );
        }

        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!("  edge_edits ({}):\n", func.edge_edits.len()),
        );
        for edge in &func.edge_edits {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "    - edge lin={} succ={} pos={:?} move {:?} -> {:?}\n",
                    edge.from_linear_op_index, edge.succ_index, edge.pos, edge.from, edge.to
                ),
            );
        }
    }

    out
}

pub(crate) fn build_decoder_ir(
    shape: &'static Shape,
    ir_decoder: &dyn Decoder,
) -> crate::ir::IrFunc {
    let mut builder = crate::ir::IrBuilder::new(shape);
    let mut lambda_shapes = Vec::new();
    collect_ir_lambda_shapes(
        shape,
        &mut std::collections::HashSet::new(),
        &mut lambda_shapes,
    );

    let mut lambda_by_shape = HashMap::new();
    lambda_by_shape.insert(shape as *const Shape, LambdaId::new(0));
    for lambda_shape in lambda_shapes.iter().copied().skip(1) {
        let lambda = builder.create_lambda(lambda_shape);
        lambda_by_shape.insert(lambda_shape as *const Shape, lambda);
    }

    let ir_lowerer = IrShapeLowerer::new(ir_decoder, lambda_by_shape);

    {
        let mut rb = builder.root_region();
        ir_lowerer.lower_shape_body(&mut rb, shape, 0);
        rb.set_results(&[]);
    }

    for lambda_shape in lambda_shapes.iter().copied().skip(1) {
        let lambda_id = ir_lowerer.lambda_for_shape(lambda_shape);
        let mut rb = builder.lambda_region(lambda_id);
        ir_lowerer.lower_shape_body(&mut rb, lambda_shape, 0);
        rb.set_results(&[]);
    }

    builder.finish()
}

pub(crate) fn run_default_passes_from_env(func: &mut crate::ir::IrFunc) {
    let pipeline_opts = PipelineOptions::from_env();
    run_configured_default_passes(func, &pipeline_opts);
}

pub(crate) fn run_configured_default_passes(
    func: &mut crate::ir::IrFunc,
    pipeline_opts: &PipelineOptions,
) {
    run_configured_default_passes_with_observer(func, pipeline_opts, |_, _| {});
}

pub(crate) fn run_configured_default_passes_with_observer<F>(
    func: &mut crate::ir::IrFunc,
    pipeline_opts: &PipelineOptions,
    mut observe_after_pass: F,
) where
    F: FnMut(&str, &crate::ir::IrFunc),
{
    // r[impl compiler.opts.all-opts]
    if !pipeline_opts.resolve_all_opts(DEFAULT_PRE_LINEARIZATION_PASSES_ENABLED) {
        return;
    }

    for pass in crate::ir_passes::default_pass_registry() {
        if !pipeline_opts.resolve_pass(pass.name, true) {
            continue;
        }
        pass.run(func);
        observe_after_pass(pass.name, func);
    }
}

// r[impl compiler.opts.regalloc]
fn maybe_disable_regalloc_edits(
    alloc: &mut crate::regalloc_engine::AllocatedProgram,
    pipeline_opts: &PipelineOptions,
) {
    if pipeline_opts.resolve_regalloc(true) {
        return;
    }

    for func in &mut alloc.functions {
        func.edits.clear();
        func.edge_edits.clear();
    }
}

fn no_regalloc_alloc_for_program(
    ra_mir: &crate::regalloc_mir::RaProgram,
) -> crate::regalloc_engine::AllocatedProgram {
    let functions = ra_mir
        .funcs
        .iter()
        .map(|func| crate::regalloc_engine::AllocatedFunction {
            lambda_id: func.lambda_id,
            num_spillslots: 0,
            edits: Vec::new(),
            inst_allocs: Vec::new(),
            inst_operands: Vec::new(),
            inst_linear_op_indices: Vec::new(),
            term_inst_indices_by_block: vec![None; func.blocks.len()],
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        })
        .collect();

    crate::regalloc_engine::AllocatedProgram {
        ra_program: ra_mir.clone(),
        functions,
    }
}

/// Compile a deserializer from already-linearized IR.
///
/// This is the first backend-adapter entrypoint used by the IR migration.
pub fn compile_linear_ir_decoder(
    ir: &crate::linearize::LinearIr,
    trusted_utf8_input: bool,
) -> CompiledDecoder {
    compile_linear_ir_decoder_with_options(ir, trusted_utf8_input, PipelineOptions::from_env())
}

fn compile_linear_ir_decoder_with_options(
    ir: &crate::linearize::LinearIr,
    trusted_utf8_input: bool,
    pipeline_opts: PipelineOptions,
) -> CompiledDecoder {
    let jit_debug = jit_debug_enabled();
    let apply_regalloc_edits = pipeline_opts.resolve_regalloc(true);

    // r[impl ir.regalloc.ra-mir]
    // Build allocator-oriented CFG IR before machine emission.
    let ra_mir = crate::regalloc_mir::lower_linear_ir(ir);
    let regalloc_alloc = if apply_regalloc_edits {
        // r[impl ir.regalloc.engine]
        // Run regalloc2 over RA-MIR and thread allocation artifacts into emission.
        let mut alloc = crate::regalloc_engine::allocate_program(&ra_mir)
            .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));
        maybe_disable_regalloc_edits(&mut alloc, &pipeline_opts);
        alloc
    } else {
        no_regalloc_alloc_for_program(&ra_mir)
    };

    let (buf, entry, source_map) = {
        let result = crate::ir_backend::compile_linear_ir_with_alloc_and_mode(
            ir,
            &ra_mir,
            &regalloc_alloc,
            apply_regalloc_edits,
        );
        materialize_backend_result(result)
    };
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
    let root_display_name = ir
        .ops
        .iter()
        .find_map(|op| match op {
            crate::linearize::LinearOp::FuncStart {
                lambda_id, shape, ..
            } if lambda_id.index() == 0 => {
                Some(format!("kajit::decode::{}", shape.type_identifier))
            }
            _ => None,
        })
        .unwrap_or_else(|| "kajit::decode::<ir-root>".to_string());
    let root_mangled_name = ir
        .ops
        .iter()
        .find_map(|op| match op {
            crate::linearize::LinearOp::FuncStart {
                lambda_id, shape, ..
            } if lambda_id.index() == 0 => Some(crate::jit_debug::rust_v0_mangle(&[
                "kajit",
                "decode",
                shape.type_identifier,
            ])),
            _ => None,
        })
        .unwrap_or_else(|| crate::jit_debug::rust_v0_mangle(&["kajit", "decode", "ir_root"]));
    let symbol = crate::jit_debug::JitSymbolEntry {
        name: root_mangled_name,
        offset: entry,
        size: buf.len().saturating_sub(entry),
    };
    let registration = if jit_debug {
        let (listing, line_by_linear_op) = build_ra_mir_listing(&ra_mir);
        let listing_path = write_ra_mir_listing_file(&root_display_name, &listing);
        let target_arch = jit_dwarf_target_arch();
        let root_shape = ir.ops.iter().find_map(|op| match op {
            crate::linearize::LinearOp::FuncStart {
                lambda_id, shape, ..
            } if lambda_id.index() == 0 => Some(*shape),
            _ => None,
        });
        let dwarf_variables = if let (Some(shape), Some(sm)) = (root_shape, source_map.as_ref()) {
            build_dwarf_variables_from_regalloc(
                ir,
                &ra_mir,
                &regalloc_alloc,
                sm,
                buf.len(),
                shape,
                target_arch,
            )
        } else {
            Vec::new()
        };
        let dwarf = listing_path.as_deref().and_then(|path| {
            build_dwarf_from_source_map(
                buf.code_ptr(),
                buf.len(),
                source_map.as_ref(),
                path,
                &line_by_linear_op,
                &root_display_name,
                &dwarf_variables,
                target_arch,
            )
        });
        crate::jit_debug::register_jit_code_with_dwarf(
            buf.code_ptr(),
            buf.len(),
            &[symbol],
            dwarf.as_ref(),
        )
    } else {
        crate::jit_debug::register_jit_code(buf.code_ptr(), buf.len(), &[symbol])
    };

    CompiledDecoder {
        buf,
        entry,
        func,
        trusted_utf8_input,
        _jit_registration: Some(registration),
    }
}

/// Compile a deserializer directly from an RaProgram (no LinearIr needed).
///
/// This is the backend for the RA-MIR text test workflow.
pub fn compile_ra_program_decoder(program: &crate::regalloc_mir::RaProgram) -> CompiledDecoder {
    let jit_debug = jit_debug_enabled();
    let pipeline_opts = PipelineOptions::from_env();
    let apply_regalloc_edits = pipeline_opts.resolve_regalloc(true);

    let alloc = if apply_regalloc_edits {
        let mut alloc = crate::regalloc_engine::allocate_program(program)
            .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));
        maybe_disable_regalloc_edits(&mut alloc, &pipeline_opts);
        alloc
    } else {
        no_regalloc_alloc_for_program(program)
    };

    let (buf, entry, source_map) = {
        let result =
            crate::ir_backend::compile_ra_program_with_mode(program, &alloc, apply_regalloc_edits);
        materialize_backend_result(result)
    };
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
    let root_display_name = "kajit::decode::<ra-mir-text>";
    let root_mangled_name = crate::jit_debug::rust_v0_mangle(&["kajit", "decode", "ra_mir_text"]);
    let symbol = crate::jit_debug::JitSymbolEntry {
        name: root_mangled_name,
        offset: entry,
        size: buf.len().saturating_sub(entry),
    };
    let registration = if jit_debug {
        let (listing, line_by_linear_op) = build_ra_mir_listing(program);
        let listing_path = write_ra_mir_listing_file(root_display_name, &listing);
        let target_arch = jit_dwarf_target_arch();
        let dwarf_variables = Vec::new();
        let dwarf = listing_path.as_deref().and_then(|path| {
            build_dwarf_from_source_map(
                buf.code_ptr(),
                buf.len(),
                source_map.as_ref(),
                path,
                &line_by_linear_op,
                root_display_name,
                &dwarf_variables,
                target_arch,
            )
        });
        crate::jit_debug::register_jit_code_with_dwarf(
            buf.code_ptr(),
            buf.len(),
            &[symbol],
            dwarf.as_ref(),
        )
    } else {
        crate::jit_debug::register_jit_code(buf.code_ptr(), buf.len(), &[symbol])
    };

    CompiledDecoder {
        buf,
        entry,
        func,
        trusted_utf8_input: false,
        _jit_registration: Some(registration),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_dwarf_sections_from_source_map_lines() {
        let mut line_by_linear_op = HashMap::new();
        line_by_linear_op.insert(0usize, 1u32);
        line_by_linear_op.insert(3usize, 2u32);

        let source_map = vec![
            kajit_emit::SourceMapEntry {
                offset: 0,
                location: kajit_emit::SourceLocation {
                    file: 0,
                    line: 1,
                    column: 1,
                },
            },
            kajit_emit::SourceMapEntry {
                offset: 8,
                location: kajit_emit::SourceLocation {
                    file: 0,
                    line: 4,
                    column: 1,
                },
            },
        ];

        let listing_path = std::env::temp_dir()
            .join(format!("kajit-debug-test-{}", std::process::id()))
            .join("sample.ra-mir");
        std::fs::create_dir_all(listing_path.parent().expect("temp listing dir")).unwrap();
        std::fs::write(&listing_path, "inst0\ninst1\n").unwrap();

        let dwarf = build_dwarf_from_source_map(
            0x1000 as *const u8,
            32,
            Some(&source_map),
            &listing_path,
            &line_by_linear_op,
            "kajit::decode::test",
            &[],
            jit_dwarf_target_arch(),
        )
        .expect("expected dwarf sections");
        assert!(!dwarf.debug_line.is_empty());
    }
}
