use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use facet::{
    ConstParamKind, Def, EnumRepr, Facet, KnownPointer, ListDef, OptionDef, PointerDef, ScalarType,
    Shape, StructKind, Type, UserType,
};
use kajit_hir as hir;

use crate::arch::EmitCtx;
use crate::format::{
    Decoder, FieldEmitInfo, FieldLowerInfo, SkippedFieldInfo, VariantEmitInfo, VariantKind,
    VariantLowerInfo,
};
use crate::intrinsics;
use crate::ir::{LambdaId, RegionBuilder, Width as IrWidth};
use crate::pipeline_opts::PipelineOptions;

/// A compiled deserializer. Owns the executable buffer containing JIT'd machine code.
pub struct CompiledDecoder {
    #[cfg(target_arch = "x86_64")]
    buf: kajit_emit::x64::FinalizedEmission,
    #[cfg(target_arch = "aarch64")]
    buf: kajit_emit::aarch64::FinalizedEmission,
    cfg_mir_line_text_by_line: Vec<String>,
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

    /// Deterministic machine-emission trace annotated with CFG-MIR provenance.
    pub fn emission_trace_text(&self) -> Result<String, kajit_emit::TraceError> {
        #[cfg(target_arch = "x86_64")]
        let entries = self.buf.trace_entries()?;

        #[cfg(target_arch = "aarch64")]
        let entries = self.buf.trace_entries()?;

        Ok(format_emission_trace_entries(
            &entries,
            &self.cfg_mir_line_text_by_line,
        ))
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
    Option<crate::ir_backend::BackendDebugInfo>,
) {
    let crate::ir_backend::LinearBackendResult {
        buf,
        entry,
        source_map,
        backend_debug_info,
    } = result;
    (buf, entry as usize, source_map, backend_debug_info)
}

#[cfg(target_arch = "x86_64")]
fn materialize_backend_result(
    result: crate::ir_backend::LinearBackendResult,
) -> (
    kajit_emit::x64::FinalizedEmission,
    usize,
    Option<kajit_emit::SourceMap>,
    Option<crate::ir_backend::BackendDebugInfo>,
) {
    let crate::ir_backend::LinearBackendResult {
        buf,
        entry,
        source_map,
        backend_debug_info,
    } = result;
    (buf, entry as usize, source_map, backend_debug_info)
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

struct CfgMirListing {
    text: String,
    line_text_by_line: Vec<String>,
}

fn build_cfg_mir_listing(
    program: &crate::regalloc_engine::cfg_mir::Program,
    registry: Option<&crate::ir::IntrinsicRegistry>,
) -> CfgMirListing {
    let lines = program.debug_line_listing_with_registry(registry);
    let mut listing = lines.join("\n");
    if !listing.is_empty() {
        listing.push('\n');
    }
    CfgMirListing {
        text: listing,
        line_text_by_line: lines,
    }
}

fn format_emission_trace_entries(
    entries: &[kajit_emit::TraceEntry],
    cfg_mir_line_text_by_line: &[String],
) -> String {
    entries
        .iter()
        .map(|entry| {
            let hex = entry
                .bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let line_text = entry
                .location
                .line
                .checked_sub(1)
                .and_then(|idx| cfg_mir_line_text_by_line.get(idx as usize))
                .map(String::as_str)
                .unwrap_or("<unknown cfg-mir provenance>");
            let bytes = if should_redact_trace_bytes(line_text) {
                format!("<redacted:{}>", entry.bytes.len())
            } else {
                hex
            };
            format!(
                "{:08x} line={} col={} bytes={} :: {}",
                entry.offset, entry.location.line, entry.location.column, bytes, line_text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn should_redact_trace_bytes(line_text: &str) -> bool {
    line_text.contains("const(@")
        || line_text.contains("call_intrinsic(@")
        || line_text.contains("call_pure(@")
}

fn write_cfg_mir_listing_file(type_name: &str, listing: &str) -> Option<PathBuf> {
    let stem = sanitize_debug_file_stem(type_name);
    let dir = Path::new("/tmp/kajit-debug");
    std::fs::create_dir_all(dir).ok()?;
    let path = dir.join(format!("{stem}.cfg-mir"));
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

fn build_jit_debug_info_from_source_map(
    code_ptr: *const u8,
    code_len: usize,
    source_map: Option<&kajit_emit::SourceMap>,
    listing_path: &Path,
    subprogram: crate::jit_dwarf::JitDebugSubprogram,
) -> Option<crate::jit_dwarf::JitDebugInfo> {
    let source_map = source_map?;
    let file_name = listing_path.file_name()?.to_str()?.to_owned();
    let directory = listing_path
        .parent()
        .and_then(Path::to_str)
        .map(str::to_owned);
    let rows = source_map
        .iter()
        .filter(|entry| entry.location.line != 0)
        .map(|entry| crate::jit_dwarf::JitDebugLineRow {
            code_offset: entry.offset,
            line: entry.location.line,
        })
        .collect();

    Some(crate::jit_dwarf::JitDebugInfo {
        target_arch: jit_dwarf_target_arch(),
        code_address: code_ptr as u64,
        code_size: code_len as u64,
        line_table: crate::jit_dwarf::JitDebugLineTable {
            file_name,
            directory,
            rows,
        },
        subprogram,
    })
}

#[derive(Debug, Clone, Copy)]
struct DeserDebugRegisterSet {
    input_ptr_hw: u8,
    input_end_hw: u8,
    out_ptr_hw: u8,
    ctx_hw: u8,
}

fn deser_debug_registers(target_arch: crate::jit_dwarf::DwarfTargetArch) -> DeserDebugRegisterSet {
    match target_arch {
        crate::jit_dwarf::DwarfTargetArch::X86_64 => DeserDebugRegisterSet {
            input_ptr_hw: 12,
            input_end_hw: 13,
            out_ptr_hw: 14,
            ctx_hw: 15,
        },
        crate::jit_dwarf::DwarfTargetArch::Aarch64 => DeserDebugRegisterSet {
            input_ptr_hw: 19,
            input_end_hw: 20,
            out_ptr_hw: 21,
            ctx_hw: 22,
        },
    }
}

fn deser_dwarf_variables(
    target_arch: crate::jit_dwarf::DwarfTargetArch,
) -> Vec<crate::jit_dwarf::DwarfVariable> {
    let regs = deser_debug_registers(target_arch);
    let input_ptr_reg =
        crate::jit_dwarf::dwarf_register_from_hw_encoding(target_arch, regs.input_ptr_hw)
            .expect("input_ptr register should map to a DWARF register");
    let input_end_reg =
        crate::jit_dwarf::dwarf_register_from_hw_encoding(target_arch, regs.input_end_hw)
            .expect("input_end register should map to a DWARF register");
    let out_ptr_reg =
        crate::jit_dwarf::dwarf_register_from_hw_encoding(target_arch, regs.out_ptr_hw)
            .expect("out_ptr register should map to a DWARF register");
    let ctx_reg = crate::jit_dwarf::dwarf_register_from_hw_encoding(target_arch, regs.ctx_hw)
        .expect("ctx register should map to a DWARF register");

    [
        ("input_ptr", crate::jit_dwarf::expr_reg(input_ptr_reg)),
        ("input_end", crate::jit_dwarf::expr_reg(input_end_reg)),
        ("out_ptr", crate::jit_dwarf::expr_reg(out_ptr_reg)),
        ("ctx", crate::jit_dwarf::expr_reg(ctx_reg)),
        (
            "error_code",
            crate::jit_dwarf::expr_breg(ctx_reg, crate::context::CTX_ERROR_CODE as i64),
        ),
        (
            "error_offset",
            crate::jit_dwarf::expr_breg(ctx_reg, crate::context::CTX_ERROR_OFFSET as i64),
        ),
    ]
    .into_iter()
    .map(|(name, expr)| crate::jit_dwarf::DwarfVariable {
        name: name.to_owned(),
        location: crate::jit_dwarf::DwarfVariableLocation::Expr(expr),
    })
    .collect()
}

fn scalar_field_dwarf_width(shape: &'static Shape) -> Option<u8> {
    let scalar_type = shape.scalar_type()?;
    if matches!(scalar_type, ScalarType::Unit) || is_string_like_scalar(scalar_type) {
        return None;
    }
    let size = shape.layout.sized_layout().ok()?.size();
    match size {
        1 | 2 | 4 | 8 => Some(size as u8),
        _ => None,
    }
}

fn dwarf_expr_for_out_field(
    target_arch: crate::jit_dwarf::DwarfTargetArch,
    offset: u32,
    size: u8,
) -> Vec<u8> {
    let regs = deser_debug_registers(target_arch);
    let out_ptr_reg =
        crate::jit_dwarf::dwarf_register_from_hw_encoding(target_arch, regs.out_ptr_hw)
            .expect("out_ptr register should map to a DWARF register");
    crate::jit_dwarf::expr_breg_deref_size_stack_value(out_ptr_reg, offset as i64, size)
}

fn cfg_semantic_field_dwarf_variables(
    root_shape: &'static Shape,
    program: &crate::regalloc_engine::cfg_mir::Program,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    code_ptr: *const u8,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
) -> Vec<ScopedDwarfVariable> {
    let Some(backend_debug_info) = backend_debug_info else {
        return Vec::new();
    };
    let root_scope = program.debug.root_scope;
    let op_ranges = backend_op_ranges_by_op(backend_debug_info, code_ptr);
    let root_lambda = crate::ir::LambdaId::new(0);
    let Some(code_end) = op_ranges
        .iter()
        .filter(|((lambda_raw, _), _)| *lambda_raw == root_lambda.index() as u32)
        .flat_map(|(_, ranges)| ranges.iter().map(|(_, end)| *end))
        .max()
    else {
        return Vec::new();
    };

    let (fields, _) = collect_fields(root_shape);
    let mut out = Vec::new();
    for field in fields {
        let Some(width) = scalar_field_dwarf_width(field.shape) else {
            continue;
        };
        let mut lexical_start = None::<u64>;
        let mut available_start = None::<u64>;

        'search: for func in &program.funcs {
            if func.lambda_id != root_lambda {
                continue;
            }
            for block in &func.blocks {
                for inst_id in &block.insts {
                    let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(*inst_id);
                    let Some(debug_value_id) = program.op_debug_value(func.lambda_id, op_id) else {
                        continue;
                    };
                    let debug_value = &program.debug.values[debug_value_id];
                    let crate::ir::DebugValueKind::Field { offset } = debug_value.kind else {
                        continue;
                    };
                    if offset != field.offset as u32 || debug_value.name != field.name {
                        continue;
                    }

                    let inst = func
                        .inst(*inst_id)
                        .expect("cfg instruction should exist for semantic debug field");
                    let writes_field = match &inst.op {
                        crate::linearize::LinearOp::WriteToField { offset, .. } => {
                            *offset == field.offset as u32
                        }
                        crate::linearize::LinearOp::CallIntrinsic { field_offset, .. } => {
                            *field_offset == field.offset as u32
                        }
                        _ => false,
                    };
                    if !writes_field {
                        continue;
                    }

                    let Some(ranges) = op_ranges.get(&(func.lambda_id.index() as u32, op_id))
                    else {
                        continue;
                    };
                    lexical_start = ranges.iter().map(|(start, _)| *start).min();
                    available_start = ranges.iter().map(|(_, end)| *end).max();
                    break 'search;
                }
            }
        }

        let (Some(lexical_start), Some(available_start)) = (lexical_start, available_start) else {
            continue;
        };
        if available_start >= code_end {
            continue;
        }

        out.push(ScopedDwarfVariable {
            scope: root_scope,
            lexical_ranges: vec![crate::jit_dwarf::JitDebugRange {
                low_pc: lexical_start,
                high_pc: code_end,
            }],
            variable: crate::jit_dwarf::DwarfVariable {
                name: field.name.to_string(),
                location: crate::jit_dwarf::DwarfVariableLocation::List(vec![
                    crate::jit_dwarf::DwarfLocationRange {
                        start: available_start,
                        end: code_end,
                        expression: dwarf_expr_for_out_field(
                            target_arch,
                            field.offset as u32,
                            width,
                        ),
                    },
                ]),
            },
        });
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn aarch64_regalloc_extra_saved_pairs(alloc: &crate::regalloc_engine::AllocatedCfgProgram) -> u32 {
    let mut max_pair = None::<u32>;
    let mut observe = |allocation: regalloc2::Allocation| {
        let Some(reg) = allocation.as_reg() else {
            return;
        };
        if reg.class() != regalloc2::RegClass::Int {
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
        for inst_allocs in func.op_allocs.values() {
            for &allocation in inst_allocs {
                observe(allocation);
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
        for &allocation in &func.return_result_allocs {
            observe(allocation);
        }
    }

    max_pair.map_or(0, |pair| pair + 1)
}

fn find_cfg_alloc_for_vreg_in_op(
    alloc_func: &crate::regalloc_engine::AllocatedCfgFunction,
    op_id: crate::regalloc_engine::cfg_mir::OpId,
    vreg: crate::ir::VReg,
    preferred_kind: Option<crate::regalloc_engine::cfg_mir::OperandKind>,
) -> Option<regalloc2::Allocation> {
    let operands = alloc_func.op_operands.get(&op_id)?;
    let allocs = alloc_func.op_allocs.get(&op_id)?;
    for ((operand_vreg, operand_kind), alloc) in operands.iter().zip(allocs.iter().copied()) {
        if *operand_vreg != vreg {
            continue;
        }
        if preferred_kind.is_none_or(|kind| *operand_kind == kind) {
            return Some(alloc);
        }
    }
    None
}

fn infer_cfg_block_param_entry_alloc(
    _func: &crate::regalloc_engine::cfg_mir::Function,
    alloc_func: &crate::regalloc_engine::AllocatedCfgFunction,
    block: &crate::regalloc_engine::cfg_mir::Block,
    param: crate::ir::VReg,
) -> Option<regalloc2::Allocation> {
    for inst_id in &block.insts {
        let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(*inst_id);
        if let Some(alloc) = find_cfg_alloc_for_vreg_in_op(
            alloc_func,
            op_id,
            param,
            Some(crate::regalloc_engine::cfg_mir::OperandKind::Use),
        ) {
            return Some(alloc);
        }
        if let Some(alloc) = find_cfg_alloc_for_vreg_in_op(
            alloc_func,
            op_id,
            param,
            Some(crate::regalloc_engine::cfg_mir::OperandKind::Def),
        ) {
            return Some(alloc);
        }
    }
    let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(block.term);
    find_cfg_alloc_for_vreg_in_op(
        alloc_func,
        term_op,
        param,
        Some(crate::regalloc_engine::cfg_mir::OperandKind::Use),
    )
}

fn dwarf_expr_for_cfg_allocation(
    program: &crate::regalloc_engine::cfg_mir::Program,
    alloc: &crate::regalloc_engine::AllocatedCfgProgram,
    allocation: regalloc2::Allocation,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
    apply_regalloc_edits: bool,
) -> Option<Vec<u8>> {
    if let Some(reg) = allocation.as_reg() {
        if reg.class() != regalloc2::RegClass::Int {
            return None;
        }
        let dwarf_reg =
            crate::jit_dwarf::dwarf_register_from_hw_encoding(target_arch, reg.hw_enc() as u8)?;
        return Some(crate::jit_dwarf::expr_reg(dwarf_reg));
    }

    let slot = allocation.as_stack()?;
    #[cfg(target_arch = "x86_64")]
    {
        let slot_base = crate::arch::BASE_FRAME;
        let spill_base = slot_base + program.slot_count * 8;
        let offset = spill_base + (slot.index() as u32) * 8;
        return Some(crate::jit_dwarf::expr_fbreg(offset as i64));
    }

    #[cfg(target_arch = "aarch64")]
    {
        let extra_saved_pairs = aarch64_regalloc_extra_saved_pairs(alloc);
        let slot_base = crate::arch::BASE_FRAME + extra_saved_pairs * 16;
        let spill_base = slot_base + program.slot_count * 8;
        let _ = apply_regalloc_edits;
        let offset = spill_base + (slot.index() as u32) * 8;
        return Some(crate::jit_dwarf::expr_fbreg(offset as i64));
    }

    #[allow(unreachable_code)]
    None
}

fn backend_op_ranges_by_op(
    backend_debug_info: &crate::ir_backend::BackendDebugInfo,
    code_ptr: *const u8,
) -> BTreeMap<(u32, crate::regalloc_engine::cfg_mir::OpId), Vec<(u64, u64)>> {
    backend_debug_info
        .op_infos
        .iter()
        .map(|op_info| {
            (
                (op_info.lambda_id, op_info.op_id),
                op_info
                    .code_ranges
                    .iter()
                    .map(|range| {
                        (
                            code_ptr as u64 + range.start_offset as u64,
                            code_ptr as u64 + range.end_offset as u64,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

#[derive(Debug, Clone)]
struct ScopedDwarfVariable {
    scope: Option<crate::ir::DebugScopeId>,
    lexical_ranges: Vec<crate::jit_dwarf::JitDebugRange>,
    variable: crate::jit_dwarf::DwarfVariable,
}

#[derive(Debug, Clone)]
struct VRegDwarfVariableInfo {
    scope: Option<crate::ir::DebugScopeId>,
    lexical_intro_ranges: Vec<crate::jit_dwarf::JitDebugRange>,
    locations: Vec<crate::jit_dwarf::DwarfLocationRange>,
}

fn merge_jit_debug_ranges(
    mut ranges: Vec<crate::jit_dwarf::JitDebugRange>,
) -> Vec<crate::jit_dwarf::JitDebugRange> {
    ranges.sort_by_key(|range| (range.low_pc, range.high_pc));
    let mut merged = Vec::<crate::jit_dwarf::JitDebugRange>::new();
    for range in ranges {
        if range.high_pc <= range.low_pc {
            continue;
        }
        if let Some(last) = merged.last_mut()
            && last.high_pc >= range.low_pc
        {
            last.high_pc = last.high_pc.max(range.high_pc);
            continue;
        }
        merged.push(range);
    }
    merged
}

fn merge_dwarf_location_ranges(
    mut locations: Vec<crate::jit_dwarf::DwarfLocationRange>,
) -> Vec<crate::jit_dwarf::DwarfLocationRange> {
    locations.sort_by_key(|location| (location.start, location.end));
    let mut merged = Vec::<crate::jit_dwarf::DwarfLocationRange>::new();
    for location in locations {
        if location.end <= location.start {
            continue;
        }
        if let Some(last) = merged.last_mut()
            && last.expression == location.expression
            && last.end >= location.start
        {
            last.end = last.end.max(location.end);
            continue;
        }
        merged.push(location);
    }
    merged
}

fn common_debug_scope(
    program: &crate::regalloc_engine::cfg_mir::Program,
    scopes: impl IntoIterator<Item = crate::ir::DebugScopeId>,
) -> Option<crate::ir::DebugScopeId> {
    let scopes = scopes.into_iter().collect::<Vec<_>>();
    let first = *scopes.first()?;
    let mut ancestors = Vec::new();
    let mut cursor = Some(first);
    while let Some(scope_id) = cursor {
        ancestors.push(scope_id);
        cursor = program.debug.scopes[scope_id].parent;
    }
    ancestors.into_iter().find(|candidate| {
        scopes.iter().all(|scope_id| {
            let mut cursor = Some(*scope_id);
            while let Some(current) = cursor {
                if current == *candidate {
                    return true;
                }
                cursor = program.debug.scopes[current].parent;
            }
            false
        })
    })
}

fn build_variable_interval_blocks(
    variables: Vec<ScopedDwarfVariable>,
) -> (
    Vec<crate::jit_dwarf::DwarfVariable>,
    Vec<crate::jit_dwarf::JitDebugLexicalBlock>,
) {
    let mut direct_variables = Vec::new();
    let mut ranged_variables = Vec::<(
        crate::jit_dwarf::DwarfVariable,
        Vec<crate::jit_dwarf::JitDebugRange>,
    )>::new();
    let mut boundaries = Vec::<u64>::new();

    for variable in variables {
        let ranges = variable.lexical_ranges;
        if ranges.is_empty() {
            direct_variables.push(variable.variable);
            continue;
        }
        for range in &ranges {
            boundaries.push(range.low_pc);
            boundaries.push(range.high_pc);
        }
        ranged_variables.push((variable.variable, ranges));
    }

    boundaries.sort_unstable();
    boundaries.dedup();

    let mut interval_blocks = Vec::<crate::jit_dwarf::JitDebugLexicalBlock>::new();
    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        if end <= start {
            continue;
        }

        let mut active_variables = ranged_variables
            .iter()
            .filter(|(_, ranges)| {
                ranges
                    .iter()
                    .any(|range| range.low_pc <= start && end <= range.high_pc)
            })
            .map(|(variable, _)| variable.clone())
            .collect::<Vec<_>>();
        if active_variables.is_empty() {
            continue;
        }
        active_variables.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));

        if let Some(last) = interval_blocks.last_mut()
            && last.lexical_blocks.is_empty()
            && last.variables == active_variables
            && last.ranges.len() == 1
            && last.ranges[0].high_pc == start
        {
            last.ranges[0].high_pc = end;
            continue;
        }

        interval_blocks.push(crate::jit_dwarf::JitDebugLexicalBlock {
            ranges: vec![crate::jit_dwarf::JitDebugRange {
                low_pc: start,
                high_pc: end,
            }],
            variables: active_variables,
            lexical_blocks: Vec::new(),
        });
    }

    (direct_variables, interval_blocks)
}

fn scope_ranges_from_backend(
    program: &crate::regalloc_engine::cfg_mir::Program,
    backend_debug_info: &crate::ir_backend::BackendDebugInfo,
    code_ptr: *const u8,
) -> BTreeMap<crate::ir::DebugScopeId, Vec<crate::jit_dwarf::JitDebugRange>> {
    let op_ranges = backend_op_ranges_by_op(backend_debug_info, code_ptr);
    let mut direct =
        BTreeMap::<crate::ir::DebugScopeId, Vec<crate::jit_dwarf::JitDebugRange>>::new();
    for ((lambda_raw, op_id), ranges) in op_ranges {
        let lambda_id = crate::ir::LambdaId::new(lambda_raw);
        let Some(scope) = program.op_debug_scope(lambda_id, op_id) else {
            continue;
        };
        let dest = direct.entry(scope).or_default();
        for (low_pc, high_pc) in ranges {
            if high_pc > low_pc {
                dest.push(crate::jit_dwarf::JitDebugRange { low_pc, high_pc });
            }
        }
    }

    let mut children_by_parent =
        BTreeMap::<crate::ir::DebugScopeId, Vec<crate::ir::DebugScopeId>>::new();
    for (scope_id, scope) in program.debug.scopes.iter() {
        if let Some(parent) = scope.parent {
            children_by_parent.entry(parent).or_default().push(scope_id);
        }
    }

    fn accumulate(
        scope_id: crate::ir::DebugScopeId,
        direct: &BTreeMap<crate::ir::DebugScopeId, Vec<crate::jit_dwarf::JitDebugRange>>,
        children_by_parent: &BTreeMap<crate::ir::DebugScopeId, Vec<crate::ir::DebugScopeId>>,
        memo: &mut BTreeMap<crate::ir::DebugScopeId, Vec<crate::jit_dwarf::JitDebugRange>>,
    ) -> Vec<crate::jit_dwarf::JitDebugRange> {
        if let Some(ranges) = memo.get(&scope_id) {
            return ranges.clone();
        }

        let mut ranges = direct.get(&scope_id).cloned().unwrap_or_default();
        if let Some(children) = children_by_parent.get(&scope_id) {
            for child in children {
                ranges.extend(accumulate(*child, direct, children_by_parent, memo));
            }
        }
        let merged = merge_jit_debug_ranges(ranges);
        memo.insert(scope_id, merged.clone());
        merged
    }

    let mut memo = BTreeMap::new();
    for (scope_id, _) in program.debug.scopes.iter() {
        let _ = accumulate(scope_id, &direct, &children_by_parent, &mut memo);
    }
    memo
}

fn cfg_vreg_dwarf_variable_infos(
    program: &crate::regalloc_engine::cfg_mir::Program,
    alloc: &crate::regalloc_engine::AllocatedCfgProgram,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    code_ptr: *const u8,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
    apply_regalloc_edits: bool,
) -> BTreeMap<crate::ir::VReg, VRegDwarfVariableInfo> {
    let Some(backend_debug_info) = backend_debug_info else {
        return BTreeMap::new();
    };
    let op_ranges = backend_op_ranges_by_op(backend_debug_info, code_ptr);
    let alloc_func_by_lambda = alloc
        .functions
        .iter()
        .map(|func| (func.lambda_id, func))
        .collect::<HashMap<_, _>>();
    let mut ranges_by_vreg =
        BTreeMap::<crate::ir::VReg, Vec<crate::jit_dwarf::DwarfLocationRange>>::new();
    let mut lexical_intro_ranges_by_vreg =
        BTreeMap::<crate::ir::VReg, Vec<crate::jit_dwarf::JitDebugRange>>::new();

    for func in &program.funcs {
        let Some(alloc_func) = alloc_func_by_lambda.get(&func.lambda_id) else {
            continue;
        };
        let lambda_key = func.lambda_id.index() as u32;
        for block in &func.blocks {
            let mut remaining_uses = BTreeMap::<crate::ir::VReg, usize>::new();
            for inst_id in &block.insts {
                let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(*inst_id);
                if let Some(operand_pairs) = alloc_func.op_operands.get(&op_id) {
                    for (vreg, operand_kind) in operand_pairs {
                        if *operand_kind == crate::regalloc_engine::cfg_mir::OperandKind::Use {
                            *remaining_uses.entry(*vreg).or_default() += 1;
                        }
                    }
                }
            }
            let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(block.term);
            if let Some(operand_pairs) = alloc_func.op_operands.get(&term_op) {
                for (vreg, operand_kind) in operand_pairs {
                    if *operand_kind == crate::regalloc_engine::cfg_mir::OperandKind::Use {
                        *remaining_uses.entry(*vreg).or_default() += 1;
                    }
                }
            }
            for &edge_id in &block.succs {
                let Some(edge) = func.edges.get(edge_id.index()) else {
                    continue;
                };
                for edge_arg in &edge.args {
                    *remaining_uses.entry(edge_arg.source).or_default() += 1;
                }
            }

            let mut live_locations = BTreeMap::<crate::ir::VReg, regalloc2::Allocation>::new();
            for &param in &block.params {
                if remaining_uses.get(&param).copied().unwrap_or(0) == 0 {
                    continue;
                }
                let Some(allocation) =
                    infer_cfg_block_param_entry_alloc(func, alloc_func, block, param)
                else {
                    continue;
                };
                live_locations.insert(param, allocation);
            }

            for inst_id in &block.insts {
                let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(*inst_id);
                let Some(op_ranges) = op_ranges.get(&(lambda_key, op_id)) else {
                    continue;
                };
                let mut used_now = Vec::<crate::ir::VReg>::new();
                let mut defs_after = Vec::<(crate::ir::VReg, regalloc2::Allocation)>::new();
                if let (Some(operand_pairs), Some(operand_allocs)) = (
                    alloc_func.op_operands.get(&op_id),
                    alloc_func.op_allocs.get(&op_id),
                ) {
                    for ((vreg, operand_kind), allocation) in
                        operand_pairs.iter().zip(operand_allocs.iter().copied())
                    {
                        match operand_kind {
                            crate::regalloc_engine::cfg_mir::OperandKind::Use => {
                                live_locations.insert(*vreg, allocation);
                                used_now.push(*vreg);
                            }
                            crate::regalloc_engine::cfg_mir::OperandKind::Def => {
                                let dest = lexical_intro_ranges_by_vreg.entry(*vreg).or_default();
                                dest.extend(op_ranges.iter().map(|(start, end)| {
                                    crate::jit_dwarf::JitDebugRange {
                                        low_pc: *start,
                                        high_pc: *end,
                                    }
                                }));
                                defs_after.push((*vreg, allocation));
                            }
                        }
                    }
                }

                for (vreg, allocation) in &live_locations {
                    if remaining_uses.get(vreg).copied().unwrap_or(0) == 0 {
                        continue;
                    }
                    let Some(expr) = dwarf_expr_for_cfg_allocation(
                        program,
                        alloc,
                        *allocation,
                        target_arch,
                        apply_regalloc_edits,
                    ) else {
                        continue;
                    };
                    let dest = ranges_by_vreg.entry(*vreg).or_default();
                    for (start, end) in op_ranges {
                        dest.push(crate::jit_dwarf::DwarfLocationRange {
                            start: *start,
                            end: *end,
                            expression: expr.clone(),
                        });
                    }
                }

                for vreg in used_now {
                    if let Some(count) = remaining_uses.get_mut(&vreg) {
                        *count = count.saturating_sub(1);
                    }
                }
                live_locations.retain(|vreg, _| remaining_uses.get(vreg).copied().unwrap_or(0) > 0);
                for (vreg, allocation) in defs_after {
                    if remaining_uses.get(&vreg).copied().unwrap_or(0) > 0 {
                        live_locations.insert(vreg, allocation);
                    }
                }
            }

            let Some(op_ranges) = op_ranges.get(&(lambda_key, term_op)) else {
                continue;
            };
            let mut used_now = Vec::<crate::ir::VReg>::new();
            if let (Some(operand_pairs), Some(operand_allocs)) = (
                alloc_func.op_operands.get(&term_op),
                alloc_func.op_allocs.get(&term_op),
            ) {
                for ((vreg, operand_kind), allocation) in
                    operand_pairs.iter().zip(operand_allocs.iter().copied())
                {
                    if *operand_kind != crate::regalloc_engine::cfg_mir::OperandKind::Use {
                        continue;
                    }
                    live_locations.insert(*vreg, allocation);
                    used_now.push(*vreg);
                }
            }
            for (vreg, allocation) in &live_locations {
                if remaining_uses.get(vreg).copied().unwrap_or(0) == 0 {
                    continue;
                }
                let Some(expr) = dwarf_expr_for_cfg_allocation(
                    program,
                    alloc,
                    *allocation,
                    target_arch,
                    apply_regalloc_edits,
                ) else {
                    continue;
                };
                let dest = ranges_by_vreg.entry(*vreg).or_default();
                for (start, end) in op_ranges {
                    dest.push(crate::jit_dwarf::DwarfLocationRange {
                        start: *start,
                        end: *end,
                        expression: expr.clone(),
                    });
                }
            }
            for vreg in used_now {
                if let Some(count) = remaining_uses.get_mut(&vreg) {
                    *count = count.saturating_sub(1);
                }
            }
        }
    }

    ranges_by_vreg
        .into_iter()
        .map(|(vreg, locations)| {
            (
                vreg,
                VRegDwarfVariableInfo {
                    scope: program.vreg_debug_scope(vreg),
                    lexical_intro_ranges: lexical_intro_ranges_by_vreg
                        .remove(&vreg)
                        .unwrap_or_default(),
                    locations: merge_dwarf_location_ranges(locations),
                },
            )
        })
        .collect()
}

fn cfg_value_dwarf_variables(
    program: &crate::regalloc_engine::cfg_mir::Program,
    alloc: &crate::regalloc_engine::AllocatedCfgProgram,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    code_ptr: *const u8,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
    apply_regalloc_edits: bool,
    suppress_semantic_vregs: bool,
) -> Vec<ScopedDwarfVariable> {
    cfg_vreg_dwarf_variable_infos(
        program,
        alloc,
        backend_debug_info,
        code_ptr,
        target_arch,
        apply_regalloc_edits,
    )
    .into_iter()
    .filter_map(|(vreg, info): (crate::ir::VReg, VRegDwarfVariableInfo)| {
        if suppress_semantic_vregs && program.vreg_debug_value(vreg).is_some() {
            return None;
        }
        if info.locations.is_empty() {
            return None;
        }
        let mut lexical_ranges = info.lexical_intro_ranges;
        lexical_ranges.extend(info.locations.iter().map(|location| {
            crate::jit_dwarf::JitDebugRange {
                low_pc: location.start,
                high_pc: location.end,
            }
        }));
        let lexical_ranges = merge_jit_debug_ranges(lexical_ranges);
        let variable = crate::jit_dwarf::DwarfVariable {
            name: format!("v{}", vreg.index()),
            location: crate::jit_dwarf::DwarfVariableLocation::List(info.locations),
        };
        Some(ScopedDwarfVariable {
            scope: info.scope,
            lexical_ranges,
            variable,
        })
    })
    .collect()
}

fn cfg_semantic_named_dwarf_variables(
    program: &crate::regalloc_engine::cfg_mir::Program,
    alloc: &crate::regalloc_engine::AllocatedCfgProgram,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    code_ptr: *const u8,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
    apply_regalloc_edits: bool,
) -> Vec<ScopedDwarfVariable> {
    let mut vregs_by_value = BTreeMap::<crate::ir::DebugValueId, Vec<crate::ir::VReg>>::new();
    for vreg_index in 0..program.vreg_count {
        let vreg = crate::ir::VReg::new(vreg_index);
        let Some(debug_value_id) = program.vreg_debug_value(vreg) else {
            continue;
        };
        let debug_value = &program.debug.values[debug_value_id];
        if !matches!(debug_value.kind, crate::ir::DebugValueKind::Named) {
            continue;
        }
        vregs_by_value.entry(debug_value_id).or_default().push(vreg);
    }

    let vreg_infos = cfg_vreg_dwarf_variable_infos(
        program,
        alloc,
        backend_debug_info,
        code_ptr,
        target_arch,
        apply_regalloc_edits,
    );

    vregs_by_value
        .into_iter()
        .filter_map(|(debug_value_id, vregs)| {
            let debug_value = &program.debug.values[debug_value_id];
            let mut scopes = Vec::new();
            let mut lexical_ranges = Vec::new();
            let mut locations = Vec::new();
            for vreg in vregs {
                let Some(info) = vreg_infos.get(&vreg) else {
                    continue;
                };
                if let Some(scope) = info.scope {
                    scopes.push(scope);
                }
                lexical_ranges.extend(info.lexical_intro_ranges.clone());
                lexical_ranges.extend(info.locations.iter().map(|location| {
                    crate::jit_dwarf::JitDebugRange {
                        low_pc: location.start,
                        high_pc: location.end,
                    }
                }));
                locations.extend(info.locations.clone());
            }
            let locations = merge_dwarf_location_ranges(locations);
            if locations.is_empty() {
                return None;
            }
            let lexical_ranges = merge_jit_debug_ranges(lexical_ranges);
            let scope = common_debug_scope(program, scopes).or(program.debug.root_scope);
            Some(ScopedDwarfVariable {
                scope,
                lexical_ranges,
                variable: crate::jit_dwarf::DwarfVariable {
                    name: debug_value.name.clone(),
                    location: crate::jit_dwarf::DwarfVariableLocation::List(locations),
                },
            })
        })
        .collect()
}

fn cfg_mir_dwarf_variables(
    root_shape: Option<&'static Shape>,
    program: &crate::regalloc_engine::cfg_mir::Program,
    alloc: &crate::regalloc_engine::AllocatedCfgProgram,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    code_ptr: *const u8,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
    apply_regalloc_edits: bool,
) -> crate::jit_dwarf::JitDebugSubprogram {
    let mut variables = deser_dwarf_variables(target_arch);
    let suppress_semantic_vregs = root_shape.is_some()
        || program
            .debug
            .values
            .iter()
            .any(|(_, value)| matches!(value.kind, crate::ir::DebugValueKind::Named));
    let mut cfg_variables = cfg_value_dwarf_variables(
        program,
        alloc,
        backend_debug_info,
        code_ptr,
        target_arch,
        apply_regalloc_edits,
        suppress_semantic_vregs,
    );
    cfg_variables.extend(cfg_semantic_named_dwarf_variables(
        program,
        alloc,
        backend_debug_info,
        code_ptr,
        target_arch,
        apply_regalloc_edits,
    ));
    if let Some(root_shape) = root_shape {
        cfg_variables.extend(cfg_semantic_field_dwarf_variables(
            root_shape,
            program,
            backend_debug_info,
            code_ptr,
            target_arch,
        ));
    }
    let (unscoped_cfg_variables, lexical_blocks) =
        cfg_mir_lexical_blocks(program, backend_debug_info, code_ptr, cfg_variables);
    variables.extend(unscoped_cfg_variables);
    crate::jit_dwarf::JitDebugSubprogram {
        name: String::new(),
        frame_base_expression: crate::jit_dwarf::expr_breg(
            crate::jit_dwarf::frame_base_register(target_arch),
            0,
        ),
        variables,
        lexical_blocks,
    }
}

fn cfg_mir_lexical_blocks(
    program: &crate::regalloc_engine::cfg_mir::Program,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    code_ptr: *const u8,
    cfg_variables: Vec<ScopedDwarfVariable>,
) -> (
    Vec<crate::jit_dwarf::DwarfVariable>,
    Vec<crate::jit_dwarf::JitDebugLexicalBlock>,
) {
    let Some(backend_debug_info) = backend_debug_info else {
        return (
            cfg_variables
                .into_iter()
                .map(|variable| variable.variable)
                .collect(),
            Vec::new(),
        );
    };
    let scope_ranges = scope_ranges_from_backend(program, backend_debug_info, code_ptr);
    let root_scope = program.debug.root_scope;
    let mut raw_vars_by_scope =
        BTreeMap::<crate::ir::DebugScopeId, Vec<ScopedDwarfVariable>>::new();
    let mut direct_vars_by_scope =
        BTreeMap::<crate::ir::DebugScopeId, Vec<crate::jit_dwarf::DwarfVariable>>::new();
    let mut interval_blocks_by_scope =
        BTreeMap::<crate::ir::DebugScopeId, Vec<crate::jit_dwarf::JitDebugLexicalBlock>>::new();
    let mut unscoped_variables = Vec::new();
    for variable in cfg_variables {
        match variable.scope {
            Some(scope) => raw_vars_by_scope.entry(scope).or_default().push(variable),
            _ => unscoped_variables.push(variable.variable),
        }
    }
    for (scope, variables) in raw_vars_by_scope {
        let (mut direct_variables, interval_blocks) = build_variable_interval_blocks(variables);
        if Some(scope) == root_scope {
            unscoped_variables.append(&mut direct_variables);
        } else {
            direct_vars_by_scope.insert(scope, direct_variables);
        }
        interval_blocks_by_scope.insert(scope, interval_blocks);
    }

    fn build_scope_blocks(
        scope_id: crate::ir::DebugScopeId,
        program: &crate::regalloc_engine::cfg_mir::Program,
        scope_ranges: &BTreeMap<crate::ir::DebugScopeId, Vec<crate::jit_dwarf::JitDebugRange>>,
        direct_vars_by_scope: &mut BTreeMap<
            crate::ir::DebugScopeId,
            Vec<crate::jit_dwarf::DwarfVariable>,
        >,
        interval_blocks_by_scope: &mut BTreeMap<
            crate::ir::DebugScopeId,
            Vec<crate::jit_dwarf::JitDebugLexicalBlock>,
        >,
    ) -> Vec<crate::jit_dwarf::JitDebugLexicalBlock> {
        let mut out = interval_blocks_by_scope
            .remove(&scope_id)
            .unwrap_or_default();
        for (child_scope_id, child_scope) in program.debug.scopes.iter() {
            if child_scope.parent != Some(scope_id) {
                continue;
            }
            let mut variables = direct_vars_by_scope
                .remove(&child_scope_id)
                .unwrap_or_default();
            variables.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
            let lexical_blocks = build_scope_blocks(
                child_scope_id,
                program,
                scope_ranges,
                direct_vars_by_scope,
                interval_blocks_by_scope,
            );
            let ranges = scope_ranges
                .get(&child_scope_id)
                .cloned()
                .unwrap_or_default();
            if ranges.is_empty() && variables.is_empty() && lexical_blocks.is_empty() {
                continue;
            }
            out.push(crate::jit_dwarf::JitDebugLexicalBlock {
                ranges,
                variables,
                lexical_blocks,
            });
        }
        out
    }

    let lexical_blocks = root_scope
        .map(|root_scope| {
            build_scope_blocks(
                root_scope,
                program,
                &scope_ranges,
                &mut direct_vars_by_scope,
                &mut interval_blocks_by_scope,
            )
        })
        .unwrap_or_default();

    for (_, mut variables) in direct_vars_by_scope {
        variables.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
        unscoped_variables.extend(variables);
    }

    (unscoped_variables, lexical_blocks)
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

pub(crate) fn symbol_registry_for_shape(shape: &'static Shape) -> crate::ir::IntrinsicRegistry {
    let mut registry = crate::ir::IntrinsicRegistry::empty();
    for (name, func) in crate::intrinsics::known_intrinsics() {
        registry.register(name, func);
    }
    for (name, func) in crate::json_intrinsics::known_intrinsics() {
        registry.register(name, func);
    }

    let mut seen = HashSet::new();
    collect_shape_symbols(shape, &mut seen, &mut registry);
    registry
}

fn collect_shape_symbols(
    shape: &'static Shape,
    seen: &mut HashSet<usize>,
    registry: &mut crate::ir::IntrinsicRegistry,
) {
    let shape_key = shape as *const Shape as usize;
    if !seen.insert(shape_key) {
        return;
    }

    if let Type::User(UserType::Struct(st)) = &shape.ty {
        for field in st.fields {
            if field.is_flattened() {
                collect_shape_symbols(field.shape(), seen, registry);
                continue;
            }

            let name = field.effective_name();
            registry.register_const(
                format!("json_key_ptr.{}", encode_symbol_bytes(name)),
                name.as_ptr() as u64,
            );
            collect_shape_symbols(field.shape(), seen, registry);
        }
    } else if let Type::User(UserType::Enum(enum_type)) = &shape.ty {
        for variant in enum_type.variants {
            for field in variant.data.fields {
                collect_shape_symbols(field.shape(), seen, registry);
            }
        }
    }

    match &shape.def {
        Def::Map(map_def) => {
            collect_shape_symbols(map_def.k, seen, registry);
            collect_shape_symbols(map_def.v, seen, registry);
        }
        Def::Set(set_def) => collect_shape_symbols(set_def.t, seen, registry),
        Def::List(list_def) => collect_shape_symbols(list_def.t, seen, registry),
        Def::Array(array_def) => collect_shape_symbols(array_def.t, seen, registry),
        Def::NdArray(ndarray_def) => collect_shape_symbols(ndarray_def.t, seen, registry),
        Def::Slice(slice_def) => collect_shape_symbols(slice_def.t, seen, registry),
        Def::Option(opt_def) => {
            let type_id = instantiated_shape_symbol_key(opt_def.t);
            registry.register_const(
                format!("option_init_none.{type_id}"),
                opt_def.vtable.init_none as *const () as usize as u64,
            );
            registry.register_const(
                format!("option_init_some.{type_id}"),
                opt_def.vtable.init_some as *const () as usize as u64,
            );
            collect_shape_symbols(opt_def.t, seen, registry);
        }
        Def::Result(result_def) => {
            collect_shape_symbols(result_def.t, seen, registry);
            collect_shape_symbols(result_def.e, seen, registry);
        }
        Def::Pointer(pointer_def) => {
            if let Some(pointee) = pointer_def.pointee {
                collect_shape_symbols(pointee, seen, registry);
            }
        }
        Def::Undefined | Def::Scalar | Def::DynamicValue(_) => {}
        _ => {}
    }
}

fn encode_symbol_bytes(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    for byte in text.as_bytes() {
        use core::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String should not fail");
    }
    out
}

fn instantiated_shape_symbol_key(shape: &'static Shape) -> String {
    let mut out = String::new();
    append_instantiated_shape_symbol_key(shape, &mut out);
    out
}

fn append_instantiated_shape_symbol_key(shape: &'static Shape, out: &mut String) {
    use core::fmt::Write as _;

    write!(out, "d{:032x}", shape.decl_id.0).expect("writing to String should not fail");

    if !shape.type_params.is_empty() {
        out.push_str("__t");
        for (index, param) in shape.type_params.iter().enumerate() {
            write!(out, "_{index}_").expect("writing to String should not fail");
            append_instantiated_shape_symbol_key(param.shape(), out);
        }
    }

    if !shape.const_params.is_empty() {
        out.push_str("__c");
        for (index, param) in shape.const_params.iter().enumerate() {
            write!(out, "_{index}_").expect("writing to String should not fail");
            out.push(const_param_kind_symbol(param.kind));
            write!(out, "{:x}", param.value).expect("writing to String should not fail");
        }
    }
}

fn const_param_kind_symbol(kind: ConstParamKind) -> char {
    match kind {
        ConstParamKind::Bool => 'b',
        ConstParamKind::Char => 'c',
        ConstParamKind::U8 => 'h',
        ConstParamKind::U16 => 't',
        ConstParamKind::U32 => 'j',
        ConstParamKind::U64 => 'm',
        ConstParamKind::Usize => 'u',
        ConstParamKind::I8 => 'a',
        ConstParamKind::I16 => 's',
        ConstParamKind::I32 => 'i',
        ConstParamKind::I64 => 'l',
        ConstParamKind::Isize => 'n',
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

    fn lower_field_value(
        &self,
        rb: &mut RegionBuilder<'_>,
        field: &FieldLowerInfo,
        f: impl FnOnce(&mut RegionBuilder<'_>),
    ) {
        let debug_value = rb.define_debug_field(field.name, field.offset as u32);
        rb.with_debug_value(Some(debug_value), f);
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
            self.lower_field_value(rb, &fields[0], |field_rb| {
                self.lower_value(field_rb, fields[0].shape, fields[0].offset);
            });
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
                            self.lower_field_value(inner_rb, field, |field_rb| {
                                self.lower_value(field_rb, field.shape, field.offset);
                            });
                        });
                } else {
                    self.decoder.lower_struct_fields(
                        rb,
                        &fields,
                        deny_unknown_fields,
                        &mut |inner_rb, field| {
                            self.lower_field_value(inner_rb, field, |field_rb| {
                                self.lower_value(field_rb, field.shape, field.offset);
                            });
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
                            self.lower_field_value(inner_rb, field, |field_rb| {
                                self.lower_value(field_rb, field.shape, field.offset);
                            });
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

/// Build IR + linear form, compile through the backend, and return a deterministic emission trace.
pub fn emission_trace_text(shape: &'static Shape, ir_decoder: &dyn Decoder) -> String {
    let pipeline_opts = PipelineOptions::from_env();
    emission_trace_text_with_options(shape, ir_decoder, &pipeline_opts)
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
    let cfg_program = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg_program)
        .unwrap_or_else(|err| panic!("regalloc2 allocation failed while counting edits: {err}"));
    alloc.functions.iter().map(|f| f.edits.len()).sum()
}

// r[impl compiler.opts.api]
pub fn emission_trace_text_with_options(
    shape: &'static Shape,
    ir_decoder: &dyn Decoder,
    pipeline_opts: &PipelineOptions,
) -> String {
    let decoder = compile_decoder_with_options(shape, ir_decoder, pipeline_opts);
    decoder
        .emission_trace_text()
        .unwrap_or_else(|err| panic!("failed to format emission trace: {err:?}"))
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
    let cfg_program = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
    let alloc = if pipeline_opts.resolve_regalloc(true) {
        let mut alloc =
            crate::regalloc_engine::allocate_cfg_program(&cfg_program).unwrap_or_else(|err| {
                panic!("regalloc2 allocation failed while formatting edits: {err}")
            });
        maybe_disable_regalloc_edits_cfg(&mut alloc, pipeline_opts);
        alloc
    } else {
        no_regalloc_alloc_for_cfg_program(&cfg_program)
    };
    format_allocated_regalloc_edits(&alloc)
}

fn format_allocated_regalloc_edits(alloc: &crate::regalloc_engine::AllocatedCfgProgram) -> String {
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
                    "    - edge e{} pos={:?} move {:?} -> {:?}\n",
                    edge.edge.0, edge.pos, edge.from, edge.to
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

struct PostcardHirLowerer {
    module: hir::Module,
    input_region: hir::RegionId,
    cursor_type: hir::TypeDefId,
    string_raw_type: Option<hir::TypeDefId>,
    bits128_raw_type: Option<hir::TypeDefId>,
    type_defs_by_shape: HashMap<*const Shape, hir::TypeDefId>,
    callables_by_name: HashMap<&'static str, hir::CallableId>,
    locals: Vec<hir::LocalDecl>,
    next_local: u32,
    next_stmt: u32,
}

impl PostcardHirLowerer {
    fn new() -> Self {
        let mut module = hir::Module::new();
        let input_region = module.add_region("input");
        let cursor_type = module.add_type_def(hir::TypeDef {
            name: "Cursor".to_owned(),
            generic_params: vec![hir::GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "bytes".to_owned(),
                        ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    },
                    hir::FieldDef {
                        name: "pos".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });

        Self {
            module,
            input_region,
            cursor_type,
            string_raw_type: None,
            bits128_raw_type: None,
            type_defs_by_shape: HashMap::new(),
            callables_by_name: HashMap::new(),
            locals: Vec::new(),
            next_local: 2,
            next_stmt: 0,
        }
    }

    fn finish(self) -> hir::Module {
        self.module
    }

    fn next_local(&mut self) -> hir::LocalId {
        let id = hir::LocalId::new(self.next_local);
        self.next_local += 1;
        id
    }

    fn next_stmt_id(&mut self) -> hir::StmtId {
        let id = hir::StmtId::new(self.next_stmt);
        self.next_stmt += 1;
        id
    }

    fn alloc_local(
        &mut self,
        name: impl Into<String>,
        ty: hir::Type,
        kind: hir::LocalKind,
    ) -> hir::LocalId {
        let local = self.next_local();
        self.locals.push(hir::LocalDecl {
            local,
            name: name.into(),
            ty,
            kind,
        });
        local
    }

    fn push_init(&mut self, statements: &mut Vec<hir::Stmt>, place: hir::Place, value: hir::Expr) {
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Init { place, value },
        });
    }

    fn shape_has_input_borrow(shape: &'static Shape) -> bool {
        match shape.scalar_type() {
            Some(ScalarType::Str | ScalarType::CowStr) => return true,
            Some(_) => return false,
            None => {}
        }

        if shape.is_transparent() {
            let (fields, _) = collect_fields(shape);
            return fields
                .iter()
                .any(|field| Self::shape_has_input_borrow(field.shape));
        }

        if let Some(opt_def) = get_option_def(shape) {
            return Self::shape_has_input_borrow(opt_def.t);
        }

        if let Some(ptr_def) = get_pointer_def(shape) {
            return ptr_def.pointee.is_some_and(Self::shape_has_input_borrow);
        }

        match &shape.def {
            Def::List(list_def) => return Self::shape_has_input_borrow(list_def.t),
            Def::Map(map_def) => {
                return Self::shape_has_input_borrow(map_def.k)
                    || Self::shape_has_input_borrow(map_def.v);
            }
            Def::Array(array_def) => return Self::shape_has_input_borrow(array_def.t),
            _ => {}
        }

        match &shape.ty {
            Type::User(UserType::Struct(_)) => {
                let (fields, _) = collect_fields(shape);
                fields
                    .iter()
                    .any(|field| Self::shape_has_input_borrow(field.shape))
            }
            Type::User(UserType::Enum(enum_type)) => collect_variants(enum_type)
                .iter()
                .flat_map(|variant| variant.fields.iter())
                .any(|field| Self::shape_has_input_borrow(field.shape)),
            _ => false,
        }
    }

    fn ensure_type_def(&mut self, shape: &'static Shape) -> hir::TypeDefId {
        let key = shape as *const Shape;
        if let Some(existing) = self.type_defs_by_shape.get(&key).copied() {
            return existing;
        }

        let generic_params = if Self::shape_has_input_borrow(shape) {
            vec![hir::GenericParam::Region {
                name: "r_input".to_owned(),
            }]
        } else {
            Vec::new()
        };

        let type_def = hir::TypeDef {
            name: shape.type_identifier.to_owned(),
            generic_params,
            kind: hir::TypeDefKind::Struct { fields: Vec::new() },
        };
        let type_id = self.module.add_type_def(type_def);
        self.type_defs_by_shape.insert(key, type_id);

        let kind = if let Def::List(list_def) = &shape.def {
            let offsets = crate::malum::discover_vec_offsets(list_def, shape);
            let mut fields = vec![
                (
                    offsets.ptr_offset,
                    hir::FieldDef {
                        name: "ptr".to_owned(),
                        ty: hir::Type::persistent_addr(),
                    },
                ),
                (
                    offsets.len_offset,
                    hir::FieldDef {
                        name: "len".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ),
                (
                    offsets.cap_offset,
                    hir::FieldDef {
                        name: "cap".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ),
            ];
            fields.sort_by_key(|(offset, _)| *offset);
            hir::TypeDefKind::Struct {
                fields: fields.into_iter().map(|(_, field)| field).collect(),
            }
        } else if shape.is_transparent() {
            let (fields, skipped) = collect_fields(shape);
            assert!(
                skipped.is_empty(),
                "postcard HIR prototype does not support transparent defaults"
            );
            hir::TypeDefKind::Struct {
                fields: fields
                    .into_iter()
                    .map(|field| hir::FieldDef {
                        name: field.name.to_owned(),
                        ty: self.lower_type(field.shape),
                    })
                    .collect(),
            }
        } else if let Some(opt_def) = get_option_def(shape) {
            hir::TypeDefKind::Enum {
                variants: vec![
                    hir::VariantDef {
                        name: "None".to_owned(),
                        fields: Vec::new(),
                    },
                    hir::VariantDef {
                        name: "Some".to_owned(),
                        fields: vec![hir::FieldDef {
                            name: "value".to_owned(),
                            ty: self.lower_type(opt_def.t),
                        }],
                    },
                ],
            }
        } else {
            match &shape.ty {
                Type::User(UserType::Struct(_)) => {
                    let (fields, skipped) = collect_fields(shape);
                    assert!(
                        skipped.is_empty(),
                        "postcard HIR prototype does not support skipped/defaulted fields"
                    );
                    hir::TypeDefKind::Struct {
                        fields: fields
                            .into_iter()
                            .map(|field| hir::FieldDef {
                                name: field.name.to_owned(),
                                ty: self.lower_type(field.shape),
                            })
                            .collect(),
                    }
                }
                Type::User(UserType::Enum(enum_type)) => hir::TypeDefKind::Enum {
                    variants: collect_variants(enum_type)
                        .into_iter()
                        .map(|variant| hir::VariantDef {
                            name: variant.name.to_owned(),
                            fields: variant
                                .fields
                                .into_iter()
                                .map(|field| hir::FieldDef {
                                    name: field.name.to_owned(),
                                    ty: self.lower_type(field.shape),
                                })
                                .collect(),
                        })
                        .collect(),
                },
                _ => panic!(
                    "postcard HIR prototype only supports struct-like composite roots for now: {}",
                    shape.type_identifier
                ),
            }
        };

        self.module.type_defs[type_id].kind = kind;
        type_id
    }

    fn ensure_string_raw_type(&mut self) -> hir::TypeDefId {
        if let Some(existing) = self.string_raw_type {
            return existing;
        }
        let offsets = crate::malum::discover_string_offsets();
        let mut fields = vec![
            (
                offsets.ptr_offset,
                hir::FieldDef {
                    name: "ptr".to_owned(),
                    ty: hir::Type::persistent_addr(),
                },
            ),
            (
                offsets.len_offset,
                hir::FieldDef {
                    name: "len".to_owned(),
                    ty: hir::Type::u(64),
                },
            ),
            (
                offsets.cap_offset,
                hir::FieldDef {
                    name: "cap".to_owned(),
                    ty: hir::Type::u(64),
                },
            ),
        ];
        fields.sort_by_key(|(offset, _)| *offset);
        let type_id = self.module.add_type_def(hir::TypeDef {
            name: "HostStringRaw".to_owned(),
            generic_params: Vec::new(),
            kind: hir::TypeDefKind::Struct {
                fields: fields.into_iter().map(|(_, field)| field).collect(),
            },
        });
        self.string_raw_type = Some(type_id);
        type_id
    }

    fn ensure_bits128_raw_type(&mut self) -> hir::TypeDefId {
        if let Some(existing) = self.bits128_raw_type {
            return existing;
        }
        let type_id = self.module.add_type_def(hir::TypeDef {
            name: "Bits128Raw".to_owned(),
            generic_params: Vec::new(),
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "lo".to_owned(),
                        ty: hir::Type::u(64),
                    },
                    hir::FieldDef {
                        name: "hi".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });
        self.bits128_raw_type = Some(type_id);
        type_id
    }

    fn lower_type(&mut self, shape: &'static Shape) -> hir::Type {
        if is_unit(shape) {
            return hir::Type::unit();
        }

        if matches!(shape.def, Def::List(_)) {
            let type_id = self.ensure_type_def(shape);
            let args = if Self::shape_has_input_borrow(shape) {
                vec![hir::GenericArg::Region(self.input_region)]
            } else {
                Vec::new()
            };
            return hir::Type::named(type_id, args);
        }

        if let Def::Array(array_def) = &shape.def {
            return hir::Type::array(self.lower_type(array_def.t), array_def.n);
        }

        if shape.is_transparent() {
            let (fields, skipped) = collect_fields(shape);
            assert!(
                skipped.is_empty() && fields.len() == 1,
                "transparent HIR prototype expects one lowered field"
            );
            return self.lower_type(fields[0].shape);
        }

        if let Some(opt_def) = get_option_def(shape) {
            let type_id = self.ensure_type_def(shape);
            let args = if Self::shape_has_input_borrow(opt_def.t) {
                vec![hir::GenericArg::Region(self.input_region)]
            } else {
                Vec::new()
            };
            return hir::Type::named(type_id, args);
        }

        if let Some(st) = shape.scalar_type() {
            return match st {
                ScalarType::Unit => hir::Type::unit(),
                ScalarType::Bool => hir::Type::bool(),
                ScalarType::U8 => hir::Type::u(8),
                ScalarType::U16 => hir::Type::u(16),
                ScalarType::U32 => hir::Type::u(32),
                ScalarType::U64 => hir::Type::u(64),
                ScalarType::U128 => hir::Type::named(self.ensure_bits128_raw_type(), Vec::new()),
                ScalarType::USize => hir::Type::u(64),
                ScalarType::I8 => hir::Type::i(8),
                ScalarType::I16 => hir::Type::i(16),
                ScalarType::I32 => hir::Type::i(32),
                ScalarType::I64 => hir::Type::i(64),
                ScalarType::I128 => hir::Type::named(self.ensure_bits128_raw_type(), Vec::new()),
                ScalarType::ISize => hir::Type::i(64),
                ScalarType::Str => hir::Type::str(self.input_region),
                ScalarType::String => hir::Type::named(self.ensure_string_raw_type(), Vec::new()),
                ScalarType::Char => hir::Type::u(32),
                ScalarType::F32 => hir::Type::u(32),
                ScalarType::F64 => hir::Type::u(64),
                ScalarType::CowStr => {
                    panic!(
                        "postcard HIR prototype does not support scalar {st:?} yet for {}",
                        shape.type_identifier
                    );
                }
                _ => panic!(
                    "postcard HIR prototype encountered unknown scalar {st:?} for {}",
                    shape.type_identifier
                ),
            };
        }

        let type_id = self.ensure_type_def(shape);
        let args = if Self::shape_has_input_borrow(shape) {
            vec![hir::GenericArg::Region(self.input_region)]
        } else {
            Vec::new()
        };
        hir::Type::named(type_id, args)
    }

    fn ensure_postcard_reader(&mut self, scalar_type: ScalarType) -> hir::CallableId {
        let name = match scalar_type {
            ScalarType::Bool => "postcard.read_bool",
            ScalarType::U8 => "postcard.read_u8",
            ScalarType::U16 => "postcard.read_u16",
            ScalarType::U32 => "postcard.read_u32",
            ScalarType::U64 => "postcard.read_u64",
            ScalarType::U128 => "postcard.read_u128",
            ScalarType::USize => "postcard.read_usize",
            ScalarType::I8 => "postcard.read_i8",
            ScalarType::I16 => "postcard.read_i16",
            ScalarType::I32 => "postcard.read_i32",
            ScalarType::I64 => "postcard.read_i64",
            ScalarType::I128 => "postcard.read_i128",
            ScalarType::ISize => "postcard.read_isize",
            ScalarType::Str => "postcard.read_str",
            other => panic!("unsupported postcard HIR reader for {other:?}"),
        };
        if let Some(existing) = self.callables_by_name.get(name).copied() {
            return existing;
        }

        let returns = vec![match scalar_type {
            ScalarType::Str => hir::Type::str(self.input_region),
            _ => self.lower_type_for_scalar(scalar_type),
        }];
        let callable = hir::CallableSpec {
            kind: hir::CallableKind::Builtin,
            name: name.to_owned(),
            signature: hir::CallSignature {
                params: vec![hir::Type::named(
                    self.cursor_type,
                    vec![hir::GenericArg::Region(self.input_region)],
                )],
                returns,
                effect_class: hir::EffectClass::Mutates,
                domain_effects: vec![hir::DomainEffect {
                    domain: "cursor".to_owned(),
                    access: hir::DomainAccess::Mutate,
                }],
                control: hir::ControlTransfer::MayFail,
                capabilities: vec!["deser.postcard".to_owned()],
                safety: hir::CallSafety::SafeCore,
            },
            docs: Some(format!(
                "Read a postcard {:?} value from the input cursor.",
                scalar_type
            )),
        };
        let callable_id = self.module.add_callable(callable);
        self.callables_by_name.insert(name, callable_id);
        callable_id
    }

    fn lower_type_for_scalar(&mut self, scalar_type: ScalarType) -> hir::Type {
        match scalar_type {
            ScalarType::Bool => hir::Type::bool(),
            ScalarType::U8 => hir::Type::u(8),
            ScalarType::U16 => hir::Type::u(16),
            ScalarType::U32 => hir::Type::u(32),
            ScalarType::U64 => hir::Type::u(64),
            ScalarType::U128 => hir::Type::named(self.ensure_bits128_raw_type(), Vec::new()),
            ScalarType::USize => hir::Type::u(64),
            ScalarType::I8 => hir::Type::i(8),
            ScalarType::I16 => hir::Type::i(16),
            ScalarType::I32 => hir::Type::i(32),
            ScalarType::I64 => hir::Type::i(64),
            ScalarType::I128 => hir::Type::named(self.ensure_bits128_raw_type(), Vec::new()),
            ScalarType::ISize => hir::Type::i(64),
            ScalarType::Char => hir::Type::u(32),
            ScalarType::F32 => hir::Type::u(32),
            ScalarType::F64 => hir::Type::u(64),
            other => panic!("unsupported postcard HIR scalar type {other:?}"),
        }
    }

    fn cursor_bytes_expr(&self, cursor_local: hir::LocalId) -> hir::Expr {
        hir::Expr::Field {
            base: Box::new(hir::Expr::Local(cursor_local)),
            field: "bytes".to_owned(),
        }
    }

    fn cursor_pos_expr(&self, cursor_local: hir::LocalId) -> hir::Expr {
        hir::Expr::Field {
            base: Box::new(hir::Expr::Local(cursor_local)),
            field: "pos".to_owned(),
        }
    }

    fn push_cursor_bounds_check(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        needed: u64,
        code: hir::ErrorCode,
    ) {
        self.push_cursor_bounds_check_expr(
            statements,
            cursor_local,
            hir::Expr::Literal(hir::Literal::Integer(needed)),
            code,
        );
    }

    fn push_cursor_bounds_check_expr(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        needed: hir::Expr,
        code: hir::ErrorCode,
    ) {
        let bytes = self.cursor_bytes_expr(cursor_local);
        let end = hir::Expr::SliceLen {
            value: Box::new(bytes),
        };
        let pos = self.cursor_pos_expr(cursor_local);
        let limit = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(pos),
            rhs: Box::new(needed),
        };
        let fail_condition = hir::Expr::Binary {
            op: hir::BinaryOp::Gt,
            lhs: Box::new(limit),
            rhs: Box::new(end),
        };
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: fail_condition,
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Fail { code },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: Vec::new(),
                }),
            },
        });
    }

    fn lower_postcard_fixed_width_scalar_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        scalar_type: ScalarType,
    ) {
        let width = match scalar_type {
            ScalarType::Bool | ScalarType::U8 | ScalarType::I8 => hir::MemoryWidth::W1,
            ScalarType::F32 => hir::MemoryWidth::W4,
            ScalarType::F64 => hir::MemoryWidth::W8,
            other => panic!("unsupported fixed-width postcard HIR scalar {other:?}"),
        };
        self.push_cursor_bounds_check(
            statements,
            cursor_local,
            u64::from(width.bytes()),
            hir::ErrorCode::UnexpectedEof,
        );

        let bytes = self.cursor_bytes_expr(cursor_local);
        let pos = self.cursor_pos_expr(cursor_local);
        let addr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(bytes),
            }),
            rhs: Box::new(pos.clone()),
        };

        let raw_local = self.alloc_local(
            format!("fixed_scalar_{}", self.locals.len()),
            match width {
                hir::MemoryWidth::W1 => hir::Type::u(8),
                hir::MemoryWidth::W2 => hir::Type::u(16),
                hir::MemoryWidth::W4 => hir::Type::u(32),
                hir::MemoryWidth::W8 => hir::Type::u(64),
            },
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(raw_local),
            hir::Expr::Load {
                addr: Box::new(addr),
                width,
            },
        );
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(pos),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(u64::from(
                        width.bytes(),
                    )))),
                },
            },
        });

        match scalar_type {
            ScalarType::Bool => {
                let invalid = hir::Expr::Binary {
                    op: hir::BinaryOp::Gt,
                    lhs: Box::new(hir::Expr::Local(raw_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                };
                statements.push(hir::Stmt {
                    id: self.next_stmt_id(),
                    kind: hir::StmtKind::If {
                        condition: invalid,
                        then_block: hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: vec![hir::Stmt {
                                id: self.next_stmt_id(),
                                kind: hir::StmtKind::Fail {
                                    code: hir::ErrorCode::InvalidBool,
                                },
                            }],
                        },
                        else_block: Some(hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: vec![hir::Stmt {
                                id: self.next_stmt_id(),
                                kind: hir::StmtKind::Init {
                                    place,
                                    value: hir::Expr::Binary {
                                        op: hir::BinaryOp::Ne,
                                        lhs: Box::new(hir::Expr::Local(raw_local)),
                                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                                    },
                                },
                            }],
                        }),
                    },
                });
            }
            ScalarType::U8 | ScalarType::I8 | ScalarType::F32 | ScalarType::F64 => {
                self.push_init(statements, place, hir::Expr::Local(raw_local));
            }
            _ => unreachable!(),
        }
    }

    fn bits128_field_place(&self, base: hir::Place, field: &str) -> hir::Place {
        hir::Place::Field {
            base: Box::new(base),
            field: field.to_owned(),
        }
    }

    fn postcard_varint128_finish_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        place: hir::Place,
        zigzag: bool,
        acc_lo_local: hir::LocalId,
        acc_hi_local: hir::LocalId,
    ) {
        let (lo_value, hi_value) = if zigzag {
            let shifted_lo = hir::Expr::Binary {
                op: hir::BinaryOp::BitOr,
                lhs: Box::new(hir::Expr::Binary {
                    op: hir::BinaryOp::Shr,
                    lhs: Box::new(hir::Expr::Local(acc_lo_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                }),
                rhs: Box::new(hir::Expr::Binary {
                    op: hir::BinaryOp::Shl,
                    lhs: Box::new(hir::Expr::Local(acc_hi_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(63))),
                }),
            };
            let shifted_hi = hir::Expr::Binary {
                op: hir::BinaryOp::Shr,
                lhs: Box::new(hir::Expr::Local(acc_hi_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
            };
            let sign = hir::Expr::Binary {
                op: hir::BinaryOp::BitAnd,
                lhs: Box::new(hir::Expr::Local(acc_lo_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
            };
            let neg_mask = hir::Expr::Binary {
                op: hir::BinaryOp::Sub,
                lhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                rhs: Box::new(sign),
            };
            (
                hir::Expr::Binary {
                    op: hir::BinaryOp::Xor,
                    lhs: Box::new(shifted_lo),
                    rhs: Box::new(neg_mask.clone()),
                },
                hir::Expr::Binary {
                    op: hir::BinaryOp::Xor,
                    lhs: Box::new(shifted_hi),
                    rhs: Box::new(neg_mask),
                },
            )
        } else {
            (
                hir::Expr::Local(acc_lo_local),
                hir::Expr::Local(acc_hi_local),
            )
        };

        self.push_init(
            statements,
            self.bits128_field_place(place.clone(), "lo"),
            lo_value,
        );
        self.push_init(statements, self.bits128_field_place(place, "hi"), hi_value);
    }

    fn postcard_varint128_finish_block(
        &mut self,
        place: hir::Place,
        zigzag: bool,
        acc_lo_local: hir::LocalId,
        acc_hi_local: hir::LocalId,
        byte_index: u64,
        raw_local: hir::LocalId,
    ) -> hir::Block {
        let mut block = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: Vec::new(),
        };

        if byte_index + 1 == Self::postcard_varint_max_bytes(128) {
            let extra_bits = hir::Expr::Binary {
                op: hir::BinaryOp::BitAnd,
                lhs: Box::new(hir::Expr::Local(raw_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7e))),
            };
            let mut ok_block = hir::Block {
                scope: hir::ScopeId::new(0),
                statements: Vec::new(),
            };
            self.postcard_varint128_finish_into_place(
                &mut ok_block.statements,
                place,
                zigzag,
                acc_lo_local,
                acc_hi_local,
            );
            block.statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(extra_bits),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::InvalidVarint,
                            },
                        }],
                    },
                    else_block: Some(ok_block),
                },
            });
            return block;
        }

        self.postcard_varint128_finish_into_place(
            &mut block.statements,
            place,
            zigzag,
            acc_lo_local,
            acc_hi_local,
        );
        block
    }

    fn lower_postcard_varint128_step(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        zigzag: bool,
        acc_lo_local: hir::LocalId,
        acc_hi_local: hir::LocalId,
        byte_index: u64,
    ) {
        self.push_cursor_bounds_check(statements, cursor_local, 1, hir::ErrorCode::UnexpectedEof);

        let pos = self.cursor_pos_expr(cursor_local);
        let addr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(self.cursor_bytes_expr(cursor_local)),
            }),
            rhs: Box::new(pos.clone()),
        };
        let raw_local = self.alloc_local(
            format!("varint128_byte_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(raw_local),
            hir::Expr::Load {
                addr: Box::new(addr),
                width: hir::MemoryWidth::W1,
            },
        );
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(pos),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                },
            },
        });

        let low = hir::Expr::Binary {
            op: hir::BinaryOp::BitAnd,
            lhs: Box::new(hir::Expr::Local(raw_local)),
            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7f))),
        };
        let shift = byte_index * 7;
        if shift < 64 {
            let lo_part = if shift == 0 {
                low.clone()
            } else {
                hir::Expr::Binary {
                    op: hir::BinaryOp::Shl,
                    lhs: Box::new(low.clone()),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(shift))),
                }
            };
            statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::Assign {
                    place: hir::Place::Local(acc_lo_local),
                    value: hir::Expr::Binary {
                        op: hir::BinaryOp::BitOr,
                        lhs: Box::new(hir::Expr::Local(acc_lo_local)),
                        rhs: Box::new(lo_part),
                    },
                },
            });

            if shift > 57 {
                let hi_part = hir::Expr::Binary {
                    op: hir::BinaryOp::Shr,
                    lhs: Box::new(low),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(64 - shift))),
                };
                statements.push(hir::Stmt {
                    id: self.next_stmt_id(),
                    kind: hir::StmtKind::Assign {
                        place: hir::Place::Local(acc_hi_local),
                        value: hir::Expr::Binary {
                            op: hir::BinaryOp::BitOr,
                            lhs: Box::new(hir::Expr::Local(acc_hi_local)),
                            rhs: Box::new(hi_part),
                        },
                    },
                });
            }
        } else {
            let hi_part = hir::Expr::Binary {
                op: hir::BinaryOp::Shl,
                lhs: Box::new(low),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(shift - 64))),
            };
            statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::Assign {
                    place: hir::Place::Local(acc_hi_local),
                    value: hir::Expr::Binary {
                        op: hir::BinaryOp::BitOr,
                        lhs: Box::new(hir::Expr::Local(acc_hi_local)),
                        rhs: Box::new(hi_part),
                    },
                },
            });
        }

        let max_bytes = Self::postcard_varint_max_bytes(128);
        let then_block = if byte_index + 1 == max_bytes {
            hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![hir::Stmt {
                    id: self.next_stmt_id(),
                    kind: hir::StmtKind::Fail {
                        code: hir::ErrorCode::InvalidVarint,
                    },
                }],
            }
        } else {
            let mut block = hir::Block {
                scope: hir::ScopeId::new(0),
                statements: Vec::new(),
            };
            self.lower_postcard_varint128_step(
                &mut block.statements,
                cursor_local,
                place.clone(),
                zigzag,
                acc_lo_local,
                acc_hi_local,
                byte_index + 1,
            );
            block
        };
        let else_block = Some(self.postcard_varint128_finish_block(
            place,
            zigzag,
            acc_lo_local,
            acc_hi_local,
            byte_index,
            raw_local,
        ));
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Ne,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitAnd,
                        lhs: Box::new(hir::Expr::Local(raw_local)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x80))),
                    }),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                },
                then_block,
                else_block,
            },
        });
    }

    fn lower_postcard_varint128_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        zigzag: bool,
    ) {
        let acc_lo_local = self.alloc_local(
            format!("varint128_lo_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        let acc_hi_local = self.alloc_local(
            format!("varint128_hi_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(acc_lo_local),
            hir::Expr::Literal(hir::Literal::Integer(0)),
        );
        self.push_init(
            statements,
            hir::Place::Local(acc_hi_local),
            hir::Expr::Literal(hir::Literal::Integer(0)),
        );
        self.lower_postcard_varint128_step(
            statements,
            cursor_local,
            place,
            zigzag,
            acc_lo_local,
            acc_hi_local,
            0,
        );
    }

    fn lower_postcard_char_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
    ) {
        let len_local = self.alloc_local(
            format!("char_len_{}", self.locals.len()),
            hir::Type::u(32),
            hir::LocalKind::Temp,
        );
        self.lower_postcard_varint_into_place(
            statements,
            cursor_local,
            hir::Place::Local(len_local),
            32,
            false,
        );

        let invalid_len = hir::Expr::Binary {
            op: hir::BinaryOp::Or,
            lhs: Box::new(hir::Expr::Binary {
                op: hir::BinaryOp::Eq,
                lhs: Box::new(hir::Expr::Local(len_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
            }),
            rhs: Box::new(hir::Expr::Binary {
                op: hir::BinaryOp::Gt,
                lhs: Box::new(hir::Expr::Local(len_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
            }),
        };
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: invalid_len,
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Fail {
                            code: hir::ErrorCode::InvalidUtf8,
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: Vec::new(),
                }),
            },
        });

        self.push_cursor_bounds_check_expr(
            statements,
            cursor_local,
            hir::Expr::Local(len_local),
            hir::ErrorCode::UnexpectedEof,
        );

        let bytes = self.cursor_bytes_expr(cursor_local);
        let pos = self.cursor_pos_expr(cursor_local);
        let data_local = self.alloc_local(
            format!("char_data_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(data_local),
            hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(hir::Expr::SliceData {
                    value: Box::new(bytes),
                }),
                rhs: Box::new(pos.clone()),
            },
        );

        let validate_utf8 = self.ensure_runtime_validate_utf8_range();
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(validate_utf8),
                args: vec![hir::Expr::Local(data_local), hir::Expr::Local(len_local)],
            })),
        });

        let raw0 = self.alloc_local(
            format!("char_raw0_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        let raw1 = self.alloc_local(
            format!("char_raw1_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        let raw2 = self.alloc_local(
            format!("char_raw2_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        let raw3 = self.alloc_local(
            format!("char_raw3_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        let code_local = self.alloc_local(
            format!("char_code_{}", self.locals.len()),
            hir::Type::u(32),
            hir::LocalKind::Temp,
        );

        self.push_init(
            statements,
            hir::Place::Local(raw0),
            hir::Expr::Load {
                addr: Box::new(hir::Expr::Local(data_local)),
                width: hir::MemoryWidth::W1,
            },
        );

        let load_byte = |data_local, offset| hir::Expr::Load {
            addr: Box::new(hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(hir::Expr::Local(data_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(offset))),
            }),
            width: hir::MemoryWidth::W1,
        };

        let one_byte = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::Init {
                    place: hir::Place::Local(code_local),
                    value: hir::Expr::Local(raw0),
                },
            }],
        };

        let mut two_byte_statements = Vec::new();
        self.push_init(
            &mut two_byte_statements,
            hir::Place::Local(raw1),
            load_byte(data_local, 1),
        );
        two_byte_statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(code_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::BitOr,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::Shl,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw0)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x1f))),
                        }),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(6))),
                    }),
                    rhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitAnd,
                        lhs: Box::new(hir::Expr::Local(raw1)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                    }),
                },
            },
        });
        let two_byte = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: two_byte_statements,
        };

        let mut three_byte_statements = Vec::new();
        self.push_init(
            &mut three_byte_statements,
            hir::Place::Local(raw1),
            load_byte(data_local, 1),
        );
        self.push_init(
            &mut three_byte_statements,
            hir::Place::Local(raw2),
            load_byte(data_local, 2),
        );
        three_byte_statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(code_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::BitOr,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::Shl,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw0)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x0f))),
                        }),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(12))),
                    }),
                    rhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitOr,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Shl,
                            lhs: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::BitAnd,
                                lhs: Box::new(hir::Expr::Local(raw1)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(6))),
                        }),
                        rhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw2)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                        }),
                    }),
                },
            },
        });
        let three_byte = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: three_byte_statements,
        };

        let mut four_byte_statements = Vec::new();
        self.push_init(
            &mut four_byte_statements,
            hir::Place::Local(raw1),
            load_byte(data_local, 1),
        );
        self.push_init(
            &mut four_byte_statements,
            hir::Place::Local(raw2),
            load_byte(data_local, 2),
        );
        self.push_init(
            &mut four_byte_statements,
            hir::Place::Local(raw3),
            load_byte(data_local, 3),
        );
        four_byte_statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(code_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::BitOr,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitOr,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Shl,
                            lhs: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::BitAnd,
                                lhs: Box::new(hir::Expr::Local(raw0)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x07))),
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(18))),
                        }),
                        rhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Shl,
                            lhs: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::BitAnd,
                                lhs: Box::new(hir::Expr::Local(raw1)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(12))),
                        }),
                    }),
                    rhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitOr,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Shl,
                            lhs: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::BitAnd,
                                lhs: Box::new(hir::Expr::Local(raw2)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(6))),
                        }),
                        rhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw3)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                        }),
                    }),
                },
            },
        });
        let four_byte = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: four_byte_statements,
        };

        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Eq,
                    lhs: Box::new(hir::Expr::Local(len_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                },
                then_block: one_byte,
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::If {
                            condition: hir::Expr::Binary {
                                op: hir::BinaryOp::Eq,
                                lhs: Box::new(hir::Expr::Local(len_local)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(2))),
                            },
                            then_block: two_byte,
                            else_block: Some(hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements: vec![hir::Stmt {
                                    id: self.next_stmt_id(),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Eq,
                                            lhs: Box::new(hir::Expr::Local(len_local)),
                                            rhs: Box::new(hir::Expr::Literal(
                                                hir::Literal::Integer(3),
                                            )),
                                        },
                                        then_block: three_byte,
                                        else_block: Some(four_byte),
                                    },
                                }],
                            }),
                        },
                    }],
                }),
            },
        });

        self.push_init(statements, place, hir::Expr::Local(code_local));
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(pos),
                    rhs: Box::new(hir::Expr::Local(len_local)),
                },
            },
        });
    }

    fn lower_postcard_option_tag_into_local(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        tag_local: hir::LocalId,
    ) {
        self.push_cursor_bounds_check(statements, cursor_local, 1, hir::ErrorCode::UnexpectedEof);

        let bytes = self.cursor_bytes_expr(cursor_local);
        let pos = self.cursor_pos_expr(cursor_local);
        let addr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(bytes),
            }),
            rhs: Box::new(pos.clone()),
        };
        let raw_local = self.alloc_local(
            format!("option_tag_raw_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(raw_local),
            hir::Expr::Load {
                addr: Box::new(addr),
                width: hir::MemoryWidth::W1,
            },
        );
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(pos),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                },
            },
        });
        let invalid = hir::Expr::Binary {
            op: hir::BinaryOp::Gt,
            lhs: Box::new(hir::Expr::Local(raw_local)),
            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
        };
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: invalid,
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Fail {
                            code: hir::ErrorCode::UnknownVariant,
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(tag_local),
                            value: hir::Expr::Binary {
                                op: hir::BinaryOp::Ne,
                                lhs: Box::new(hir::Expr::Local(raw_local)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                            },
                        },
                    }],
                }),
            },
        });
    }

    fn ensure_runtime_validate_utf8_range(&mut self) -> hir::CallableId {
        const NAME: &str = "runtime.validate_utf8_range";
        if let Some(existing) = self.callables_by_name.get(NAME).copied() {
            return existing;
        }

        let callable = hir::CallableSpec {
            kind: hir::CallableKind::Host,
            name: NAME.to_owned(),
            signature: hir::CallSignature {
                params: vec![hir::Type::u(64), hir::Type::u(32)],
                returns: vec![],
                effect_class: hir::EffectClass::Reads,
                domain_effects: vec![hir::DomainEffect {
                    domain: "input".to_owned(),
                    access: hir::DomainAccess::Read,
                }],
                control: hir::ControlTransfer::MayFail,
                capabilities: vec!["runtime.utf8".to_owned()],
                safety: hir::CallSafety::OpaqueHost,
            },
            docs: Some("Validate that a borrowed byte range is UTF-8.".to_owned()),
        };
        let callable_id = self.module.add_callable(callable);
        self.callables_by_name.insert(NAME, callable_id);
        callable_id
    }

    fn ensure_runtime_string_validate_alloc_copy(&mut self) -> hir::CallableId {
        const NAME: &str = "runtime.string_validate_alloc_copy";
        if let Some(existing) = self.callables_by_name.get(NAME).copied() {
            return existing;
        }

        let callable = hir::CallableSpec {
            kind: hir::CallableKind::Host,
            name: NAME.to_owned(),
            signature: hir::CallSignature {
                params: vec![hir::Type::u(64), hir::Type::u(32)],
                returns: vec![hir::Type::persistent_addr()],
                effect_class: hir::EffectClass::Barrier,
                domain_effects: vec![
                    hir::DomainEffect {
                        domain: "input".to_owned(),
                        access: hir::DomainAccess::Read,
                    },
                    hir::DomainEffect {
                        domain: "persistent_heap".to_owned(),
                        access: hir::DomainAccess::Mutate,
                    },
                ],
                control: hir::ControlTransfer::MayFail,
                capabilities: vec!["runtime.alloc".to_owned(), "runtime.utf8".to_owned()],
                safety: hir::CallSafety::OpaqueHost,
            },
            docs: Some(
                "Validate a postcard string range, allocate persistent bytes, and copy them."
                    .to_owned(),
            ),
        };
        let callable_id = self.module.add_callable(callable);
        self.callables_by_name.insert(NAME, callable_id);
        callable_id
    }

    fn ensure_runtime_alloc_persistent(&mut self) -> hir::CallableId {
        const NAME: &str = "runtime.alloc_persistent";
        if let Some(existing) = self.callables_by_name.get(NAME).copied() {
            return existing;
        }
        let callable = hir::CallableSpec {
            kind: hir::CallableKind::Host,
            name: NAME.to_owned(),
            signature: hir::CallSignature {
                params: vec![hir::Type::u(64), hir::Type::u(64)],
                returns: vec![hir::Type::persistent_addr()],
                effect_class: hir::EffectClass::Mutates,
                domain_effects: vec![hir::DomainEffect {
                    domain: "persistent_heap".to_owned(),
                    access: hir::DomainAccess::Mutate,
                }],
                control: hir::ControlTransfer::MayFail,
                capabilities: vec!["runtime.alloc".to_owned()],
                safety: hir::CallSafety::OpaqueHost,
            },
            docs: Some("Allocate persistent memory that may escape in the result.".to_owned()),
        };
        let callable_id = self.module.add_callable(callable);
        self.callables_by_name.insert(NAME, callable_id);
        callable_id
    }

    fn ensure_runtime_vec_from_raw_parts(&mut self) -> hir::CallableId {
        const NAME: &str = "runtime.vec_from_raw_parts";
        if let Some(existing) = self.callables_by_name.get(NAME).copied() {
            return existing;
        }
        let callable = hir::CallableSpec {
            kind: hir::CallableKind::Host,
            name: NAME.to_owned(),
            signature: hir::CallSignature {
                params: vec![
                    hir::Type::persistent_addr(),
                    hir::Type::u(64),
                    hir::Type::u(64),
                    hir::Type::u(64),
                ],
                returns: vec![hir::Type::u(64)],
                effect_class: hir::EffectClass::Barrier,
                domain_effects: vec![hir::DomainEffect {
                    domain: "persistent_heap".to_owned(),
                    access: hir::DomainAccess::Mutate,
                }],
                control: hir::ControlTransfer::MayFail,
                capabilities: vec!["runtime.alloc".to_owned()],
                safety: hir::CallSafety::OpaqueHost,
            },
            docs: Some("Materialize a Vec-like host value from persistent raw parts.".to_owned()),
        };
        let callable_id = self.module.add_callable(callable);
        self.callables_by_name.insert(NAME, callable_id);
        callable_id
    }

    fn lower_postcard_str_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
    ) {
        let len_local = self.alloc_local(
            format!("str_len_{}", self.locals.len()),
            hir::Type::u(32),
            hir::LocalKind::Temp,
        );
        let data_local = self.alloc_local(
            format!("str_data_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        self.lower_postcard_varint_into_place(
            statements,
            cursor_local,
            hir::Place::Local(len_local),
            32,
            false,
        );

        self.push_cursor_bounds_check_expr(
            statements,
            cursor_local,
            hir::Expr::Local(len_local),
            hir::ErrorCode::UnexpectedEof,
        );

        let bytes = self.cursor_bytes_expr(cursor_local);
        let pos = self.cursor_pos_expr(cursor_local);
        let data_expr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(bytes),
            }),
            rhs: Box::new(pos.clone()),
        };
        self.push_init(statements, hir::Place::Local(data_local), data_expr);

        let validate_utf8 = self.ensure_runtime_validate_utf8_range();
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(validate_utf8),
                args: vec![hir::Expr::Local(data_local), hir::Expr::Local(len_local)],
            })),
        });

        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(pos),
                    rhs: Box::new(hir::Expr::Local(len_local)),
                },
            },
        });

        self.push_init(
            statements,
            place,
            hir::Expr::Str {
                data: Box::new(hir::Expr::Local(data_local)),
                len: Box::new(hir::Expr::Local(len_local)),
            },
        );
    }

    fn lower_postcard_owned_string_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
    ) {
        let len_local = self.alloc_local(
            format!("string_len_{}", self.locals.len()),
            hir::Type::u(32),
            hir::LocalKind::Temp,
        );
        let data_local = self.alloc_local(
            format!("string_data_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        let ptr_local = self.alloc_local(
            format!("string_ptr_{}", self.locals.len()),
            hir::Type::persistent_addr(),
            hir::LocalKind::Temp,
        );
        let string_raw_type = self.ensure_string_raw_type();
        let raw_string_local = self.alloc_local(
            format!("string_raw_{}", self.locals.len()),
            hir::Type::named(string_raw_type, Vec::new()),
            hir::LocalKind::Temp,
        );

        self.lower_postcard_varint_into_place(
            statements,
            cursor_local,
            hir::Place::Local(len_local),
            32,
            false,
        );
        self.push_cursor_bounds_check_expr(
            statements,
            cursor_local,
            hir::Expr::Local(len_local),
            hir::ErrorCode::UnexpectedEof,
        );

        let bytes = self.cursor_bytes_expr(cursor_local);
        let pos = self.cursor_pos_expr(cursor_local);
        let data_expr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(bytes),
            }),
            rhs: Box::new(pos.clone()),
        };
        self.push_init(statements, hir::Place::Local(data_local), data_expr);
        let validate_alloc = self.ensure_runtime_string_validate_alloc_copy();
        self.push_init(
            statements,
            hir::Place::Local(ptr_local),
            hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(validate_alloc),
                args: vec![hir::Expr::Local(data_local), hir::Expr::Local(len_local)],
            }),
        );

        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(pos),
                    rhs: Box::new(hir::Expr::Local(len_local)),
                },
            },
        });

        self.push_init(
            statements,
            hir::Place::Field {
                base: Box::new(hir::Place::Local(raw_string_local)),
                field: "ptr".to_owned(),
            },
            hir::Expr::Local(ptr_local),
        );
        self.push_init(
            statements,
            hir::Place::Field {
                base: Box::new(hir::Place::Local(raw_string_local)),
                field: "len".to_owned(),
            },
            hir::Expr::Local(len_local),
        );
        self.push_init(
            statements,
            hir::Place::Field {
                base: Box::new(hir::Place::Local(raw_string_local)),
                field: "cap".to_owned(),
            },
            hir::Expr::Local(len_local),
        );
        self.push_init(statements, place, hir::Expr::Local(raw_string_local));
    }

    fn push_store(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        addr: hir::Expr,
        width: hir::MemoryWidth,
        value: hir::Expr,
    ) {
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Store { addr, width, value },
        });
    }

    fn add_addr_offset(base: hir::Expr, offset: usize) -> hir::Expr {
        if offset == 0 {
            base
        } else {
            hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(base),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(offset as u64))),
            }
        }
    }

    fn store_expr_into_addr(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        base_addr: hir::Expr,
        shape: &'static Shape,
        value: hir::Expr,
    ) {
        if is_unit(shape) {
            return;
        }

        if shape.is_transparent() {
            let (fields, skipped) = collect_fields(shape);
            assert!(
                skipped.is_empty() && fields.len() == 1,
                "transparent HIR store expects one lowered field"
            );
            self.store_expr_into_addr(
                statements,
                base_addr,
                fields[0].shape,
                hir::Expr::Field {
                    base: Box::new(value),
                    field: fields[0].name.to_owned(),
                },
            );
            return;
        }

        if let Def::Array(array_def) = &shape.def {
            let elem_layout = array_def
                .t
                .layout
                .sized_layout()
                .expect("array element must be Sized");
            for index in 0..array_def.n {
                self.store_expr_into_addr(
                    statements,
                    Self::add_addr_offset(base_addr.clone(), index * elem_layout.size()),
                    array_def.t,
                    hir::Expr::Index {
                        base: Box::new(value.clone()),
                        index: Box::new(hir::Expr::Literal(hir::Literal::Integer(index as u64))),
                    },
                );
            }
            return;
        }

        if let Some(st) = shape.scalar_type() {
            match st {
                ScalarType::Bool | ScalarType::U8 | ScalarType::I8 => {
                    self.push_store(statements, base_addr, hir::MemoryWidth::W1, value);
                    return;
                }
                ScalarType::U16 | ScalarType::I16 => {
                    self.push_store(statements, base_addr, hir::MemoryWidth::W2, value);
                    return;
                }
                ScalarType::U32 | ScalarType::I32 | ScalarType::F32 | ScalarType::Char => {
                    self.push_store(statements, base_addr, hir::MemoryWidth::W4, value);
                    return;
                }
                ScalarType::U64
                | ScalarType::I64
                | ScalarType::USize
                | ScalarType::ISize
                | ScalarType::F64 => {
                    self.push_store(statements, base_addr, hir::MemoryWidth::W8, value);
                    return;
                }
                ScalarType::U128 | ScalarType::I128 => {
                    self.push_store(
                        statements,
                        base_addr.clone(),
                        hir::MemoryWidth::W8,
                        hir::Expr::Field {
                            base: Box::new(value.clone()),
                            field: "lo".to_owned(),
                        },
                    );
                    self.push_store(
                        statements,
                        Self::add_addr_offset(base_addr, 8),
                        hir::MemoryWidth::W8,
                        hir::Expr::Field {
                            base: Box::new(value),
                            field: "hi".to_owned(),
                        },
                    );
                    return;
                }
                ScalarType::Str => {
                    self.push_store(
                        statements,
                        base_addr.clone(),
                        hir::MemoryWidth::W8,
                        hir::Expr::SliceData {
                            value: Box::new(value.clone()),
                        },
                    );
                    self.push_store(
                        statements,
                        Self::add_addr_offset(base_addr, 8),
                        hir::MemoryWidth::W8,
                        hir::Expr::SliceLen {
                            value: Box::new(value),
                        },
                    );
                    return;
                }
                ScalarType::String => {
                    let offsets = crate::malum::discover_string_offsets();
                    for (offset, field) in [
                        (offsets.ptr_offset as usize, "ptr"),
                        (offsets.len_offset as usize, "len"),
                        (offsets.cap_offset as usize, "cap"),
                    ] {
                        self.push_store(
                            statements,
                            Self::add_addr_offset(base_addr.clone(), offset),
                            hir::MemoryWidth::W8,
                            hir::Expr::Field {
                                base: Box::new(value.clone()),
                                field: field.to_owned(),
                            },
                        );
                    }
                    return;
                }
                _ => panic!(
                    "postcard HIR store into addr does not support scalar {st:?} yet for {}",
                    shape.type_identifier
                ),
            }
        }

        match &shape.ty {
            Type::User(UserType::Struct(_)) => {
                let (fields, skipped) = collect_fields(shape);
                assert!(
                    skipped.is_empty(),
                    "postcard HIR addr-store does not support skipped/defaulted fields"
                );
                for field in fields {
                    self.store_expr_into_addr(
                        statements,
                        Self::add_addr_offset(base_addr.clone(), field.offset),
                        field.shape,
                        hir::Expr::Field {
                            base: Box::new(value.clone()),
                            field: field.name.to_owned(),
                        },
                    );
                }
            }
            _ => panic!(
                "postcard HIR addr-store does not support shape {} yet",
                shape.type_identifier
            ),
        }
    }

    fn lower_postcard_list_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        list_def: &ListDef,
        _shape: &'static Shape,
    ) {
        let elem_layout = list_def
            .t
            .layout
            .sized_layout()
            .expect("postcard HIR list element must be Sized");
        let len_local = self.alloc_local(
            format!("list_len_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        let bytes_local = self.alloc_local(
            format!("list_bytes_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        let ptr_local = self.alloc_local(
            format!("list_ptr_{}", self.locals.len()),
            hir::Type::persistent_addr(),
            hir::LocalKind::Temp,
        );
        let index_local = self.alloc_local(
            format!("list_index_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        let elem_ty = self.lower_type(list_def.t);
        let elem_local = self.alloc_local(
            format!("list_elem_{}", self.locals.len()),
            elem_ty,
            hir::LocalKind::Temp,
        );

        self.lower_postcard_varint_into_place(
            statements,
            cursor_local,
            hir::Place::Local(len_local),
            64,
            false,
        );
        self.push_init(
            statements,
            hir::Place::Local(bytes_local),
            hir::Expr::Binary {
                op: hir::BinaryOp::Mul,
                lhs: Box::new(hir::Expr::Local(len_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                    elem_layout.size() as u64,
                ))),
            },
        );
        let alloc_persistent = self.ensure_runtime_alloc_persistent();
        self.push_init(
            statements,
            hir::Place::Local(ptr_local),
            hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(alloc_persistent),
                args: vec![
                    hir::Expr::Local(bytes_local),
                    hir::Expr::Literal(hir::Literal::Integer(elem_layout.align() as u64)),
                ],
            }),
        );
        self.push_init(
            statements,
            hir::Place::Local(index_local),
            hir::Expr::Literal(hir::Literal::Integer(0)),
        );

        let mut loop_body = Vec::new();
        loop_body.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Eq,
                    lhs: Box::new(hir::Expr::Local(index_local)),
                    rhs: Box::new(hir::Expr::Local(len_local)),
                },
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Break,
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: Vec::new(),
                }),
            },
        });
        self.lower_shape_into_place(
            &mut loop_body,
            cursor_local,
            hir::Place::Local(elem_local),
            list_def.t,
        );
        let elem_addr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::Local(ptr_local)),
            rhs: Box::new(hir::Expr::Binary {
                op: hir::BinaryOp::Mul,
                lhs: Box::new(hir::Expr::Local(index_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                    elem_layout.size() as u64,
                ))),
            }),
        };
        self.store_expr_into_addr(
            &mut loop_body,
            elem_addr,
            list_def.t,
            hir::Expr::Local(elem_local),
        );
        loop_body.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Local(index_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(hir::Expr::Local(index_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                },
            },
        });
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Loop {
                body: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: loop_body,
                },
            },
        });

        let vec_from_raw_parts = self.ensure_runtime_vec_from_raw_parts();
        self.push_init(
            statements,
            place,
            hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(vec_from_raw_parts),
                args: vec![
                    hir::Expr::Local(ptr_local),
                    hir::Expr::Local(len_local),
                    hir::Expr::Local(len_local),
                    hir::Expr::Literal(hir::Literal::Integer(elem_layout.align() as u64)),
                ],
            }),
        );
    }

    fn postcard_varint_max_bytes(bits: u32) -> u64 {
        match bits {
            16 => 3,
            32 => 5,
            64 => 10,
            128 => 19,
            _ => panic!("unsupported postcard HIR varint width {bits}"),
        }
    }

    fn postcard_varint_finish_expr(&self, acc_local: hir::LocalId, zigzag: bool) -> hir::Expr {
        let acc = hir::Expr::Local(acc_local);
        if !zigzag {
            return acc;
        }
        let shifted = hir::Expr::Binary {
            op: hir::BinaryOp::Shr,
            lhs: Box::new(acc.clone()),
            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
        };
        let sign = hir::Expr::Binary {
            op: hir::BinaryOp::BitAnd,
            lhs: Box::new(acc),
            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
        };
        let neg_sign = hir::Expr::Binary {
            op: hir::BinaryOp::Sub,
            lhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
            rhs: Box::new(sign),
        };
        hir::Expr::Binary {
            op: hir::BinaryOp::Xor,
            lhs: Box::new(shifted),
            rhs: Box::new(neg_sign),
        }
    }

    fn postcard_varint_finish_block(
        &mut self,
        place: hir::Place,
        bits: u32,
        zigzag: bool,
        acc_local: hir::LocalId,
        byte_index: u64,
        raw_local: hir::LocalId,
    ) -> hir::Block {
        let mut block = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: Vec::new(),
        };

        if bits == 64 && byte_index + 1 == Self::postcard_varint_max_bytes(bits) {
            let extra_bits = hir::Expr::Binary {
                op: hir::BinaryOp::BitAnd,
                lhs: Box::new(hir::Expr::Local(raw_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7e))),
            };
            block.statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(extra_bits),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::InvalidVarint,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Init {
                                place,
                                value: self.postcard_varint_finish_expr(acc_local, zigzag),
                            },
                        }],
                    }),
                },
            });
            return block;
        }

        if bits < 64 {
            let upper = hir::Expr::Binary {
                op: hir::BinaryOp::Shr,
                lhs: Box::new(hir::Expr::Local(acc_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(bits as u64))),
            };
            block.statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(upper),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::NumberOutOfRange,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Init {
                                place,
                                value: self.postcard_varint_finish_expr(acc_local, zigzag),
                            },
                        }],
                    }),
                },
            });
            return block;
        }

        self.push_init(
            &mut block.statements,
            place,
            self.postcard_varint_finish_expr(acc_local, zigzag),
        );
        block
    }

    fn lower_postcard_varint_step(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        bits: u32,
        zigzag: bool,
        acc_local: hir::LocalId,
        byte_index: u64,
    ) {
        self.push_cursor_bounds_check(statements, cursor_local, 1, hir::ErrorCode::UnexpectedEof);

        let pos = self.cursor_pos_expr(cursor_local);
        let addr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(self.cursor_bytes_expr(cursor_local)),
            }),
            rhs: Box::new(pos.clone()),
        };
        let raw_local = self.alloc_local(
            format!("varint_byte_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(raw_local),
            hir::Expr::Load {
                addr: Box::new(addr),
                width: hir::MemoryWidth::W1,
            },
        );
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(pos),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                },
            },
        });

        let low = hir::Expr::Binary {
            op: hir::BinaryOp::BitAnd,
            lhs: Box::new(hir::Expr::Local(raw_local)),
            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7f))),
        };
        let part = if byte_index == 0 {
            low
        } else {
            hir::Expr::Binary {
                op: hir::BinaryOp::Shl,
                lhs: Box::new(low),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(byte_index * 7))),
            }
        };
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Local(acc_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::BitOr,
                    lhs: Box::new(hir::Expr::Local(acc_local)),
                    rhs: Box::new(part),
                },
            },
        });

        let max_bytes = Self::postcard_varint_max_bytes(bits);
        let then_block = if byte_index + 1 == max_bytes {
            hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![hir::Stmt {
                    id: self.next_stmt_id(),
                    kind: hir::StmtKind::Fail {
                        code: hir::ErrorCode::InvalidVarint,
                    },
                }],
            }
        } else {
            let mut block = hir::Block {
                scope: hir::ScopeId::new(0),
                statements: Vec::new(),
            };
            self.lower_postcard_varint_step(
                &mut block.statements,
                cursor_local,
                place.clone(),
                bits,
                zigzag,
                acc_local,
                byte_index + 1,
            );
            block
        };
        let else_block =
            Some(self.postcard_varint_finish_block(
                place, bits, zigzag, acc_local, byte_index, raw_local,
            ));
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Ne,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitAnd,
                        lhs: Box::new(hir::Expr::Local(raw_local)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x80))),
                    }),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                },
                then_block,
                else_block,
            },
        });
    }

    fn lower_postcard_varint_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        bits: u32,
        zigzag: bool,
    ) {
        let acc_local = self.alloc_local(
            format!("varint_acc_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(acc_local),
            hir::Expr::Literal(hir::Literal::Integer(0)),
        );
        self.lower_postcard_varint_step(
            statements,
            cursor_local,
            place,
            bits,
            zigzag,
            acc_local,
            0,
        );
    }

    fn lower_shape_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        shape: &'static Shape,
    ) {
        if is_unit(shape) {
            return;
        }

        if let Def::List(list_def) = &shape.def {
            self.lower_postcard_list_into_place(statements, cursor_local, place, list_def, shape);
            return;
        }

        if shape.is_transparent() {
            let (fields, skipped) = collect_fields(shape);
            assert!(
                skipped.is_empty() && fields.len() == 1,
                "transparent HIR prototype expects one lowered field"
            );
            self.lower_shape_into_place(statements, cursor_local, place, fields[0].shape);
            return;
        }

        if let Def::Array(array_def) = &shape.def {
            for index in 0..array_def.n {
                let elem_place = hir::Place::Index {
                    base: Box::new(place.clone()),
                    index: Box::new(hir::Expr::Literal(hir::Literal::Integer(index as u64))),
                };
                self.lower_shape_into_place(statements, cursor_local, elem_place, array_def.t);
            }
            return;
        }

        if let Some(opt_def) = get_option_def(shape) {
            let option_def = self.ensure_type_def(shape);
            let tag_local = self.alloc_local(
                format!("option_is_some_{}", self.locals.len()),
                hir::Type::bool(),
                hir::LocalKind::Temp,
            );
            self.lower_postcard_option_tag_into_local(statements, cursor_local, tag_local);

            let mut then_block = hir::Block {
                scope: hir::ScopeId::new(0),
                statements: Vec::new(),
            };

            if is_unit(opt_def.t) {
                self.push_init(
                    &mut then_block.statements,
                    place.clone(),
                    hir::Expr::Variant {
                        def: option_def,
                        variant: "Some".to_owned(),
                        fields: vec![("value".to_owned(), hir::Expr::Literal(hir::Literal::Unit))],
                    },
                );
            } else {
                let payload_ty = self.lower_type(opt_def.t);
                let payload_local = self.alloc_local(
                    format!("option_value_{}", self.locals.len()),
                    payload_ty,
                    hir::LocalKind::Temp,
                );
                self.lower_shape_into_place(
                    &mut then_block.statements,
                    cursor_local,
                    hir::Place::Local(payload_local),
                    opt_def.t,
                );
                self.push_init(
                    &mut then_block.statements,
                    place.clone(),
                    hir::Expr::Variant {
                        def: option_def,
                        variant: "Some".to_owned(),
                        fields: vec![("value".to_owned(), hir::Expr::Local(payload_local))],
                    },
                );
            }

            let mut else_block = hir::Block {
                scope: hir::ScopeId::new(0),
                statements: Vec::new(),
            };
            self.push_init(
                &mut else_block.statements,
                place,
                hir::Expr::Variant {
                    def: option_def,
                    variant: "None".to_owned(),
                    fields: Vec::new(),
                },
            );

            statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Local(tag_local),
                    then_block,
                    else_block: Some(else_block),
                },
            });
            return;
        }

        if let Some(st) = shape.scalar_type() {
            match st {
                ScalarType::Bool | ScalarType::U8 | ScalarType::I8 => {
                    self.lower_postcard_fixed_width_scalar_into_place(
                        statements,
                        cursor_local,
                        place,
                        st,
                    );
                    return;
                }
                ScalarType::U16 => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        16,
                        false,
                    );
                    return;
                }
                ScalarType::U32 => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        32,
                        false,
                    );
                    return;
                }
                ScalarType::U64 | ScalarType::USize => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        64,
                        false,
                    );
                    return;
                }
                ScalarType::U128 => {
                    self.lower_postcard_varint128_into_place(
                        statements,
                        cursor_local,
                        place,
                        false,
                    );
                    return;
                }
                ScalarType::I16 => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        16,
                        true,
                    );
                    return;
                }
                ScalarType::I32 => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        32,
                        true,
                    );
                    return;
                }
                ScalarType::I64 | ScalarType::ISize => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        64,
                        true,
                    );
                    return;
                }
                ScalarType::I128 => {
                    self.lower_postcard_varint128_into_place(statements, cursor_local, place, true);
                    return;
                }
                ScalarType::F32 | ScalarType::F64 => {
                    self.lower_postcard_fixed_width_scalar_into_place(
                        statements,
                        cursor_local,
                        place,
                        st,
                    );
                    return;
                }
                ScalarType::Char => {
                    self.lower_postcard_char_into_place(statements, cursor_local, place);
                    return;
                }
                ScalarType::Str => {
                    self.lower_postcard_str_into_place(statements, cursor_local, place);
                    return;
                }
                ScalarType::String => {
                    self.lower_postcard_owned_string_into_place(statements, cursor_local, place);
                    return;
                }
                _ => {}
            }
            let callable = self.ensure_postcard_reader(st);
            self.push_init(
                statements,
                place,
                hir::Expr::Call(hir::CallExpr {
                    target: hir::CallTarget::Callable(callable),
                    args: vec![hir::Expr::Local(cursor_local)],
                }),
            );
            return;
        }

        match &shape.ty {
            Type::User(UserType::Struct(_)) => {
                let (fields, skipped) = collect_fields(shape);
                assert!(
                    skipped.is_empty(),
                    "postcard HIR prototype does not support skipped/defaulted fields"
                );
                for field in fields {
                    let field_place = hir::Place::Field {
                        base: Box::new(place.clone()),
                        field: field.name.to_owned(),
                    };
                    self.lower_shape_into_place(statements, cursor_local, field_place, field.shape);
                }
            }
            Type::User(UserType::Enum(enum_type)) => {
                let variants = collect_variants(enum_type);
                let enum_def = self.ensure_type_def(shape);
                let disc_local = self.alloc_local(
                    format!("enum_discriminant_{}", self.locals.len()),
                    hir::Type::u(32),
                    hir::LocalKind::Temp,
                );
                self.lower_postcard_varint_into_place(
                    statements,
                    cursor_local,
                    hir::Place::Local(disc_local),
                    32,
                    false,
                );

                statements.push(hir::Stmt {
                    id: self.next_stmt_id(),
                    kind: hir::StmtKind::Match {
                        scrutinee: hir::Expr::Local(disc_local),
                        arms: variants
                            .into_iter()
                            .map(|variant| hir::MatchArm {
                                pattern: hir::Pattern::Integer(
                                    variant
                                        .rust_discriminant
                                        .try_into()
                                        .expect("enum discriminant must fit in u64"),
                                ),
                                body: hir::Block {
                                    scope: hir::ScopeId::new(0),
                                    statements: {
                                        let mut statements = Vec::new();
                                        let mut variant_fields = Vec::new();
                                        for field in &variant.fields {
                                            if is_unit(field.shape) {
                                                variant_fields.push((
                                                    field.name.to_owned(),
                                                    hir::Expr::Literal(hir::Literal::Unit),
                                                ));
                                                continue;
                                            }
                                            let field_ty = self.lower_type(field.shape);
                                            let field_local = self.alloc_local(
                                                format!(
                                                    "variant_{}_{}_{}",
                                                    variant.name,
                                                    field.name,
                                                    self.locals.len()
                                                ),
                                                field_ty,
                                                hir::LocalKind::Temp,
                                            );
                                            self.lower_shape_into_place(
                                                &mut statements,
                                                cursor_local,
                                                hir::Place::Local(field_local),
                                                field.shape,
                                            );
                                            variant_fields.push((
                                                field.name.to_owned(),
                                                hir::Expr::Local(field_local),
                                            ));
                                        }
                                        statements.push(hir::Stmt {
                                            id: self.next_stmt_id(),
                                            kind: hir::StmtKind::Init {
                                                place: place.clone(),
                                                value: hir::Expr::Variant {
                                                    def: enum_def,
                                                    variant: variant.name.to_owned(),
                                                    fields: variant_fields,
                                                },
                                            },
                                        });
                                        statements
                                    },
                                },
                            })
                            .collect(),
                    },
                });
            }
            _ => panic!(
                "postcard HIR prototype does not support shape {} yet",
                shape.type_identifier
            ),
        }
    }
}

pub(crate) fn build_postcard_decoder_hir(shape: &'static Shape) -> hir::Module {
    let mut lowerer = PostcardHirLowerer::new();
    let root_type = lowerer.lower_type(shape);
    let cursor_local = lowerer.next_local();
    let out_local = lowerer.next_local();
    let root_scope = hir::ScopeId::new(0);
    let mut statements = Vec::new();
    lowerer.lower_shape_into_place(
        &mut statements,
        cursor_local,
        hir::Place::Local(out_local),
        shape,
    );
    statements.push(hir::Stmt {
        id: lowerer.next_stmt_id(),
        kind: hir::StmtKind::Return(None),
    });
    let locals = std::mem::take(&mut lowerer.locals);

    lowerer.module.add_function(hir::Function {
        name: format!("decode_{}", shape.type_identifier.replace("::", "_")),
        region_params: vec![lowerer.input_region],
        store_params: Vec::new(),
        params: vec![
            hir::Parameter {
                local: cursor_local,
                name: "cursor".to_owned(),
                ty: hir::Type::named(
                    lowerer.cursor_type,
                    vec![hir::GenericArg::Region(lowerer.input_region)],
                ),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: out_local,
                name: "out".to_owned(),
                ty: root_type,
                kind: hir::LocalKind::Destination,
            },
        ],
        locals,
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: root_scope,
            parent: None,
            comment: Some(format!(
                "Postcard prototype HIR for {}",
                shape.type_identifier
            )),
        }],
        body: hir::Block {
            scope: root_scope,
            statements,
        },
    });

    lowerer.finish()
}

pub(crate) fn build_json_decoder_hir(shape: &'static Shape) -> hir::Module {
    match shape.scalar_type() {
        Some(ScalarType::Bool) => build_json_root_bool_decoder_hir(shape),
        Some(ScalarType::U32) => build_json_root_u32_decoder_hir(shape),
        other => panic!("unsupported JSON HIR prototype shape: {other:?}"),
    }
}

fn build_json_root_bool_decoder_hir(shape: &'static Shape) -> hir::Module {
    let mut module = hir::Module::new();
    let input_region = module.add_region("input");
    let cursor_type = module.add_type_def(hir::TypeDef {
        name: "Cursor".to_owned(),
        generic_params: vec![hir::GenericParam::Region {
            name: "r_input".to_owned(),
        }],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "bytes".to_owned(),
                    ty: hir::Type::slice(input_region, hir::Type::u(8)),
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
    });

    let cursor_local = hir::LocalId::new(0);
    let out_local = hir::LocalId::new(1);
    let byte_local = hir::LocalId::new(2);
    let root_scope = hir::ScopeId::new(0);

    let cursor_bytes = || hir::Expr::Field {
        base: Box::new(hir::Expr::Local(cursor_local)),
        field: "bytes".to_owned(),
    };
    let cursor_pos = || hir::Expr::Field {
        base: Box::new(hir::Expr::Local(cursor_local)),
        field: "pos".to_owned(),
    };
    let byte_at_cursor = || hir::Expr::Load {
        addr: Box::new(hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(cursor_bytes()),
            }),
            rhs: Box::new(cursor_pos()),
        }),
        width: hir::MemoryWidth::W1,
    };
    let advance_cursor_stmt = |stmt_id: u32, delta: u64| -> hir::Stmt {
        hir::Stmt {
            id: hir::StmtId::new(stmt_id),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(cursor_pos()),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(delta))),
                },
            },
        }
    };
    let cursor_bounds_if =
        |stmt_id: u32, need: u64, fail_stmt: u32, error: hir::ErrorCode| -> hir::Stmt {
            hir::Stmt {
                id: hir::StmtId::new(stmt_id),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Gt,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(cursor_pos()),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(need))),
                        }),
                        rhs: Box::new(hir::Expr::SliceLen {
                            value: Box::new(cursor_bytes()),
                        }),
                    },
                    then_block: hir::Block {
                        scope: root_scope,
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(fail_stmt),
                            kind: hir::StmtKind::Fail { code: error },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: root_scope,
                        statements: Vec::new(),
                    }),
                },
            }
        };
    let matches_ascii = |text: &[u8], start_stmt: &mut u32| -> Vec<hir::Stmt> {
        let mut statements = Vec::new();
        for (index, expected) in text.iter().copied().enumerate() {
            let mismatch_stmt = *start_stmt;
            *start_stmt += 2;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(mismatch_stmt),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(hir::Expr::Load {
                            addr: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::Add,
                                lhs: Box::new(hir::Expr::SliceData {
                                    value: Box::new(cursor_bytes()),
                                }),
                                rhs: Box::new(hir::Expr::Binary {
                                    op: hir::BinaryOp::Add,
                                    lhs: Box::new(cursor_pos()),
                                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                        index as u64,
                                    ))),
                                }),
                            }),
                            width: hir::MemoryWidth::W1,
                        }),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(expected as u64))),
                    },
                    then_block: hir::Block {
                        scope: root_scope,
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(mismatch_stmt + 1),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::InvalidBool,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: root_scope,
                        statements: Vec::new(),
                    }),
                },
            });
        }
        statements
    };

    module.add_function(hir::Function {
        name: format!("decode_{}", shape.type_identifier.replace("::", "_")),
        region_params: vec![input_region],
        store_params: Vec::new(),
        params: vec![
            hir::Parameter {
                local: cursor_local,
                name: "cursor".to_owned(),
                ty: hir::Type::named(cursor_type, vec![hir::GenericArg::Region(input_region)]),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: out_local,
                name: "out".to_owned(),
                ty: hir::Type::bool(),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![hir::LocalDecl {
            local: byte_local,
            name: "byte".to_owned(),
            ty: hir::Type::u(8),
            kind: hir::LocalKind::Temp,
        }],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: root_scope,
            parent: None,
            comment: Some(format!("Prototype JSON HIR for {}", shape.type_identifier)),
        }],
        body: hir::Block {
            scope: root_scope,
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Loop {
                        body: hir::Block {
                            scope: root_scope,
                            statements: vec![
                                cursor_bounds_if(1, 1, 2, hir::ErrorCode::UnexpectedEof),
                                hir::Stmt {
                                    id: hir::StmtId::new(3),
                                    kind: hir::StmtKind::Assign {
                                        place: hir::Place::Local(byte_local),
                                        value: byte_at_cursor(),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(4),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Or,
                                            lhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Eq,
                                                lhs: Box::new(hir::Expr::Local(byte_local)),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(b' ' as u64),
                                                )),
                                            }),
                                            rhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Or,
                                                lhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Eq,
                                                    lhs: Box::new(hir::Expr::Local(byte_local)),
                                                    rhs: Box::new(hir::Expr::Literal(
                                                        hir::Literal::Integer(b'\n' as u64),
                                                    )),
                                                }),
                                                rhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Or,
                                                    lhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Eq,
                                                        lhs: Box::new(hir::Expr::Local(byte_local)),
                                                        rhs: Box::new(hir::Expr::Literal(
                                                            hir::Literal::Integer(b'\r' as u64),
                                                        )),
                                                    }),
                                                    rhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Eq,
                                                        lhs: Box::new(hir::Expr::Local(byte_local)),
                                                        rhs: Box::new(hir::Expr::Literal(
                                                            hir::Literal::Integer(b'\t' as u64),
                                                        )),
                                                    }),
                                                }),
                                            }),
                                        },
                                        then_block: hir::Block {
                                            scope: root_scope,
                                            statements: vec![
                                                advance_cursor_stmt(5, 1),
                                                hir::Stmt {
                                                    id: hir::StmtId::new(6),
                                                    kind: hir::StmtKind::Continue,
                                                },
                                            ],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: root_scope,
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(7),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        }),
                                    },
                                },
                            ],
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(8),
                    kind: hir::StmtKind::If {
                        condition: hir::Expr::Binary {
                            op: hir::BinaryOp::Eq,
                            lhs: Box::new(byte_at_cursor()),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(b't' as u64))),
                        },
                        then_block: {
                            let mut statements =
                                vec![cursor_bounds_if(9, 4, 10, hir::ErrorCode::UnexpectedEof)];
                            let mut next_stmt = 11;
                            statements.extend(matches_ascii(b"true", &mut next_stmt));
                            statements.push(advance_cursor_stmt(next_stmt, 4));
                            statements.push(hir::Stmt {
                                id: hir::StmtId::new(next_stmt + 1),
                                kind: hir::StmtKind::Init {
                                    place: hir::Place::Local(out_local),
                                    value: hir::Expr::Literal(hir::Literal::Bool(true)),
                                },
                            });
                            statements.push(hir::Stmt {
                                id: hir::StmtId::new(next_stmt + 2),
                                kind: hir::StmtKind::Return(None),
                            });
                            hir::Block {
                                scope: root_scope,
                                statements,
                            }
                        },
                        else_block: Some(hir::Block {
                            scope: root_scope,
                            statements: vec![hir::Stmt {
                                id: hir::StmtId::new(17),
                                kind: hir::StmtKind::If {
                                    condition: hir::Expr::Binary {
                                        op: hir::BinaryOp::Eq,
                                        lhs: Box::new(byte_at_cursor()),
                                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                            b'f' as u64,
                                        ))),
                                    },
                                    then_block: {
                                        let mut statements = vec![cursor_bounds_if(
                                            18,
                                            5,
                                            19,
                                            hir::ErrorCode::UnexpectedEof,
                                        )];
                                        let mut next_stmt = 20;
                                        statements.extend(matches_ascii(b"false", &mut next_stmt));
                                        statements.push(advance_cursor_stmt(next_stmt, 5));
                                        statements.push(hir::Stmt {
                                            id: hir::StmtId::new(next_stmt + 1),
                                            kind: hir::StmtKind::Init {
                                                place: hir::Place::Local(out_local),
                                                value: hir::Expr::Literal(hir::Literal::Bool(
                                                    false,
                                                )),
                                            },
                                        });
                                        statements.push(hir::Stmt {
                                            id: hir::StmtId::new(next_stmt + 2),
                                            kind: hir::StmtKind::Return(None),
                                        });
                                        hir::Block {
                                            scope: root_scope,
                                            statements,
                                        }
                                    },
                                    else_block: Some(hir::Block {
                                        scope: root_scope,
                                        statements: vec![hir::Stmt {
                                            id: hir::StmtId::new(28),
                                            kind: hir::StmtKind::Fail {
                                                code: hir::ErrorCode::InvalidBool,
                                            },
                                        }],
                                    }),
                                },
                            }],
                        }),
                    },
                },
            ],
        },
    });

    module
}

fn build_json_root_u32_decoder_hir(shape: &'static Shape) -> hir::Module {
    let mut module = hir::Module::new();
    let input_region = module.add_region("input");
    let cursor_type = module.add_type_def(hir::TypeDef {
        name: "Cursor".to_owned(),
        generic_params: vec![hir::GenericParam::Region {
            name: "r_input".to_owned(),
        }],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "bytes".to_owned(),
                    ty: hir::Type::slice(input_region, hir::Type::u(8)),
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
    });

    let cursor_local = hir::LocalId::new(0);
    let out_local = hir::LocalId::new(1);
    let byte_local = hir::LocalId::new(2);
    let acc_local = hir::LocalId::new(3);
    let digit_count_local = hir::LocalId::new(4);
    let root_scope = hir::ScopeId::new(0);

    let cursor_bytes = || hir::Expr::Field {
        base: Box::new(hir::Expr::Local(cursor_local)),
        field: "bytes".to_owned(),
    };
    let cursor_pos = || hir::Expr::Field {
        base: Box::new(hir::Expr::Local(cursor_local)),
        field: "pos".to_owned(),
    };
    let byte_at_cursor = || hir::Expr::Load {
        addr: Box::new(hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(cursor_bytes()),
            }),
            rhs: Box::new(cursor_pos()),
        }),
        width: hir::MemoryWidth::W1,
    };
    let advance_cursor_stmt = |stmt_id: u32, delta: u64| -> hir::Stmt {
        hir::Stmt {
            id: hir::StmtId::new(stmt_id),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(cursor_pos()),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(delta))),
                },
            },
        }
    };
    let cursor_bounds_if =
        |stmt_id: u32, need: u64, fail_stmt: u32, error: hir::ErrorCode| -> hir::Stmt {
            hir::Stmt {
                id: hir::StmtId::new(stmt_id),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Gt,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(cursor_pos()),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(need))),
                        }),
                        rhs: Box::new(hir::Expr::SliceLen {
                            value: Box::new(cursor_bytes()),
                        }),
                    },
                    then_block: hir::Block {
                        scope: root_scope,
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(fail_stmt),
                            kind: hir::StmtKind::Fail { code: error },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: root_scope,
                        statements: Vec::new(),
                    }),
                },
            }
        };

    module.add_function(hir::Function {
        name: format!("decode_{}", shape.type_identifier.replace("::", "_")),
        region_params: vec![input_region],
        store_params: Vec::new(),
        params: vec![
            hir::Parameter {
                local: cursor_local,
                name: "cursor".to_owned(),
                ty: hir::Type::named(cursor_type, vec![hir::GenericArg::Region(input_region)]),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: out_local,
                name: "out".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: byte_local,
                name: "byte".to_owned(),
                ty: hir::Type::u(8),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: acc_local,
                name: "acc".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: digit_count_local,
                name: "digit_count".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: root_scope,
            parent: None,
            comment: Some(format!(
                "Prototype JSON HIR for {}",
                shape.type_identifier
            )),
        }],
        body: hir::Block {
            scope: root_scope,
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Loop {
                        body: hir::Block {
                            scope: root_scope,
                            statements: vec![
                                cursor_bounds_if(1, 1, 2, hir::ErrorCode::UnexpectedEof),
                                hir::Stmt {
                                    id: hir::StmtId::new(3),
                                    kind: hir::StmtKind::Assign {
                                        place: hir::Place::Local(byte_local),
                                        value: byte_at_cursor(),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(4),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Or,
                                            lhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Eq,
                                                lhs: Box::new(hir::Expr::Local(byte_local)),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(b' ' as u64),
                                                )),
                                            }),
                                            rhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Or,
                                                lhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Eq,
                                                    lhs: Box::new(hir::Expr::Local(byte_local)),
                                                    rhs: Box::new(hir::Expr::Literal(
                                                        hir::Literal::Integer(b'\n' as u64),
                                                    )),
                                                }),
                                                rhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Or,
                                                    lhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Eq,
                                                        lhs: Box::new(hir::Expr::Local(byte_local)),
                                                        rhs: Box::new(hir::Expr::Literal(
                                                            hir::Literal::Integer(b'\r' as u64),
                                                        )),
                                                    }),
                                                    rhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Eq,
                                                        lhs: Box::new(hir::Expr::Local(byte_local)),
                                                        rhs: Box::new(hir::Expr::Literal(
                                                            hir::Literal::Integer(b'\t' as u64),
                                                        )),
                                                    }),
                                                }),
                                            }),
                                        },
                                        then_block: hir::Block {
                                            scope: root_scope,
                                            statements: vec![
                                                advance_cursor_stmt(5, 1),
                                                hir::Stmt {
                                                    id: hir::StmtId::new(6),
                                                    kind: hir::StmtKind::Continue,
                                                },
                                            ],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: root_scope,
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(7),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        }),
                                    },
                                },
                            ],
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(8),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(acc_local),
                        value: hir::Expr::Literal(hir::Literal::Integer(0)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(9),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(digit_count_local),
                        value: hir::Expr::Literal(hir::Literal::Integer(0)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(10),
                    kind: hir::StmtKind::Loop {
                        body: hir::Block {
                            scope: root_scope,
                            statements: vec![
                                hir::Stmt {
                                    id: hir::StmtId::new(11),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Ge,
                                            lhs: Box::new(cursor_pos()),
                                            rhs: Box::new(hir::Expr::SliceLen {
                                                value: Box::new(cursor_bytes()),
                                            }),
                                        },
                                        then_block: hir::Block {
                                            scope: root_scope,
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(12),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: root_scope,
                                            statements: Vec::new(),
                                        }),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(13),
                                    kind: hir::StmtKind::Assign {
                                        place: hir::Place::Local(byte_local),
                                        value: byte_at_cursor(),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(14),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Or,
                                            lhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Lt,
                                                lhs: Box::new(hir::Expr::Local(byte_local)),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(b'0' as u64),
                                                )),
                                            }),
                                            rhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Gt,
                                                lhs: Box::new(hir::Expr::Local(byte_local)),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(b'9' as u64),
                                                )),
                                            }),
                                        },
                                        then_block: hir::Block {
                                            scope: root_scope,
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(15),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: root_scope,
                                            statements: vec![
                                                hir::Stmt {
                                                    id: hir::StmtId::new(16),
                                                    kind: hir::StmtKind::Assign {
                                                        place: hir::Place::Local(acc_local),
                                                        value: hir::Expr::Binary {
                                                            op: hir::BinaryOp::Add,
                                                            lhs: Box::new(hir::Expr::Binary {
                                                                op: hir::BinaryOp::Mul,
                                                                lhs: Box::new(hir::Expr::Local(
                                                                    acc_local,
                                                                )),
                                                                rhs: Box::new(hir::Expr::Literal(
                                                                    hir::Literal::Integer(10),
                                                                )),
                                                            }),
                                                            rhs: Box::new(hir::Expr::Binary {
                                                                op: hir::BinaryOp::Sub,
                                                                lhs: Box::new(hir::Expr::Local(
                                                                    byte_local,
                                                                )),
                                                                rhs: Box::new(hir::Expr::Literal(
                                                                    hir::Literal::Integer(
                                                                        b'0' as u64,
                                                                    ),
                                                                )),
                                                            }),
                                                        },
                                                    },
                                                },
                                                hir::Stmt {
                                                    id: hir::StmtId::new(17),
                                                    kind: hir::StmtKind::If {
                                                        condition: hir::Expr::Binary {
                                                            op: hir::BinaryOp::Gt,
                                                            lhs: Box::new(hir::Expr::Local(
                                                                acc_local,
                                                            )),
                                                            rhs: Box::new(hir::Expr::Literal(
                                                                hir::Literal::Integer(
                                                                    u32::MAX as u64,
                                                                ),
                                                            )),
                                                        },
                                                        then_block: hir::Block {
                                                            scope: root_scope,
                                                            statements: vec![hir::Stmt {
                                                                id: hir::StmtId::new(18),
                                                                kind: hir::StmtKind::Fail {
                                                                    code: hir::ErrorCode::NumberOutOfRange,
                                                                },
                                                            }],
                                                        },
                                                        else_block: Some(hir::Block {
                                                            scope: root_scope,
                                                            statements: Vec::new(),
                                                        }),
                                                    },
                                                },
                                                hir::Stmt {
                                                    id: hir::StmtId::new(19),
                                                    kind: hir::StmtKind::Assign {
                                                        place: hir::Place::Local(digit_count_local),
                                                        value: hir::Expr::Binary {
                                                            op: hir::BinaryOp::Add,
                                                            lhs: Box::new(hir::Expr::Local(
                                                                digit_count_local,
                                                            )),
                                                            rhs: Box::new(hir::Expr::Literal(
                                                                hir::Literal::Integer(1),
                                                            )),
                                                        },
                                                    },
                                                },
                                                advance_cursor_stmt(20, 1),
                                                hir::Stmt {
                                                    id: hir::StmtId::new(21),
                                                    kind: hir::StmtKind::Continue,
                                                },
                                            ],
                                        }),
                                    },
                                },
                            ],
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(22),
                    kind: hir::StmtKind::If {
                        condition: hir::Expr::Binary {
                            op: hir::BinaryOp::Eq,
                            lhs: Box::new(hir::Expr::Local(digit_count_local)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                        },
                        then_block: hir::Block {
                            scope: root_scope,
                            statements: vec![hir::Stmt {
                                id: hir::StmtId::new(23),
                                kind: hir::StmtKind::Fail {
                                    code: hir::ErrorCode::InvalidJsonNumber,
                                },
                            }],
                        },
                        else_block: Some(hir::Block {
                            scope: root_scope,
                            statements: vec![],
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(24),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(out_local),
                        value: hir::Expr::Local(acc_local),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(25),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    module
}

pub(crate) fn build_postcard_decoder_ir_via_hir(shape: &'static Shape) -> crate::ir::IrFunc {
    let module = build_postcard_decoder_hir(shape);
    build_structural_hir_ir(shape, &module)
}

#[derive(Clone, Copy)]
struct StructuralLocalStorage {
    base_slot: crate::ir::SlotId,
}

struct StructuralHirIrLowerer<'a> {
    module: &'a hir::Module,
    cursor_local: hir::LocalId,
    local_slots: std::collections::HashMap<hir::LocalId, StructuralLocalStorage>,
    local_types: std::collections::HashMap<hir::LocalId, &'a hir::Type>,
    cursor_bytes_ptr_slot: Option<crate::ir::SlotId>,
    _marker: std::marker::PhantomData<&'a hir::Module>,
}

enum ResolvedStructuralPlace<'a> {
    Destination {
        shape: &'static Shape,
        offset: usize,
    },
    Local {
        ty: &'a hir::Type,
        storage: StructuralLocalStorage,
        slot_offset: usize,
    },
}

enum ResolvedDynamicIndex<'a> {
    Destination {
        shape: &'static Shape,
        addr: crate::ir::PortSource,
    },
    Local {
        ty: &'a hir::Type,
        addr: crate::ir::PortSource,
    },
}

impl<'a> StructuralHirIrLowerer<'a> {
    fn new(
        rb: &mut RegionBuilder<'_>,
        module: &'a hir::Module,
        function: &'a hir::Function,
    ) -> Self {
        let cursor_local = function
            .params
            .iter()
            .find(|param| param.kind == hir::LocalKind::Param)
            .map(|param| param.local)
            .expect("structural HIR function should have a cursor param");
        let mut local_slots = std::collections::HashMap::new();
        let mut local_types = std::collections::HashMap::new();
        for param in &function.params {
            if !param.is_destination() {
                local_slots.insert(
                    param.local,
                    Self::alloc_local_storage(rb, module, &param.ty),
                );
            }
            local_types.insert(param.local, &param.ty);
        }
        for local in &function.locals {
            local_slots.insert(
                local.local,
                Self::alloc_local_storage(rb, module, &local.ty),
            );
            local_types.insert(local.local, &local.ty);
        }
        let (cursor_bytes_ptr_slot, _cursor_bytes_len_slot, _cursor_pos_slot) =
            Self::initialize_cursor_shadow(rb, module, cursor_local, &local_slots, &local_types);
        Self {
            module,
            cursor_local,
            local_slots,
            local_types,
            cursor_bytes_ptr_slot,
            _marker: std::marker::PhantomData,
        }
    }

    fn alloc_local_storage(
        rb: &mut RegionBuilder<'_>,
        module: &'a hir::Module,
        ty: &hir::Type,
    ) -> StructuralLocalStorage {
        let slot_count = Self::slot_count_for_type(module, ty);
        let base_slot = rb.alloc_slot();
        for _ in 1..slot_count {
            let _ = rb.alloc_slot();
        }
        StructuralLocalStorage { base_slot }
    }

    fn initialize_cursor_shadow(
        rb: &mut RegionBuilder<'_>,
        module: &'a hir::Module,
        cursor_local: hir::LocalId,
        local_slots: &std::collections::HashMap<hir::LocalId, StructuralLocalStorage>,
        local_types: &std::collections::HashMap<hir::LocalId, &'a hir::Type>,
    ) -> (
        Option<crate::ir::SlotId>,
        Option<crate::ir::SlotId>,
        Option<crate::ir::SlotId>,
    ) {
        let Some(cursor_ty) = local_types.get(&cursor_local).copied() else {
            return (None, None, None);
        };
        let Some(storage) = local_slots.get(&cursor_local).copied() else {
            return (None, None, None);
        };
        let bytes_offset = match Self::struct_field_slot_offset(module, cursor_ty, "bytes") {
            Some(offset) => offset,
            None => return (None, None, None),
        };
        let pos_offset = match Self::struct_field_slot_offset(module, cursor_ty, "pos") {
            Some(offset) => offset,
            None => return (None, None, None),
        };

        let bytes_ptr_slot = Self::slot_at(storage, bytes_offset);
        let bytes_len_slot = Self::slot_at(storage, bytes_offset + 1);
        let pos_slot = Self::slot_at(storage, pos_offset);
        let base = rb.save_cursor();
        let end = rb.save_input_end();
        let len = rb.binop(crate::ir::IrOp::Sub, end, base);
        let zero = rb.const_val(0);
        rb.write_to_slot(bytes_ptr_slot, base);
        rb.write_to_slot(bytes_len_slot, len);
        rb.write_to_slot(pos_slot, zero);
        (Some(bytes_ptr_slot), Some(bytes_len_slot), Some(pos_slot))
    }

    fn struct_field_slot_offset(
        module: &'a hir::Module,
        ty: &hir::Type,
        field_name: &str,
    ) -> Option<usize> {
        let hir::Type::Named { def, .. } = ty else {
            return None;
        };
        let hir::TypeDefKind::Struct { fields } = &module.type_defs[*def].kind else {
            return None;
        };
        let mut slot_offset = 0usize;
        for field in fields {
            if field.name == field_name {
                return Some(slot_offset);
            }
            slot_offset += Self::slot_count_for_type(module, &field.ty);
        }
        None
    }

    fn slot_count_for_type(module: &'a hir::Module, ty: &hir::Type) -> usize {
        match ty {
            hir::Type::Unit
            | hir::Type::Bool
            | hir::Type::Integer(_)
            | hir::Type::Address { .. } => 1,
            hir::Type::Array { element, len } => Self::slot_count_for_type(module, element)
                .saturating_mul(*len)
                .max(1),
            hir::Type::Str { .. } | hir::Type::Slice { .. } => 2,
            hir::Type::Handle { .. } => 1,
            hir::Type::Named { def, .. } => match &module.type_defs[*def].kind {
                hir::TypeDefKind::Struct { fields } => fields
                    .iter()
                    .map(|field| Self::slot_count_for_type(module, &field.ty))
                    .sum::<usize>()
                    .max(1),
                hir::TypeDefKind::Enum { variants } => {
                    let payload_slots = variants
                        .iter()
                        .map(|variant| {
                            variant
                                .fields
                                .iter()
                                .map(|field| Self::slot_count_for_type(module, &field.ty))
                                .sum::<usize>()
                        })
                        .max()
                        .unwrap_or(0);
                    (1 + payload_slots).max(1)
                }
            },
        }
    }

    fn callable_name(&self, call: &hir::CallExpr) -> &str {
        match call.target {
            hir::CallTarget::Callable(callable) => &self.module.callables[callable].name,
        }
    }

    fn slot_at(storage: StructuralLocalStorage, slot_offset: usize) -> crate::ir::SlotId {
        crate::ir::SlotId::new(storage.base_slot.index() as u32 + slot_offset as u32)
    }

    fn lower_block(
        &self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        for stmt in statements {
            self.lower_stmt(rb, stmt, dest_local, dest_shape);
        }
    }

    fn lower_stmt(
        &self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        match &stmt.kind {
            hir::StmtKind::Init { place, value } | hir::StmtKind::Assign { place, value } => {
                self.lower_assign_like(rb, place, value, dest_local, dest_shape);
            }
            hir::StmtKind::Expr(expr) => self.lower_effect_expr(rb, expr, dest_local, dest_shape),
            hir::StmtKind::Fail { code } => rb.error_exit(*code),
            hir::StmtKind::Store { addr, width, value } => {
                let addr = self.lower_scalar_expr(rb, addr, dest_local, dest_shape);
                let value = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
                rb.store_to_addr(addr, value, self.ir_width_for_memory_width(*width));
            }
            hir::StmtKind::If {
                condition,
                then_block,
                else_block,
            } => {
                let predicate = self.lower_scalar_expr(rb, condition, dest_local, dest_shape);
                let else_block = else_block
                    .as_ref()
                    .expect("structural HIR subset requires else");
                let _ = rb.gamma(predicate, &[], 2, |branch_idx, branch| {
                    match branch_idx {
                        0 => {
                            self.lower_block(branch, &else_block.statements, dest_local, dest_shape)
                        }
                        1 => {
                            self.lower_block(branch, &then_block.statements, dest_local, dest_shape)
                        }
                        _ => unreachable!(),
                    }
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Match { scrutinee, arms } => {
                let predicate = self.lower_scalar_expr(rb, scrutinee, dest_local, dest_shape);
                for (expected, arm) in arms.iter().enumerate() {
                    let hir::Pattern::Integer(value) = arm.pattern else {
                        panic!("structural HIR subset only supports integer match patterns");
                    };
                    assert_eq!(
                        value, expected as u64,
                        "structural HIR subset requires contiguous integer match arms starting at 0"
                    );
                }
                let _ = rb.gamma(predicate, &[], arms.len(), |branch_idx, branch| {
                    self.lower_block(
                        branch,
                        &arms[branch_idx].body.statements,
                        dest_local,
                        dest_shape,
                    );
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Loop { body } => {
                let active_slot = rb.alloc_slot();
                let continue_slot = rb.alloc_slot();
                let _ = rb.theta(&[], |body_rb| {
                    let one = body_rb.const_val(1);
                    body_rb.write_to_slot(active_slot, one);
                    body_rb.write_to_slot(continue_slot, one);
                    self.lower_loop_block(
                        body_rb,
                        &body.statements,
                        dest_local,
                        dest_shape,
                        active_slot,
                        continue_slot,
                    );
                    let predicate = body_rb.read_from_slot(continue_slot);
                    body_rb.set_results(&[predicate]);
                });
            }
            hir::StmtKind::Return(None) => {}
            other => panic!("unsupported structural HIR statement: {other:?}"),
        }
    }

    fn lower_loop_block(
        &self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
        active_slot: crate::ir::SlotId,
        continue_slot: crate::ir::SlotId,
    ) {
        for stmt in statements {
            self.lower_loop_stmt(rb, stmt, dest_local, dest_shape, active_slot, continue_slot);
        }
    }

    fn lower_loop_stmt(
        &self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
        active_slot: crate::ir::SlotId,
        continue_slot: crate::ir::SlotId,
    ) {
        self.with_active_guard(rb, active_slot, |guard_rb| match &stmt.kind {
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
                let predicate = self.lower_scalar_expr(guard_rb, condition, dest_local, dest_shape);
                let else_block = else_block
                    .as_ref()
                    .expect("structural HIR loop subset requires else");
                let _ = guard_rb.gamma(predicate, &[], 2, |branch_idx, branch| {
                    match branch_idx {
                        0 => self.lower_loop_block(
                            branch,
                            &else_block.statements,
                            dest_local,
                            dest_shape,
                            active_slot,
                            continue_slot,
                        ),
                        1 => self.lower_loop_block(
                            branch,
                            &then_block.statements,
                            dest_local,
                            dest_shape,
                            active_slot,
                            continue_slot,
                        ),
                        _ => unreachable!(),
                    }
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Match { scrutinee, arms } => {
                let predicate = self.lower_scalar_expr(guard_rb, scrutinee, dest_local, dest_shape);
                for (expected, arm) in arms.iter().enumerate() {
                    let hir::Pattern::Integer(value) = arm.pattern else {
                        panic!("structural HIR loop subset only supports integer match patterns");
                    };
                    assert_eq!(
                        value, expected as u64,
                        "structural HIR loop subset requires contiguous integer match arms starting at 0"
                    );
                }
                let _ = guard_rb.gamma(predicate, &[], arms.len(), |branch_idx, branch| {
                    self.lower_loop_block(
                        branch,
                        &arms[branch_idx].body.statements,
                        dest_local,
                        dest_shape,
                        active_slot,
                        continue_slot,
                    );
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Loop { body } => {
                let nested_active_slot = guard_rb.alloc_slot();
                let nested_continue_slot = guard_rb.alloc_slot();
                let _ = guard_rb.theta(&[], |body_rb| {
                    let one = body_rb.const_val(1);
                    body_rb.write_to_slot(nested_active_slot, one);
                    body_rb.write_to_slot(nested_continue_slot, one);
                    self.lower_loop_block(
                        body_rb,
                        &body.statements,
                        dest_local,
                        dest_shape,
                        nested_active_slot,
                        nested_continue_slot,
                    );
                    let predicate = body_rb.read_from_slot(nested_continue_slot);
                    body_rb.set_results(&[predicate]);
                });
            }
            hir::StmtKind::Return(_) => {
                panic!("structural HIR loops do not support return in loop bodies yet");
            }
            _ => self.lower_stmt(guard_rb, stmt, dest_local, dest_shape),
        });
    }

    fn with_active_guard(
        &self,
        rb: &mut RegionBuilder<'_>,
        active_slot: crate::ir::SlotId,
        f: impl FnOnce(&mut RegionBuilder<'_>),
    ) {
        let active = rb.read_from_slot(active_slot);
        let mut f = Some(f);
        let _ = rb.gamma(active, &[], 2, |branch_idx, branch| {
            if branch_idx == 1 {
                f.take().expect("active branch should lower exactly once")(branch);
            }
            branch.set_results(&[]);
        });
    }

    fn lower_effect_expr(
        &self,
        rb: &mut RegionBuilder<'_>,
        expr: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        match expr {
            hir::Expr::Call(call) => self.lower_effect_call(rb, call, dest_local, dest_shape),
            other => panic!("unsupported structural HIR effect expression: {other:?}"),
        }
    }

    fn lower_assign_like(
        &self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
        value: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        self.lower_place_write(rb, place, value, dest_local, dest_shape);
    }

    fn lower_place_write(
        &self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
        value: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        if let hir::Place::Index { base, index } = place
            && !matches!(**index, hir::Expr::Literal(hir::Literal::Integer(_)))
        {
            if let hir::Expr::Local(local) = value
                && Self::slot_count_for_type(self.module, self.local_types[local]) > 1
            {
                self.lower_dynamic_index_write_from_local(
                    rb, base, index, *local, dest_local, dest_shape,
                );
            } else {
                self.lower_dynamic_index_write(rb, base, index, value, dest_local, dest_shape);
            }
            return;
        }
        let resolved = self.resolve_place(place, dest_local, dest_shape);
        match value {
            hir::Expr::Local(local) => match resolved {
                ResolvedStructuralPlace::Destination { shape, offset } => {
                    self.copy_local_into_shape_offset(rb, *local, shape, offset);
                }
                ResolvedStructuralPlace::Local {
                    ty,
                    storage,
                    slot_offset,
                } => {
                    self.copy_local_into_local(rb, *local, ty, storage, slot_offset);
                }
            },
            hir::Expr::Call(call) => {
                if self.is_vec_from_raw_parts(call) {
                    match resolved {
                        ResolvedStructuralPlace::Destination { shape, offset } => {
                            self.lower_vec_from_raw_parts_at_offset(
                                rb, call, shape, offset, dest_local, dest_shape,
                            );
                        }
                        ResolvedStructuralPlace::Local { .. } => {
                            panic!("local vec materialization is not supported yet");
                        }
                    }
                    return;
                }
                let scalar = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
                match resolved {
                    ResolvedStructuralPlace::Destination { shape, offset } => {
                        let width = self.scalar_width_for_shape(shape);
                        rb.write_to_field(scalar, offset as u32, width);
                    }
                    ResolvedStructuralPlace::Local {
                        ty,
                        storage,
                        slot_offset,
                    } => {
                        assert_eq!(
                            Self::slot_count_for_type(self.module, ty),
                            1,
                            "structural local scalar write requires single-slot type"
                        );
                        rb.write_to_slot(Self::slot_at(storage, slot_offset), scalar);
                        self.maybe_sync_cursor_position(rb, place, scalar);
                    }
                }
            }
            hir::Expr::Str { data, len } => match resolved {
                ResolvedStructuralPlace::Destination { shape, offset } => {
                    assert_eq!(
                        shape.scalar_type(),
                        Some(ScalarType::Str),
                        "str materialization requires a str destination, got {}",
                        shape.type_identifier
                    );
                    let data = self.lower_scalar_expr(rb, data, dest_local, dest_shape);
                    let len = self.lower_scalar_expr(rb, len, dest_local, dest_shape);
                    rb.write_to_field(data, offset as u32, crate::ir::Width::W8);
                    rb.write_to_field(len, (offset + 8) as u32, crate::ir::Width::W8);
                }
                ResolvedStructuralPlace::Local {
                    ty,
                    storage,
                    slot_offset,
                } => {
                    assert!(
                        matches!(ty, hir::Type::Str { .. }),
                        "str materialization requires a local str type, got {ty:?}"
                    );
                    let data = self.lower_scalar_expr(rb, data, dest_local, dest_shape);
                    let len = self.lower_scalar_expr(rb, len, dest_local, dest_shape);
                    rb.write_to_slot(Self::slot_at(storage, slot_offset), data);
                    rb.write_to_slot(Self::slot_at(storage, slot_offset + 1), len);
                }
            },
            hir::Expr::Variant {
                variant, fields, ..
            } => match resolved {
                ResolvedStructuralPlace::Destination { shape, offset } => {
                    if let Some(opt_def) = get_option_def(shape) {
                        self.lower_option_variant_write(rb, offset, *opt_def, variant, fields);
                        return;
                    }
                    let Type::User(UserType::Enum(enum_type)) = &shape.ty else {
                        panic!("variant init must target an enum place");
                    };
                    let variant_info = collect_variants(enum_type)
                        .into_iter()
                        .find(|candidate| candidate.name == variant.as_str())
                        .unwrap_or_else(|| panic!("missing enum variant {variant}"));
                    let disc_width =
                        ir_width_from_disc_size(discriminant_size(enum_type.enum_repr));
                    let disc = variant_info
                        .rust_discriminant
                        .try_into()
                        .expect("enum discriminant must fit in u64");
                    let value = rb.const_val(disc);
                    rb.write_to_field(value, offset as u32, disc_width);
                    for field in &variant_info.fields {
                        let (_, expr) = fields
                            .iter()
                            .find(|(name, _)| name == field.name)
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing enum payload field {} for variant {variant}",
                                    field.name
                                )
                            });
                        self.lower_value_into_shape_offset(
                            rb,
                            field.shape,
                            field.offset,
                            expr,
                            dest_local,
                            dest_shape,
                        );
                    }
                }
                ResolvedStructuralPlace::Local { .. } => {
                    panic!("local enum writes are not supported yet");
                }
            },
            hir::Expr::Index { .. } => match resolved {
                ResolvedStructuralPlace::Destination { shape, offset } => {
                    self.lower_value_into_shape_offset(
                        rb, shape, offset, value, dest_local, dest_shape,
                    );
                }
                ResolvedStructuralPlace::Local {
                    ty,
                    storage,
                    slot_offset,
                } => {
                    let scalar = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
                    assert_eq!(
                        Self::slot_count_for_type(self.module, ty),
                        1,
                        "structural local indexed write requires single-slot type"
                    );
                    rb.write_to_slot(Self::slot_at(storage, slot_offset), scalar);
                }
            },
            hir::Expr::Literal(hir::Literal::Unit) => {}
            _ => {
                let scalar = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
                match resolved {
                    ResolvedStructuralPlace::Destination { shape, offset } => {
                        let width = self.scalar_width_for_shape(shape);
                        rb.write_to_field(scalar, offset as u32, width);
                    }
                    ResolvedStructuralPlace::Local {
                        ty,
                        storage,
                        slot_offset,
                    } => {
                        assert_eq!(
                            Self::slot_count_for_type(self.module, ty),
                            1,
                            "structural local scalar write requires single-slot type"
                        );
                        rb.write_to_slot(Self::slot_at(storage, slot_offset), scalar);
                        self.maybe_sync_cursor_position(rb, place, scalar);
                    }
                }
            }
        }
    }

    fn copy_local_into_local(
        &self,
        rb: &mut RegionBuilder<'_>,
        source_local: hir::LocalId,
        dest_ty: &hir::Type,
        dest_storage: StructuralLocalStorage,
        dest_slot_offset: usize,
    ) {
        let source_storage = self.local_slots[&source_local];
        let source_ty = self.local_types[&source_local];
        let source_slots = Self::slot_count_for_type(self.module, source_ty);
        let dest_slots = Self::slot_count_for_type(self.module, dest_ty);
        assert_eq!(
            source_slots, dest_slots,
            "structural local copy requires matching slot counts"
        );
        for slot_index in 0..source_slots {
            let value = rb.read_from_slot(Self::slot_at(source_storage, slot_index));
            rb.write_to_slot(
                Self::slot_at(dest_storage, dest_slot_offset + slot_index),
                value,
            );
        }
    }

    fn maybe_sync_cursor_position(
        &self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
        value: crate::ir::PortSource,
    ) {
        let Some(bytes_ptr_slot) = self.cursor_bytes_ptr_slot else {
            return;
        };
        if !self.is_cursor_pos_place(place) {
            return;
        }
        let base = rb.read_from_slot(bytes_ptr_slot);
        let absolute = rb.binop(crate::ir::IrOp::Add, base, value);
        rb.restore_cursor(absolute);
    }

    fn is_cursor_pos_place(&self, place: &hir::Place) -> bool {
        matches!(
            place,
            hir::Place::Field { base, field }
                if field == "pos" && matches!(&**base, hir::Place::Local(local) if *local == self.cursor_local)
        )
    }

    fn lower_option_variant_write(
        &self,
        rb: &mut RegionBuilder<'_>,
        offset: usize,
        opt_def: OptionDef,
        variant: &str,
        fields: &[(String, hir::Expr)],
    ) {
        let offset = offset as u32;
        match variant {
            "None" => {
                assert!(
                    fields.is_empty(),
                    "Option::None should not carry payload fields"
                );
                let init_fn = rb.const_val(opt_def.vtable.init_none as *const () as usize as u64);
                rb.call_intrinsic(
                    crate::ir::IntrinsicFn(
                        intrinsics::kajit_option_init_none_ctx as *const () as usize,
                    ),
                    &[init_fn],
                    offset,
                    false,
                );
            }
            "Some" => {
                assert_eq!(
                    fields.len(),
                    1,
                    "Option::Some should carry exactly one payload field"
                );
                let payload_ptr = match &fields[0].1 {
                    hir::Expr::Local(local) => rb.slot_addr(self.local_slots[local].base_slot),
                    hir::Expr::Literal(hir::Literal::Unit) => {
                        let slot = rb.alloc_slot();
                        rb.slot_addr(slot)
                    }
                    hir::Expr::Literal(hir::Literal::Bool(value)) => {
                        let slot = rb.alloc_slot();
                        let value = rb.const_val(u64::from(*value));
                        rb.write_to_slot(slot, value);
                        rb.slot_addr(slot)
                    }
                    hir::Expr::Literal(hir::Literal::Integer(value)) => {
                        let slot = rb.alloc_slot();
                        let value = rb.const_val(*value);
                        rb.write_to_slot(slot, value);
                        rb.slot_addr(slot)
                    }
                    other => panic!("unsupported structural Option payload: {other:?}"),
                };
                let init_fn = rb.const_val(opt_def.vtable.init_some as *const () as usize as u64);
                rb.call_intrinsic(
                    crate::ir::IntrinsicFn(
                        intrinsics::kajit_option_init_some_ctx as *const () as usize,
                    ),
                    &[init_fn, payload_ptr],
                    offset,
                    false,
                );
            }
            other => panic!("unsupported structural Option variant {other}"),
        }
    }

    fn lower_value_into_shape_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        shape: &'static Shape,
        offset: usize,
        expr: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        match expr {
            hir::Expr::Call(call) if self.is_vec_from_raw_parts(call) => {
                self.lower_vec_from_raw_parts_at_offset(
                    rb, call, shape, offset, dest_local, dest_shape,
                );
            }
            hir::Expr::Call(_) => {
                let scalar = self.lower_scalar_expr(rb, expr, dest_local, dest_shape);
                let width = self.scalar_width_for_shape(shape);
                rb.write_to_field(scalar, offset as u32, width);
            }
            hir::Expr::Local(local) => self.copy_local_into_shape_offset(rb, *local, shape, offset),
            hir::Expr::Index { base, index } => {
                let base = self.expr_to_place(base);
                if matches!(&**index, hir::Expr::Literal(hir::Literal::Integer(_))) {
                    let place = hir::Place::Index {
                        base: Box::new(base),
                        index: index.clone(),
                    };
                    match self.resolve_place(&place, dest_local, dest_shape) {
                        ResolvedStructuralPlace::Destination {
                            shape: source_shape,
                            offset: source_offset,
                        } => {
                            self.copy_shape_bytes_to_shape_offset(
                                rb,
                                source_shape,
                                source_offset,
                                offset,
                            );
                        }
                        ResolvedStructuralPlace::Local {
                            ty: source_ty,
                            storage,
                            slot_offset,
                        } => {
                            let slot_count = Self::slot_count_for_type(self.module, source_ty);
                            for slot_index in 0..slot_count {
                                let slot = Self::slot_at(storage, slot_offset + slot_index);
                                let value = rb.read_from_slot(slot);
                                rb.write_to_field(
                                    value,
                                    (offset + slot_index * 8) as u32,
                                    crate::ir::Width::W8,
                                );
                            }
                        }
                    }
                } else {
                    self.copy_dynamic_index_into_shape_offset(
                        rb, &base, index, offset, dest_local, dest_shape,
                    );
                }
            }
            hir::Expr::Literal(hir::Literal::Unit) => {}
            hir::Expr::Literal(hir::Literal::Bool(value)) => {
                let value = rb.const_val(u64::from(*value));
                rb.write_to_field(value, offset as u32, crate::ir::Width::W1);
            }
            hir::Expr::Literal(hir::Literal::Integer(value)) => {
                let value = rb.const_val(*value);
                let width = self.scalar_width_for_shape(shape);
                rb.write_to_field(value, offset as u32, width);
            }
            hir::Expr::Str { data, len } => {
                assert_eq!(
                    shape.scalar_type(),
                    Some(ScalarType::Str),
                    "str materialization requires a str destination, got {}",
                    shape.type_identifier
                );
                let data = self.lower_scalar_expr(rb, data, dest_local, dest_shape);
                let len = self.lower_scalar_expr(rb, len, dest_local, dest_shape);
                rb.write_to_field(data, offset as u32, crate::ir::Width::W8);
                rb.write_to_field(len, (offset + 8) as u32, crate::ir::Width::W8);
            }
            hir::Expr::Variant {
                variant, fields, ..
            } => {
                if let Some(opt_def) = get_option_def(shape) {
                    self.lower_option_variant_write(rb, offset, *opt_def, variant, fields);
                } else {
                    panic!("nested non-Option variant payloads are not supported yet");
                }
            }
            other => panic!("unsupported structural payload expression: {other:?}"),
        }
    }

    fn copy_shape_bytes_to_shape_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        source_shape: &'static Shape,
        source_offset: usize,
        target_offset: usize,
    ) {
        let size = source_shape
            .layout
            .sized_layout()
            .expect("indexed destination element must be Sized")
            .size();
        let full_words = size / 8;
        let remainder = size % 8;
        for word_index in 0..full_words {
            let value = rb.read_from_field(
                (source_offset + word_index * 8) as u32,
                crate::ir::Width::W8,
            );
            rb.write_to_field(
                value,
                (target_offset + word_index * 8) as u32,
                crate::ir::Width::W8,
            );
        }
        if remainder != 0 {
            let width = match remainder {
                1 => crate::ir::Width::W1,
                2 => crate::ir::Width::W2,
                4 => crate::ir::Width::W4,
                _ => panic!("unsupported indexed destination remainder width {remainder}"),
            };
            let value = rb.read_from_field((source_offset + full_words * 8) as u32, width);
            rb.write_to_field(value, (target_offset + full_words * 8) as u32, width);
        }
    }

    fn copy_local_into_shape_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        local: hir::LocalId,
        shape: &'static Shape,
        offset: usize,
    ) {
        let storage = self.local_slots[&local];
        if let Some(st) = shape.scalar_type() {
            if !is_string_like_scalar(st) {
                let value = rb.read_from_slot(storage.base_slot);
                let width = self.scalar_width_for_shape(shape);
                rb.write_to_field(value, offset as u32, width);
                return;
            }
        }

        let size = shape
            .layout
            .sized_layout()
            .expect("structural local copy requires Sized layout")
            .size();
        let full_slots = size / 8;
        let remainder = size % 8;

        for slot_index in 0..full_slots {
            let slot = crate::ir::SlotId::new(storage.base_slot.index() as u32 + slot_index as u32);
            let value = rb.read_from_slot(slot);
            rb.write_to_field(
                value,
                (offset + slot_index * 8) as u32,
                crate::ir::Width::W8,
            );
        }

        if remainder != 0 {
            let slot = crate::ir::SlotId::new(storage.base_slot.index() as u32 + full_slots as u32);
            let value = rb.read_from_slot(slot);
            let width = match remainder {
                1 => crate::ir::Width::W1,
                2 => crate::ir::Width::W2,
                4 => crate::ir::Width::W4,
                _ => panic!("unsupported remainder width {remainder}"),
            };
            rb.write_to_field(value, (offset + full_slots * 8) as u32, width);
        }
    }

    fn is_vec_from_raw_parts(&self, call: &hir::CallExpr) -> bool {
        self.callable_name(call) == "runtime.vec_from_raw_parts"
    }

    fn lower_vec_from_raw_parts_at_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
        shape: &'static Shape,
        offset: usize,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        let Def::List(list_def) = &shape.def else {
            panic!("runtime.vec_from_raw_parts requires a list destination");
        };
        assert_eq!(
            call.args.len(),
            4,
            "runtime.vec_from_raw_parts expects ptr, len, cap, align"
        );
        let ptr = self.lower_scalar_expr(rb, &call.args[0], dest_local, dest_shape);
        let len = self.lower_scalar_expr(rb, &call.args[1], dest_local, dest_shape);
        let cap = self.lower_scalar_expr(rb, &call.args[2], dest_local, dest_shape);
        let align = self.lower_scalar_expr(rb, &call.args[3], dest_local, dest_shape);
        let offsets = crate::malum::discover_vec_offsets(list_def, shape);
        let usize_width = if core::mem::size_of::<usize>() == 8 {
            crate::ir::Width::W8
        } else {
            crate::ir::Width::W4
        };

        let zero = rb.const_val(0);
        let cap_nonzero = rb.binop(crate::ir::IrOp::CmpNe, cap, zero);
        rb.gamma(cap_nonzero, &[], 2, |branch_idx, branch| {
            let ptr_value = match branch_idx {
                0 => align,
                1 => ptr,
                _ => unreachable!(),
            };
            branch.write_to_field(ptr_value, (offset as u32) + offsets.ptr_offset, usize_width);
            branch.write_to_field(len, (offset as u32) + offsets.len_offset, usize_width);
            branch.write_to_field(cap, (offset as u32) + offsets.cap_offset, usize_width);
            branch.set_results(&[]);
        });
    }

    fn lower_scalar_expr(
        &self,
        rb: &mut RegionBuilder<'_>,
        expr: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) -> crate::ir::PortSource {
        match expr {
            hir::Expr::Literal(hir::Literal::Bool(value)) => rb.const_val(u64::from(*value)),
            hir::Expr::Literal(hir::Literal::Integer(value)) => rb.const_val(*value),
            hir::Expr::Local(local) => {
                let slot = self.local_slots[local].base_slot;
                rb.read_from_slot(slot)
            }
            hir::Expr::Load { addr, width } => {
                let addr = self.lower_scalar_expr(rb, addr, dest_local, dest_shape);
                rb.load_from_addr(addr, self.ir_width_for_memory_width(*width))
            }
            hir::Expr::SliceData { value } => {
                self.lower_view_component(rb, value, 0, dest_local, dest_shape)
            }
            hir::Expr::SliceLen { value } => {
                self.lower_view_component(rb, value, 1, dest_local, dest_shape)
            }
            hir::Expr::Field { .. } | hir::Expr::Index { .. } => {
                let place = self.expr_to_place(expr);
                if let hir::Place::Index { base, index } = &place
                    && !matches!(**index, hir::Expr::Literal(hir::Literal::Integer(_)))
                {
                    return self.lower_dynamic_index_read(rb, base, index, dest_local, dest_shape);
                }
                match self.resolve_place(&place, dest_local, dest_shape) {
                    ResolvedStructuralPlace::Destination { shape, offset } => {
                        let width = self.scalar_width_for_shape(shape);
                        rb.read_from_field(offset as u32, width)
                    }
                    ResolvedStructuralPlace::Local {
                        ty,
                        storage,
                        slot_offset,
                    } => {
                        assert_eq!(
                            Self::slot_count_for_type(self.module, ty),
                            1,
                            "structural local scalar read requires single-slot type"
                        );
                        rb.read_from_slot(Self::slot_at(storage, slot_offset))
                    }
                }
            }
            hir::Expr::Binary { op, lhs, rhs } => {
                let lhs = self.lower_scalar_expr(rb, lhs, dest_local, dest_shape);
                let rhs = self.lower_scalar_expr(rb, rhs, dest_local, dest_shape);
                let ir_op = match op {
                    hir::BinaryOp::Add => crate::ir::IrOp::Add,
                    hir::BinaryOp::Sub => crate::ir::IrOp::Sub,
                    hir::BinaryOp::Mul => crate::ir::IrOp::Mul,
                    hir::BinaryOp::BitAnd => crate::ir::IrOp::And,
                    hir::BinaryOp::BitOr => crate::ir::IrOp::Or,
                    hir::BinaryOp::Xor => crate::ir::IrOp::Xor,
                    hir::BinaryOp::Shl => crate::ir::IrOp::Shl,
                    hir::BinaryOp::Shr => crate::ir::IrOp::Shr,
                    hir::BinaryOp::Eq => crate::ir::IrOp::CmpEq,
                    hir::BinaryOp::And => crate::ir::IrOp::And,
                    hir::BinaryOp::Or => crate::ir::IrOp::Or,
                    hir::BinaryOp::Ne => crate::ir::IrOp::CmpNe,
                    hir::BinaryOp::Lt => crate::ir::IrOp::CmpLt,
                    hir::BinaryOp::Le => crate::ir::IrOp::CmpLe,
                    hir::BinaryOp::Gt => crate::ir::IrOp::CmpGt,
                    hir::BinaryOp::Ge => crate::ir::IrOp::CmpGe,
                    other => panic!("unsupported structural HIR binary op: {other:?}"),
                };
                rb.binop(ir_op, lhs, rhs)
            }
            hir::Expr::Call(call) => self.lower_scalar_call_expr(rb, call, dest_local, dest_shape),
            other => panic!("unsupported structural HIR scalar expression: {other:?}"),
        }
    }

    fn lower_scalar_call_expr(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) -> crate::ir::PortSource {
        let args = call
            .args
            .iter()
            .map(|arg| self.lower_scalar_expr(rb, arg, dest_local, dest_shape))
            .collect::<Vec<_>>();
        let func = match self.callable_name(call) {
            "runtime.alloc_persistent" => {
                crate::ir::IntrinsicFn(intrinsics::kajit_alloc_persistent as *const () as usize)
            }
            "runtime.string_validate_alloc_copy" => crate::ir::IntrinsicFn(
                intrinsics::kajit_string_validate_alloc_copy as *const () as usize,
            ),
            other => panic!("unsupported structural HIR scalar call {other}"),
        };
        rb.call_intrinsic(func, &args, 0, true)
            .expect("scalar intrinsic call should return a value")
    }

    fn lower_effect_call(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        let args = call
            .args
            .iter()
            .map(|arg| self.lower_scalar_expr(rb, arg, dest_local, dest_shape))
            .collect::<Vec<_>>();
        let func = match self.callable_name(call) {
            "runtime.validate_utf8_range" => {
                crate::ir::IntrinsicFn(intrinsics::kajit_validate_utf8_range as *const () as usize)
            }
            other => panic!("unsupported structural HIR effect call {other}"),
        };
        rb.call_intrinsic(func, &args, 0, false);
    }

    fn expr_to_place(&self, expr: &hir::Expr) -> hir::Place {
        match expr {
            hir::Expr::Local(local) => hir::Place::Local(*local),
            hir::Expr::Field { base, field } => hir::Place::Field {
                base: Box::new(self.expr_to_place(base)),
                field: field.clone(),
            },
            hir::Expr::Index { base, index } => hir::Place::Index {
                base: Box::new(self.expr_to_place(base)),
                index: index.clone(),
            },
            other => panic!("unsupported structural HIR place expression: {other:?}"),
        }
    }

    fn lower_view_component(
        &self,
        rb: &mut RegionBuilder<'_>,
        value: &hir::Expr,
        word_index: usize,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) -> crate::ir::PortSource {
        let place = self.expr_to_place(value);
        match self.resolve_place(&place, dest_local, dest_shape) {
            ResolvedStructuralPlace::Local {
                ty,
                storage,
                slot_offset,
            } => {
                assert!(
                    matches!(ty, hir::Type::Slice { .. } | hir::Type::Str { .. }),
                    "slice_data/slice_len require a local Slice/str, got {ty:?}"
                );
                rb.read_from_slot(Self::slot_at(storage, slot_offset + word_index))
            }
            ResolvedStructuralPlace::Destination { shape, offset } => {
                let width = crate::ir::Width::W8;
                match shape.scalar_type() {
                    Some(ScalarType::Str) => {
                        rb.read_from_field((offset + word_index * 8) as u32, width)
                    }
                    _ => panic!(
                        "slice_data/slice_len require a slice-like destination, got {}",
                        shape.type_identifier
                    ),
                }
            }
        }
    }

    fn add_scaled_index(
        &self,
        rb: &mut RegionBuilder<'_>,
        base_addr: crate::ir::PortSource,
        index: crate::ir::PortSource,
        stride_bytes: usize,
    ) -> crate::ir::PortSource {
        if stride_bytes == 1 {
            rb.binop(crate::ir::IrOp::Add, base_addr, index)
        } else {
            let stride = rb.const_val(stride_bytes as u64);
            let scaled = rb.binop(crate::ir::IrOp::Mul, index, stride);
            rb.binop(crate::ir::IrOp::Add, base_addr, scaled)
        }
    }

    fn add_byte_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        base_addr: crate::ir::PortSource,
        offset: usize,
    ) -> crate::ir::PortSource {
        if offset == 0 {
            base_addr
        } else {
            let offset = rb.const_val(offset as u64);
            rb.binop(crate::ir::IrOp::Add, base_addr, offset)
        }
    }

    fn lower_dynamic_index_read(
        &self,
        rb: &mut RegionBuilder<'_>,
        base: &hir::Place,
        index: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) -> crate::ir::PortSource {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_shape);
        match resolved {
            ResolvedDynamicIndex::Destination { shape, addr } => {
                let width = self.scalar_width_for_shape(shape);
                rb.load_from_addr(addr, width)
            }
            ResolvedDynamicIndex::Local { ty, addr } => {
                let width = self.scalar_width_for_hir_type(ty);
                rb.load_from_addr(addr, width)
            }
        }
    }

    fn lower_dynamic_index_write(
        &self,
        rb: &mut RegionBuilder<'_>,
        base: &hir::Place,
        index: &hir::Expr,
        value: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_shape);
        let value = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
        match resolved {
            ResolvedDynamicIndex::Destination { shape, addr } => {
                let width = self.scalar_width_for_shape(shape);
                rb.store_to_addr(addr, value, width);
            }
            ResolvedDynamicIndex::Local { ty, addr } => {
                let width = self.scalar_width_for_hir_type(ty);
                rb.store_to_addr(addr, value, width);
            }
        }
    }

    fn lower_dynamic_index_write_from_local(
        &self,
        rb: &mut RegionBuilder<'_>,
        base: &hir::Place,
        index: &hir::Expr,
        local: hir::LocalId,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_shape);
        let storage = self.local_slots[&local];
        let slot_count = Self::slot_count_for_type(self.module, self.local_types[&local]);
        let base_addr = match resolved {
            ResolvedDynamicIndex::Destination { addr, .. }
            | ResolvedDynamicIndex::Local { addr, .. } => addr,
        };
        for slot_index in 0..slot_count {
            let slot = Self::slot_at(storage, slot_index);
            let value = rb.read_from_slot(slot);
            let dst_addr = self.add_byte_offset(
                rb,
                base_addr,
                slot_index * crate::ir::SLOT_ADDR_STRIDE_BYTES,
            );
            rb.store_to_addr(dst_addr, value, crate::ir::Width::W8);
        }
    }

    fn copy_dynamic_index_into_shape_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        base: &hir::Place,
        index: &hir::Expr,
        target_offset: usize,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_shape);
        match resolved {
            ResolvedDynamicIndex::Destination { shape, addr } => {
                let size = shape
                    .layout
                    .sized_layout()
                    .expect("dynamic indexed destination element must be Sized")
                    .size();
                let full_words = size / 8;
                let remainder = size % 8;
                for word_index in 0..full_words {
                    let src_addr = self.add_byte_offset(rb, addr, word_index * 8);
                    let value = rb.load_from_addr(src_addr, crate::ir::Width::W8);
                    rb.write_to_field(
                        value,
                        (target_offset + word_index * 8) as u32,
                        crate::ir::Width::W8,
                    );
                }
                if remainder != 0 {
                    let src_addr = self.add_byte_offset(rb, addr, full_words * 8);
                    let width = match remainder {
                        1 => crate::ir::Width::W1,
                        2 => crate::ir::Width::W2,
                        4 => crate::ir::Width::W4,
                        _ => panic!(
                            "unsupported dynamic indexed destination remainder width {remainder}"
                        ),
                    };
                    let value = rb.load_from_addr(src_addr, width);
                    rb.write_to_field(value, (target_offset + full_words * 8) as u32, width);
                }
            }
            ResolvedDynamicIndex::Local { ty, addr } => {
                let slot_count = Self::slot_count_for_type(self.module, ty);
                for slot_index in 0..slot_count {
                    let src_addr = self.add_byte_offset(
                        rb,
                        addr,
                        slot_index * crate::ir::SLOT_ADDR_STRIDE_BYTES,
                    );
                    let value = rb.load_from_addr(src_addr, crate::ir::Width::W8);
                    rb.write_to_field(
                        value,
                        (target_offset + slot_index * 8) as u32,
                        crate::ir::Width::W8,
                    );
                }
            }
        }
    }

    fn lower_dynamic_index_addr(
        &self,
        rb: &mut RegionBuilder<'_>,
        base: &hir::Place,
        index: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) -> ResolvedDynamicIndex<'a> {
        let index = self.lower_scalar_expr(rb, index, dest_local, dest_shape);
        match self.resolve_place(base, dest_local, dest_shape) {
            ResolvedStructuralPlace::Destination { shape, offset } => {
                let Def::Array(array_def) = &shape.def else {
                    panic!(
                        "dynamic indexed structural HIR place requires an array destination, got {}",
                        shape.type_identifier
                    );
                };
                let elem_layout = array_def
                    .t
                    .layout
                    .sized_layout()
                    .expect("array element must be Sized");
                let mut base_addr = rb.save_out_ptr();
                if offset != 0 {
                    let offset_val = rb.const_val(offset as u64);
                    base_addr = rb.binop(crate::ir::IrOp::Add, base_addr, offset_val);
                }
                let addr = self.add_scaled_index(rb, base_addr, index, elem_layout.size());
                ResolvedDynamicIndex::Destination {
                    shape: array_def.t,
                    addr,
                }
            }
            ResolvedStructuralPlace::Local {
                ty,
                storage,
                slot_offset,
            } => {
                let hir::Type::Array { element, .. } = ty else {
                    panic!("dynamic indexed local place requires an HIR array type");
                };
                let base_slot = Self::slot_at(storage, slot_offset);
                let base_addr = rb.slot_addr(base_slot);
                let elem_slots = Self::slot_count_for_type(self.module, element);
                let addr = self.add_scaled_index(
                    rb,
                    base_addr,
                    index,
                    elem_slots * crate::ir::SLOT_ADDR_STRIDE_BYTES,
                );
                ResolvedDynamicIndex::Local { ty: element, addr }
            }
        }
    }

    fn resolve_place(
        &self,
        place: &hir::Place,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) -> ResolvedStructuralPlace<'a> {
        match place {
            hir::Place::Local(local) => {
                if *local == dest_local {
                    ResolvedStructuralPlace::Destination {
                        shape: dest_shape,
                        offset: 0,
                    }
                } else {
                    ResolvedStructuralPlace::Local {
                        ty: self.local_types[local],
                        storage: self.local_slots[local],
                        slot_offset: 0,
                    }
                }
            }
            hir::Place::Field { base, field } => {
                match self.resolve_place(base, dest_local, dest_shape) {
                    ResolvedStructuralPlace::Destination { shape, offset } => {
                        if matches!(
                            shape.scalar_type(),
                            Some(ScalarType::U128 | ScalarType::I128)
                        ) {
                            let field_offset = match field.as_str() {
                                "lo" => 0,
                                "hi" => 8,
                                _ => panic!(
                                    "missing raw128 field {field} while lowering structural HIR place for {}",
                                    shape.type_identifier
                                ),
                            };
                            return ResolvedStructuralPlace::Destination {
                                shape: u64::SHAPE,
                                offset: offset + field_offset,
                            };
                        }
                        let (fields, skipped) = collect_fields(shape);
                        assert!(
                            skipped.is_empty(),
                            "structural HIR subset does not support skipped/defaulted fields"
                        );
                        let field_info = fields
                            .into_iter()
                            .find(|candidate| candidate.name == field.as_str())
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing field {field} while lowering structural HIR place for {}",
                                    shape.type_identifier
                                )
                            });
                        ResolvedStructuralPlace::Destination {
                            shape: field_info.shape,
                            offset: offset + field_info.offset,
                        }
                    }
                    ResolvedStructuralPlace::Local {
                        ty,
                        storage,
                        slot_offset,
                    } => {
                        let hir::Type::Named { def, .. } = ty else {
                            panic!(
                                "local field place requires a named struct type, got {ty:?} for field {field}"
                            );
                        };
                        let hir::TypeDefKind::Struct { fields } = &self.module.type_defs[*def].kind
                        else {
                            panic!("local field place requires a struct type");
                        };
                        let mut running_slots = 0usize;
                        let field_info = fields
                            .iter()
                            .find_map(|candidate| {
                                let found = (candidate.name == field.as_str())
                                    .then_some((&candidate.ty, running_slots));
                                running_slots +=
                                    Self::slot_count_for_type(self.module, &candidate.ty);
                                found
                            })
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing HIR struct field {field} while resolving local place"
                                )
                            });
                        ResolvedStructuralPlace::Local {
                            ty: field_info.0,
                            storage,
                            slot_offset: slot_offset + field_info.1,
                        }
                    }
                }
            }
            hir::Place::Index { base, index } => {
                let hir::Expr::Literal(hir::Literal::Integer(index)) = &**index else {
                    panic!("structural HIR array indices must be integer literals");
                };
                let index = usize::try_from(*index).expect("array index must fit in usize");
                match self.resolve_place(base, dest_local, dest_shape) {
                    ResolvedStructuralPlace::Destination { shape, offset } => {
                        let Def::Array(array_def) = &shape.def else {
                            panic!(
                                "indexed structural HIR place requires an array base, got {}",
                                shape.type_identifier
                            );
                        };
                        assert!(
                            index < array_def.n,
                            "array index {index} out of bounds for {}",
                            shape.type_identifier
                        );
                        let elem_layout = array_def
                            .t
                            .layout
                            .sized_layout()
                            .expect("array element must be Sized");
                        let stride = elem_layout.size();
                        ResolvedStructuralPlace::Destination {
                            shape: array_def.t,
                            offset: offset + index * stride,
                        }
                    }
                    ResolvedStructuralPlace::Local {
                        ty,
                        storage,
                        slot_offset,
                    } => {
                        let hir::Type::Array { element, len } = ty else {
                            panic!("indexed local place requires an HIR array type");
                        };
                        assert!(
                            index < *len,
                            "local array index {index} out of bounds for {len}"
                        );
                        let elem_slots = Self::slot_count_for_type(self.module, element);
                        ResolvedStructuralPlace::Local {
                            ty: element,
                            storage,
                            slot_offset: slot_offset + index * elem_slots,
                        }
                    }
                }
            }
        }
    }

    fn scalar_width_for_shape(&self, shape: &'static Shape) -> crate::ir::Width {
        match shape.scalar_type() {
            Some(ScalarType::Bool | ScalarType::U8 | ScalarType::I8) => crate::ir::Width::W1,
            Some(ScalarType::U16 | ScalarType::I16) => crate::ir::Width::W2,
            Some(ScalarType::U32 | ScalarType::I32 | ScalarType::F32 | ScalarType::Char) => {
                crate::ir::Width::W4
            }
            Some(
                ScalarType::U64
                | ScalarType::I64
                | ScalarType::USize
                | ScalarType::ISize
                | ScalarType::U128
                | ScalarType::I128
                | ScalarType::F64,
            ) => crate::ir::Width::W8,
            _ => panic!(
                "unsupported structural HIR scalar width for {}",
                shape.type_identifier
            ),
        }
    }

    fn scalar_width_for_hir_type(&self, ty: &hir::Type) -> crate::ir::Width {
        match ty {
            hir::Type::Bool => crate::ir::Width::W1,
            hir::Type::Integer(kind) => match kind.bits {
                8 => crate::ir::Width::W1,
                16 => crate::ir::Width::W2,
                32 => crate::ir::Width::W4,
                64 => crate::ir::Width::W8,
                other => panic!("unsupported structural HIR integer width: {other}"),
            },
            hir::Type::Address { .. } => crate::ir::Width::W8,
            _ => panic!("unsupported structural HIR scalar local type: {ty:?}"),
        }
    }

    fn ir_width_for_memory_width(&self, width: hir::MemoryWidth) -> crate::ir::Width {
        match width {
            hir::MemoryWidth::W1 => crate::ir::Width::W1,
            hir::MemoryWidth::W2 => crate::ir::Width::W2,
            hir::MemoryWidth::W4 => crate::ir::Width::W4,
            hir::MemoryWidth::W8 => crate::ir::Width::W8,
        }
    }
}

pub(crate) fn build_structural_hir_ir(
    shape: &'static Shape,
    module: &hir::Module,
) -> crate::ir::IrFunc {
    let (_, function) = module
        .functions
        .iter()
        .next()
        .expect("structural HIR module should contain one function");
    let dest_local = function
        .destination_param()
        .map(|param| param.local)
        .expect("structural HIR function should have a destination param");

    let mut builder = crate::ir::IrBuilder::new(shape);
    let _ = builder.add_state_domain(crate::ir::MEMORY_STATE_DOMAIN_NAME);
    {
        let mut rb = builder.root_region();
        let lowerer = StructuralHirIrLowerer::new(&mut rb, module, function);
        lowerer.lower_block(&mut rb, &function.body.statements, dest_local, shape);
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

fn maybe_disable_regalloc_edits_cfg(
    alloc: &mut crate::regalloc_engine::AllocatedCfgProgram,
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

fn no_regalloc_alloc_for_cfg_program(
    cfg_program: &crate::regalloc_engine::cfg_mir::Program,
) -> crate::regalloc_engine::AllocatedCfgProgram {
    let functions = cfg_program
        .funcs
        .iter()
        .map(|func| crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: func.lambda_id,
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::new(),
            op_operands: std::collections::HashMap::new(),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        })
        .collect();

    crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: cfg_program.clone(),
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

/// Compile a deserializer directly from CFG-MIR.
///
/// This is primarily intended for regression tests and minimization workflows
/// where a failing CFG-MIR program is edited by hand and recompiled quickly.
pub fn compile_cfg_mir_decoder(
    cfg_program: &crate::regalloc_engine::cfg_mir::Program,
    trusted_utf8_input: bool,
) -> CompiledDecoder {
    compile_cfg_mir_decoder_with_registry(cfg_program, None, trusted_utf8_input)
}

pub(crate) fn compile_cfg_mir_decoder_with_registry(
    cfg_program: &crate::regalloc_engine::cfg_mir::Program,
    registry: Option<&crate::ir::IntrinsicRegistry>,
    trusted_utf8_input: bool,
) -> CompiledDecoder {
    compile_cfg_mir_decoder_with_options(
        cfg_program,
        registry,
        trusted_utf8_input,
        PipelineOptions::from_env(),
    )
}

fn compile_linear_ir_decoder_with_options(
    ir: &crate::linearize::LinearIr,
    trusted_utf8_input: bool,
    pipeline_opts: PipelineOptions,
) -> CompiledDecoder {
    let jit_debug = jit_debug_enabled();
    let apply_regalloc_edits = pipeline_opts.resolve_regalloc(true);

    let cfg_program = crate::regalloc_engine::cfg_mir::lower_linear_ir(ir);
    let regalloc_alloc = if apply_regalloc_edits {
        let mut alloc = crate::regalloc_engine::allocate_cfg_program(&cfg_program)
            .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));
        maybe_disable_regalloc_edits_cfg(&mut alloc, &pipeline_opts);
        alloc
    } else {
        no_regalloc_alloc_for_cfg_program(&cfg_program)
    };

    let (buf, entry, source_map, backend_debug_info) = {
        let result = crate::ir_backend::compile_linear_ir_with_alloc_and_mode(
            ir,
            &cfg_program,
            &regalloc_alloc,
            apply_regalloc_edits,
        );
        materialize_backend_result(result)
    };
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
    let root_shape = ir.ops.iter().find_map(|op| match op {
        crate::linearize::LinearOp::FuncStart {
            lambda_id, shape, ..
        } if lambda_id.index() == 0 => Some(*shape),
        _ => None,
    });
    let registry = root_shape.map(symbol_registry_for_shape);
    let listing = build_cfg_mir_listing(&cfg_program, registry.as_ref());
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
        let listing_path = write_cfg_mir_listing_file(&root_display_name, &listing.text);
        let mut debug_subprogram = cfg_mir_dwarf_variables(
            root_shape,
            &cfg_program,
            &regalloc_alloc,
            backend_debug_info.as_ref(),
            buf.code_ptr(),
            jit_dwarf_target_arch(),
            apply_regalloc_edits,
        );
        debug_subprogram.name = root_display_name.clone();
        let dwarf = listing_path.as_deref().and_then(|path| {
            let debug_info = build_jit_debug_info_from_source_map(
                buf.code_ptr(),
                buf.len(),
                source_map.as_ref(),
                path,
                debug_subprogram.clone(),
            )?;
            crate::jit_dwarf::build_jit_dwarf_sections_from_debug_info(&debug_info).ok()
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
        cfg_mir_line_text_by_line: listing.line_text_by_line,
        entry,
        func,
        trusted_utf8_input,
        _jit_registration: Some(registration),
    }
}

fn compile_cfg_mir_decoder_with_options(
    cfg_program: &crate::regalloc_engine::cfg_mir::Program,
    registry: Option<&crate::ir::IntrinsicRegistry>,
    trusted_utf8_input: bool,
    pipeline_opts: PipelineOptions,
) -> CompiledDecoder {
    let jit_debug = jit_debug_enabled();
    let apply_regalloc_edits = pipeline_opts.resolve_regalloc(true);

    let regalloc_alloc = if apply_regalloc_edits {
        let mut alloc = crate::regalloc_engine::allocate_cfg_program(cfg_program)
            .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));
        maybe_disable_regalloc_edits_cfg(&mut alloc, &pipeline_opts);
        alloc
    } else {
        no_regalloc_alloc_for_cfg_program(cfg_program)
    };

    let shim_linear = crate::linearize::LinearIr {
        ops: Vec::new(),
        label_count: 0,
        vreg_count: cfg_program.vreg_count,
        slot_count: cfg_program.slot_count,
        debug: Default::default(),
    };
    let (buf, entry, source_map, backend_debug_info) = {
        let result = crate::ir_backend::compile_linear_ir_with_alloc_and_mode(
            &shim_linear,
            cfg_program,
            &regalloc_alloc,
            apply_regalloc_edits,
        );
        materialize_backend_result(result)
    };
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
    let listing = build_cfg_mir_listing(cfg_program, registry);

    let root_display_name = "kajit::decode::cfg_mir_text".to_string();
    let root_mangled_name = crate::jit_debug::rust_v0_mangle(&["kajit", "decode", "cfg_mir_text"]);
    let symbol = crate::jit_debug::JitSymbolEntry {
        name: root_mangled_name,
        offset: entry,
        size: buf.len().saturating_sub(entry),
    };
    let registration = if jit_debug {
        let listing_path = write_cfg_mir_listing_file(&root_display_name, &listing.text);
        let mut debug_subprogram = cfg_mir_dwarf_variables(
            None,
            cfg_program,
            &regalloc_alloc,
            backend_debug_info.as_ref(),
            buf.code_ptr(),
            jit_dwarf_target_arch(),
            apply_regalloc_edits,
        );
        debug_subprogram.name = root_display_name.clone();
        let dwarf = listing_path.as_deref().and_then(|path| {
            let debug_info = build_jit_debug_info_from_source_map(
                buf.code_ptr(),
                buf.len(),
                source_map.as_ref(),
                path,
                debug_subprogram.clone(),
            )?;
            crate::jit_dwarf::build_jit_dwarf_sections_from_debug_info(&debug_info).ok()
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
        cfg_mir_line_text_by_line: listing.line_text_by_line,
        entry,
        func,
        trusted_utf8_input,
        _jit_registration: Some(registration),
    }
}

#[cfg(test)]
mod tests {
    use kajit_hir_text::parse_hir;
    use serde::Serialize;

    use super::*;
    use facet::Facet;

    #[derive(Facet)]
    struct Wrapper<T> {
        inner: T,
    }

    #[derive(Facet)]
    struct ConstWrapper<const N: usize> {
        inner: [u8; N],
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct BorrowedHeader<'a> {
        len: u32,
        name: &'a str,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct OwnedHeader {
        len: u32,
        name: String,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct OwnedAddress {
        city: String,
        zip: u32,
    }

    #[derive(Debug, PartialEq, Facet)]
    struct FloatHeader {
        a: f32,
        b: f64,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct CharHeader {
        ch: char,
    }

    #[derive(Debug, PartialEq, Eq, Facet, Serialize)]
    struct BigUnsigned {
        value: u128,
    }

    #[derive(Debug, PartialEq, Eq, Facet, Serialize)]
    struct BigSigned {
        value: i128,
    }

    #[derive(Debug, PartialEq, Eq, Facet, Serialize)]
    struct MaybeBigUnsigned {
        value: Option<u128>,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct MaybeBorrowedName<'a> {
        name: Option<&'a str>,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct MaybeCount {
        count: Option<u32>,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    #[repr(u8)]
    enum UnitAnimal {
        Cat,
        Dog,
        Parrot,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    #[repr(u8)]
    enum PayloadAnimal<'a> {
        Cat,
        Count(u32),
        Name(&'a str),
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct ConstantNumber {
        value: u32,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct ScalarNumber {
        value: u32,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct BoolHeader {
        value: bool,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct ScalarArrayHolder {
        values: [u32; 4],
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct BorrowedArrayHolder<'a> {
        values: [&'a str; 2],
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct BranchyAnimal {
        animal: UnitAnimal,
        value: u32,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct MaskSummary {
        masked: u32,
        shifted: u32,
        toggled: u32,
        combined: u32,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct ScratchSummary {
        mask: u32,
        done: u32,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct DynamicIndexSummary {
        selected: u32,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct DynamicDestinationSummary {
        values: [u32; 4],
        selected: u32,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct PersistentBufferSummary {
        ptr: usize,
        len: usize,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct VecHolder {
        values: Vec<u32>,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct Pair {
        lo: u64,
        hi: u64,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct DynamicAggregateSummary {
        pair: Pair,
    }

    #[derive(Debug, PartialEq, Eq, Facet)]
    struct DynamicAggregateDestinationSummary {
        pairs: [Pair; 2],
        selected: Pair,
    }

    fn compile_structural_hir_decoder(
        shape: &'static Shape,
        module: &hir::Module,
    ) -> CompiledDecoder {
        let mut func = build_structural_hir_ir(shape, module);
        run_default_passes_from_env(&mut func);
        let linear = crate::linearize::linearize(&mut func);
        compile_linear_ir_decoder(&linear, false)
    }

    fn compile_postcard_decoder_via_structural_hir(shape: &'static Shape) -> CompiledDecoder {
        let module = build_postcard_decoder_hir(shape);
        compile_structural_hir_decoder(shape, &module)
    }

    fn compile_json_decoder_via_structural_hir(shape: &'static Shape) -> CompiledDecoder {
        let module = build_json_decoder_hir(shape);
        compile_structural_hir_decoder(shape, &module)
    }

    fn build_structural_json_bool_module() -> hir::Module {
        let mut module = hir::Module::new();
        let input_region = module.add_region("input");
        let cursor_def = module.add_type_def(hir::TypeDef {
            name: "Cursor".to_owned(),
            generic_params: vec![hir::GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "bytes".to_owned(),
                        ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    },
                    hir::FieldDef {
                        name: "pos".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });
        let root_def = module.add_type_def(hir::TypeDef {
            name: <BoolHeader>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "value".to_owned(),
                    ty: hir::Type::bool(),
                }],
            },
        });

        let cursor_local = hir::LocalId::new(0);
        let out_local = hir::LocalId::new(1);
        let byte_local = hir::LocalId::new(2);

        let cursor_bytes = || hir::Expr::Field {
            base: Box::new(hir::Expr::Local(cursor_local)),
            field: "bytes".to_owned(),
        };
        let cursor_pos = || hir::Expr::Field {
            base: Box::new(hir::Expr::Local(cursor_local)),
            field: "pos".to_owned(),
        };
        let byte_at_cursor = || hir::Expr::Load {
            addr: Box::new(hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(hir::Expr::SliceData {
                    value: Box::new(cursor_bytes()),
                }),
                rhs: Box::new(cursor_pos()),
            }),
            width: hir::MemoryWidth::W1,
        };
        let advance_cursor_stmt = |stmt_id: u32, delta: u64| hir::Stmt {
            id: hir::StmtId::new(stmt_id),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(cursor_pos()),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(delta))),
                },
            },
        };
        let cursor_bounds_if =
            |stmt_id: u32, need: u64, fail_stmt: u32, error: hir::ErrorCode| -> hir::Stmt {
                hir::Stmt {
                    id: hir::StmtId::new(stmt_id),
                    kind: hir::StmtKind::If {
                        condition: hir::Expr::Binary {
                            op: hir::BinaryOp::Gt,
                            lhs: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::Add,
                                lhs: Box::new(cursor_pos()),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(need))),
                            }),
                            rhs: Box::new(hir::Expr::SliceLen {
                                value: Box::new(cursor_bytes()),
                            }),
                        },
                        then_block: hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: vec![hir::Stmt {
                                id: hir::StmtId::new(fail_stmt),
                                kind: hir::StmtKind::Fail { code: error },
                            }],
                        },
                        else_block: Some(hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: Vec::new(),
                        }),
                    },
                }
            };
        let matches_ascii = |text: &[u8], start_stmt: &mut u32| -> Vec<hir::Stmt> {
            let mut statements = Vec::new();
            for (index, expected) in text.iter().copied().enumerate() {
                let mismatch_stmt = *start_stmt;
                *start_stmt += 2;
                statements.push(hir::Stmt {
                    id: hir::StmtId::new(mismatch_stmt),
                    kind: hir::StmtKind::If {
                        condition: hir::Expr::Binary {
                            op: hir::BinaryOp::Ne,
                            lhs: Box::new(hir::Expr::Load {
                                addr: Box::new(hir::Expr::Binary {
                                    op: hir::BinaryOp::Add,
                                    lhs: Box::new(hir::Expr::SliceData {
                                        value: Box::new(cursor_bytes()),
                                    }),
                                    rhs: Box::new(hir::Expr::Binary {
                                        op: hir::BinaryOp::Add,
                                        lhs: Box::new(cursor_pos()),
                                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                            index as u64,
                                        ))),
                                    }),
                                }),
                                width: hir::MemoryWidth::W1,
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                expected as u64,
                            ))),
                        },
                        then_block: hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: vec![hir::Stmt {
                                id: hir::StmtId::new(mismatch_stmt + 1),
                                kind: hir::StmtKind::Fail {
                                    code: hir::ErrorCode::InvalidBool,
                                },
                            }],
                        },
                        else_block: Some(hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: Vec::new(),
                        }),
                    },
                });
            }
            statements
        };

        module.add_function(hir::Function {
            name: "json_bool".to_owned(),
            region_params: vec![input_region],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: cursor_local,
                    name: "cursor".to_owned(),
                    ty: hir::Type::named(cursor_def, vec![hir::GenericArg::Region(input_region)]),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: out_local,
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![hir::LocalDecl {
                local: byte_local,
                name: "byte".to_owned(),
                ty: hir::Type::u(8),
                kind: hir::LocalKind::Temp,
            }],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("structural JSON bool parser".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Loop {
                            body: hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements: vec![
                                    cursor_bounds_if(1, 1, 2, hir::ErrorCode::UnexpectedEof),
                                    hir::Stmt {
                                        id: hir::StmtId::new(3),
                                        kind: hir::StmtKind::Assign {
                                            place: hir::Place::Local(byte_local),
                                            value: byte_at_cursor(),
                                        },
                                    },
                                    hir::Stmt {
                                        id: hir::StmtId::new(4),
                                        kind: hir::StmtKind::If {
                                            condition: hir::Expr::Binary {
                                                op: hir::BinaryOp::Or,
                                                lhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Eq,
                                                    lhs: Box::new(hir::Expr::Local(byte_local)),
                                                    rhs: Box::new(hir::Expr::Literal(
                                                        hir::Literal::Integer(b' ' as u64),
                                                    )),
                                                }),
                                                rhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Or,
                                                    lhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Eq,
                                                        lhs: Box::new(hir::Expr::Local(byte_local)),
                                                        rhs: Box::new(hir::Expr::Literal(
                                                            hir::Literal::Integer(b'\n' as u64),
                                                        )),
                                                    }),
                                                    rhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Or,
                                                        lhs: Box::new(hir::Expr::Binary {
                                                            op: hir::BinaryOp::Eq,
                                                            lhs: Box::new(hir::Expr::Local(
                                                                byte_local,
                                                            )),
                                                            rhs: Box::new(hir::Expr::Literal(
                                                                hir::Literal::Integer(b'\r' as u64),
                                                            )),
                                                        }),
                                                        rhs: Box::new(hir::Expr::Binary {
                                                            op: hir::BinaryOp::Eq,
                                                            lhs: Box::new(hir::Expr::Local(
                                                                byte_local,
                                                            )),
                                                            rhs: Box::new(hir::Expr::Literal(
                                                                hir::Literal::Integer(b'\t' as u64),
                                                            )),
                                                        }),
                                                    }),
                                                }),
                                            },
                                            then_block: hir::Block {
                                                scope: hir::ScopeId::new(0),
                                                statements: vec![
                                                    advance_cursor_stmt(5, 1),
                                                    hir::Stmt {
                                                        id: hir::StmtId::new(6),
                                                        kind: hir::StmtKind::Continue,
                                                    },
                                                ],
                                            },
                                            else_block: Some(hir::Block {
                                                scope: hir::ScopeId::new(0),
                                                statements: vec![hir::Stmt {
                                                    id: hir::StmtId::new(7),
                                                    kind: hir::StmtKind::Break,
                                                }],
                                            }),
                                        },
                                    },
                                ],
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(8),
                        kind: hir::StmtKind::If {
                            condition: hir::Expr::Binary {
                                op: hir::BinaryOp::Eq,
                                lhs: Box::new(byte_at_cursor()),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                    b't' as u64,
                                ))),
                            },
                            then_block: {
                                let mut statements =
                                    vec![cursor_bounds_if(9, 4, 10, hir::ErrorCode::UnexpectedEof)];
                                let mut next_stmt = 11;
                                statements.extend(matches_ascii(b"true", &mut next_stmt));
                                statements.push(advance_cursor_stmt(next_stmt, 4));
                                statements.push(hir::Stmt {
                                    id: hir::StmtId::new(next_stmt + 1),
                                    kind: hir::StmtKind::Init {
                                        place: hir::Place::Field {
                                            base: Box::new(hir::Place::Local(out_local)),
                                            field: "value".to_owned(),
                                        },
                                        value: hir::Expr::Literal(hir::Literal::Bool(true)),
                                    },
                                });
                                statements.push(hir::Stmt {
                                    id: hir::StmtId::new(next_stmt + 2),
                                    kind: hir::StmtKind::Return(None),
                                });
                                hir::Block {
                                    scope: hir::ScopeId::new(0),
                                    statements,
                                }
                            },
                            else_block: Some(hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements: vec![hir::Stmt {
                                    id: hir::StmtId::new(17),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Eq,
                                            lhs: Box::new(byte_at_cursor()),
                                            rhs: Box::new(hir::Expr::Literal(
                                                hir::Literal::Integer(b'f' as u64),
                                            )),
                                        },
                                        then_block: {
                                            let mut statements = vec![cursor_bounds_if(
                                                18,
                                                5,
                                                19,
                                                hir::ErrorCode::UnexpectedEof,
                                            )];
                                            let mut next_stmt = 20;
                                            statements
                                                .extend(matches_ascii(b"false", &mut next_stmt));
                                            statements.push(advance_cursor_stmt(next_stmt, 5));
                                            statements.push(hir::Stmt {
                                                id: hir::StmtId::new(next_stmt + 1),
                                                kind: hir::StmtKind::Init {
                                                    place: hir::Place::Field {
                                                        base: Box::new(hir::Place::Local(
                                                            out_local,
                                                        )),
                                                        field: "value".to_owned(),
                                                    },
                                                    value: hir::Expr::Literal(hir::Literal::Bool(
                                                        false,
                                                    )),
                                                },
                                            });
                                            statements.push(hir::Stmt {
                                                id: hir::StmtId::new(next_stmt + 2),
                                                kind: hir::StmtKind::Return(None),
                                            });
                                            hir::Block {
                                                scope: hir::ScopeId::new(0),
                                                statements,
                                            }
                                        },
                                        else_block: Some(hir::Block {
                                            scope: hir::ScopeId::new(0),
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(28),
                                                kind: hir::StmtKind::Fail {
                                                    code: hir::ErrorCode::InvalidBool,
                                                },
                                            }],
                                        }),
                                    },
                                }],
                            }),
                        },
                    },
                ],
            },
        });

        module
    }

    #[test]
    fn instantiated_shape_symbol_key_distinguishes_generic_instantiations() {
        let u32_key = instantiated_shape_symbol_key(<Wrapper<u32>>::SHAPE);
        let string_key = instantiated_shape_symbol_key(<Wrapper<String>>::SHAPE);

        assert_ne!(u32_key, string_key);
        assert!(u32_key.starts_with('d'));
        assert!(u32_key.contains("__t_0_d"));
        assert_eq!(
            <Wrapper<u32>>::SHAPE.decl_id,
            <Wrapper<String>>::SHAPE.decl_id
        );
    }

    #[test]
    fn instantiated_shape_symbol_key_includes_const_params() {
        let n4_key = instantiated_shape_symbol_key(<ConstWrapper<4>>::SHAPE);
        let n8_key = instantiated_shape_symbol_key(<ConstWrapper<8>>::SHAPE);

        assert_ne!(n4_key, n8_key);
        assert!(n4_key.contains("__c_0_u4"));
        assert!(n8_key.contains("__c_0_u8"));
    }

    #[test]
    fn postcard_hir_models_borrowed_output_structs() {
        let module = build_postcard_decoder_hir(<BorrowedHeader<'static>>::SHAPE);
        assert_eq!(module.functions.len(), 1);

        let (_, function) = module.functions.iter().next().unwrap();
        assert_eq!(function.params.len(), 2);
        assert_eq!(function.region_params.len(), 1);

        let destination = function.destination_param().unwrap();
        let hir::Type::Named { def, args } = &destination.ty else {
            panic!("expected named root output type");
        };
        assert_eq!(
            module.type_defs[*def].name,
            <BorrowedHeader<'static>>::SHAPE.type_identifier
        );
        assert_eq!(
            args,
            &vec![hir::GenericArg::Region(function.region_params[0])]
        );

        let statements = &function.body.statements;
        assert!(
            module
                .callable_named("runtime.validate_utf8_range")
                .is_some(),
            "borrowed string lowering should install runtime UTF-8 validation"
        );
        assert!(
            module.callable_named("postcard.read_str").is_none(),
            "borrowed string lowering should not use postcard.read_str"
        );
        assert!(
            module.callable_named("postcard.read_u32").is_none(),
            "borrowed header lowering should not use postcard.read_u32"
        );
        assert!(
            statements.iter().any(|stmt| matches!(
                &stmt.kind,
                hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                    target: hir::CallTarget::Callable(callable_id),
                    ..
                })) if module.callables[*callable_id].name == "runtime.validate_utf8_range"
            )),
            "borrowed string lowering should validate UTF-8 explicitly"
        );
        assert!(
            statements.iter().any(|stmt| matches!(
                &stmt.kind,
                hir::StmtKind::Init {
                    place: hir::Place::Field { field, .. },
                    value: hir::Expr::Str { .. },
                } if field == "name"
            )),
            "borrowed string lowering should materialize a str value directly into the destination"
        );
    }

    #[test]
    fn json_hir_models_root_bool_without_reader_calls() {
        let module = build_json_decoder_hir(<bool>::SHAPE);
        let (_, function) = module.functions.iter().next().unwrap();

        assert!(
            module.callable_named("postcard.read_bool").is_none(),
            "json root bool HIR should not mention postcard bool readers"
        );
        assert!(
            module.callable_named("kajit_json_read_bool").is_none(),
            "json root bool HIR should not call the old JSON bool intrinsic"
        );
        assert!(
            module.callables.is_empty(),
            "root bool HIR should be leaf-free"
        );
        assert!(
            function
                .body
                .statements
                .iter()
                .any(|stmt| matches!(&stmt.kind, hir::StmtKind::Loop { .. })),
            "json root bool HIR should spell out whitespace skipping as control flow"
        );
        assert!(
            function
                .body
                .statements
                .iter()
                .any(|stmt| matches!(&stmt.kind, hir::StmtKind::If { .. })),
            "json root bool HIR should spell out token dispatch as control flow"
        );
    }

    #[test]
    fn json_hir_models_root_u32_without_reader_calls() {
        let module = build_json_decoder_hir(<u32>::SHAPE);
        let (_, function) = module.functions.iter().next().unwrap();

        assert!(
            module.callable_named("postcard.read_u32").is_none(),
            "json root u32 HIR should not mention postcard integer readers"
        );
        assert!(
            module.callable_named("kajit_json_read_u32").is_none(),
            "json root u32 HIR should not call the old JSON u32 intrinsic"
        );
        assert!(
            module.callables.is_empty(),
            "root u32 HIR should be leaf-free"
        );
        assert!(
            function
                .body
                .statements
                .iter()
                .any(|stmt| matches!(&stmt.kind, hir::StmtKind::Loop { .. })),
            "json root u32 HIR should spell out whitespace and digit scanning as control flow"
        );
    }

    #[test]
    fn postcard_hir_models_owned_output_strings() {
        let module = build_postcard_decoder_hir(<OwnedHeader>::SHAPE);
        let (_, function) = module.functions.iter().next().unwrap();

        assert!(
            module
                .callable_named("runtime.string_validate_alloc_copy")
                .is_some(),
            "owned string lowering should install the raw string allocation helper"
        );
        assert!(
            module.callable_named("postcard.read_str").is_none(),
            "owned string lowering should not use postcard.read_str"
        );
        assert!(
            function
                .locals
                .iter()
                .any(|local| matches!(local.ty, hir::Type::Address { .. })),
            "owned string lowering should allocate a persistent data pointer local"
        );
        assert!(
            function.body.statements.iter().any(|stmt| matches!(
                &stmt.kind,
                hir::StmtKind::Init {
                    value: hir::Expr::Call(hir::CallExpr {
                        target: hir::CallTarget::Callable(callable_id),
                        ..
                    }),
                    ..
                } if module.callables[*callable_id].name == "runtime.string_validate_alloc_copy"
            )),
            "owned string lowering should compute string storage through the lean helper"
        );
    }

    #[test]
    fn postcard_hir_models_float_scalars_without_reader_calls() {
        let module = build_postcard_decoder_hir(<FloatHeader>::SHAPE);

        assert!(
            module.callable_named("postcard.read_f32").is_none(),
            "float lowering should not use postcard.read_f32"
        );
        assert!(
            module.callable_named("postcard.read_f64").is_none(),
            "float lowering should not use postcard.read_f64"
        );
    }

    #[test]
    fn postcard_hir_models_char_without_reader_calls() {
        let module = build_postcard_decoder_hir(<CharHeader>::SHAPE);

        assert!(
            module.callable_named("postcard.read_char").is_none(),
            "char lowering should not use postcard.read_char"
        );
        assert!(
            module
                .callable_named("runtime.validate_utf8_range")
                .is_some(),
            "char lowering should validate UTF-8 explicitly"
        );
    }

    #[test]
    fn postcard_hir_models_128bit_scalars_without_reader_calls() {
        let unsigned = build_postcard_decoder_hir(<BigUnsigned>::SHAPE);
        let signed = build_postcard_decoder_hir(<BigSigned>::SHAPE);
        let optional = build_postcard_decoder_hir(<MaybeBigUnsigned>::SHAPE);

        assert!(
            unsigned.callable_named("postcard.read_u128").is_none(),
            "u128 lowering should not use postcard.read_u128"
        );
        assert!(
            signed.callable_named("postcard.read_i128").is_none(),
            "i128 lowering should not use postcard.read_i128"
        );
        assert!(
            optional.callable_named("postcard.read_u128").is_none(),
            "Option<u128> lowering should not use postcard.read_u128"
        );
    }

    #[test]
    fn postcard_hir_models_option_borrowed_fields() {
        let module = build_postcard_decoder_hir(<MaybeBorrowedName<'static>>::SHAPE);
        let (_, function) = module.functions.iter().next().unwrap();
        let input_region = function.region_params[0];

        assert!(
            module.callable_named("postcard.read_option_tag").is_none(),
            "option lowering should not use postcard.read_option_tag"
        );

        assert!(function.locals.len() >= 4);
        assert!(
            function
                .locals
                .iter()
                .any(|local| local.ty == hir::Type::bool())
        );
        assert!(
            function
                .locals
                .iter()
                .any(|local| local.ty == hir::Type::u(8))
        );
        assert!(
            function
                .locals
                .iter()
                .any(|local| local.ty == hir::Type::str(input_region))
        );

        let (_, then_block, else_block) = function
            .body
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::If {
                    condition,
                    then_block,
                    else_block,
                } if matches!(condition, hir::Expr::Local(_)) => {
                    Some((condition, then_block, else_block))
                }
                _ => None,
            })
            .expect("expected option if statement");

        let Some(else_block) = else_block else {
            panic!("expected explicit option else block");
        };
        assert!(then_block.statements.len() >= 2);
        assert_eq!(else_block.statements.len(), 1);
        assert!(
            then_block.statements.iter().any(|stmt| matches!(
                &stmt.kind,
                hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                    target: hir::CallTarget::Callable(callable_id),
                    ..
                })) if module.callables[*callable_id].name == "runtime.validate_utf8_range"
            )),
            "borrowed option payload should validate UTF-8 explicitly"
        );

        let value = then_block
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "Some") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected Some variant init");
        let hir::Expr::Variant {
            def,
            variant,
            fields,
        } = value
        else {
            panic!("expected Some variant expression");
        };
        assert_eq!(variant, "Some");
        assert_eq!(fields.len(), 1);
        let hir::Expr::Local(payload_local) = &fields[0].1 else {
            panic!("expected Some variant payload local");
        };
        assert_eq!(
            function
                .locals
                .iter()
                .find(|local| local.ty == hir::Type::str(input_region))
                .expect("expected borrowed payload local")
                .local,
            *payload_local
        );

        let hir::TypeDefKind::Enum { variants } = &module.type_defs[*def].kind else {
            panic!("expected Option HIR enum type");
        };
        assert_eq!(variants[0].name, "None");
        assert_eq!(variants[1].name, "Some");
        assert_eq!(variants[1].fields[0].ty, hir::Type::str(input_region));

        let value = else_block
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "None") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected None variant init");
        let hir::Expr::Variant {
            variant, fields, ..
        } = value
        else {
            panic!("expected None variant expression");
        };
        assert_eq!(variant, "None");
        assert!(fields.is_empty());
    }

    #[test]
    fn postcard_hir_text_round_trips() {
        std::thread::Builder::new()
            .name("postcard_hir_text_round_trips".to_owned())
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let module = build_postcard_decoder_hir(<MaybeBorrowedName<'static>>::SHAPE);
                let text = module.to_string();
                let reparsed = parse_hir(&text).expect("postcard HIR text should parse");

                assert_eq!(reparsed, module);
            })
            .expect("thread should spawn")
            .join()
            .expect("round-trip thread should succeed");
    }

    #[test]
    fn postcard_hir_ir_path_decodes_option_borrowed_fields() {
        let decoder = crate::compile_postcard_decoder_via_hir(<MaybeBorrowedName<'static>>::SHAPE);

        let some = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[1, 2, b'h', b'i'])
            .expect("HIR->RVSDG postcard decoder should decode Some(&str)");
        assert_eq!(some, MaybeBorrowedName { name: Some("hi") });

        let none = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[0])
            .expect("HIR->RVSDG postcard decoder should decode None");
        assert_eq!(none, MaybeBorrowedName { name: None });
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_float_fields() {
        let decoder = compile_postcard_decoder_via_structural_hir(<FloatHeader>::SHAPE);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3.14f32.to_le_bytes());
        bytes.extend_from_slice(&2.718281828459045f64.to_le_bytes());

        let value = crate::deserialize::<FloatHeader>(&decoder, &bytes)
            .expect("structural HIR postcard decoder should decode float fields");
        assert_eq!(value.a.to_bits(), 3.14f32.to_bits());
        assert_eq!(value.b.to_bits(), 2.718281828459045f64.to_bits());
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_char_field() {
        let decoder = compile_postcard_decoder_via_structural_hir(<CharHeader>::SHAPE);

        let value = crate::deserialize::<CharHeader>(&decoder, &[2, 0xC3, 0x9F])
            .expect("structural HIR postcard decoder should decode char fields");
        assert_eq!(value, CharHeader { ch: 'ß' });
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_128bit_fields() {
        let unsigned = BigUnsigned {
            value: (1_u128 << 100) | 0x1234_5678_9abc_def0_u128,
        };
        let signed = BigSigned {
            value: -((1_i128 << 97) - 0x1234_5678_9abc_i128),
        };

        let unsigned_decoder = compile_postcard_decoder_via_structural_hir(<BigUnsigned>::SHAPE);
        let unsigned_bytes = postcard::to_allocvec(&unsigned)
            .expect("postcard should encode unsigned 128-bit sample");
        let unsigned_value = crate::deserialize::<BigUnsigned>(&unsigned_decoder, &unsigned_bytes)
            .expect("structural HIR postcard decoder should decode u128 fields");
        assert_eq!(unsigned_value, unsigned);

        let signed_decoder = compile_postcard_decoder_via_structural_hir(<BigSigned>::SHAPE);
        let signed_bytes =
            postcard::to_allocvec(&signed).expect("postcard should encode signed 128-bit sample");
        let signed_value = crate::deserialize::<BigSigned>(&signed_decoder, &signed_bytes)
            .expect("structural HIR postcard decoder should decode i128 fields");
        assert_eq!(signed_value, signed);
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_option_u128_field() {
        let decoder = compile_postcard_decoder_via_structural_hir(<MaybeBigUnsigned>::SHAPE);
        let sample = MaybeBigUnsigned {
            value: Some((1_u128 << 72) | 0x55aa_33cc_77ee_u128),
        };
        let bytes = postcard::to_allocvec(&sample).expect("postcard should encode Option<u128>");
        let value = crate::deserialize::<MaybeBigUnsigned>(&decoder, &bytes)
            .expect("structural HIR postcard decoder should decode Option<u128>");
        assert_eq!(value, sample);
    }

    #[test]
    fn postcard_hir_models_unit_enums() {
        let module = build_postcard_decoder_hir(<UnitAnimal>::SHAPE);
        let (_, function) = module.functions.iter().next().unwrap();

        assert!(
            module
                .callable_named("postcard.read_discriminant")
                .is_none(),
            "unit enum lowering should not use postcard.read_discriminant"
        );

        assert!(
            function
                .locals
                .iter()
                .any(|local| local.ty == hir::Type::u(32)),
            "unit enum lowering should use a scalar discriminant local"
        );
        let (scrutinee, arms) = function
            .body
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Match { scrutinee, arms } => Some((scrutinee, arms)),
                _ => None,
            })
            .expect("expected enum match statement");
        let hir::Expr::Local(disc_local) = scrutinee else {
            panic!("expected discriminant local");
        };
        assert!(
            function
                .locals
                .iter()
                .any(|local| local.local == *disc_local),
            "match should scrutinee the decoded discriminant local"
        );
        assert_eq!(arms.len(), 3);
        assert!(matches!(arms[0].pattern, hir::Pattern::Integer(0)));
        assert!(matches!(arms[1].pattern, hir::Pattern::Integer(1)));
        assert!(matches!(arms[2].pattern, hir::Pattern::Integer(2)));

        let hir::StmtKind::Init { value, .. } = &arms[1].body.statements[0].kind else {
            panic!("expected unit variant init");
        };
        let hir::Expr::Variant {
            variant, fields, ..
        } = value
        else {
            panic!("expected unit variant expression");
        };
        assert_eq!(variant, "Dog");
        assert!(fields.is_empty());
    }

    #[test]
    fn postcard_hir_models_payload_enums() {
        let module = build_postcard_decoder_hir(<PayloadAnimal<'static>>::SHAPE);
        let (_, function) = module.functions.iter().next().unwrap();

        let arms = function
            .body
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Match { arms, .. } => Some(arms),
                _ => None,
            })
            .expect("expected enum match statement");
        assert_eq!(arms.len(), 3);

        let count_arm = &arms[1];
        let value = count_arm
            .body
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "Count") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected Count variant init");
        let hir::Expr::Variant {
            variant, fields, ..
        } = value
        else {
            panic!("expected Count variant expression");
        };
        assert_eq!(variant, "Count");
        assert_eq!(fields.len(), 1);

        let name_arm = &arms[2];
        assert!(name_arm.body.statements.len() >= 2);
        let value = name_arm
            .body
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "Name") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected Name variant init");
        let hir::Expr::Variant {
            variant, fields, ..
        } = value
        else {
            panic!("expected Name variant expression");
        };
        assert_eq!(variant, "Name");
        assert_eq!(fields.len(), 1);
        assert!(
            name_arm.body.statements.iter().any(|stmt| matches!(
                &stmt.kind,
                hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                    target: hir::CallTarget::Callable(callable_id),
                    ..
                })) if module.callables[*callable_id].name == "runtime.validate_utf8_range"
            )),
            "borrowed enum payload should validate UTF-8 explicitly"
        );
    }

    #[test]
    fn postcard_hir_models_arrays() {
        let module = build_postcard_decoder_hir(<BorrowedArrayHolder<'static>>::SHAPE);
        let (_, function) = module.functions.iter().next().unwrap();

        let array_inits = function
            .body
            .statements
            .iter()
            .filter_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init {
                    place: hir::Place::Index { .. },
                    value,
                } => Some(value),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(array_inits.len(), 2);
        for value in array_inits {
            assert!(matches!(value, hir::Expr::Str { .. } | hir::Expr::Local(_)));
        }
        assert!(
            module.callable_named("postcard.read_str").is_none(),
            "borrowed array lowering should not use postcard.read_str"
        );
    }

    #[test]
    fn postcard_hir_ir_path_decodes_unit_enums() {
        let decoder = crate::compile_postcard_decoder_via_hir(<UnitAnimal>::SHAPE);

        let cat = crate::deserialize::<UnitAnimal>(&decoder, &[0])
            .expect("HIR->RVSDG postcard decoder should decode Cat");
        assert_eq!(cat, UnitAnimal::Cat);

        let dog = crate::deserialize::<UnitAnimal>(&decoder, &[1])
            .expect("HIR->RVSDG postcard decoder should decode Dog");
        assert_eq!(dog, UnitAnimal::Dog);

        let parrot = crate::deserialize::<UnitAnimal>(&decoder, &[2])
            .expect("HIR->RVSDG postcard decoder should decode Parrot");
        assert_eq!(parrot, UnitAnimal::Parrot);
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_scalar_field() {
        let decoder = compile_postcard_decoder_via_structural_hir(<ScalarNumber>::SHAPE);

        let value = crate::deserialize::<ScalarNumber>(&decoder, &[42])
            .expect("structural HIR postcard decoder should decode a scalar field");
        assert_eq!(value, ScalarNumber { value: 42 });
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_borrowed_header() {
        let decoder = compile_postcard_decoder_via_structural_hir(<BorrowedHeader<'static>>::SHAPE);

        let value = crate::deserialize::<BorrowedHeader<'_>>(&decoder, &[7, 2, b'h', b'i'])
            .expect("structural HIR postcard decoder should decode direct borrowed fields");
        assert_eq!(value, BorrowedHeader { len: 7, name: "hi" });
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_owned_header() {
        let decoder = compile_postcard_decoder_via_structural_hir(<OwnedHeader>::SHAPE);

        let value = crate::deserialize::<OwnedHeader>(&decoder, &[7, 2, b'h', b'i'])
            .expect("structural HIR postcard decoder should decode direct owned string fields");
        assert_eq!(
            value,
            OwnedHeader {
                len: 7,
                name: "hi".to_owned(),
            }
        );
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_root_vec_u32() {
        let decoder = compile_postcard_decoder_via_structural_hir(<Vec<u32>>::SHAPE);

        let value = crate::deserialize::<Vec<u32>>(&decoder, &[3, 1, 2, 3])
            .expect("structural HIR postcard decoder should decode root Vec<u32>");
        assert_eq!(value, vec![1, 2, 3]);
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_root_vec_string() {
        let decoder = compile_postcard_decoder_via_structural_hir(<Vec<String>>::SHAPE);

        let value = crate::deserialize::<Vec<String>>(&decoder, &[2, 2, b'h', b'i', 2, b'b', b'y'])
            .expect("structural HIR postcard decoder should decode root Vec<String>");
        assert_eq!(value, vec!["hi".to_owned(), "by".to_owned()]);
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_root_vec_structs() {
        let decoder = compile_postcard_decoder_via_structural_hir(<Vec<OwnedAddress>>::SHAPE);

        let value = crate::deserialize::<Vec<OwnedAddress>>(
            &decoder,
            &[2, 2, b'P', b'A', 75, 2, b'L', b'Y', 13],
        )
        .expect("structural HIR postcard decoder should decode root Vec<struct>");
        assert_eq!(
            value,
            vec![
                OwnedAddress {
                    city: "PA".to_owned(),
                    zip: 75,
                },
                OwnedAddress {
                    city: "LY".to_owned(),
                    zip: 13,
                },
            ]
        );
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_unit_enums() {
        let decoder = compile_postcard_decoder_via_structural_hir(<UnitAnimal>::SHAPE);

        let cat = crate::deserialize::<UnitAnimal>(&decoder, &[0])
            .expect("structural HIR postcard decoder should decode Cat");
        assert_eq!(cat, UnitAnimal::Cat);

        let dog = crate::deserialize::<UnitAnimal>(&decoder, &[1])
            .expect("structural HIR postcard decoder should decode Dog");
        assert_eq!(dog, UnitAnimal::Dog);

        let parrot = crate::deserialize::<UnitAnimal>(&decoder, &[2])
            .expect("structural HIR postcard decoder should decode Parrot");
        assert_eq!(parrot, UnitAnimal::Parrot);
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_option_scalar_field() {
        let decoder = compile_postcard_decoder_via_structural_hir(<MaybeCount>::SHAPE);

        let some = crate::deserialize::<MaybeCount>(&decoder, &[1, 42])
            .expect("structural HIR postcard decoder should decode Some(u32)");
        assert_eq!(some, MaybeCount { count: Some(42) });

        let none = crate::deserialize::<MaybeCount>(&decoder, &[0])
            .expect("structural HIR postcard decoder should decode None");
        assert_eq!(none, MaybeCount { count: None });
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_option_borrowed_field() {
        let decoder =
            compile_postcard_decoder_via_structural_hir(<MaybeBorrowedName<'static>>::SHAPE);

        let some = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[1, 2, b'h', b'i'])
            .expect("structural HIR postcard decoder should decode Some(&str)");
        assert_eq!(some, MaybeBorrowedName { name: Some("hi") });

        let none = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[0])
            .expect("structural HIR postcard decoder should decode None");
        assert_eq!(none, MaybeBorrowedName { name: None });
    }

    #[test]
    fn structural_hir_ir_path_decodes_constant_output() {
        let mut module = hir::Module::new();
        let root_def = module.add_type_def(hir::TypeDef {
            name: <ConstantNumber>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "value".to_owned(),
                    ty: hir::Type::u(32),
                }],
            },
        });
        module.add_function(hir::Function {
            name: "const_number".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("constant structural HIR".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "value".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(42)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<ConstantNumber>::SHAPE, &module);
        let value = crate::deserialize::<ConstantNumber>(&decoder, &[])
            .expect("structural HIR decoder should ignore input and write a constant");
        assert_eq!(value, ConstantNumber { value: 42 });
    }

    #[test]
    fn structural_hir_ir_path_preserves_local_scalar_across_empty_else_if() {
        let mut module = hir::Module::new();
        let root_def = module.add_type_def(hir::TypeDef {
            name: <ConstantNumber>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "value".to_owned(),
                    ty: hir::Type::u(32),
                }],
            },
        });

        module.add_function(hir::Function {
            name: "local_across_if".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "tmp".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Temp,
            }],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("local scalar across empty else".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(2)),
                            value: hir::Expr::Literal(hir::Literal::Integer(2)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::If {
                            condition: hir::Expr::Literal(hir::Literal::Bool(false)),
                            then_block: hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements: vec![hir::Stmt {
                                    id: hir::StmtId::new(2),
                                    kind: hir::StmtKind::Fail {
                                        code: hir::ErrorCode::InvalidBool,
                                    },
                                }],
                            },
                            else_block: Some(hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements: Vec::new(),
                            }),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Assign {
                            place: hir::Place::Local(hir::LocalId::new(2)),
                            value: hir::Expr::Binary {
                                op: hir::BinaryOp::BitOr,
                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(4),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "value".to_owned(),
                            },
                            value: hir::Expr::Local(hir::LocalId::new(2)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(5),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<ConstantNumber>::SHAPE, &module);
        let value = crate::deserialize::<ConstantNumber>(&decoder, &[])
            .expect("structural HIR decoder should preserve local scalars across if");
        assert_eq!(value, ConstantNumber { value: 6 });
    }

    #[test]
    fn structural_hir_ir_path_executes_unrolled_varint_shape() {
        let mut module = hir::Module::new();
        let root_def = module.add_type_def(hir::TypeDef {
            name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "values".to_owned(),
                    ty: hir::Type::array(hir::Type::u(32), 4),
                }],
            },
        });

        let mut locals = Vec::new();
        let mut statements = Vec::new();
        let mut next_stmt = 0u32;
        let mut next_local = 2u32;
        let make_local = |locals: &mut Vec<hir::LocalDecl>,
                          next_local: &mut u32,
                          name: String,
                          ty: hir::Type|
         -> hir::LocalId {
            let local = hir::LocalId::new(*next_local);
            *next_local += 1;
            locals.push(hir::LocalDecl {
                local,
                name,
                ty,
                kind: hir::LocalKind::Temp,
            });
            local
        };

        for (index, raw_value) in [1_u64, 2, 3, 4].into_iter().enumerate() {
            let acc_local = make_local(
                &mut locals,
                &mut next_local,
                format!("acc_{index}"),
                hir::Type::u(64),
            );
            let raw_local = make_local(
                &mut locals,
                &mut next_local,
                format!("raw_{index}"),
                hir::Type::u(8),
            );
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::Init {
                    place: hir::Place::Local(acc_local),
                    value: hir::Expr::Literal(hir::Literal::Integer(0)),
                },
            });
            next_stmt += 1;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::Init {
                    place: hir::Place::Local(raw_local),
                    value: hir::Expr::Literal(hir::Literal::Integer(raw_value)),
                },
            });
            next_stmt += 1;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::Assign {
                    place: hir::Place::Local(acc_local),
                    value: hir::Expr::Binary {
                        op: hir::BinaryOp::BitOr,
                        lhs: Box::new(hir::Expr::Local(acc_local)),
                        rhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw_local)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7f))),
                        }),
                    },
                },
            });
            next_stmt += 1;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw_local)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x80))),
                        }),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(next_stmt + 1),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::InvalidVarint,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(next_stmt + 2),
                            kind: hir::StmtKind::Init {
                                place: hir::Place::Index {
                                    base: Box::new(hir::Place::Field {
                                        base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                        field: "values".to_owned(),
                                    }),
                                    index: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                        index as u64,
                                    ))),
                                },
                                value: hir::Expr::Local(acc_local),
                            },
                        }],
                    }),
                },
            });
            next_stmt += 3;
        }
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Return(None),
        });

        module.add_function(hir::Function {
            name: "unrolled_varint_shape".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals,
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("unrolled varint shape".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements,
            },
        });

        let decoder = compile_structural_hir_decoder(<ScalarArrayHolder>::SHAPE, &module);
        let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[])
            .expect("structural HIR decoder should execute unrolled varint shape");
        assert_eq!(
            value,
            ScalarArrayHolder {
                values: [1, 2, 3, 4]
            }
        );
    }

    #[test]
    fn structural_hir_ir_path_reads_bytes_via_cursor_shadow() {
        let mut module = hir::Module::new();
        let input_region = module.add_region("input");
        let cursor_def = module.add_type_def(hir::TypeDef {
            name: "Cursor".to_owned(),
            generic_params: vec![hir::GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "bytes".to_owned(),
                        ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    },
                    hir::FieldDef {
                        name: "pos".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });
        let root_def = module.add_type_def(hir::TypeDef {
            name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "values".to_owned(),
                    ty: hir::Type::array(hir::Type::u(32), 4),
                }],
            },
        });

        let mut statements = Vec::new();
        let mut next_stmt = 0u32;
        for index in 0..4_u64 {
            let pos_expr = hir::Expr::Field {
                base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                field: "pos".to_owned(),
            };
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Gt,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(pos_expr.clone()),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                        }),
                        rhs: Box::new(hir::Expr::SliceLen {
                            value: Box::new(hir::Expr::Field {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                                field: "bytes".to_owned(),
                            }),
                        }),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(next_stmt + 1),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::UnexpectedEof,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: Vec::new(),
                    }),
                },
            });
            next_stmt += 2;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::Init {
                    place: hir::Place::Index {
                        base: Box::new(hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "values".to_owned(),
                        }),
                        index: Box::new(hir::Expr::Literal(hir::Literal::Integer(index))),
                    },
                    value: hir::Expr::Load {
                        addr: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(hir::Expr::SliceData {
                                value: Box::new(hir::Expr::Field {
                                    base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                                    field: "bytes".to_owned(),
                                }),
                            }),
                            rhs: Box::new(pos_expr.clone()),
                        }),
                        width: hir::MemoryWidth::W1,
                    },
                },
            });
            next_stmt += 1;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::Assign {
                    place: hir::Place::Field {
                        base: Box::new(hir::Place::Local(hir::LocalId::new(0))),
                        field: "pos".to_owned(),
                    },
                    value: hir::Expr::Binary {
                        op: hir::BinaryOp::Add,
                        lhs: Box::new(pos_expr),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                    },
                },
            });
            next_stmt += 1;
        }
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Return(None),
        });

        module.add_function(hir::Function {
            name: "cursor_shadow_reads".to_owned(),
            region_params: vec![input_region],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::named(cursor_def, vec![hir::GenericArg::Region(input_region)]),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("cursor shadow byte loads".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements,
            },
        });

        let decoder = compile_structural_hir_decoder(<ScalarArrayHolder>::SHAPE, &module);
        let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
            .expect("structural HIR decoder should read bytes via the cursor shadow");
        assert_eq!(
            value,
            ScalarArrayHolder {
                values: [1, 2, 3, 4]
            }
        );
    }

    #[test]
    fn structural_hir_ir_path_parses_json_bool_tokens() {
        let module = build_structural_json_bool_module();
        let decoder = compile_structural_hir_decoder(<BoolHeader>::SHAPE, &module);

        let t = crate::deserialize::<BoolHeader>(&decoder, br#"true"#)
            .expect("json bool kernel should parse true");
        assert_eq!(t, BoolHeader { value: true });

        let f = crate::deserialize::<BoolHeader>(&decoder, b" \n\tfalse")
            .expect("json bool kernel should skip leading whitespace and parse false");
        assert_eq!(f, BoolHeader { value: false });
    }

    #[test]
    fn structural_hir_ir_path_rejects_invalid_json_bool_tokens() {
        let module = build_structural_json_bool_module();
        let decoder = compile_structural_hir_decoder(<BoolHeader>::SHAPE, &module);

        let err = crate::deserialize::<BoolHeader>(&decoder, br#"trux"#)
            .expect_err("json bool kernel should reject invalid bool tokens");
        assert_eq!(err.code, crate::context::ErrorCode::InvalidBool);

        let err = crate::deserialize::<BoolHeader>(&decoder, b"   ")
            .expect_err("json bool kernel should reject whitespace-only input");
        assert_eq!(err.code, crate::context::ErrorCode::UnexpectedEof);
    }

    #[test]
    fn json_structural_hir_ir_path_decodes_root_bool() {
        let decoder = compile_json_decoder_via_structural_hir(<bool>::SHAPE);

        let t = crate::deserialize::<bool>(&decoder, br#"true"#)
            .expect("shape-driven JSON HIR should parse true");
        assert!(t);

        let f = crate::deserialize::<bool>(&decoder, b"\r\n false")
            .expect("shape-driven JSON HIR should skip leading whitespace and parse false");
        assert!(!f);
    }

    #[test]
    fn json_structural_hir_ir_path_decodes_root_u32() {
        let decoder = compile_json_decoder_via_structural_hir(<u32>::SHAPE);

        let zero = crate::deserialize::<u32>(&decoder, b"0")
            .expect("shape-driven JSON HIR should parse zero");
        assert_eq!(zero, 0);

        let forty_two = crate::deserialize::<u32>(&decoder, b" \n42")
            .expect("shape-driven JSON HIR should skip leading whitespace and parse digits");
        assert_eq!(forty_two, 42);

        let err = crate::deserialize::<u32>(&decoder, b"")
            .expect_err("shape-driven JSON HIR should reject empty input");
        assert_eq!(err.code, crate::context::ErrorCode::UnexpectedEof);

        let err = crate::deserialize::<u32>(&decoder, b"abc")
            .expect_err("shape-driven JSON HIR should reject non-digit input");
        assert_eq!(err.code, crate::context::ErrorCode::InvalidJsonNumber);

        let err = crate::deserialize::<u32>(&decoder, b"4294967296")
            .expect_err("shape-driven JSON HIR should reject out-of-range u32 values");
        assert_eq!(err.code, crate::context::ErrorCode::NumberOutOfRange);
    }

    #[test]
    fn structural_hir_ir_path_executes_cursor_shadow_varint_array() {
        let mut module = hir::Module::new();
        let input_region = module.add_region("input");
        let cursor_def = module.add_type_def(hir::TypeDef {
            name: "Cursor".to_owned(),
            generic_params: vec![hir::GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "bytes".to_owned(),
                        ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    },
                    hir::FieldDef {
                        name: "pos".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });
        let root_def = module.add_type_def(hir::TypeDef {
            name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "values".to_owned(),
                    ty: hir::Type::array(hir::Type::u(32), 4),
                }],
            },
        });

        let mut locals = Vec::new();
        let mut statements = Vec::new();
        let mut next_stmt = 0u32;
        let mut next_local = 2u32;
        let make_local = |locals: &mut Vec<hir::LocalDecl>,
                          next_local: &mut u32,
                          name: String,
                          ty: hir::Type|
         -> hir::LocalId {
            let local = hir::LocalId::new(*next_local);
            *next_local += 1;
            locals.push(hir::LocalDecl {
                local,
                name,
                ty,
                kind: hir::LocalKind::Temp,
            });
            local
        };

        for index in 0..4_u64 {
            let acc_local = make_local(
                &mut locals,
                &mut next_local,
                format!("acc_{index}"),
                hir::Type::u(64),
            );
            let raw_local = make_local(
                &mut locals,
                &mut next_local,
                format!("raw_{index}"),
                hir::Type::u(8),
            );
            let pos_expr = hir::Expr::Field {
                base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                field: "pos".to_owned(),
            };
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Gt,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(pos_expr.clone()),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                        }),
                        rhs: Box::new(hir::Expr::SliceLen {
                            value: Box::new(hir::Expr::Field {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                                field: "bytes".to_owned(),
                            }),
                        }),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(next_stmt + 1),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::UnexpectedEof,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: Vec::new(),
                    }),
                },
            });
            next_stmt += 2;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::Init {
                    place: hir::Place::Local(raw_local),
                    value: hir::Expr::Load {
                        addr: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(hir::Expr::SliceData {
                                value: Box::new(hir::Expr::Field {
                                    base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                                    field: "bytes".to_owned(),
                                }),
                            }),
                            rhs: Box::new(pos_expr.clone()),
                        }),
                        width: hir::MemoryWidth::W1,
                    },
                },
            });
            next_stmt += 1;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::Assign {
                    place: hir::Place::Field {
                        base: Box::new(hir::Place::Local(hir::LocalId::new(0))),
                        field: "pos".to_owned(),
                    },
                    value: hir::Expr::Binary {
                        op: hir::BinaryOp::Add,
                        lhs: Box::new(pos_expr),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                    },
                },
            });
            next_stmt += 1;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::Init {
                    place: hir::Place::Local(acc_local),
                    value: hir::Expr::Binary {
                        op: hir::BinaryOp::BitAnd,
                        lhs: Box::new(hir::Expr::Local(raw_local)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7f))),
                    },
                },
            });
            next_stmt += 1;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw_local)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x80))),
                        }),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(next_stmt + 1),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::InvalidVarint,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(next_stmt + 2),
                            kind: hir::StmtKind::Init {
                                place: hir::Place::Index {
                                    base: Box::new(hir::Place::Field {
                                        base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                        field: "values".to_owned(),
                                    }),
                                    index: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                        index,
                                    ))),
                                },
                                value: hir::Expr::Local(acc_local),
                            },
                        }],
                    }),
                },
            });
            next_stmt += 3;
        }

        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Return(None),
        });

        module.add_function(hir::Function {
            name: "cursor_shadow_varint_array".to_owned(),
            region_params: vec![input_region],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::named(cursor_def, vec![hir::GenericArg::Region(input_region)]),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals,
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("cursor shadow varint array".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements,
            },
        });

        let decoder = compile_structural_hir_decoder(<ScalarArrayHolder>::SHAPE, &module);
        let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
            .expect("structural HIR decoder should execute cursor shadow varint array");
        assert_eq!(
            value,
            ScalarArrayHolder {
                values: [1, 2, 3, 4]
            }
        );
    }

    #[test]
    fn structural_hir_ir_path_executes_cursor_shadow_range_checked_varint_array() {
        let mut module = hir::Module::new();
        let input_region = module.add_region("input");
        let cursor_def = module.add_type_def(hir::TypeDef {
            name: "Cursor".to_owned(),
            generic_params: vec![hir::GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "bytes".to_owned(),
                        ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    },
                    hir::FieldDef {
                        name: "pos".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });
        let root_def = module.add_type_def(hir::TypeDef {
            name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "values".to_owned(),
                    ty: hir::Type::array(hir::Type::u(32), 4),
                }],
            },
        });

        let mut locals = Vec::new();
        let mut statements = Vec::new();
        let mut next_stmt = 0u32;
        let mut next_local = 2u32;
        let make_local = |locals: &mut Vec<hir::LocalDecl>,
                          next_local: &mut u32,
                          name: String,
                          ty: hir::Type|
         -> hir::LocalId {
            let local = hir::LocalId::new(*next_local);
            *next_local += 1;
            locals.push(hir::LocalDecl {
                local,
                name,
                ty,
                kind: hir::LocalKind::Temp,
            });
            local
        };

        for index in 0..4_u64 {
            let acc_local = make_local(
                &mut locals,
                &mut next_local,
                format!("acc_{index}"),
                hir::Type::u(64),
            );
            let raw_local = make_local(
                &mut locals,
                &mut next_local,
                format!("raw_{index}"),
                hir::Type::u(8),
            );
            let pos_expr = hir::Expr::Field {
                base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                field: "pos".to_owned(),
            };
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Gt,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(pos_expr.clone()),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                        }),
                        rhs: Box::new(hir::Expr::SliceLen {
                            value: Box::new(hir::Expr::Field {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                                field: "bytes".to_owned(),
                            }),
                        }),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(next_stmt + 1),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::UnexpectedEof,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: Vec::new(),
                    }),
                },
            });
            next_stmt += 2;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::Init {
                    place: hir::Place::Local(raw_local),
                    value: hir::Expr::Load {
                        addr: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(hir::Expr::SliceData {
                                value: Box::new(hir::Expr::Field {
                                    base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                                    field: "bytes".to_owned(),
                                }),
                            }),
                            rhs: Box::new(pos_expr.clone()),
                        }),
                        width: hir::MemoryWidth::W1,
                    },
                },
            });
            next_stmt += 1;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::Assign {
                    place: hir::Place::Field {
                        base: Box::new(hir::Place::Local(hir::LocalId::new(0))),
                        field: "pos".to_owned(),
                    },
                    value: hir::Expr::Binary {
                        op: hir::BinaryOp::Add,
                        lhs: Box::new(pos_expr),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                    },
                },
            });
            next_stmt += 1;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::Init {
                    place: hir::Place::Local(acc_local),
                    value: hir::Expr::Binary {
                        op: hir::BinaryOp::BitAnd,
                        lhs: Box::new(hir::Expr::Local(raw_local)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7f))),
                    },
                },
            });
            next_stmt += 1;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(next_stmt),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw_local)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x80))),
                        }),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(next_stmt + 1),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::InvalidVarint,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(next_stmt + 2),
                            kind: hir::StmtKind::If {
                                condition: hir::Expr::Binary {
                                    op: hir::BinaryOp::Ne,
                                    lhs: Box::new(hir::Expr::Binary {
                                        op: hir::BinaryOp::Shr,
                                        lhs: Box::new(hir::Expr::Local(acc_local)),
                                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                            32,
                                        ))),
                                    }),
                                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                                },
                                then_block: hir::Block {
                                    scope: hir::ScopeId::new(0),
                                    statements: vec![hir::Stmt {
                                        id: hir::StmtId::new(next_stmt + 3),
                                        kind: hir::StmtKind::Fail {
                                            code: hir::ErrorCode::NumberOutOfRange,
                                        },
                                    }],
                                },
                                else_block: Some(hir::Block {
                                    scope: hir::ScopeId::new(0),
                                    statements: vec![hir::Stmt {
                                        id: hir::StmtId::new(next_stmt + 4),
                                        kind: hir::StmtKind::Init {
                                            place: hir::Place::Index {
                                                base: Box::new(hir::Place::Field {
                                                    base: Box::new(hir::Place::Local(
                                                        hir::LocalId::new(1),
                                                    )),
                                                    field: "values".to_owned(),
                                                }),
                                                index: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(index),
                                                )),
                                            },
                                            value: hir::Expr::Local(acc_local),
                                        },
                                    }],
                                }),
                            },
                        }],
                    }),
                },
            });
            next_stmt += 5;
        }

        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Return(None),
        });

        module.add_function(hir::Function {
            name: "cursor_shadow_range_checked_varint_array".to_owned(),
            region_params: vec![input_region],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::named(cursor_def, vec![hir::GenericArg::Region(input_region)]),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals,
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("cursor shadow range checked varint array".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements,
            },
        });

        let decoder = compile_structural_hir_decoder(<ScalarArrayHolder>::SHAPE, &module);
        let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4]).expect(
            "structural HIR decoder should execute range checked cursor shadow varint array",
        );
        assert_eq!(
            value,
            ScalarArrayHolder {
                values: [1, 2, 3, 4]
            }
        );
    }

    #[test]
    fn structural_hir_ir_path_executes_exact_postcard_varint_array_shape() {
        let mut module = hir::Module::new();
        let input_region = module.add_region("input");
        let cursor_def = module.add_type_def(hir::TypeDef {
            name: "Cursor".to_owned(),
            generic_params: vec![hir::GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "bytes".to_owned(),
                        ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    },
                    hir::FieldDef {
                        name: "pos".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });
        let root_def = module.add_type_def(hir::TypeDef {
            name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "values".to_owned(),
                    ty: hir::Type::array(hir::Type::u(32), 4),
                }],
            },
        });

        let mut lowerer = PostcardHirLowerer::new();
        let cursor_local = hir::LocalId::new(0);
        let out_local = hir::LocalId::new(1);
        let mut statements = Vec::new();
        let (fields, _) = collect_fields(<ScalarArrayHolder>::SHAPE);
        lowerer.lower_shape_into_place(
            &mut statements,
            cursor_local,
            hir::Place::Field {
                base: Box::new(hir::Place::Local(out_local)),
                field: "values".to_owned(),
            },
            fields[0].shape,
        );
        statements.push(hir::Stmt {
            id: lowerer.next_stmt_id(),
            kind: hir::StmtKind::Return(None),
        });

        module.add_function(hir::Function {
            name: "exact_postcard_varint_array_shape".to_owned(),
            region_params: vec![input_region],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: cursor_local,
                    name: "cursor".to_owned(),
                    ty: hir::Type::named(cursor_def, vec![hir::GenericArg::Region(input_region)]),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: out_local,
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: lowerer.locals,
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("exact postcard varint array shape".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements,
            },
        });

        let decoder = compile_structural_hir_decoder(<ScalarArrayHolder>::SHAPE, &module);
        let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
            .expect("structural HIR decoder should execute exact postcard array shape");
        assert_eq!(
            value,
            ScalarArrayHolder {
                values: [1, 2, 3, 4]
            }
        );
    }

    #[test]
    fn structural_hir_ir_path_preserves_temp_after_cursor_sync() {
        let mut module = hir::Module::new();
        let input_region = module.add_region("input");
        let cursor_def = module.add_type_def(hir::TypeDef {
            name: "Cursor".to_owned(),
            generic_params: vec![hir::GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "bytes".to_owned(),
                        ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    },
                    hir::FieldDef {
                        name: "pos".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });
        let root_def = module.add_type_def(hir::TypeDef {
            name: <ConstantNumber>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "value".to_owned(),
                    ty: hir::Type::u(32),
                }],
            },
        });

        module.add_function(hir::Function {
            name: "temp_after_cursor_sync".to_owned(),
            region_params: vec![input_region],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::named(cursor_def, vec![hir::GenericArg::Region(input_region)]),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "raw".to_owned(),
                ty: hir::Type::u(8),
                kind: hir::LocalKind::Temp,
            }],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("temp survives cursor sync".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(2)),
                            value: hir::Expr::Load {
                                addr: Box::new(hir::Expr::SliceData {
                                    value: Box::new(hir::Expr::Field {
                                        base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                                        field: "bytes".to_owned(),
                                    }),
                                }),
                                width: hir::MemoryWidth::W1,
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Assign {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(0))),
                                field: "pos".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(1)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "value".to_owned(),
                            },
                            value: hir::Expr::Local(hir::LocalId::new(2)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<ConstantNumber>::SHAPE, &module);
        let value = crate::deserialize::<ConstantNumber>(&decoder, &[42])
            .expect("temp should survive cursor sync");
        assert_eq!(value, ConstantNumber { value: 42 });
    }

    #[test]
    fn structural_hir_ir_path_builds_persistent_buffer_kernel() {
        let mut module = hir::Module::new();
        let root_def = module.add_type_def(hir::TypeDef {
            name: <PersistentBufferSummary>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "ptr".to_owned(),
                        ty: hir::Type::u(64),
                    },
                    hir::FieldDef {
                        name: "len".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });
        let runtime = module.install_runtime_memory_callables();
        module.add_function(hir::Function {
            name: "build_persistent_buffer".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![
                hir::LocalDecl {
                    local: hir::LocalId::new(2),
                    name: "len".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Temp,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(3),
                    name: "bytes".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Temp,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(4),
                    name: "ptr".to_owned(),
                    ty: hir::Type::persistent_addr(),
                    kind: hir::LocalKind::Temp,
                },
            ],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("persistent buffer kernel".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(2)),
                            value: hir::Expr::Literal(hir::Literal::Integer(2)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(3)),
                            value: hir::Expr::Binary {
                                op: hir::BinaryOp::Mul,
                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(4)),
                            value: hir::Expr::Call(hir::CallExpr {
                                target: hir::CallTarget::Callable(runtime.alloc_persistent),
                                args: vec![
                                    hir::Expr::Local(hir::LocalId::new(3)),
                                    hir::Expr::Literal(hir::Literal::Integer(4)),
                                ],
                            }),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Store {
                            addr: hir::Expr::Local(hir::LocalId::new(4)),
                            width: hir::MemoryWidth::W4,
                            value: hir::Expr::Literal(hir::Literal::Integer(10)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(4),
                        kind: hir::StmtKind::Store {
                            addr: hir::Expr::Binary {
                                op: hir::BinaryOp::Add,
                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(4))),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                            },
                            width: hir::MemoryWidth::W4,
                            value: hir::Expr::Literal(hir::Literal::Integer(20)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(5),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "ptr".to_owned(),
                            },
                            value: hir::Expr::Local(hir::LocalId::new(4)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(6),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "len".to_owned(),
                            },
                            value: hir::Expr::Local(hir::LocalId::new(2)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(7),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<PersistentBufferSummary>::SHAPE, &module);
        let value = crate::deserialize::<PersistentBufferSummary>(&decoder, &[])
            .expect("structural HIR decoder should build a persistent buffer kernel");
        assert_eq!(value.len, 2);
        assert_ne!(value.ptr, 0);

        let ptr = value.ptr as *const u32;
        let words = unsafe { std::slice::from_raw_parts(ptr, value.len) };
        assert_eq!(words, &[10, 20]);

        let layout = std::alloc::Layout::from_size_align(8, 4).unwrap();
        unsafe { std::alloc::dealloc(value.ptr as *mut u8, layout) };
    }

    #[test]
    fn structural_hir_ir_path_materializes_vec_from_raw_parts() {
        let mut module = hir::Module::new();
        let root_def = module.add_type_def(hir::TypeDef {
            name: <VecHolder>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "values".to_owned(),
                    ty: hir::Type::u(64),
                }],
            },
        });
        let runtime = module.install_runtime_memory_callables();
        module.add_function(hir::Function {
            name: "build_vec_holder".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![
                hir::LocalDecl {
                    local: hir::LocalId::new(2),
                    name: "len".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Temp,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(3),
                    name: "bytes".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Temp,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(4),
                    name: "ptr".to_owned(),
                    ty: hir::Type::persistent_addr(),
                    kind: hir::LocalKind::Temp,
                },
            ],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("vec materialization kernel".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(2)),
                            value: hir::Expr::Literal(hir::Literal::Integer(2)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(3)),
                            value: hir::Expr::Binary {
                                op: hir::BinaryOp::Mul,
                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(4)),
                            value: hir::Expr::Call(hir::CallExpr {
                                target: hir::CallTarget::Callable(runtime.alloc_persistent),
                                args: vec![
                                    hir::Expr::Local(hir::LocalId::new(3)),
                                    hir::Expr::Literal(hir::Literal::Integer(4)),
                                ],
                            }),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Store {
                            addr: hir::Expr::Local(hir::LocalId::new(4)),
                            width: hir::MemoryWidth::W4,
                            value: hir::Expr::Literal(hir::Literal::Integer(10)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(4),
                        kind: hir::StmtKind::Store {
                            addr: hir::Expr::Binary {
                                op: hir::BinaryOp::Add,
                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(4))),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                            },
                            width: hir::MemoryWidth::W4,
                            value: hir::Expr::Literal(hir::Literal::Integer(20)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(5),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "values".to_owned(),
                            },
                            value: hir::Expr::Call(hir::CallExpr {
                                target: hir::CallTarget::Callable(runtime.vec_from_raw_parts),
                                args: vec![
                                    hir::Expr::Local(hir::LocalId::new(4)),
                                    hir::Expr::Local(hir::LocalId::new(2)),
                                    hir::Expr::Local(hir::LocalId::new(2)),
                                    hir::Expr::Literal(hir::Literal::Integer(4)),
                                ],
                            }),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(6),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<VecHolder>::SHAPE, &module);
        let value = crate::deserialize::<VecHolder>(&decoder, &[])
            .expect("structural HIR decoder should materialize a Vec from raw parts");
        assert_eq!(value.values, vec![10, 20]);
    }

    #[test]
    fn structural_hir_ir_path_materializes_root_vec_from_raw_parts() {
        let mut module = hir::Module::new();
        let runtime = module.install_runtime_memory_callables();
        module.add_function(hir::Function {
            name: "build_root_vec".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![
                hir::LocalDecl {
                    local: hir::LocalId::new(2),
                    name: "len".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Temp,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(3),
                    name: "bytes".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Temp,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(4),
                    name: "ptr".to_owned(),
                    ty: hir::Type::persistent_addr(),
                    kind: hir::LocalKind::Temp,
                },
            ],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("root vec materialization kernel".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(2)),
                            value: hir::Expr::Literal(hir::Literal::Integer(2)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(3)),
                            value: hir::Expr::Binary {
                                op: hir::BinaryOp::Mul,
                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(4)),
                            value: hir::Expr::Call(hir::CallExpr {
                                target: hir::CallTarget::Callable(runtime.alloc_persistent),
                                args: vec![
                                    hir::Expr::Local(hir::LocalId::new(3)),
                                    hir::Expr::Literal(hir::Literal::Integer(4)),
                                ],
                            }),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Store {
                            addr: hir::Expr::Local(hir::LocalId::new(4)),
                            width: hir::MemoryWidth::W4,
                            value: hir::Expr::Literal(hir::Literal::Integer(10)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(4),
                        kind: hir::StmtKind::Store {
                            addr: hir::Expr::Binary {
                                op: hir::BinaryOp::Add,
                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(4))),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                            },
                            width: hir::MemoryWidth::W4,
                            value: hir::Expr::Literal(hir::Literal::Integer(20)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(5),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(1)),
                            value: hir::Expr::Call(hir::CallExpr {
                                target: hir::CallTarget::Callable(runtime.vec_from_raw_parts),
                                args: vec![
                                    hir::Expr::Local(hir::LocalId::new(4)),
                                    hir::Expr::Local(hir::LocalId::new(2)),
                                    hir::Expr::Local(hir::LocalId::new(2)),
                                    hir::Expr::Literal(hir::Literal::Integer(4)),
                                ],
                            }),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(6),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<Vec<u32>>::SHAPE, &module);
        let value = crate::deserialize::<Vec<u32>>(&decoder, &[])
            .expect("structural HIR decoder should materialize a root Vec from raw parts");
        assert_eq!(value, vec![10, 20]);
    }

    #[test]
    fn structural_hir_ir_path_materializes_empty_vec_from_raw_parts() {
        let mut module = hir::Module::new();
        let root_def = module.add_type_def(hir::TypeDef {
            name: <VecHolder>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "values".to_owned(),
                    ty: hir::Type::u(64),
                }],
            },
        });
        let runtime = module.install_runtime_memory_callables();
        module.add_function(hir::Function {
            name: "build_empty_vec_holder".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("empty vec materialization kernel".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "values".to_owned(),
                            },
                            value: hir::Expr::Call(hir::CallExpr {
                                target: hir::CallTarget::Callable(runtime.vec_from_raw_parts),
                                args: vec![
                                    hir::Expr::Literal(hir::Literal::Integer(0)),
                                    hir::Expr::Literal(hir::Literal::Integer(0)),
                                    hir::Expr::Literal(hir::Literal::Integer(0)),
                                    hir::Expr::Literal(hir::Literal::Integer(4)),
                                ],
                            }),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<VecHolder>::SHAPE, &module);
        let value = crate::deserialize::<VecHolder>(&decoder, &[])
            .expect("structural HIR decoder should materialize an empty Vec from raw parts");
        assert!(value.values.is_empty());
    }

    #[test]
    fn structural_hir_ir_path_loads_from_persistent_buffer() {
        let mut module = hir::Module::new();
        let root_def = module.add_type_def(hir::TypeDef {
            name: <ScalarNumber>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "value".to_owned(),
                    ty: hir::Type::u(32),
                }],
            },
        });
        let runtime = module.install_runtime_memory_callables();
        module.add_function(hir::Function {
            name: "load_persistent_word".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "ptr".to_owned(),
                ty: hir::Type::persistent_addr(),
                kind: hir::LocalKind::Temp,
            }],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("persistent buffer load kernel".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(2)),
                            value: hir::Expr::Call(hir::CallExpr {
                                target: hir::CallTarget::Callable(runtime.alloc_persistent),
                                args: vec![
                                    hir::Expr::Literal(hir::Literal::Integer(4)),
                                    hir::Expr::Literal(hir::Literal::Integer(4)),
                                ],
                            }),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Store {
                            addr: hir::Expr::Local(hir::LocalId::new(2)),
                            width: hir::MemoryWidth::W4,
                            value: hir::Expr::Literal(hir::Literal::Integer(0x1234)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "value".to_owned(),
                            },
                            value: hir::Expr::Load {
                                addr: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                width: hir::MemoryWidth::W4,
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<ScalarNumber>::SHAPE, &module);
        let value = crate::deserialize::<ScalarNumber>(&decoder, &[])
            .expect("structural HIR decoder should load from persistent buffer");
        assert_eq!(value, ScalarNumber { value: 0x1234 });
    }

    #[test]
    fn structural_hir_ir_path_executes_loop_break_and_continue() {
        let mut module = hir::Module::new();
        let root_def = module.add_type_def(hir::TypeDef {
            name: <ConstantNumber>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "value".to_owned(),
                    ty: hir::Type::u(32),
                }],
            },
        });

        module.add_function(hir::Function {
            name: "loop_break_continue".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![
                hir::LocalDecl {
                    local: hir::LocalId::new(2),
                    name: "i".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Temp,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(3),
                    name: "sum".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Temp,
                },
            ],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("loop break/continue kernel".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(2)),
                            value: hir::Expr::Literal(hir::Literal::Integer(0)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(3)),
                            value: hir::Expr::Literal(hir::Literal::Integer(0)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Loop {
                            body: hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements: vec![
                                    hir::Stmt {
                                        id: hir::StmtId::new(3),
                                        kind: hir::StmtKind::If {
                                            condition: hir::Expr::Binary {
                                                op: hir::BinaryOp::Eq,
                                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(
                                                    2,
                                                ))),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(5),
                                                )),
                                            },
                                            then_block: hir::Block {
                                                scope: hir::ScopeId::new(0),
                                                statements: vec![hir::Stmt {
                                                    id: hir::StmtId::new(4),
                                                    kind: hir::StmtKind::Break,
                                                }],
                                            },
                                            else_block: Some(hir::Block {
                                                scope: hir::ScopeId::new(0),
                                                statements: vec![],
                                            }),
                                        },
                                    },
                                    hir::Stmt {
                                        id: hir::StmtId::new(5),
                                        kind: hir::StmtKind::Assign {
                                            place: hir::Place::Local(hir::LocalId::new(2)),
                                            value: hir::Expr::Binary {
                                                op: hir::BinaryOp::Add,
                                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(
                                                    2,
                                                ))),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(1),
                                                )),
                                            },
                                        },
                                    },
                                    hir::Stmt {
                                        id: hir::StmtId::new(6),
                                        kind: hir::StmtKind::If {
                                            condition: hir::Expr::Binary {
                                                op: hir::BinaryOp::Eq,
                                                lhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::BitAnd,
                                                    lhs: Box::new(hir::Expr::Local(
                                                        hir::LocalId::new(2),
                                                    )),
                                                    rhs: Box::new(hir::Expr::Literal(
                                                        hir::Literal::Integer(1),
                                                    )),
                                                }),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(0),
                                                )),
                                            },
                                            then_block: hir::Block {
                                                scope: hir::ScopeId::new(0),
                                                statements: vec![hir::Stmt {
                                                    id: hir::StmtId::new(7),
                                                    kind: hir::StmtKind::Continue,
                                                }],
                                            },
                                            else_block: Some(hir::Block {
                                                scope: hir::ScopeId::new(0),
                                                statements: vec![],
                                            }),
                                        },
                                    },
                                    hir::Stmt {
                                        id: hir::StmtId::new(8),
                                        kind: hir::StmtKind::Assign {
                                            place: hir::Place::Local(hir::LocalId::new(3)),
                                            value: hir::Expr::Binary {
                                                op: hir::BinaryOp::Add,
                                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(
                                                    3,
                                                ))),
                                                rhs: Box::new(hir::Expr::Local(hir::LocalId::new(
                                                    2,
                                                ))),
                                            },
                                        },
                                    },
                                ],
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(9),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "value".to_owned(),
                            },
                            value: hir::Expr::Local(hir::LocalId::new(3)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(10),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<ConstantNumber>::SHAPE, &module);
        let value = crate::deserialize::<ConstantNumber>(&decoder, &[])
            .expect("structural HIR decoder should execute loops with break/continue");
        assert_eq!(value, ConstantNumber { value: 9 });
    }

    #[test]
    fn structural_hir_ir_path_decodes_if_and_match() {
        let mut module = hir::Module::new();
        let animal_def = module.add_type_def(hir::TypeDef {
            name: <UnitAnimal>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Enum {
                variants: vec![
                    hir::VariantDef {
                        name: "Cat".to_owned(),
                        fields: vec![],
                    },
                    hir::VariantDef {
                        name: "Dog".to_owned(),
                        fields: vec![],
                    },
                    hir::VariantDef {
                        name: "Parrot".to_owned(),
                        fields: vec![],
                    },
                ],
            },
        });
        let root_def = module.add_type_def(hir::TypeDef {
            name: <BranchyAnimal>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "animal".to_owned(),
                        ty: hir::Type::named(animal_def, Vec::new()),
                    },
                    hir::FieldDef {
                        name: "value".to_owned(),
                        ty: hir::Type::u(32),
                    },
                ],
            },
        });

        module.add_function(hir::Function {
            name: "branchy_animal".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![
                hir::LocalDecl {
                    local: hir::LocalId::new(2),
                    name: "flag".to_owned(),
                    ty: hir::Type::bool(),
                    kind: hir::LocalKind::Let,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(3),
                    name: "tag".to_owned(),
                    ty: hir::Type::u(32),
                    kind: hir::LocalKind::Let,
                },
            ],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("structural if/match HIR".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(2)),
                            value: hir::Expr::Literal(hir::Literal::Bool(true)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::If {
                            condition: hir::Expr::Local(hir::LocalId::new(2)),
                            then_block: hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements: vec![hir::Stmt {
                                    id: hir::StmtId::new(2),
                                    kind: hir::StmtKind::Init {
                                        place: hir::Place::Field {
                                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                            field: "animal".to_owned(),
                                        },
                                        value: hir::Expr::Variant {
                                            def: animal_def,
                                            variant: "Dog".to_owned(),
                                            fields: vec![],
                                        },
                                    },
                                }],
                            },
                            else_block: Some(hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements: vec![hir::Stmt {
                                    id: hir::StmtId::new(3),
                                    kind: hir::StmtKind::Init {
                                        place: hir::Place::Field {
                                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                            field: "animal".to_owned(),
                                        },
                                        value: hir::Expr::Variant {
                                            def: animal_def,
                                            variant: "Cat".to_owned(),
                                            fields: vec![],
                                        },
                                    },
                                }],
                            }),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(4),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(3)),
                            value: hir::Expr::Literal(hir::Literal::Integer(1)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(5),
                        kind: hir::StmtKind::Match {
                            scrutinee: hir::Expr::Local(hir::LocalId::new(3)),
                            arms: vec![
                                hir::MatchArm {
                                    pattern: hir::Pattern::Integer(0),
                                    body: hir::Block {
                                        scope: hir::ScopeId::new(0),
                                        statements: vec![hir::Stmt {
                                            id: hir::StmtId::new(6),
                                            kind: hir::StmtKind::Init {
                                                place: hir::Place::Field {
                                                    base: Box::new(hir::Place::Local(
                                                        hir::LocalId::new(1),
                                                    )),
                                                    field: "value".to_owned(),
                                                },
                                                value: hir::Expr::Literal(hir::Literal::Integer(7)),
                                            },
                                        }],
                                    },
                                },
                                hir::MatchArm {
                                    pattern: hir::Pattern::Integer(1),
                                    body: hir::Block {
                                        scope: hir::ScopeId::new(0),
                                        statements: vec![hir::Stmt {
                                            id: hir::StmtId::new(7),
                                            kind: hir::StmtKind::Init {
                                                place: hir::Place::Field {
                                                    base: Box::new(hir::Place::Local(
                                                        hir::LocalId::new(1),
                                                    )),
                                                    field: "value".to_owned(),
                                                },
                                                value: hir::Expr::Literal(hir::Literal::Integer(
                                                    42,
                                                )),
                                            },
                                        }],
                                    },
                                },
                            ],
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(8),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<BranchyAnimal>::SHAPE, &module);
        let value = crate::deserialize::<BranchyAnimal>(&decoder, &[])
            .expect("structural HIR decoder should lower if+match");
        assert_eq!(
            value,
            BranchyAnimal {
                animal: UnitAnimal::Dog,
                value: 42,
            }
        );
    }

    #[test]
    fn structural_hir_ir_path_computes_bit_masks() {
        let mut module = hir::Module::new();
        let root_def = module.add_type_def(hir::TypeDef {
            name: <MaskSummary>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "masked".to_owned(),
                        ty: hir::Type::u(32),
                    },
                    hir::FieldDef {
                        name: "shifted".to_owned(),
                        ty: hir::Type::u(32),
                    },
                    hir::FieldDef {
                        name: "toggled".to_owned(),
                        ty: hir::Type::u(32),
                    },
                    hir::FieldDef {
                        name: "combined".to_owned(),
                        ty: hir::Type::u(32),
                    },
                ],
            },
        });

        module.add_function(hir::Function {
            name: "mask_summary".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![
                hir::LocalDecl {
                    local: hir::LocalId::new(2),
                    name: "mask".to_owned(),
                    ty: hir::Type::u(32),
                    kind: hir::LocalKind::Let,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(3),
                    name: "masked".to_owned(),
                    ty: hir::Type::u(32),
                    kind: hir::LocalKind::Let,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(4),
                    name: "shifted".to_owned(),
                    ty: hir::Type::u(32),
                    kind: hir::LocalKind::Let,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(5),
                    name: "toggled".to_owned(),
                    ty: hir::Type::u(32),
                    kind: hir::LocalKind::Let,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(6),
                    name: "combined".to_owned(),
                    ty: hir::Type::u(32),
                    kind: hir::LocalKind::Let,
                },
            ],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("structural bit-mask HIR".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(2)),
                            value: hir::Expr::Literal(hir::Literal::Integer(0b1111)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(3)),
                            value: hir::Expr::Binary {
                                op: hir::BinaryOp::BitAnd,
                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0b1011))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(4)),
                            value: hir::Expr::Binary {
                                op: hir::BinaryOp::Shr,
                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(5)),
                            value: hir::Expr::Binary {
                                op: hir::BinaryOp::Xor,
                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0b0011))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(4),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(6)),
                            value: hir::Expr::Binary {
                                op: hir::BinaryOp::BitOr,
                                lhs: Box::new(hir::Expr::Binary {
                                    op: hir::BinaryOp::Shl,
                                    lhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(3))),
                                }),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(5),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "masked".to_owned(),
                            },
                            value: hir::Expr::Local(hir::LocalId::new(3)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(6),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "shifted".to_owned(),
                            },
                            value: hir::Expr::Local(hir::LocalId::new(4)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(7),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "toggled".to_owned(),
                            },
                            value: hir::Expr::Local(hir::LocalId::new(5)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(8),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "combined".to_owned(),
                            },
                            value: hir::Expr::Local(hir::LocalId::new(6)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(9),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<MaskSummary>::SHAPE, &module);
        let value = crate::deserialize::<MaskSummary>(&decoder, &[])
            .expect("structural HIR decoder should compute bit-mask values");
        assert_eq!(
            value,
            MaskSummary {
                masked: 0b1011,
                shifted: 0b0101,
                toggled: 0b1000,
                combined: 0b1001,
            }
        );
    }

    #[test]
    fn structural_hir_ir_path_updates_local_scratch_struct_fields() {
        let mut module = hir::Module::new();
        let scratch_def = module.add_type_def(hir::TypeDef {
            name: "ScratchState".to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "mask".to_owned(),
                        ty: hir::Type::u(32),
                    },
                    hir::FieldDef {
                        name: "done".to_owned(),
                        ty: hir::Type::u(32),
                    },
                ],
            },
        });
        let root_def = module.add_type_def(hir::TypeDef {
            name: <ScratchSummary>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "mask".to_owned(),
                        ty: hir::Type::u(32),
                    },
                    hir::FieldDef {
                        name: "done".to_owned(),
                        ty: hir::Type::u(32),
                    },
                ],
            },
        });

        module.add_function(hir::Function {
            name: "scratch_summary".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "scratch".to_owned(),
                ty: hir::Type::named(scratch_def, Vec::new()),
                kind: hir::LocalKind::Let,
            }],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("structural local scratch-state HIR".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                field: "mask".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(0b1111)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                field: "done".to_owned(),
                            },
                            value: hir::Expr::Binary {
                                op: hir::BinaryOp::BitAnd,
                                lhs: Box::new(hir::Expr::Field {
                                    base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                    field: "mask".to_owned(),
                                }),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0b0011))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "mask".to_owned(),
                            },
                            value: hir::Expr::Field {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                field: "mask".to_owned(),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "done".to_owned(),
                            },
                            value: hir::Expr::Field {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                field: "done".to_owned(),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(4),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<ScratchSummary>::SHAPE, &module);
        let value = crate::deserialize::<ScratchSummary>(&decoder, &[])
            .expect("structural HIR decoder should support local scratch-state fields");
        assert_eq!(
            value,
            ScratchSummary {
                mask: 0b1111,
                done: 0b0011,
            }
        );
    }

    #[test]
    fn structural_hir_ir_path_updates_dynamic_local_array_elements() {
        let mut module = hir::Module::new();
        let root_def = module.add_type_def(hir::TypeDef {
            name: <DynamicIndexSummary>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "selected".to_owned(),
                    ty: hir::Type::u(32),
                }],
            },
        });

        module.add_function(hir::Function {
            name: "dynamic_index_summary".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![
                hir::LocalDecl {
                    local: hir::LocalId::new(2),
                    name: "scratch".to_owned(),
                    ty: hir::Type::array(hir::Type::u(32), 4),
                    kind: hir::LocalKind::Let,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(3),
                    name: "idx".to_owned(),
                    ty: hir::Type::u(32),
                    kind: hir::LocalKind::Let,
                },
            ],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("structural dynamic indexed scratch-array HIR".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(3)),
                            value: hir::Expr::Literal(hir::Literal::Integer(2)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Assign {
                            place: hir::Place::Index {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                index: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(42)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "selected".to_owned(),
                            },
                            value: hir::Expr::Index {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                index: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<DynamicIndexSummary>::SHAPE, &module);
        let value = crate::deserialize::<DynamicIndexSummary>(&decoder, &[])
            .expect("structural HIR decoder should support computed local array indexing");
        assert_eq!(value, DynamicIndexSummary { selected: 42 });
    }

    #[test]
    fn structural_hir_ir_path_updates_dynamic_destination_array_elements() {
        let mut module = hir::Module::new();
        let root_def = module.add_type_def(hir::TypeDef {
            name: <DynamicDestinationSummary>::SHAPE
                .type_identifier
                .to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "values".to_owned(),
                        ty: hir::Type::array(hir::Type::u(32), 4),
                    },
                    hir::FieldDef {
                        name: "selected".to_owned(),
                        ty: hir::Type::u(32),
                    },
                ],
            },
        });

        module.add_function(hir::Function {
            name: "dynamic_destination_summary".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "idx".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Let,
            }],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("structural dynamic indexed destination-array HIR".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(2)),
                            value: hir::Expr::Literal(hir::Literal::Integer(1)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Index {
                                base: Box::new(hir::Place::Field {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                    field: "values".to_owned(),
                                }),
                                index: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(5)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Assign {
                            place: hir::Place::Index {
                                base: Box::new(hir::Place::Field {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                    field: "values".to_owned(),
                                }),
                                index: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(7)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Index {
                                base: Box::new(hir::Place::Field {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                    field: "values".to_owned(),
                                }),
                                index: Box::new(hir::Expr::Literal(hir::Literal::Integer(2))),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(11)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(4),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Index {
                                base: Box::new(hir::Place::Field {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                    field: "values".to_owned(),
                                }),
                                index: Box::new(hir::Expr::Literal(hir::Literal::Integer(3))),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(13)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(5),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "selected".to_owned(),
                            },
                            value: hir::Expr::Index {
                                base: Box::new(hir::Expr::Field {
                                    base: Box::new(hir::Expr::Local(hir::LocalId::new(1))),
                                    field: "values".to_owned(),
                                }),
                                index: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(6),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<DynamicDestinationSummary>::SHAPE, &module);
        let value = crate::deserialize::<DynamicDestinationSummary>(&decoder, &[])
            .expect("structural HIR decoder should support computed destination array indexing");
        assert_eq!(
            value,
            DynamicDestinationSummary {
                values: [5, 7, 11, 13],
                selected: 7,
            }
        );
    }

    #[test]
    fn structural_hir_ir_path_reads_dynamic_local_aggregate_elements() {
        let mut module = hir::Module::new();
        let pair_def = module.add_type_def(hir::TypeDef {
            name: "Pair".to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "lo".to_owned(),
                        ty: hir::Type::u(64),
                    },
                    hir::FieldDef {
                        name: "hi".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });
        let root_def = module.add_type_def(hir::TypeDef {
            name: <DynamicAggregateSummary>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "pair".to_owned(),
                    ty: hir::Type::named(pair_def, Vec::new()),
                }],
            },
        });

        module.add_function(hir::Function {
            name: "dynamic_aggregate_summary".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![
                hir::LocalDecl {
                    local: hir::LocalId::new(2),
                    name: "pairs".to_owned(),
                    ty: hir::Type::array(hir::Type::named(pair_def, Vec::new()), 2),
                    kind: hir::LocalKind::Let,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(3),
                    name: "idx".to_owned(),
                    ty: hir::Type::u(32),
                    kind: hir::LocalKind::Let,
                },
            ],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("structural dynamic indexed aggregate-array HIR".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Index {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                    index: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                                }),
                                field: "lo".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(1)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Index {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                    index: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                                }),
                                field: "hi".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(2)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Index {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                    index: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                                }),
                                field: "lo".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(3)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Index {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                    index: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                                }),
                                field: "hi".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(4)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(4),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(3)),
                            value: hir::Expr::Literal(hir::Literal::Integer(1)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(5),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "pair".to_owned(),
                            },
                            value: hir::Expr::Index {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                index: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(6),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<DynamicAggregateSummary>::SHAPE, &module);
        let value = crate::deserialize::<DynamicAggregateSummary>(&decoder, &[]).expect(
            "structural HIR decoder should support computed aggregate local array indexing",
        );
        assert_eq!(
            value,
            DynamicAggregateSummary {
                pair: Pair { lo: 3, hi: 4 },
            }
        );
    }

    #[test]
    fn structural_hir_ir_path_writes_dynamic_local_aggregate_elements() {
        let mut module = hir::Module::new();
        let pair_def = module.add_type_def(hir::TypeDef {
            name: "Pair".to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "lo".to_owned(),
                        ty: hir::Type::u(64),
                    },
                    hir::FieldDef {
                        name: "hi".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });
        let root_def = module.add_type_def(hir::TypeDef {
            name: <DynamicAggregateSummary>::SHAPE.type_identifier.to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![hir::FieldDef {
                    name: "pair".to_owned(),
                    ty: hir::Type::named(pair_def, Vec::new()),
                }],
            },
        });

        module.add_function(hir::Function {
            name: "dynamic_aggregate_write_summary".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![
                hir::LocalDecl {
                    local: hir::LocalId::new(2),
                    name: "pairs".to_owned(),
                    ty: hir::Type::array(hir::Type::named(pair_def, Vec::new()), 2),
                    kind: hir::LocalKind::Let,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(3),
                    name: "pair".to_owned(),
                    ty: hir::Type::named(pair_def, Vec::new()),
                    kind: hir::LocalKind::Let,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(4),
                    name: "idx".to_owned(),
                    ty: hir::Type::u(32),
                    kind: hir::LocalKind::Let,
                },
            ],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some("structural dynamic indexed aggregate-array write HIR".to_owned()),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(3))),
                                field: "lo".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(9)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(3))),
                                field: "hi".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(10)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(4)),
                            value: hir::Expr::Literal(hir::Literal::Integer(1)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Assign {
                            place: hir::Place::Index {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                index: Box::new(hir::Expr::Local(hir::LocalId::new(4))),
                            },
                            value: hir::Expr::Local(hir::LocalId::new(3)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(4),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "pair".to_owned(),
                            },
                            value: hir::Expr::Index {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                index: Box::new(hir::Expr::Local(hir::LocalId::new(4))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(5),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder = compile_structural_hir_decoder(<DynamicAggregateSummary>::SHAPE, &module);
        let value = crate::deserialize::<DynamicAggregateSummary>(&decoder, &[])
            .expect("structural HIR decoder should support computed aggregate local array writes");
        assert_eq!(
            value,
            DynamicAggregateSummary {
                pair: Pair { lo: 9, hi: 10 },
            }
        );
    }

    #[test]
    fn structural_hir_ir_path_writes_dynamic_destination_aggregate_elements() {
        let mut module = hir::Module::new();
        let pair_def = module.add_type_def(hir::TypeDef {
            name: "Pair".to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "lo".to_owned(),
                        ty: hir::Type::u(64),
                    },
                    hir::FieldDef {
                        name: "hi".to_owned(),
                        ty: hir::Type::u(64),
                    },
                ],
            },
        });
        let root_def = module.add_type_def(hir::TypeDef {
            name: <DynamicAggregateDestinationSummary>::SHAPE
                .type_identifier
                .to_owned(),
            generic_params: vec![],
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "pairs".to_owned(),
                        ty: hir::Type::array(hir::Type::named(pair_def, Vec::new()), 2),
                    },
                    hir::FieldDef {
                        name: "selected".to_owned(),
                        ty: hir::Type::named(pair_def, Vec::new()),
                    },
                ],
            },
        });

        module.add_function(hir::Function {
            name: "dynamic_aggregate_destination_summary".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                hir::Parameter {
                    local: hir::LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: hir::Type::u(64),
                    kind: hir::LocalKind::Param,
                },
                hir::Parameter {
                    local: hir::LocalId::new(1),
                    name: "out".to_owned(),
                    ty: hir::Type::named(root_def, Vec::new()),
                    kind: hir::LocalKind::Destination,
                },
            ],
            locals: vec![
                hir::LocalDecl {
                    local: hir::LocalId::new(2),
                    name: "pair".to_owned(),
                    ty: hir::Type::named(pair_def, Vec::new()),
                    kind: hir::LocalKind::Let,
                },
                hir::LocalDecl {
                    local: hir::LocalId::new(3),
                    name: "idx".to_owned(),
                    ty: hir::Type::u(32),
                    kind: hir::LocalKind::Let,
                },
            ],
            return_type: hir::Type::unit(),
            scopes: vec![hir::Scope {
                id: hir::ScopeId::new(0),
                parent: None,
                comment: Some(
                    "structural dynamic indexed destination aggregate-array HIR".to_owned(),
                ),
            }],
            body: hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![
                    hir::Stmt {
                        id: hir::StmtId::new(0),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                field: "lo".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(21)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(1),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                field: "hi".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(22)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(hir::LocalId::new(3)),
                            value: hir::Expr::Literal(hir::Literal::Integer(1)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(3),
                        kind: hir::StmtKind::Assign {
                            place: hir::Place::Index {
                                base: Box::new(hir::Place::Field {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                    field: "pairs".to_owned(),
                                }),
                                index: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                            },
                            value: hir::Expr::Local(hir::LocalId::new(2)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(4),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Index {
                                    base: Box::new(hir::Place::Field {
                                        base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                        field: "pairs".to_owned(),
                                    }),
                                    index: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                                }),
                                field: "lo".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(1)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(5),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Index {
                                    base: Box::new(hir::Place::Field {
                                        base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                        field: "pairs".to_owned(),
                                    }),
                                    index: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                                }),
                                field: "hi".to_owned(),
                            },
                            value: hir::Expr::Literal(hir::Literal::Integer(2)),
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(6),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "selected".to_owned(),
                            },
                            value: hir::Expr::Index {
                                base: Box::new(hir::Expr::Field {
                                    base: Box::new(hir::Expr::Local(hir::LocalId::new(1))),
                                    field: "pairs".to_owned(),
                                }),
                                index: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                            },
                        },
                    },
                    hir::Stmt {
                        id: hir::StmtId::new(7),
                        kind: hir::StmtKind::Return(None),
                    },
                ],
            },
        });

        let decoder =
            compile_structural_hir_decoder(<DynamicAggregateDestinationSummary>::SHAPE, &module);
        let value = crate::deserialize::<DynamicAggregateDestinationSummary>(&decoder, &[])
            .expect("structural HIR decoder should support computed aggregate destination writes");
        assert_eq!(
            value,
            DynamicAggregateDestinationSummary {
                pairs: [Pair { lo: 1, hi: 2 }, Pair { lo: 21, hi: 22 }],
                selected: Pair { lo: 21, hi: 22 },
            }
        );
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_scalar_arrays() {
        let decoder = compile_postcard_decoder_via_structural_hir(<ScalarArrayHolder>::SHAPE);

        let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
            .expect("structural HIR postcard decoder should decode scalar arrays");
        assert_eq!(
            value,
            ScalarArrayHolder {
                values: [1, 2, 3, 4],
            }
        );
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_borrowed_arrays() {
        let decoder =
            compile_postcard_decoder_via_structural_hir(<BorrowedArrayHolder<'static>>::SHAPE);

        let value = crate::deserialize::<BorrowedArrayHolder<'_>>(
            &decoder,
            &[2, b'h', b'i', 2, b'o', b'k'],
        )
        .expect("structural HIR postcard decoder should decode borrowed arrays");
        assert_eq!(
            value,
            BorrowedArrayHolder {
                values: ["hi", "ok"],
            }
        );
    }

    #[test]
    fn postcard_structural_hir_ir_path_decodes_payload_enums() {
        let decoder = compile_postcard_decoder_via_structural_hir(<PayloadAnimal<'static>>::SHAPE);

        let cat = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[0])
            .expect("structural HIR postcard decoder should decode unit enum variant");
        assert_eq!(cat, PayloadAnimal::Cat);

        let count = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[1, 42])
            .expect("structural HIR postcard decoder should decode scalar payload enum variant");
        assert_eq!(count, PayloadAnimal::Count(42));

        let name = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[2, 2, b'h', b'i'])
            .expect("structural HIR postcard decoder should decode borrowed payload enum variant");
        assert_eq!(name, PayloadAnimal::Name("hi"));
    }

    #[test]
    fn postcard_structural_hir_array_path_matches_jit_differential_harness() {
        let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
        let linear = crate::linearize::linearize(&mut func);
        let report = crate::differential_check_linear_ir_vs_jit(&linear, &[1, 2, 3, 4])
            .expect("differential harness should execute structural HIR postcard array decoder");
        assert!(
            report.is_match(),
            "unexpected differential mismatch: {:?}",
            report.mismatch
        );
    }

    #[test]
    fn postcard_structural_hir_array_path_matches_post_regalloc_simulation() {
        let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
        let linear = crate::linearize::linearize(&mut func);
        let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
        let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
            .expect("regalloc should allocate structural HIR postcard array cfg");
        let result = crate::regalloc_engine::differential_check_cfg(&cfg, &alloc, &[1, 2, 3, 4]);
        assert!(
            matches!(
                result,
                crate::regalloc_engine::DifferentialCheckResult::Match { .. }
            ),
            "unexpected interpreter/post-regalloc mismatch: {result:?}"
        );
    }

    #[test]
    fn postcard_structural_hir_array_path_without_backend_edit_emission() {
        let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
        let linear = crate::linearize::linearize(&mut func);
        let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
        let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
            .expect("regalloc should allocate structural HIR postcard array cfg");
        let result =
            crate::ir_backend::compile_linear_ir_with_alloc_and_mode(&linear, &cfg, &alloc, false);
        let (buf, entry, _source_map, _backend_debug_info) = materialize_backend_result(result);
        let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
            unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
        let decoder = CompiledDecoder {
            buf,
            cfg_mir_line_text_by_line: Default::default(),
            entry,
            func,
            trusted_utf8_input: false,
            _jit_registration: None,
        };

        let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
            .expect("structural HIR postcard array decoder should execute without backend edits");
        assert_eq!(
            value,
            ScalarArrayHolder {
                values: [1, 2, 3, 4]
            }
        );
    }

    #[test]
    fn debug_scalar_array_regalloc_edits() {
        let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
        let linear = crate::linearize::linearize(&mut func);
        let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
        let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
            .expect("regalloc should allocate structural HIR postcard array cfg");
        println!("{}", format_allocated_regalloc_edits(&alloc));
    }

    #[test]
    fn debug_scalar_array_emission_trace() {
        let decoder = compile_postcard_decoder_via_structural_hir(<ScalarArrayHolder>::SHAPE);
        println!(
            "{}",
            decoder
                .emission_trace_text()
                .expect("emission trace should render")
        );
    }

    #[test]
    fn builds_dwarf_sections_from_source_map_lines() {
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
                    line: 2,
                    column: 1,
                },
            },
        ];

        let listing_path = std::env::temp_dir()
            .join(format!("kajit-debug-test-{}", std::process::id()))
            .join("sample.cfg-mir");
        std::fs::create_dir_all(listing_path.parent().expect("temp listing dir")).unwrap();
        std::fs::write(&listing_path, "inst0\ninst1\n").unwrap();

        let debug_info = build_jit_debug_info_from_source_map(
            0x1000 as *const u8,
            32,
            Some(&source_map),
            &listing_path,
            crate::jit_dwarf::JitDebugSubprogram {
                name: "kajit::decode::test".to_string(),
                frame_base_expression: crate::jit_dwarf::expr_breg(
                    crate::jit_dwarf::frame_base_register(jit_dwarf_target_arch()),
                    0,
                ),
                variables: Vec::new(),
                lexical_blocks: Vec::new(),
            },
        )
        .expect("expected debug info");
        let dwarf = crate::jit_dwarf::build_jit_dwarf_sections_from_debug_info(&debug_info)
            .expect("expected dwarf sections");
        assert!(!dwarf.debug_line.is_empty());
    }

    #[test]
    fn deser_dwarf_variables_cover_fixed_runtime_state() {
        let vars = deser_dwarf_variables(jit_dwarf_target_arch());
        let names = vars.iter().map(|var| var.name.as_str()).collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "input_ptr",
                "input_end",
                "out_ptr",
                "ctx",
                "error_code",
                "error_offset",
            ]
        );
        for var in vars {
            match var.location {
                crate::jit_dwarf::DwarfVariableLocation::Expr(expr) => {
                    assert!(!expr.is_empty());
                }
                crate::jit_dwarf::DwarfVariableLocation::List(_) => {
                    panic!("deserializer runtime-state vars should use inline exprloc")
                }
            }
        }
    }

    #[test]
    fn cfg_value_dwarf_variables_cover_def_vregs() {
        let v0 = crate::ir::VReg::new(0);
        let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
        let inst_id_2 = crate::regalloc_engine::cfg_mir::InstId::new(1);
        let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
        let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
        let func = crate::regalloc_engine::cfg_mir::Function {
            id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
            lambda_id: crate::ir::LambdaId::new(0),
            entry: block_id,
            data_args: Vec::new(),
            data_results: Vec::new(),
            blocks: vec![crate::regalloc_engine::cfg_mir::Block {
                id: block_id,
                params: Vec::new(),
                insts: vec![inst_id, inst_id_2],
                term: term_id,
                preds: Vec::new(),
                succs: Vec::new(),
            }],
            edges: Vec::new(),
            insts: vec![
                crate::regalloc_engine::cfg_mir::Inst {
                    id: inst_id,
                    op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                    operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
                },
                crate::regalloc_engine::cfg_mir::Inst {
                    id: inst_id_2,
                    op: crate::linearize::LinearOp::Copy {
                        dst: crate::ir::VReg::new(1),
                        src: v0,
                    },
                    operands: vec![
                        crate::regalloc_engine::cfg_mir::Operand {
                            vreg: v0,
                            kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                            class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                            fixed: None,
                        },
                        crate::regalloc_engine::cfg_mir::Operand {
                            vreg: crate::ir::VReg::new(1),
                            kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                            class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                            fixed: None,
                        },
                    ],
                    clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
                },
            ],
            terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
        };
        let root_scope = crate::ir::DebugScopeId::new(0);
        let block_scope = crate::ir::DebugScopeId::new(1);
        let mut scopes = crate::ir::Arena::new();
        scopes.push(crate::ir::DebugScope {
            parent: None,
            kind: crate::ir::DebugScopeKind::LambdaBody {
                lambda_id: crate::ir::LambdaId::new(0),
            },
        });
        scopes.push(crate::ir::DebugScope {
            parent: Some(root_scope),
            kind: crate::ir::DebugScopeKind::ThetaBody,
        });
        let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
        let op_id_2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id_2);
        let program = crate::regalloc_engine::cfg_mir::Program {
            funcs: vec![func],
            vreg_count: 2,
            slot_count: 0,
            debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
                scopes,
                values: crate::ir::Arena::new(),
                root_scope: Some(root_scope),
                op_scopes: std::collections::HashMap::from([
                    ((crate::ir::LambdaId::new(0), op_id), block_scope),
                    ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                    (
                        (
                            crate::ir::LambdaId::new(0),
                            crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                        ),
                        block_scope,
                    ),
                ]),
                op_values: std::collections::HashMap::new(),
                vreg_scopes: vec![Some(block_scope), Some(root_scope)],
                vreg_values: vec![None, None],
            },
        };
        #[cfg(target_arch = "aarch64")]
        let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
        #[cfg(target_arch = "x86_64")]
        let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
        #[cfg(target_arch = "aarch64")]
        let reg_2 = regalloc2::PReg::new(20, regalloc2::RegClass::Int);
        #[cfg(target_arch = "x86_64")]
        let reg_2 = regalloc2::PReg::new(13, regalloc2::RegClass::Int);
        let alloc = crate::regalloc_engine::AllocatedCfgProgram {
            cfg_program: program.clone(),
            functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
                lambda_id: crate::ir::LambdaId::new(0),
                num_spillslots: 0,
                edits: Vec::new(),
                op_allocs: std::collections::HashMap::from([
                    (op_id, vec![regalloc2::Allocation::reg(reg)]),
                    (
                        op_id_2,
                        vec![
                            regalloc2::Allocation::reg(reg),
                            regalloc2::Allocation::reg(reg_2),
                        ],
                    ),
                ]),
                op_operands: std::collections::HashMap::from([
                    (
                        op_id,
                        vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                    ),
                    (
                        op_id_2,
                        vec![
                            (v0, crate::regalloc_engine::cfg_mir::OperandKind::Use),
                            (
                                crate::ir::VReg::new(1),
                                crate::regalloc_engine::cfg_mir::OperandKind::Def,
                            ),
                        ],
                    ),
                ]),
                edge_edits: Vec::new(),
                return_result_allocs: Vec::new(),
            }],
        };
        let backend_debug_info = crate::ir_backend::BackendDebugInfo {
            op_infos: vec![
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id,
                    line: 1,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 0,
                        end_offset: 4,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: op_id_2,
                    line: 2,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 4,
                        end_offset: 8,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                    line: 3,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 8,
                        end_offset: 12,
                    }],
                },
            ],
        };

        let vars = cfg_value_dwarf_variables(
            &program,
            &alloc,
            Some(&backend_debug_info),
            0x1000 as *const u8,
            jit_dwarf_target_arch(),
            true,
            false,
        );

        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].variable.name, "v0");
        assert_eq!(
            vars[0].lexical_ranges,
            vec![crate::jit_dwarf::JitDebugRange {
                low_pc: 0x1000,
                high_pc: 0x1008,
            }]
        );
        match &vars[0].variable.location {
            crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].start, 0x1004);
                assert_eq!(locations[0].end, 0x1008);
                let dwarf_reg = crate::jit_dwarf::dwarf_register_from_hw_encoding(
                    jit_dwarf_target_arch(),
                    reg.hw_enc() as u8,
                )
                .unwrap();
                assert_eq!(
                    locations[0].expression,
                    crate::jit_dwarf::expr_reg(dwarf_reg)
                );
            }
            crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
                panic!("cfg def vregs should use ranged locations")
            }
        }
    }

    #[test]
    fn cfg_value_dwarf_variables_keep_edge_carried_defs_live() {
        let v0 = crate::ir::VReg::new(0);
        let v1 = crate::ir::VReg::new(1);
        let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
        let inst_id_2 = crate::regalloc_engine::cfg_mir::InstId::new(1);
        let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
        let return_term_id = crate::regalloc_engine::cfg_mir::TermId::new(1);
        let entry_block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
        let exit_block_id = crate::regalloc_engine::cfg_mir::BlockId::new(1);
        let edge_id = crate::regalloc_engine::cfg_mir::EdgeId::new(0);
        let func = crate::regalloc_engine::cfg_mir::Function {
            id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
            lambda_id: crate::ir::LambdaId::new(0),
            entry: entry_block_id,
            data_args: Vec::new(),
            data_results: Vec::new(),
            blocks: vec![
                crate::regalloc_engine::cfg_mir::Block {
                    id: entry_block_id,
                    params: Vec::new(),
                    insts: vec![inst_id, inst_id_2],
                    term: term_id,
                    preds: Vec::new(),
                    succs: vec![edge_id],
                },
                crate::regalloc_engine::cfg_mir::Block {
                    id: exit_block_id,
                    params: vec![v0],
                    insts: Vec::new(),
                    term: return_term_id,
                    preds: vec![edge_id],
                    succs: Vec::new(),
                },
            ],
            edges: vec![crate::regalloc_engine::cfg_mir::Edge {
                id: edge_id,
                from: entry_block_id,
                to: exit_block_id,
                args: vec![crate::regalloc_engine::cfg_mir::EdgeArg {
                    target: v0,
                    source: v0,
                }],
            }],
            insts: vec![
                crate::regalloc_engine::cfg_mir::Inst {
                    id: inst_id,
                    op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                    operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
                },
                crate::regalloc_engine::cfg_mir::Inst {
                    id: inst_id_2,
                    op: crate::linearize::LinearOp::Const { dst: v1, value: 9 },
                    operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v1,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
                },
            ],
            terms: vec![
                crate::regalloc_engine::cfg_mir::Terminator::Branch { edge: edge_id },
                crate::regalloc_engine::cfg_mir::Terminator::Return,
            ],
        };
        let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
        let op_id_2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id_2);
        let root_scope = crate::ir::DebugScopeId::new(0);
        let block_scope = crate::ir::DebugScopeId::new(1);
        let mut scopes = crate::ir::Arena::new();
        scopes.push(crate::ir::DebugScope {
            parent: None,
            kind: crate::ir::DebugScopeKind::LambdaBody {
                lambda_id: crate::ir::LambdaId::new(0),
            },
        });
        scopes.push(crate::ir::DebugScope {
            parent: Some(root_scope),
            kind: crate::ir::DebugScopeKind::ThetaBody,
        });
        let program = crate::regalloc_engine::cfg_mir::Program {
            funcs: vec![func],
            vreg_count: 2,
            slot_count: 0,
            debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
                scopes,
                values: crate::ir::Arena::new(),
                root_scope: Some(root_scope),
                op_scopes: std::collections::HashMap::from([
                    ((crate::ir::LambdaId::new(0), op_id), block_scope),
                    ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                    (
                        (
                            crate::ir::LambdaId::new(0),
                            crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                        ),
                        block_scope,
                    ),
                ]),
                op_values: std::collections::HashMap::new(),
                vreg_scopes: vec![Some(block_scope), Some(root_scope)],
                vreg_values: vec![None, None],
            },
        };
        let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
        #[cfg(target_arch = "aarch64")]
        let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
        #[cfg(target_arch = "x86_64")]
        let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
        #[cfg(target_arch = "aarch64")]
        let reg_2 = regalloc2::PReg::new(20, regalloc2::RegClass::Int);
        #[cfg(target_arch = "x86_64")]
        let reg_2 = regalloc2::PReg::new(13, regalloc2::RegClass::Int);
        let alloc = crate::regalloc_engine::AllocatedCfgProgram {
            cfg_program: program.clone(),
            functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
                lambda_id: crate::ir::LambdaId::new(0),
                num_spillslots: 0,
                edits: Vec::new(),
                op_allocs: std::collections::HashMap::from([
                    (op_id, vec![regalloc2::Allocation::reg(reg)]),
                    (op_id_2, vec![regalloc2::Allocation::reg(reg_2)]),
                ]),
                op_operands: std::collections::HashMap::from([
                    (
                        op_id,
                        vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                    ),
                    (
                        op_id_2,
                        vec![(v1, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                    ),
                    (term_op, Vec::new()),
                ]),
                edge_edits: Vec::new(),
                return_result_allocs: Vec::new(),
            }],
        };
        let backend_debug_info = crate::ir_backend::BackendDebugInfo {
            op_infos: vec![
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id,
                    line: 1,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 0,
                        end_offset: 4,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: op_id_2,
                    line: 2,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 4,
                        end_offset: 8,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: term_op,
                    line: 3,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 8,
                        end_offset: 12,
                    }],
                },
            ],
        };

        let vars = cfg_value_dwarf_variables(
            &program,
            &alloc,
            Some(&backend_debug_info),
            0x1000 as *const u8,
            jit_dwarf_target_arch(),
            true,
            false,
        );

        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].variable.name, "v0");
        assert_eq!(
            vars[0].lexical_ranges,
            vec![crate::jit_dwarf::JitDebugRange {
                low_pc: 0x1000,
                high_pc: 0x100c,
            }]
        );
        match &vars[0].variable.location {
            crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].start, 0x1004);
                assert_eq!(locations[0].end, 0x100c);
                let dwarf_reg = crate::jit_dwarf::dwarf_register_from_hw_encoding(
                    jit_dwarf_target_arch(),
                    reg.hw_enc() as u8,
                )
                .unwrap();
                assert_eq!(
                    locations[0].expression,
                    crate::jit_dwarf::expr_reg(dwarf_reg)
                );
            }
            crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
                panic!("cfg edge-carried vregs should use ranged locations")
            }
        }
    }

    #[test]
    fn cfg_mir_dwarf_variables_place_block_local_vregs_in_lexical_blocks() {
        let v0 = crate::ir::VReg::new(0);
        let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
        let inst_id_2 = crate::regalloc_engine::cfg_mir::InstId::new(1);
        let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
        let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
        let func = crate::regalloc_engine::cfg_mir::Function {
            id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
            lambda_id: crate::ir::LambdaId::new(0),
            entry: block_id,
            data_args: Vec::new(),
            data_results: Vec::new(),
            blocks: vec![crate::regalloc_engine::cfg_mir::Block {
                id: block_id,
                params: Vec::new(),
                insts: vec![inst_id, inst_id_2],
                term: term_id,
                preds: Vec::new(),
                succs: Vec::new(),
            }],
            edges: Vec::new(),
            insts: vec![
                crate::regalloc_engine::cfg_mir::Inst {
                    id: inst_id,
                    op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                    operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
                },
                crate::regalloc_engine::cfg_mir::Inst {
                    id: inst_id_2,
                    op: crate::linearize::LinearOp::Copy {
                        dst: crate::ir::VReg::new(1),
                        src: v0,
                    },
                    operands: vec![
                        crate::regalloc_engine::cfg_mir::Operand {
                            vreg: v0,
                            kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                            class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                            fixed: None,
                        },
                        crate::regalloc_engine::cfg_mir::Operand {
                            vreg: crate::ir::VReg::new(1),
                            kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                            class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                            fixed: None,
                        },
                    ],
                    clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
                },
            ],
            terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
        };
        let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
        let op_id_2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id_2);
        let root_scope = crate::ir::DebugScopeId::new(0);
        let block_scope = crate::ir::DebugScopeId::new(1);
        let mut scopes = crate::ir::Arena::new();
        scopes.push(crate::ir::DebugScope {
            parent: None,
            kind: crate::ir::DebugScopeKind::LambdaBody {
                lambda_id: crate::ir::LambdaId::new(0),
            },
        });
        scopes.push(crate::ir::DebugScope {
            parent: Some(root_scope),
            kind: crate::ir::DebugScopeKind::ThetaBody,
        });
        let program = crate::regalloc_engine::cfg_mir::Program {
            funcs: vec![func],
            vreg_count: 2,
            slot_count: 0,
            debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
                scopes,
                values: crate::ir::Arena::new(),
                root_scope: Some(root_scope),
                op_scopes: std::collections::HashMap::from([
                    ((crate::ir::LambdaId::new(0), op_id), block_scope),
                    ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                    (
                        (
                            crate::ir::LambdaId::new(0),
                            crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                        ),
                        block_scope,
                    ),
                ]),
                op_values: std::collections::HashMap::new(),
                vreg_scopes: vec![Some(block_scope), Some(root_scope)],
                vreg_values: vec![None, None],
            },
        };
        #[cfg(target_arch = "aarch64")]
        let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
        #[cfg(target_arch = "x86_64")]
        let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
        #[cfg(target_arch = "aarch64")]
        let reg_2 = regalloc2::PReg::new(20, regalloc2::RegClass::Int);
        #[cfg(target_arch = "x86_64")]
        let reg_2 = regalloc2::PReg::new(13, regalloc2::RegClass::Int);
        let alloc = crate::regalloc_engine::AllocatedCfgProgram {
            cfg_program: program.clone(),
            functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
                lambda_id: crate::ir::LambdaId::new(0),
                num_spillslots: 0,
                edits: Vec::new(),
                op_allocs: std::collections::HashMap::from([
                    (op_id, vec![regalloc2::Allocation::reg(reg)]),
                    (
                        op_id_2,
                        vec![
                            regalloc2::Allocation::reg(reg),
                            regalloc2::Allocation::reg(reg_2),
                        ],
                    ),
                ]),
                op_operands: std::collections::HashMap::from([
                    (
                        op_id,
                        vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                    ),
                    (
                        op_id_2,
                        vec![
                            (v0, crate::regalloc_engine::cfg_mir::OperandKind::Use),
                            (
                                crate::ir::VReg::new(1),
                                crate::regalloc_engine::cfg_mir::OperandKind::Def,
                            ),
                        ],
                    ),
                ]),
                edge_edits: Vec::new(),
                return_result_allocs: Vec::new(),
            }],
        };
        let backend_debug_info = crate::ir_backend::BackendDebugInfo {
            op_infos: vec![
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id,
                    line: 1,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 0,
                        end_offset: 4,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: op_id_2,
                    line: 2,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 4,
                        end_offset: 8,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                    line: 3,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 8,
                        end_offset: 12,
                    }],
                },
            ],
        };

        let subprogram = cfg_mir_dwarf_variables(
            None,
            &program,
            &alloc,
            Some(&backend_debug_info),
            0x1000 as *const u8,
            jit_dwarf_target_arch(),
            true,
        );

        assert!(
            !subprogram
                .variables
                .iter()
                .any(|variable| variable.name == "v0")
        );
        assert_eq!(subprogram.lexical_blocks.len(), 1);
        assert_eq!(subprogram.lexical_blocks[0].ranges.len(), 1);
        assert_eq!(subprogram.lexical_blocks[0].ranges[0].low_pc, 0x1000);
        assert_eq!(subprogram.lexical_blocks[0].ranges[0].high_pc, 0x100c);
        assert!(subprogram.lexical_blocks[0].variables.is_empty());
        assert_eq!(subprogram.lexical_blocks[0].lexical_blocks.len(), 1);
        assert_eq!(
            subprogram.lexical_blocks[0].lexical_blocks[0].ranges.len(),
            1
        );
        assert_eq!(
            subprogram.lexical_blocks[0].lexical_blocks[0].ranges[0].low_pc,
            0x1000
        );
        assert_eq!(
            subprogram.lexical_blocks[0].lexical_blocks[0].ranges[0].high_pc,
            0x1008
        );
        assert_eq!(
            subprogram.lexical_blocks[0].lexical_blocks[0]
                .variables
                .len(),
            1
        );
        assert_eq!(
            subprogram.lexical_blocks[0].lexical_blocks[0].variables[0].name,
            "v0"
        );
    }

    #[test]
    fn cfg_semantic_field_dwarf_variables_follow_field_debug_values() {
        #[derive(Facet)]
        struct Bools {
            a: bool,
            b: bool,
        }

        let inst_a = crate::regalloc_engine::cfg_mir::InstId::new(0);
        let inst_b = crate::regalloc_engine::cfg_mir::InstId::new(1);
        let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
        let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
        let op_a = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_a);
        let op_b = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_b);
        let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
        let root_scope = crate::ir::DebugScopeId::new(0);

        let func = crate::regalloc_engine::cfg_mir::Function {
            id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
            lambda_id: crate::ir::LambdaId::new(0),
            entry: block_id,
            data_args: Vec::new(),
            data_results: Vec::new(),
            blocks: vec![crate::regalloc_engine::cfg_mir::Block {
                id: block_id,
                params: Vec::new(),
                insts: vec![inst_a, inst_b],
                term: term_id,
                preds: Vec::new(),
                succs: Vec::new(),
            }],
            edges: Vec::new(),
            insts: vec![
                crate::regalloc_engine::cfg_mir::Inst {
                    id: inst_a,
                    op: crate::linearize::LinearOp::CallIntrinsic {
                        func: crate::ir::IntrinsicFn(
                            crate::json_intrinsics::kajit_json_read_bool as *const () as usize,
                        ),
                        args: Vec::new(),
                        dst: None,
                        field_offset: 0,
                    },
                    operands: Vec::new(),
                    clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
                },
                crate::regalloc_engine::cfg_mir::Inst {
                    id: inst_b,
                    op: crate::linearize::LinearOp::CallIntrinsic {
                        func: crate::ir::IntrinsicFn(
                            crate::json_intrinsics::kajit_json_read_bool as *const () as usize,
                        ),
                        args: Vec::new(),
                        dst: None,
                        field_offset: 1,
                    },
                    operands: Vec::new(),
                    clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
                },
            ],
            terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
        };

        let mut scopes = crate::ir::Arena::new();
        scopes.push(crate::ir::DebugScope {
            parent: None,
            kind: crate::ir::DebugScopeKind::LambdaBody {
                lambda_id: crate::ir::LambdaId::new(0),
            },
        });
        let mut values = crate::ir::Arena::new();
        let debug_a = values.push(crate::ir::DebugValue {
            name: "a".to_string(),
            kind: crate::ir::DebugValueKind::Field { offset: 0 },
        });
        let debug_b = values.push(crate::ir::DebugValue {
            name: "b".to_string(),
            kind: crate::ir::DebugValueKind::Field { offset: 1 },
        });
        let program = crate::regalloc_engine::cfg_mir::Program {
            funcs: vec![func],
            vreg_count: 0,
            slot_count: 0,
            debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
                scopes,
                values,
                root_scope: Some(root_scope),
                op_scopes: std::collections::HashMap::new(),
                op_values: std::collections::HashMap::from([
                    ((crate::ir::LambdaId::new(0), op_a), debug_a),
                    ((crate::ir::LambdaId::new(0), op_b), debug_b),
                ]),
                vreg_scopes: Vec::new(),
                vreg_values: Vec::new(),
            },
        };
        let backend_debug_info = crate::ir_backend::BackendDebugInfo {
            op_infos: vec![
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: op_a,
                    line: 10,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 0,
                        end_offset: 8,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: op_b,
                    line: 20,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 16,
                        end_offset: 24,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: term_op,
                    line: 30,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 32,
                        end_offset: 40,
                    }],
                },
            ],
        };

        let vars = cfg_semantic_field_dwarf_variables(
            <Bools as Facet>::SHAPE,
            &program,
            Some(&backend_debug_info),
            0x1000 as *const u8,
            jit_dwarf_target_arch(),
        );

        assert_eq!(vars.len(), 2);
        assert_eq!(vars[0].scope, Some(root_scope));
        assert_eq!(vars[0].variable.name, "a");
        assert_eq!(
            vars[0].lexical_ranges,
            vec![crate::jit_dwarf::JitDebugRange {
                low_pc: 0x1000,
                high_pc: 0x1028,
            }]
        );
        assert_eq!(vars[1].scope, Some(root_scope));
        assert_eq!(vars[1].variable.name, "b");
        assert_eq!(
            vars[1].lexical_ranges,
            vec![crate::jit_dwarf::JitDebugRange {
                low_pc: 0x1010,
                high_pc: 0x1028,
            }]
        );

        let expected_expr_a = dwarf_expr_for_out_field(jit_dwarf_target_arch(), 0, 1);
        let expected_expr_b = dwarf_expr_for_out_field(jit_dwarf_target_arch(), 1, 1);
        match &vars[0].variable.location {
            crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].start, 0x1008);
                assert_eq!(locations[0].end, 0x1028);
                assert_eq!(locations[0].expression, expected_expr_a);
            }
            crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
                panic!("semantic field vars should use ranged locations")
            }
        }
        match &vars[1].variable.location {
            crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].start, 0x1018);
                assert_eq!(locations[0].end, 0x1028);
                assert_eq!(locations[0].expression, expected_expr_b);
            }
            crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
                panic!("semantic field vars should use ranged locations")
            }
        }
    }

    #[test]
    fn cfg_value_dwarf_variables_can_hide_semantic_owned_vregs() {
        let v0 = crate::ir::VReg::new(0);
        let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
        let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
        let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
        let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
        let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
        let root_scope = crate::ir::DebugScopeId::new(0);
        let func = crate::regalloc_engine::cfg_mir::Function {
            id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
            lambda_id: crate::ir::LambdaId::new(0),
            entry: block_id,
            data_args: Vec::new(),
            data_results: Vec::new(),
            blocks: vec![crate::regalloc_engine::cfg_mir::Block {
                id: block_id,
                params: Vec::new(),
                insts: vec![inst_id],
                term: term_id,
                preds: Vec::new(),
                succs: Vec::new(),
            }],
            edges: Vec::new(),
            insts: vec![crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 1 },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            }],
            terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
        };
        let mut scopes = crate::ir::Arena::new();
        scopes.push(crate::ir::DebugScope {
            parent: None,
            kind: crate::ir::DebugScopeKind::LambdaBody {
                lambda_id: crate::ir::LambdaId::new(0),
            },
        });
        let mut values = crate::ir::Arena::new();
        let debug_a = values.push(crate::ir::DebugValue {
            name: "a".to_string(),
            kind: crate::ir::DebugValueKind::Field { offset: 0 },
        });
        let program = crate::regalloc_engine::cfg_mir::Program {
            funcs: vec![func],
            vreg_count: 1,
            slot_count: 0,
            debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
                scopes,
                values,
                root_scope: Some(root_scope),
                op_scopes: std::collections::HashMap::from([
                    ((crate::ir::LambdaId::new(0), op_id), root_scope),
                    ((crate::ir::LambdaId::new(0), term_op), root_scope),
                ]),
                op_values: std::collections::HashMap::new(),
                vreg_scopes: vec![Some(root_scope)],
                vreg_values: vec![Some(debug_a)],
            },
        };
        #[cfg(target_arch = "aarch64")]
        let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
        #[cfg(target_arch = "x86_64")]
        let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
        let alloc = crate::regalloc_engine::AllocatedCfgProgram {
            cfg_program: program.clone(),
            functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
                lambda_id: crate::ir::LambdaId::new(0),
                num_spillslots: 0,
                edits: Vec::new(),
                op_allocs: std::collections::HashMap::from([(
                    op_id,
                    vec![regalloc2::Allocation::reg(reg)],
                )]),
                op_operands: std::collections::HashMap::from([
                    (
                        op_id,
                        vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                    ),
                    (term_op, Vec::new()),
                ]),
                edge_edits: Vec::new(),
                return_result_allocs: Vec::new(),
            }],
        };
        let backend_debug_info = crate::ir_backend::BackendDebugInfo {
            op_infos: vec![
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id,
                    line: 1,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 0,
                        end_offset: 4,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: term_op,
                    line: 2,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 4,
                        end_offset: 8,
                    }],
                },
            ],
        };

        let vars = cfg_value_dwarf_variables(
            &program,
            &alloc,
            Some(&backend_debug_info),
            0x1000 as *const u8,
            jit_dwarf_target_arch(),
            true,
            true,
        );

        assert!(vars.is_empty(), "semantic-owned vregs should be hidden");
    }

    #[test]
    fn cfg_semantic_named_dwarf_variables_merge_shared_vregs() {
        let v0 = crate::ir::VReg::new(0);
        let v1 = crate::ir::VReg::new(1);
        let inst0 = crate::regalloc_engine::cfg_mir::InstId::new(0);
        let inst1 = crate::regalloc_engine::cfg_mir::InstId::new(1);
        let inst2 = crate::regalloc_engine::cfg_mir::InstId::new(2);
        let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
        let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
        let op0 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst0);
        let op1 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst1);
        let op2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst2);
        let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
        let root_scope = crate::ir::DebugScopeId::new(0);
        let func = crate::regalloc_engine::cfg_mir::Function {
            id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
            lambda_id: crate::ir::LambdaId::new(0),
            entry: block_id,
            data_args: Vec::new(),
            data_results: Vec::new(),
            blocks: vec![crate::regalloc_engine::cfg_mir::Block {
                id: block_id,
                params: Vec::new(),
                insts: vec![inst0, inst1, inst2],
                term: term_id,
                preds: Vec::new(),
                succs: Vec::new(),
            }],
            edges: Vec::new(),
            insts: vec![
                crate::regalloc_engine::cfg_mir::Inst {
                    id: inst0,
                    op: crate::linearize::LinearOp::Const { dst: v0, value: 1 },
                    operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
                },
                crate::regalloc_engine::cfg_mir::Inst {
                    id: inst1,
                    op: crate::linearize::LinearOp::Copy { dst: v1, src: v0 },
                    operands: vec![
                        crate::regalloc_engine::cfg_mir::Operand {
                            vreg: v0,
                            kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                            class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                            fixed: None,
                        },
                        crate::regalloc_engine::cfg_mir::Operand {
                            vreg: v1,
                            kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                            class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                            fixed: None,
                        },
                    ],
                    clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
                },
                crate::regalloc_engine::cfg_mir::Inst {
                    id: inst2,
                    op: crate::linearize::LinearOp::WriteToField {
                        src: v1,
                        offset: 0,
                        width: crate::ir::Width::W1,
                    },
                    operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v1,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
                },
            ],
            terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
        };
        let mut scopes = crate::ir::Arena::new();
        scopes.push(crate::ir::DebugScope {
            parent: None,
            kind: crate::ir::DebugScopeKind::LambdaBody {
                lambda_id: crate::ir::LambdaId::new(0),
            },
        });
        let mut values = crate::ir::Arena::new();
        let debug_flag = values.push(crate::ir::DebugValue {
            name: "flag".to_string(),
            kind: crate::ir::DebugValueKind::Named,
        });
        let program = crate::regalloc_engine::cfg_mir::Program {
            funcs: vec![func],
            vreg_count: 2,
            slot_count: 0,
            debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
                scopes,
                values,
                root_scope: Some(root_scope),
                op_scopes: std::collections::HashMap::from([
                    ((crate::ir::LambdaId::new(0), op0), root_scope),
                    ((crate::ir::LambdaId::new(0), op1), root_scope),
                    ((crate::ir::LambdaId::new(0), op2), root_scope),
                    ((crate::ir::LambdaId::new(0), term_op), root_scope),
                ]),
                op_values: std::collections::HashMap::from([
                    ((crate::ir::LambdaId::new(0), op0), debug_flag),
                    ((crate::ir::LambdaId::new(0), op1), debug_flag),
                ]),
                vreg_scopes: vec![Some(root_scope), Some(root_scope)],
                vreg_values: vec![Some(debug_flag), Some(debug_flag)],
            },
        };
        #[cfg(target_arch = "aarch64")]
        let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
        #[cfg(target_arch = "x86_64")]
        let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
        let alloc = crate::regalloc_engine::AllocatedCfgProgram {
            cfg_program: program.clone(),
            functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
                lambda_id: crate::ir::LambdaId::new(0),
                num_spillslots: 0,
                edits: Vec::new(),
                op_allocs: std::collections::HashMap::from([
                    (op0, vec![regalloc2::Allocation::reg(reg)]),
                    (
                        op1,
                        vec![
                            regalloc2::Allocation::reg(reg),
                            regalloc2::Allocation::reg(reg),
                        ],
                    ),
                    (op2, vec![regalloc2::Allocation::reg(reg)]),
                ]),
                op_operands: std::collections::HashMap::from([
                    (
                        op0,
                        vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                    ),
                    (
                        op1,
                        vec![
                            (v0, crate::regalloc_engine::cfg_mir::OperandKind::Use),
                            (v1, crate::regalloc_engine::cfg_mir::OperandKind::Def),
                        ],
                    ),
                    (
                        op2,
                        vec![(v1, crate::regalloc_engine::cfg_mir::OperandKind::Use)],
                    ),
                    (term_op, Vec::new()),
                ]),
                edge_edits: Vec::new(),
                return_result_allocs: Vec::new(),
            }],
        };
        let backend_debug_info = crate::ir_backend::BackendDebugInfo {
            op_infos: vec![
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: op0,
                    line: 1,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 0,
                        end_offset: 4,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: op1,
                    line: 2,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 4,
                        end_offset: 8,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: op2,
                    line: 3,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 8,
                        end_offset: 12,
                    }],
                },
                crate::ir_backend::BackendOpDebugInfo {
                    lambda_id: 0,
                    op_id: term_op,
                    line: 4,
                    code_ranges: vec![crate::ir_backend::BackendCodeRange {
                        start_offset: 12,
                        end_offset: 16,
                    }],
                },
            ],
        };

        let vars = cfg_semantic_named_dwarf_variables(
            &program,
            &alloc,
            Some(&backend_debug_info),
            0x1000 as *const u8,
            jit_dwarf_target_arch(),
            true,
        );

        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].scope, Some(root_scope));
        assert_eq!(vars[0].variable.name, "flag");
        assert_eq!(
            vars[0].lexical_ranges,
            vec![crate::jit_dwarf::JitDebugRange {
                low_pc: 0x1000,
                high_pc: 0x100c,
            }]
        );
        match &vars[0].variable.location {
            crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].start, 0x1004);
                assert_eq!(locations[0].end, 0x100c);
            }
            crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
                panic!("semantic named vars should use ranged locations")
            }
        }
    }
}
