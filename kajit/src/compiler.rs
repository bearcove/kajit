use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use facet::{
    ConstParamKind, Def, EnumRepr, KnownPointer, OptionDef, PointerDef, ScalarType, Shape,
    StructKind, Type, UserType,
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
            type_defs_by_shape: HashMap::new(),
            callables_by_name: HashMap::new(),
            locals: Vec::new(),
            next_local: 0,
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

        let kind = if shape.is_transparent() {
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

    fn lower_type(&mut self, shape: &'static Shape) -> hir::Type {
        if is_unit(shape) {
            return hir::Type::unit();
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
                ScalarType::U128 => hir::Type::u(128),
                ScalarType::USize => hir::Type::u(64),
                ScalarType::I8 => hir::Type::i(8),
                ScalarType::I16 => hir::Type::i(16),
                ScalarType::I32 => hir::Type::i(32),
                ScalarType::I64 => hir::Type::i(64),
                ScalarType::I128 => hir::Type::i(128),
                ScalarType::ISize => hir::Type::i(64),
                ScalarType::Str => hir::Type::str(self.input_region),
                ScalarType::String
                | ScalarType::CowStr
                | ScalarType::Char
                | ScalarType::F32
                | ScalarType::F64 => {
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

    fn ensure_postcard_option_tag_reader(&mut self) -> hir::CallableId {
        const NAME: &str = "postcard.read_option_tag";
        if let Some(existing) = self.callables_by_name.get(NAME).copied() {
            return existing;
        }

        let callable_id = self.module.add_callable(hir::CallableSpec {
            kind: hir::CallableKind::Builtin,
            name: NAME.to_owned(),
            signature: hir::CallSignature {
                params: vec![hir::Type::named(
                    self.cursor_type,
                    vec![hir::GenericArg::Region(self.input_region)],
                )],
                returns: vec![hir::Type::bool()],
                effect_class: hir::EffectClass::Mutates,
                domain_effects: vec![hir::DomainEffect {
                    domain: "cursor".to_owned(),
                    access: hir::DomainAccess::Mutate,
                }],
                control: hir::ControlTransfer::MayFail,
                capabilities: vec!["deser.postcard".to_owned()],
                safety: hir::CallSafety::SafeCore,
            },
            docs: Some("Read and validate a postcard Option tag.".to_owned()),
        });
        self.callables_by_name.insert(NAME, callable_id);
        callable_id
    }

    fn ensure_postcard_discriminant_reader(&mut self) -> hir::CallableId {
        const NAME: &str = "postcard.read_discriminant";
        if let Some(existing) = self.callables_by_name.get(NAME).copied() {
            return existing;
        }

        let callable_id = self.module.add_callable(hir::CallableSpec {
            kind: hir::CallableKind::Builtin,
            name: NAME.to_owned(),
            signature: hir::CallSignature {
                params: vec![hir::Type::named(
                    self.cursor_type,
                    vec![hir::GenericArg::Region(self.input_region)],
                )],
                returns: vec![hir::Type::u(32)],
                effect_class: hir::EffectClass::Mutates,
                domain_effects: vec![hir::DomainEffect {
                    domain: "cursor".to_owned(),
                    access: hir::DomainAccess::Mutate,
                }],
                control: hir::ControlTransfer::MayFail,
                capabilities: vec!["deser.postcard".to_owned()],
                safety: hir::CallSafety::SafeCore,
            },
            docs: Some("Read a postcard enum discriminant.".to_owned()),
        });
        self.callables_by_name.insert(NAME, callable_id);
        callable_id
    }

    fn lower_type_for_scalar(&self, scalar_type: ScalarType) -> hir::Type {
        match scalar_type {
            ScalarType::Bool => hir::Type::bool(),
            ScalarType::U8 => hir::Type::u(8),
            ScalarType::U16 => hir::Type::u(16),
            ScalarType::U32 => hir::Type::u(32),
            ScalarType::U64 => hir::Type::u(64),
            ScalarType::U128 => hir::Type::u(128),
            ScalarType::USize => hir::Type::u(64),
            ScalarType::I8 => hir::Type::i(8),
            ScalarType::I16 => hir::Type::i(16),
            ScalarType::I32 => hir::Type::i(32),
            ScalarType::I64 => hir::Type::i(64),
            ScalarType::I128 => hir::Type::i(128),
            ScalarType::ISize => hir::Type::i(64),
            other => panic!("unsupported postcard HIR scalar type {other:?}"),
        }
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

        if shape.is_transparent() {
            let (fields, skipped) = collect_fields(shape);
            assert!(
                skipped.is_empty() && fields.len() == 1,
                "transparent HIR prototype expects one lowered field"
            );
            self.lower_shape_into_place(statements, cursor_local, place, fields[0].shape);
            return;
        }

        if let Some(opt_def) = get_option_def(shape) {
            let option_def = self.ensure_type_def(shape);
            let tag_local = self.alloc_local(
                format!("option_is_some_{}", self.locals.len()),
                hir::Type::bool(),
                hir::LocalKind::Temp,
            );
            let tag_reader = self.ensure_postcard_option_tag_reader();
            self.push_init(
                statements,
                hir::Place::Local(tag_local),
                hir::Expr::Call(hir::CallExpr {
                    target: hir::CallTarget::Callable(tag_reader),
                    args: vec![hir::Expr::Local(cursor_local)],
                }),
            );

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
                let disc_reader = self.ensure_postcard_discriminant_reader();
                self.push_init(
                    statements,
                    hir::Place::Local(disc_local),
                    hir::Expr::Call(hir::CallExpr {
                        target: hir::CallTarget::Callable(disc_reader),
                        args: vec![hir::Expr::Local(cursor_local)],
                    }),
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
                ty: hir::Type::place(root_type),
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

pub(crate) fn build_postcard_decoder_ir_via_hir(shape: &'static Shape) -> crate::ir::IrFunc {
    let module = build_postcard_decoder_hir(shape);
    build_structural_hir_ir(shape, &module)
}

#[derive(Clone, Copy)]
struct StructuralLocalStorage {
    base_slot: crate::ir::SlotId,
    slot_count: usize,
}

struct StructuralHirIrLowerer<'a> {
    module: &'a hir::Module,
    decoder: crate::postcard::KajitPostcard,
    cursor_local: hir::LocalId,
    local_slots: std::collections::HashMap<hir::LocalId, StructuralLocalStorage>,
    _marker: std::marker::PhantomData<&'a hir::Module>,
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
        for param in &function.params {
            if param.kind != hir::LocalKind::Destination {
                local_slots.insert(
                    param.local,
                    Self::alloc_local_storage(rb, module, &param.ty),
                );
            }
        }
        for local in &function.locals {
            local_slots.insert(
                local.local,
                Self::alloc_local_storage(rb, module, &local.ty),
            );
        }
        Self {
            module,
            decoder: crate::postcard::KajitPostcard,
            cursor_local,
            local_slots,
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
        StructuralLocalStorage {
            base_slot,
            slot_count,
        }
    }

    fn slot_count_for_type(module: &'a hir::Module, ty: &hir::Type) -> usize {
        match ty {
            hir::Type::Unit | hir::Type::Bool | hir::Type::Integer(_) => 1,
            hir::Type::Str { .. } | hir::Type::Slice { .. } => 2,
            hir::Type::Place(inner) => Self::slot_count_for_type(module, inner),
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
            hir::StmtKind::Return(None) => {}
            other => panic!("unsupported structural HIR statement: {other:?}"),
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
        match place {
            hir::Place::Local(local) => {
                if *local == dest_local {
                    self.lower_place_write(rb, place, value, dest_local, dest_shape);
                } else {
                    let storage = self.local_slots[local];
                    match value {
                        hir::Expr::Call(call) => {
                            self.lower_postcard_call_into_slot(rb, call, storage)
                        }
                        _ => {
                            let scalar = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
                            rb.write_to_slot(storage.base_slot, scalar);
                        }
                    }
                }
            }
            hir::Place::Field { .. } => {
                self.lower_place_write(rb, place, value, dest_local, dest_shape);
            }
            hir::Place::Index { .. } => {
                panic!("structural HIR subset does not support indexed places")
            }
        }
    }

    fn lower_place_write(
        &self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
        value: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        let (place_shape, offset, _) = self.resolve_place(place, dest_local, dest_shape);
        match value {
            hir::Expr::Call(call) => {
                self.lower_postcard_call_at_offset(rb, call, offset);
            }
            hir::Expr::Variant {
                variant, fields, ..
            } => {
                if let Some(opt_def) = get_option_def(place_shape) {
                    self.lower_option_variant_write(rb, offset, *opt_def, variant, fields);
                    return;
                }
                let Type::User(UserType::Enum(enum_type)) = &place_shape.ty else {
                    panic!("variant init must target an enum place");
                };
                let variant_info = collect_variants(enum_type)
                    .into_iter()
                    .find(|candidate| candidate.name == variant.as_str())
                    .unwrap_or_else(|| panic!("missing enum variant {variant}"));
                let disc_width = ir_width_from_disc_size(discriminant_size(enum_type.enum_repr));
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
            hir::Expr::Literal(hir::Literal::Unit) => {}
            _ => {
                let scalar = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
                let width = self.scalar_width_for_shape(place_shape);
                rb.write_to_field(scalar, offset as u32, width);
            }
        }
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
        _dest_local: hir::LocalId,
        _dest_shape: &'static Shape,
    ) {
        match expr {
            hir::Expr::Call(call) => self.lower_postcard_call_at_offset(rb, call, offset),
            hir::Expr::Local(local) => self.copy_local_into_shape_offset(rb, *local, shape, offset),
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

    fn lower_postcard_call_into_slot(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
        storage: StructuralLocalStorage,
    ) {
        for slot_index in 0..storage.slot_count {
            let zero = rb.const_val(0);
            rb.write_to_slot(
                crate::ir::SlotId::new(storage.base_slot.index() as u32 + slot_index as u32),
                zero,
            );
        }
        let saved_out = rb.save_out_ptr();
        let slot_ptr = rb.slot_addr(storage.base_slot);
        rb.set_out_ptr(slot_ptr);
        self.lower_postcard_call_at_offset(rb, call, 0);
        rb.set_out_ptr(saved_out);
    }

    fn lower_postcard_call_at_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
        offset: usize,
    ) {
        assert_eq!(
            call.args,
            vec![hir::Expr::Local(self.cursor_local)],
            "structural postcard readers should consume the cursor local directly"
        );
        match self.callable_name(call) {
            "postcard.read_bool" => self.decoder.lower_read_scalar(rb, offset, ScalarType::Bool),
            "postcard.read_u8" => self.decoder.lower_read_scalar(rb, offset, ScalarType::U8),
            "postcard.read_u16" => self.decoder.lower_read_scalar(rb, offset, ScalarType::U16),
            "postcard.read_u32" => self.decoder.lower_read_scalar(rb, offset, ScalarType::U32),
            "postcard.read_u64" => self.decoder.lower_read_scalar(rb, offset, ScalarType::U64),
            "postcard.read_u128" => self.decoder.lower_read_scalar(rb, offset, ScalarType::U128),
            "postcard.read_usize" => self
                .decoder
                .lower_read_scalar(rb, offset, ScalarType::USize),
            "postcard.read_i8" => self.decoder.lower_read_scalar(rb, offset, ScalarType::I8),
            "postcard.read_i16" => self.decoder.lower_read_scalar(rb, offset, ScalarType::I16),
            "postcard.read_i32" => self.decoder.lower_read_scalar(rb, offset, ScalarType::I32),
            "postcard.read_i64" => self.decoder.lower_read_scalar(rb, offset, ScalarType::I64),
            "postcard.read_i128" => self.decoder.lower_read_scalar(rb, offset, ScalarType::I128),
            "postcard.read_isize" => self
                .decoder
                .lower_read_scalar(rb, offset, ScalarType::ISize),
            "postcard.read_str" => self.decoder.lower_read_string(rb, offset, ScalarType::Str),
            "postcard.read_option_tag" => self.lower_postcard_option_tag(rb, offset),
            "postcard.read_discriminant" => {
                self.decoder.lower_read_scalar(rb, offset, ScalarType::U32)
            }
            other => panic!("unsupported structural postcard reader {other}"),
        }
    }

    fn lower_postcard_option_tag(&self, rb: &mut RegionBuilder<'_>, offset: usize) {
        rb.bounds_check(1);
        let tag = rb.read_bytes(1);
        let width = crate::ir::Width::W1;
        rb.gamma(tag, &[], 3, |branch_idx, branch| {
            match branch_idx {
                0 | 1 => {
                    let value = branch.const_val(branch_idx as u64);
                    branch.write_to_field(value, offset as u32, width);
                }
                2 => branch.error_exit(crate::context::ErrorCode::UnknownVariant),
                _ => unreachable!(),
            }
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
            hir::Expr::Field { base, field } => {
                let place = self.expr_field_to_place(base, field);
                let (shape, offset, _) = self.resolve_place(&place, dest_local, dest_shape);
                let width = self.scalar_width_for_shape(shape);
                rb.read_from_field(offset as u32, width)
            }
            hir::Expr::Binary { op, lhs, rhs } => {
                let lhs = self.lower_scalar_expr(rb, lhs, dest_local, dest_shape);
                let rhs = self.lower_scalar_expr(rb, rhs, dest_local, dest_shape);
                let ir_op = match op {
                    hir::BinaryOp::Add => crate::ir::IrOp::Add,
                    hir::BinaryOp::Sub => crate::ir::IrOp::Sub,
                    hir::BinaryOp::And => crate::ir::IrOp::And,
                    hir::BinaryOp::Or => crate::ir::IrOp::Or,
                    hir::BinaryOp::Ne => crate::ir::IrOp::CmpNe,
                    other => panic!("unsupported structural HIR binary op: {other:?}"),
                };
                rb.binop(ir_op, lhs, rhs)
            }
            other => panic!("unsupported structural HIR scalar expression: {other:?}"),
        }
    }

    fn expr_field_to_place(&self, base: &hir::Expr, field: &str) -> hir::Place {
        match base {
            hir::Expr::Local(local) => hir::Place::Field {
                base: Box::new(hir::Place::Local(*local)),
                field: field.to_owned(),
            },
            hir::Expr::Field {
                base: inner_base,
                field: inner_field,
            } => hir::Place::Field {
                base: Box::new(self.expr_field_to_place(inner_base, inner_field)),
                field: field.to_owned(),
            },
            other => panic!("unsupported structural HIR field base: {other:?}"),
        }
    }

    fn resolve_place(
        &self,
        place: &hir::Place,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) -> (&'static Shape, usize, Option<&'static str>) {
        match place {
            hir::Place::Local(local) => {
                assert_eq!(
                    *local, dest_local,
                    "structural HIR subset only supports local root on the destination parameter"
                );
                (dest_shape, 0, None)
            }
            hir::Place::Field { base, field } => {
                let (base_shape, base_offset, _) = self.resolve_place(base, dest_local, dest_shape);
                let (fields, skipped) = collect_fields(base_shape);
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
                            base_shape.type_identifier
                        )
                    });
                (
                    field_info.shape,
                    base_offset + field_info.offset,
                    Some(field_info.name),
                )
            }
            hir::Place::Index { .. } => panic!("indexed places are unsupported"),
        }
    }

    fn scalar_width_for_shape(&self, shape: &'static Shape) -> crate::ir::Width {
        match shape.scalar_type() {
            Some(ScalarType::Bool | ScalarType::U8 | ScalarType::I8) => crate::ir::Width::W1,
            Some(ScalarType::U16 | ScalarType::I16) => crate::ir::Width::W2,
            Some(ScalarType::U32 | ScalarType::I32) => crate::ir::Width::W4,
            Some(
                ScalarType::U64
                | ScalarType::I64
                | ScalarType::USize
                | ScalarType::ISize
                | ScalarType::U128
                | ScalarType::I128,
            ) => crate::ir::Width::W8,
            _ => panic!(
                "unsupported structural HIR scalar width for {}",
                shape.type_identifier
            ),
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
        .params
        .iter()
        .find(|param| param.kind == hir::LocalKind::Destination)
        .map(|param| param.local)
        .expect("structural HIR function should have a destination param");

    let mut builder = crate::ir::IrBuilder::new(shape);
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
    struct BranchyAnimal {
        animal: UnitAnimal,
        value: u32,
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

        let hir::Type::Place(out_ty) = &function.params[1].ty else {
            panic!("expected destination Place<T> param");
        };
        let hir::Type::Named { def, args } = &**out_ty else {
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
        assert_eq!(statements.len(), 3);

        let hir::StmtKind::Init { place, value } = &statements[0].kind else {
            panic!("expected first init statement");
        };
        let hir::Place::Field { field, .. } = place else {
            panic!("expected field place");
        };
        assert_eq!(field, "len");
        let hir::Expr::Call(call) = value else {
            panic!("expected scalar read call");
        };
        let hir::CallTarget::Callable(callable_id) = call.target;
        assert_eq!(module.callables[callable_id].name, "postcard.read_u32");

        let hir::StmtKind::Init { place, value } = &statements[1].kind else {
            panic!("expected second init statement");
        };
        let hir::Place::Field { field, .. } = place else {
            panic!("expected field place");
        };
        assert_eq!(field, "name");
        let hir::Expr::Call(call) = value else {
            panic!("expected borrowed string read call");
        };
        let hir::CallTarget::Callable(callable_id) = call.target;
        let callable = &module.callables[callable_id];
        assert_eq!(callable.name, "postcard.read_str");
        assert_eq!(
            callable.signature.returns,
            vec![hir::Type::str(function.region_params[0])]
        );
        assert_eq!(
            callable.signature.domain_effects,
            vec![hir::DomainEffect {
                domain: "cursor".to_owned(),
                access: hir::DomainAccess::Mutate,
            }]
        );
    }

    #[test]
    fn postcard_hir_models_option_borrowed_fields() {
        let module = build_postcard_decoder_hir(<MaybeBorrowedName<'static>>::SHAPE);
        let (_, function) = module.functions.iter().next().unwrap();

        assert_eq!(function.locals.len(), 2);
        assert_eq!(function.body.statements.len(), 3);

        let hir::StmtKind::Init { value, .. } = &function.body.statements[0].kind else {
            panic!("expected option-tag init");
        };
        let hir::Expr::Call(call) = value else {
            panic!("expected option-tag call");
        };
        let hir::CallTarget::Callable(tag_reader) = call.target;
        let tag_reader = &module.callables[tag_reader];
        assert_eq!(tag_reader.name, "postcard.read_option_tag");
        assert_eq!(tag_reader.signature.returns, vec![hir::Type::bool()]);

        let hir::StmtKind::If {
            condition,
            then_block,
            else_block,
        } = &function.body.statements[1].kind
        else {
            panic!("expected option if statement");
        };
        let hir::Expr::Local(tag_local) = condition else {
            panic!("expected local option tag condition");
        };
        assert_eq!(function.locals[0].local, *tag_local);
        assert_eq!(function.locals[0].ty, hir::Type::bool());

        let Some(else_block) = else_block else {
            panic!("expected explicit option else block");
        };
        assert_eq!(then_block.statements.len(), 2);
        assert_eq!(else_block.statements.len(), 1);

        let hir::StmtKind::Init { value, .. } = &then_block.statements[0].kind else {
            panic!("expected option payload init");
        };
        let hir::Expr::Call(call) = value else {
            panic!("expected borrowed payload read call");
        };
        let hir::CallTarget::Callable(payload_reader) = call.target;
        assert_eq!(module.callables[payload_reader].name, "postcard.read_str");
        assert_eq!(
            function.locals[1].ty,
            hir::Type::str(function.region_params[0])
        );

        let hir::StmtKind::Init { value, .. } = &then_block.statements[1].kind else {
            panic!("expected Some variant init");
        };
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
        assert_eq!(function.locals[1].local, *payload_local);

        let hir::TypeDefKind::Enum { variants } = &module.type_defs[*def].kind else {
            panic!("expected Option HIR enum type");
        };
        assert_eq!(variants[0].name, "None");
        assert_eq!(variants[1].name, "Some");
        assert_eq!(
            variants[1].fields[0].ty,
            hir::Type::str(function.region_params[0])
        );

        let hir::StmtKind::Init { value, .. } = &else_block.statements[0].kind else {
            panic!("expected None variant init");
        };
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
        let module = build_postcard_decoder_hir(<MaybeBorrowedName<'static>>::SHAPE);
        let text = module.to_string();
        let reparsed = parse_hir(&text).expect("postcard HIR text should parse");

        assert_eq!(reparsed, module);
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
    fn postcard_hir_models_unit_enums() {
        let module = build_postcard_decoder_hir(<UnitAnimal>::SHAPE);
        let (_, function) = module.functions.iter().next().unwrap();

        assert_eq!(function.locals.len(), 1);
        assert_eq!(function.body.statements.len(), 3);

        let hir::StmtKind::Init { value, .. } = &function.body.statements[0].kind else {
            panic!("expected enum discriminant init");
        };
        let hir::Expr::Call(call) = value else {
            panic!("expected discriminant reader call");
        };
        let hir::CallTarget::Callable(callable_id) = call.target;
        assert_eq!(
            module.callables[callable_id].name,
            "postcard.read_discriminant"
        );

        let hir::StmtKind::Match { scrutinee, arms } = &function.body.statements[1].kind else {
            panic!("expected enum match statement");
        };
        let hir::Expr::Local(disc_local) = scrutinee else {
            panic!("expected discriminant local");
        };
        assert_eq!(function.locals[0].local, *disc_local);
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

        let hir::StmtKind::Match { arms, .. } = &function.body.statements[1].kind else {
            panic!("expected enum match statement");
        };
        assert_eq!(arms.len(), 3);

        let count_arm = &arms[1];
        assert_eq!(count_arm.body.statements.len(), 2);
        let hir::StmtKind::Init { value, .. } = &count_arm.body.statements[0].kind else {
            panic!("expected payload init in Count arm");
        };
        let hir::Expr::Call(call) = value else {
            panic!("expected Count payload reader call");
        };
        let hir::CallTarget::Callable(callable_id) = call.target;
        assert_eq!(module.callables[callable_id].name, "postcard.read_u32");

        let hir::StmtKind::Init { value, .. } = &count_arm.body.statements[1].kind else {
            panic!("expected Count variant init");
        };
        let hir::Expr::Variant {
            variant, fields, ..
        } = value
        else {
            panic!("expected Count variant expression");
        };
        assert_eq!(variant, "Count");
        assert_eq!(fields.len(), 1);

        let name_arm = &arms[2];
        assert_eq!(name_arm.body.statements.len(), 2);
        let hir::StmtKind::Init { value, .. } = &name_arm.body.statements[0].kind else {
            panic!("expected payload init in Name arm");
        };
        let hir::Expr::Call(call) = value else {
            panic!("expected Name payload reader call");
        };
        let hir::CallTarget::Callable(callable_id) = call.target;
        assert_eq!(module.callables[callable_id].name, "postcard.read_str");
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
                    ty: hir::Type::place(hir::Type::named(root_def, Vec::new())),
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
                    ty: hir::Type::place(hir::Type::named(root_def, Vec::new())),
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
