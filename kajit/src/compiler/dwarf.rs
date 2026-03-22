//! DWARF debug info generation for JIT-compiled code.

use super::*;

pub(super) fn jit_debug_enabled() -> bool {
    let Ok(raw) = std::env::var("KAJIT_DEBUG") else {
        return false;
    };
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub(super) fn sanitize_debug_file_stem(name: &str) -> String {
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

pub(super) struct CfgMirListing {
    pub(super) text: String,
    pub(super) line_text_by_line: Vec<String>,
}

pub(super) fn build_cfg_mir_listing(
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

pub(super) fn format_emission_trace_entries(
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

pub(super) fn should_redact_trace_bytes(line_text: &str) -> bool {
    line_text.contains("const(@")
        || line_text.contains("call_intrinsic(@")
        || line_text.contains("call_pure(@")
}

pub(super) fn write_cfg_mir_listing_file(type_name: &str, listing: &str) -> Option<PathBuf> {
    let stem = sanitize_debug_file_stem(type_name);
    let dir = Path::new("/tmp/kajit-debug");
    std::fs::create_dir_all(dir).ok()?;
    let path = dir.join(format!("{stem}.cfg-mir"));
    std::fs::write(&path, listing).ok()?;
    Some(path)
}

pub(super) fn jit_dwarf_target_arch() -> crate::jit_dwarf::DwarfTargetArch {
    if cfg!(target_arch = "x86_64") {
        crate::jit_dwarf::DwarfTargetArch::X86_64
    } else if cfg!(target_arch = "aarch64") {
        crate::jit_dwarf::DwarfTargetArch::Aarch64
    } else {
        panic!("unsupported target architecture for DWARF generation")
    }
}

pub(super) fn build_jit_debug_info_from_source_map(
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
pub(super) struct DeserDebugRegisterSet {
    pub(super) input_ptr_hw: u8,
    pub(super) input_end_hw: u8,
    pub(super) out_ptr_hw: u8,
    pub(super) ctx_hw: u8,
}

pub(super) fn deser_debug_registers(
    target_arch: crate::jit_dwarf::DwarfTargetArch,
) -> DeserDebugRegisterSet {
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

pub(super) fn deser_dwarf_variables(
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

pub(super) fn scalar_field_dwarf_width(shape: &'static Shape) -> Option<u8> {
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

pub(super) fn dwarf_expr_for_out_field(
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

pub(super) fn cfg_semantic_field_dwarf_variables(
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
pub(super) fn aarch64_regalloc_extra_saved_pairs(
    alloc: &crate::regalloc_engine::AllocatedCfgProgram,
) -> u32 {
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

pub(super) fn find_cfg_alloc_for_vreg_in_op(
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

pub(super) fn infer_cfg_block_param_entry_alloc(
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

pub(super) fn dwarf_expr_for_cfg_allocation(
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

pub(super) fn backend_op_ranges_by_op(
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
pub(super) struct ScopedDwarfVariable {
    pub(super) scope: Option<crate::ir::DebugScopeId>,
    pub(super) lexical_ranges: Vec<crate::jit_dwarf::JitDebugRange>,
    pub(super) variable: crate::jit_dwarf::DwarfVariable,
}

#[derive(Debug, Clone)]
pub(super) struct VRegDwarfVariableInfo {
    pub(super) scope: Option<crate::ir::DebugScopeId>,
    pub(super) lexical_intro_ranges: Vec<crate::jit_dwarf::JitDebugRange>,
    pub(super) locations: Vec<crate::jit_dwarf::DwarfLocationRange>,
}

pub(super) fn merge_jit_debug_ranges(
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

pub(super) fn merge_dwarf_location_ranges(
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

pub(super) fn common_debug_scope(
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

pub(super) fn build_variable_interval_blocks(
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

pub(super) fn scope_ranges_from_backend(
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

pub(super) fn cfg_vreg_dwarf_variable_infos(
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

pub(super) fn cfg_value_dwarf_variables(
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

pub(super) fn cfg_semantic_named_dwarf_variables(
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

pub(super) fn cfg_mir_dwarf_variables(
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

pub(super) fn cfg_mir_lexical_blocks(
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
