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
    let rows = normalize_debug_line_rows(source_map);

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

pub(super) fn normalize_debug_line_rows(
    source_map: &kajit_emit::SourceMap,
) -> Vec<crate::jit_dwarf::JitDebugLineRow> {
    let mut rows = source_map
        .iter()
        .filter(|entry| entry.location.line != 0)
        .map(|entry| crate::jit_dwarf::JitDebugLineRow {
            code_offset: entry.offset,
            line: entry.location.line,
        })
        .collect::<Vec<_>>();

    if let Some(first) = rows.first().cloned()
        && first.code_offset != 0
    {
        rows.insert(
            0,
            crate::jit_dwarf::JitDebugLineRow {
                code_offset: 0,
                line: first.line,
            },
        );
    }

    rows
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
    location_map: &crate::harness::LocationMap,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    code_ptr: *const u8,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
) -> Vec<ScopedDwarfVariable> {
    let Some(backend_debug_info) = backend_debug_info else {
        return Vec::new();
    };
    const PTR_WIDTH_BYTES: u8 = 8;

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

    let out_ptr = program
        .funcs
        .iter()
        .find(|func| func.lambda_id == root_lambda)
        .and_then(|func| func.data_args.first().copied());
    let Some(out_ptr) = out_ptr else {
        return Vec::new();
    };

    let out_ptr_locations = backend_debug_info
        .op_infos
        .iter()
        .filter(|op_info| op_info.lambda_id == root_lambda.index() as u32)
        .filter_map(|op_info| {
            let out_ptr_loc = location_map.location_at(op_info.line, out_ptr.index() as u32)?;
            Some((op_info, out_ptr_loc.clone()))
        })
        .flat_map(|(op_info, out_ptr_loc)| {
            op_info.code_ranges.iter().map(move |range| {
                let start = (code_ptr as u64) + (range.start_offset as u64);
                let end = (code_ptr as u64) + (range.end_offset as u64);
                (start, end, out_ptr_loc.clone())
            })
        })
        .collect::<Vec<_>>();
    let out_ptr_locations = {
        let mut ranges = out_ptr_locations;
        ranges.sort_by_key(|(start, _, _)| *start);
        let mut out = Vec::<(u64, u64, crate::harness::VRegLocation)>::new();
        for (start, end, loc) in ranges {
            if end <= start {
                continue;
            }
            let mut gap_fill = None::<(u64, u64, crate::harness::VRegLocation)>;
            if let Some((_, prev_end, prev_loc)) = out.last_mut() {
                // If there's a gap, conservatively extend the previous location through it.
                if *prev_end < start {
                    gap_fill = Some((*prev_end, start, prev_loc.clone()));
                }
                // Merge adjacent/overlapping segments with identical locations.
                if prev_loc == &loc && start <= *prev_end {
                    *prev_end = (*prev_end).max(end);
                    continue;
                }
            }
            if let Some(gap_fill) = gap_fill {
                out.push(gap_fill);
            }
            out.push((start, end, loc));
        }
        // Ensure coverage up to code_end if we have any location info at all.
        if let Some((_, last_end, last_loc)) = out.last().cloned()
            && last_end < code_end
        {
            out.push((last_end, code_end, last_loc));
        }
        out
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
            for block in func.live_blocks() {
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
                        // General-purpose IR: field writes become a generic StoreToAddr (or a Call)
                        // with debug provenance attached. At this point we only need to find a
                        // single op that (a) carries the field debug value and (b) is actually
                        // an effectful write/call site.
                        crate::linearize::LinearOp::StoreToAddr { .. }
                        | crate::linearize::LinearOp::CallLambda { .. } => true,
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

        let mut locations = Vec::<crate::jit_dwarf::DwarfLocationRange>::new();
        for (start, end, out_ptr_loc) in &out_ptr_locations {
            let start = (*start).max(available_start);
            let end = (*end).min(code_end);
            if end <= start {
                continue;
            }
            let expr = match out_ptr_loc {
                crate::harness::VRegLocation::Register(preg) => {
                    let Some(out_ptr_reg) =
                        crate::jit_dwarf::dwarf_register_from_hw_encoding(target_arch, *preg)
                    else {
                        continue;
                    };
                    crate::jit_dwarf::expr_breg_deref_size_stack_value(
                        out_ptr_reg,
                        field.offset as i64,
                        width,
                    )
                }
                crate::harness::VRegLocation::StackSlot(offset) => {
                    let mut expr =
                        crate::jit_dwarf::expr_fbreg_deref_size(*offset as i64, PTR_WIDTH_BYTES);
                    expr.extend(crate::jit_dwarf::expr_plus_uconst(field.offset as u64));
                    expr.extend(crate::jit_dwarf::expr_deref_size(width));
                    expr.extend(crate::jit_dwarf::expr_stack_value());
                    expr
                }
                crate::harness::VRegLocation::Constant(_) => continue,
            };
            locations.push(crate::jit_dwarf::DwarfLocationRange {
                start,
                end,
                expression: expr,
            });
        }
        let locations = merge_dwarf_location_ranges(locations);
        if locations.is_empty() {
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
                location: crate::jit_dwarf::DwarfVariableLocation::List(locations),
            },
        });
    }
    out
}

pub(super) fn dwarf_expr_for_vreg_location(
    location: &crate::harness::VRegLocation,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
) -> Option<Vec<u8>> {
    match location {
        crate::harness::VRegLocation::Register(preg) => {
            let dwarf_reg = crate::jit_dwarf::dwarf_register_from_hw_encoding(target_arch, *preg)?;
            Some(crate::jit_dwarf::expr_reg(dwarf_reg))
        }
        crate::harness::VRegLocation::StackSlot(offset) => {
            Some(crate::jit_dwarf::expr_fbreg(*offset as i64))
        }
        crate::harness::VRegLocation::Constant(_) => None,
    }
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

pub(super) fn backend_line_by_op(
    backend_debug_info: &crate::ir_backend::BackendDebugInfo,
) -> BTreeMap<(u32, crate::regalloc_engine::cfg_mir::OpId), u32> {
    backend_debug_info
        .op_infos
        .iter()
        .map(|op_info| ((op_info.lambda_id, op_info.op_id), op_info.line))
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
    location_map: &crate::harness::LocationMap,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    code_ptr: *const u8,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
) -> BTreeMap<crate::ir::VReg, VRegDwarfVariableInfo> {
    let Some(backend_debug_info) = backend_debug_info else {
        return BTreeMap::new();
    };
    let op_ranges = backend_op_ranges_by_op(backend_debug_info, code_ptr);
    let op_lines = backend_line_by_op(backend_debug_info);
    let mut ranges_by_vreg =
        BTreeMap::<crate::ir::VReg, Vec<crate::jit_dwarf::DwarfLocationRange>>::new();
    let mut lexical_intro_ranges_by_vreg =
        BTreeMap::<crate::ir::VReg, Vec<crate::jit_dwarf::JitDebugRange>>::new();

    for func in &program.funcs {
        let lambda_key = func.lambda_id.index() as u32;
        for block in func.live_blocks() {
            let mut remaining_uses = BTreeMap::<crate::ir::VReg, usize>::new();
            for inst_id in &block.insts {
                for operand in &func.insts[inst_id.index()].operands {
                    if operand.kind == crate::regalloc_engine::cfg_mir::OperandKind::Use {
                        *remaining_uses.entry(operand.vreg).or_default() += 1;
                    }
                }
            }
            let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(block.term);
            if let Some(term_inst) = func.term(block.term) {
                match term_inst {
                    crate::regalloc_engine::cfg_mir::Terminator::BranchIf { cond, .. }
                    | crate::regalloc_engine::cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                        *remaining_uses.entry(*cond).or_default() += 1;
                    }
                    crate::regalloc_engine::cfg_mir::Terminator::JumpTable {
                        predicate, ..
                    } => {
                        *remaining_uses.entry(*predicate).or_default() += 1;
                    }
                    crate::regalloc_engine::cfg_mir::Terminator::Return
                    | crate::regalloc_engine::cfg_mir::Terminator::ErrorExit { .. }
                    | crate::regalloc_engine::cfg_mir::Terminator::Branch { .. } => {}
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

            let mut live_vregs = BTreeMap::<crate::ir::VReg, ()>::new();
            for &param in &block.params {
                if remaining_uses.get(&param).copied().unwrap_or(0) == 0 {
                    continue;
                }
                if !location_map
                    .static_locations
                    .contains_key(&(param.index() as u32))
                {
                    continue;
                }
                live_vregs.insert(param, ());
            }

            for inst_id in &block.insts {
                let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(*inst_id);
                let Some(op_ranges) = op_ranges.get(&(lambda_key, op_id)) else {
                    continue;
                };
                let Some(op_line) = op_lines.get(&(lambda_key, op_id)).copied() else {
                    continue;
                };
                let mut used_now = Vec::<crate::ir::VReg>::new();
                let mut defs_after = Vec::<crate::ir::VReg>::new();
                for operand in &func.insts[inst_id.index()].operands {
                    match operand.kind {
                        crate::regalloc_engine::cfg_mir::OperandKind::Use => {
                            if location_map
                                .static_locations
                                .contains_key(&(operand.vreg.index() as u32))
                            {
                                live_vregs.insert(operand.vreg, ());
                            }
                            used_now.push(operand.vreg);
                        }
                        crate::regalloc_engine::cfg_mir::OperandKind::Def => {
                            let dest = lexical_intro_ranges_by_vreg
                                .entry(operand.vreg)
                                .or_default();
                            dest.extend(op_ranges.iter().map(|(start, end)| {
                                crate::jit_dwarf::JitDebugRange {
                                    low_pc: *start,
                                    high_pc: *end,
                                }
                            }));
                            defs_after.push(operand.vreg);
                        }
                    }
                }

                for vreg in live_vregs.keys() {
                    if remaining_uses.get(vreg).copied().unwrap_or(0) == 0 {
                        continue;
                    }
                    let Some(location) = location_map.location_at(op_line, vreg.index() as u32)
                    else {
                        continue;
                    };
                    let Some(expr) = dwarf_expr_for_vreg_location(location, target_arch) else {
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
                live_vregs.retain(|vreg, _| remaining_uses.get(vreg).copied().unwrap_or(0) > 0);
                for vreg in defs_after {
                    if remaining_uses.get(&vreg).copied().unwrap_or(0) > 0 {
                        live_vregs.insert(vreg, ());
                    }
                }
            }

            let Some(op_ranges) = op_ranges.get(&(lambda_key, term_op)) else {
                continue;
            };
            let Some(op_line) = op_lines.get(&(lambda_key, term_op)).copied() else {
                continue;
            };
            let mut used_now = Vec::<crate::ir::VReg>::new();
            if let Some(term_inst) = func.term(block.term) {
                match term_inst {
                    crate::regalloc_engine::cfg_mir::Terminator::BranchIf { cond, .. }
                    | crate::regalloc_engine::cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                        if location_map
                            .static_locations
                            .contains_key(&(cond.index() as u32))
                        {
                            live_vregs.insert(*cond, ());
                            used_now.push(*cond);
                        }
                    }
                    crate::regalloc_engine::cfg_mir::Terminator::JumpTable {
                        predicate, ..
                    } => {
                        if location_map
                            .static_locations
                            .contains_key(&(predicate.index() as u32))
                        {
                            live_vregs.insert(*predicate, ());
                            used_now.push(*predicate);
                        }
                    }
                    crate::regalloc_engine::cfg_mir::Terminator::Return
                    | crate::regalloc_engine::cfg_mir::Terminator::ErrorExit { .. }
                    | crate::regalloc_engine::cfg_mir::Terminator::Branch { .. } => {}
                }
            }
            for vreg in live_vregs.keys() {
                if remaining_uses.get(vreg).copied().unwrap_or(0) == 0 {
                    continue;
                }
                let Some(location) = location_map.location_at(op_line, vreg.index() as u32) else {
                    continue;
                };
                let Some(expr) = dwarf_expr_for_vreg_location(location, target_arch) else {
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
    location_map: &crate::harness::LocationMap,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    code_ptr: *const u8,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
    suppress_semantic_vregs: bool,
) -> Vec<ScopedDwarfVariable> {
    cfg_vreg_dwarf_variable_infos(
        program,
        location_map,
        backend_debug_info,
        code_ptr,
        target_arch,
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
    location_map: &crate::harness::LocationMap,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    code_ptr: *const u8,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
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
        location_map,
        backend_debug_info,
        code_ptr,
        target_arch,
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
    location_map: &crate::harness::LocationMap,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    code_ptr: *const u8,
    target_arch: crate::jit_dwarf::DwarfTargetArch,
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
        location_map,
        backend_debug_info,
        code_ptr,
        target_arch,
        suppress_semantic_vregs,
    );
    cfg_variables.extend(cfg_semantic_named_dwarf_variables(
        program,
        location_map,
        backend_debug_info,
        code_ptr,
        target_arch,
    ));
    if let Some(root_shape) = root_shape {
        cfg_variables.extend(cfg_semantic_field_dwarf_variables(
            root_shape,
            program,
            location_map,
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
