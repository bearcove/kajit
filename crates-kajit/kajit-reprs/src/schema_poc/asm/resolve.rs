use std::collections::HashMap;

use super::*;
use crate::schema_poc::{ResolutionSet, ResolvedRef, SymbolDef, SymbolKind, SymbolRef};

fn span_bounds(prov: &Prov) -> Option<(u32, u32)> {
    let span = prov.span.as_ref()?;
    Some((span.start, span.end))
}

fn push_label_definitions(items: &[X64Item], definitions: &mut Vec<SymbolDef>) {
    for item in items {
        if let X64Item::Label { name, .. } = item
            && let Some((start, end)) = span_bounds(&name.prov)
        {
            definitions.push(SymbolDef {
                name: name.text.clone(),
                kind: SymbolKind::Label,
                start,
                end,
                detail: None,
                docs: None,
            });
        }
    }
}

fn push_a64_label_definitions(items: &[A64Item], definitions: &mut Vec<SymbolDef>) {
    for item in items {
        if let A64Item::Label { name, .. } = item
            && let Some((start, end)) = span_bounds(&name.prov)
        {
            definitions.push(SymbolDef {
                name: name.text.clone(),
                kind: SymbolKind::Label,
                start,
                end,
                detail: None,
                docs: None,
            });
        }
    }
}

fn push_x64_references(
    items: &[X64Item],
    label_targets: &HashMap<String, usize>,
    references: &mut Vec<ResolvedRef>,
) {
    for item in items {
        if let X64Item::Jmp { target, .. } = item
            && let Some((start, end)) = span_bounds(&target.prov)
        {
            references.push(ResolvedRef {
                reference: SymbolRef {
                    name: target.text.clone(),
                    kind: SymbolKind::Label,
                    start,
                    end,
                },
                target: label_targets.get(&target.text).copied(),
            });
        }
    }
}

fn push_a64_references(
    items: &[A64Item],
    label_targets: &HashMap<String, usize>,
    references: &mut Vec<ResolvedRef>,
) {
    for item in items {
        if let A64Item::B { target, .. } = item
            && let Some((start, end)) = span_bounds(&target.prov)
        {
            references.push(ResolvedRef {
                reference: SymbolRef {
                    name: target.text.clone(),
                    kind: SymbolKind::Label,
                    start,
                    end,
                },
                target: label_targets.get(&target.text).copied(),
            });
        }
    }
}

pub fn resolve_program(program: &Program) -> ResolutionSet {
    let mut definitions = Vec::new();
    match program {
        Program::AArch64 { items, .. } => push_a64_label_definitions(items, &mut definitions),
        Program::X86_64 { items, .. } => push_label_definitions(items, &mut definitions),
    }

    let label_targets = definitions
        .iter()
        .enumerate()
        .map(|(idx, def)| (def.name.clone(), idx))
        .collect::<HashMap<_, _>>();

    let mut references = Vec::new();
    match program {
        Program::AArch64 { items, .. } => {
            push_a64_references(items, &label_targets, &mut references)
        }
        Program::X86_64 { items, .. } => {
            push_x64_references(items, &label_targets, &mut references)
        }
    }

    ResolutionSet {
        definitions,
        references,
    }
}

pub fn resolve(source: &str) -> Result<ResolutionSet, String> {
    let program = parse_root_text(source)?;
    Ok(resolve_program(&program))
}
