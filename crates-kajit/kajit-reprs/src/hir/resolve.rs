use std::collections::HashMap;

use super::*;
use crate::{ResolutionSet, ResolvedRef, SymbolDef, SymbolKind, SymbolRef};

fn span_bounds(prov: &Prov) -> Option<(u32, u32)> {
    let span = prov.span.as_ref()?;
    Some((span.start, span.end))
}

fn symbol_lookup_key(text: &str) -> &str {
    text.strip_prefix('@').unwrap_or(text)
}

fn doc_markdown(docs: &Option<DocBlock>) -> Option<String> {
    docs.as_ref().map(|docs| docs.0.join("\n"))
}

fn function_signature(function: &Function) -> String {
    let params = function
        .params
        .iter()
        .map(|param| format!("{}: {}", param.name.text, param.ty.text))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "fn {}({params}) -> {}",
        function.name.text, function.return_type.text
    )
}

fn type_signature(ty: &TypeDef) -> String {
    let params = if ty.params.is_empty() {
        String::new()
    } else {
        format!(
            "<{}>",
            ty.params
                .iter()
                .map(|param| param.text.clone())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    format!("type {}{}", ty.name.text, params)
}

fn push_expr_refs(
    expr: &Expr,
    function_targets: &HashMap<String, usize>,
    references: &mut Vec<ResolvedRef>,
) {
    match expr {
        Expr::Binary { lhs, rhs, .. } => {
            push_expr_refs(lhs, function_targets, references);
            push_expr_refs(rhs, function_targets, references);
        }
        Expr::Call { args, callee, .. } => {
            if let Some((start, end)) = span_bounds(&callee.prov) {
                let key = symbol_lookup_key(&callee.text);
                references.push(ResolvedRef {
                    reference: SymbolRef {
                        name: key.to_owned(),
                        kind: SymbolKind::Function,
                        start,
                        end,
                    },
                    target: function_targets.get(key).copied(),
                });
            }
            for arg in args {
                push_expr_refs(arg, function_targets, references);
            }
        }
        Expr::Field { base, .. } => push_expr_refs(base, function_targets, references),
        Expr::Literal { .. } | Expr::Local { .. } => {}
    }
}

fn push_place_refs(
    place: &Place,
    references: &mut Vec<ResolvedRef>,
    type_targets: &HashMap<String, usize>,
) {
    match place {
        Place::Field { base, .. } => push_place_refs(base, references, type_targets),
        Place::Local { .. } => {
            let _ = (references, type_targets);
        }
    }
}

fn push_stmt_refs(
    stmt: &Stmt,
    function_targets: &HashMap<String, usize>,
    references: &mut Vec<ResolvedRef>,
    type_targets: &HashMap<String, usize>,
) {
    match stmt {
        Stmt::Assign { place, value, .. } | Stmt::Init { place, value, .. } => {
            push_place_refs(place, references, type_targets);
            push_expr_refs(value, function_targets, references);
        }
        Stmt::Expr { value, .. } => push_expr_refs(value, function_targets, references),
        Stmt::If {
            condition,
            then,
            r#else,
            ..
        } => {
            push_expr_refs(condition, function_targets, references);
            push_block_refs(then, function_targets, references, type_targets);
            if let Some(r#else) = r#else {
                push_block_refs(r#else, function_targets, references, type_targets);
            }
        }
        Stmt::Return { value, .. } => {
            if let Some(value) = value {
                push_expr_refs(value, function_targets, references);
            }
        }
    }
}

fn push_block_refs(
    block: &Block,
    function_targets: &HashMap<String, usize>,
    references: &mut Vec<ResolvedRef>,
    type_targets: &HashMap<String, usize>,
) {
    for stmt in &block.statements {
        push_stmt_refs(stmt, function_targets, references, type_targets);
    }
}

fn push_type_ref(
    ty: &Type,
    type_targets: &HashMap<String, usize>,
    references: &mut Vec<ResolvedRef>,
) {
    if let Some((start, end)) = span_bounds(&ty.prov) {
        references.push(ResolvedRef {
            reference: SymbolRef {
                name: ty.text.clone(),
                kind: SymbolKind::Type,
                start,
                end,
            },
            target: type_targets.get(&ty.text).copied(),
        });
    }
}

pub fn resolve_module(module: &Module) -> ResolutionSet {
    let mut definitions = Vec::new();

    for function in &module.functions {
        if let Some((start, end)) = span_bounds(&function.name.prov) {
            definitions.push(SymbolDef {
                name: function.name.text.clone(),
                kind: SymbolKind::Function,
                start,
                end,
                detail: Some(function_signature(function)),
                docs: doc_markdown(&function.docs),
            });
        }
    }

    for ty in &module.type_defs {
        if let Some((start, end)) = span_bounds(&ty.name.prov) {
            definitions.push(SymbolDef {
                name: ty.name.text.clone(),
                kind: SymbolKind::Type,
                start,
                end,
                detail: Some(type_signature(ty)),
                docs: doc_markdown(&ty.docs),
            });
        }
    }

    let function_targets = definitions
        .iter()
        .enumerate()
        .filter(|(_, def)| def.kind == SymbolKind::Function)
        .map(|(idx, def)| (def.name.clone(), idx))
        .collect::<HashMap<_, _>>();
    let type_targets = definitions
        .iter()
        .enumerate()
        .filter(|(_, def)| def.kind == SymbolKind::Type)
        .map(|(idx, def)| (def.name.clone(), idx))
        .collect::<HashMap<_, _>>();

    let mut references = Vec::new();
    for function in &module.functions {
        push_type_ref(&function.return_type, &type_targets, &mut references);
        for param in &function.params {
            push_type_ref(&param.ty, &type_targets, &mut references);
        }
        for local in &function.locals {
            push_type_ref(&local.ty, &type_targets, &mut references);
        }
        push_block_refs(
            &function.body,
            &function_targets,
            &mut references,
            &type_targets,
        );
    }

    ResolutionSet {
        definitions,
        references,
    }
}

pub fn resolve(source: &str) -> Result<ResolutionSet, String> {
    let module = parse_root_text(source)?;
    Ok(resolve_module(&module))
}
