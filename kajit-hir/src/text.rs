use std::fmt;

use crate::{
    AllocationDomain, BinaryOp, Block, CallExpr, CallSafety, CallTarget, CallableKind,
    ControlTransfer, DomainAccess, EffectClass, Expr, Function, GenericArg, GenericParam, Literal,
    LocalId, LocalKind, MatchArm, Module, Pattern, PatternField, Place, Scope, Stmt, StmtKind,
    Type, TypeDef, TypeDefKind, UnaryOp, VariantDef,
};

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "hir_module {{")?;
        fmt_regions(self, f, 1)?;
        fmt_stores(self, f, 1)?;
        fmt_types(self, f, 1)?;
        fmt_callables(self, f, 1)?;
        fmt_functions(self, f, 1)?;
        writeln!(f, "}}")
    }
}

fn pad(indent: usize) -> String {
    "  ".repeat(indent)
}

fn quoted(text: &str) -> String {
    format!("{text:?}")
}

fn fmt_regions(module: &Module, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
    let pad = pad(indent);
    writeln!(f, "{pad}regions [")?;
    for (id, region) in module.regions.iter() {
        writeln!(f, "{}  r{} {}", pad, id.index(), quoted(&region.name))?;
    }
    writeln!(f, "{pad}]")
}

fn fmt_stores(module: &Module, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
    let pad = pad(indent);
    writeln!(f, "{pad}stores [")?;
    for (id, store) in module.stores.iter() {
        writeln!(f, "{}  store{} {}", pad, id.index(), quoted(&store.name))?;
    }
    writeln!(f, "{pad}]")
}

fn fmt_types(module: &Module, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
    let pad = pad(indent);
    writeln!(f, "{pad}types [")?;
    for (id, type_def) in module.type_defs.iter() {
        fmt_type_def(module, id.index(), type_def, f, indent + 1)?;
    }
    writeln!(f, "{pad}]")
}

fn fmt_type_def(
    module: &Module,
    index: usize,
    type_def: &TypeDef,
    f: &mut fmt::Formatter<'_>,
    indent: usize,
) -> fmt::Result {
    let pad = pad(indent);
    write!(f, "{pad}type t{index} {}", quoted(&type_def.name))?;
    fmt_generic_params(&type_def.generic_params, f)?;
    match &type_def.kind {
        TypeDefKind::Struct { fields } => {
            writeln!(f, " = struct {{")?;
            for field in fields {
                writeln!(
                    f,
                    "{}  {}: {}",
                    pad,
                    quoted(&field.name),
                    TypeDisplay {
                        module,
                        ty: &field.ty
                    }
                )?;
            }
            writeln!(f, "{pad}}}")
        }
        TypeDefKind::Enum { variants } => {
            writeln!(f, " = enum {{")?;
            for variant in variants {
                fmt_variant_def(module, variant, f, indent + 1)?;
            }
            writeln!(f, "{pad}}}")
        }
    }
}

fn fmt_variant_def(
    module: &Module,
    variant: &VariantDef,
    f: &mut fmt::Formatter<'_>,
    indent: usize,
) -> fmt::Result {
    let pad = pad(indent);
    write!(f, "{pad}{} ", quoted(&variant.name))?;
    if variant.fields.is_empty() {
        writeln!(f, "{{}}")
    } else {
        writeln!(f, "{{")?;
        for field in &variant.fields {
            writeln!(
                f,
                "{}  {}: {}",
                pad,
                quoted(&field.name),
                TypeDisplay {
                    module,
                    ty: &field.ty
                }
            )?;
        }
        writeln!(f, "{pad}}}")
    }
}

fn fmt_generic_params(params: &[GenericParam], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if params.is_empty() {
        return Ok(());
    }
    write!(f, " <")?;
    for (index, param) in params.iter().enumerate() {
        if index > 0 {
            write!(f, ", ")?;
        }
        match param {
            GenericParam::Type { name } => write!(f, "type {}", quoted(name))?,
            GenericParam::Region { name } => write!(f, "region {}", quoted(name))?,
            GenericParam::Store { name } => write!(f, "store {}", quoted(name))?,
        }
    }
    write!(f, ">")
}

fn fmt_callables(module: &Module, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
    let pad = pad(indent);
    writeln!(f, "{pad}callables [")?;
    for (id, callable) in module.callables.iter() {
        let kind = match callable.kind {
            CallableKind::Builtin => "builtin",
            CallableKind::Host => "host",
        };
        writeln!(
            f,
            "{}  callable c{} {} {} {{",
            pad,
            id.index(),
            kind,
            quoted(&callable.name)
        )?;
        writeln!(
            f,
            "{}    params {}",
            pad,
            fmt_type_list(module, &callable.signature.params)
        )?;
        writeln!(
            f,
            "{}    returns {}",
            pad,
            fmt_type_list(module, &callable.signature.returns)
        )?;
        writeln!(
            f,
            "{}    effect {}",
            pad,
            match callable.signature.effect_class {
                EffectClass::Pure => "pure",
                EffectClass::Reads => "reads",
                EffectClass::Mutates => "mutates",
                EffectClass::Barrier => "barrier",
            }
        )?;
        write!(f, "{}    domains [", pad)?;
        for (i, domain) in callable.signature.domain_effects.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(
                f,
                "{}:{}",
                quoted(&domain.domain),
                match domain.access {
                    DomainAccess::Read => "read",
                    DomainAccess::Mutate => "mutate",
                }
            )?;
        }
        writeln!(f, "]")?;
        writeln!(
            f,
            "{}    control {}",
            pad,
            match callable.signature.control {
                ControlTransfer::Returns => "returns",
                ControlTransfer::MayFail => "may_fail",
                ControlTransfer::NeverReturns => "never_returns",
            }
        )?;
        write!(f, "{}    capabilities [", pad)?;
        for (i, capability) in callable.signature.capabilities.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", quoted(capability))?;
        }
        writeln!(f, "]")?;
        writeln!(
            f,
            "{}    safety {}",
            pad,
            match callable.signature.safety {
                CallSafety::SafeCore => "safe_core",
                CallSafety::OpaqueHost => "opaque_host",
                CallSafety::UnsafeInterop => "unsafe_interop",
            }
        )?;
        match &callable.docs {
            Some(docs) => writeln!(f, "{}    docs {}", pad, quoted(docs))?,
            None => writeln!(f, "{}    docs none", pad)?,
        }
        writeln!(f, "{}  }}", pad)?;
    }
    writeln!(f, "{pad}]")
}

fn fmt_functions(module: &Module, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
    let pad = pad(indent);
    writeln!(f, "{pad}functions [")?;
    for (id, function) in module.functions.iter() {
        fmt_function(module, id.index(), function, f, indent + 1)?;
    }
    writeln!(f, "{pad}]")
}

fn fmt_function(
    module: &Module,
    index: usize,
    function: &Function,
    f: &mut fmt::Formatter<'_>,
    indent: usize,
) -> fmt::Result {
    let pad = pad(indent);
    writeln!(f, "{pad}function f{index} {} {{", quoted(&function.name))?;
    write!(f, "{}  regions [", pad)?;
    for (i, region) in function.region_params.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "r{}", region.index())?;
    }
    writeln!(f, "]")?;
    write!(f, "{}  stores [", pad)?;
    for (i, store) in function.store_params.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "store{}", store.index())?;
    }
    writeln!(f, "]")?;
    writeln!(f, "{}  params [", pad)?;
    for param in &function.params {
        fmt_local_decl(
            module,
            param.local,
            &param.name,
            &param.ty,
            param.kind,
            f,
            indent + 2,
        )?;
    }
    writeln!(f, "{}  ]", pad)?;
    writeln!(f, "{}  locals [", pad)?;
    for local in &function.locals {
        fmt_local_decl(
            module,
            local.local,
            &local.name,
            &local.ty,
            local.kind,
            f,
            indent + 2,
        )?;
    }
    writeln!(f, "{}  ]", pad)?;
    writeln!(
        f,
        "{}  return {}",
        pad,
        TypeDisplay {
            module,
            ty: &function.return_type
        }
    )?;
    writeln!(f, "{}  scopes [", pad)?;
    for scope in &function.scopes {
        fmt_scope(scope, f, indent + 2)?;
    }
    writeln!(f, "{}  ]", pad)?;
    write!(f, "{}  body ", pad)?;
    fmt_block(module, &function.body, f, indent + 1)?;
    writeln!(f)?;
    writeln!(f, "{pad}}}")
}

fn fmt_local_decl(
    module: &Module,
    local: LocalId,
    name: &str,
    ty: &Type,
    kind: LocalKind,
    f: &mut fmt::Formatter<'_>,
    indent: usize,
) -> fmt::Result {
    let pad = pad(indent);
    writeln!(
        f,
        "{pad}l{} {} {}: {}",
        local.index(),
        match kind {
            LocalKind::Param => "param",
            LocalKind::Let => "let",
            LocalKind::Temp => "temp",
            LocalKind::Destination => "destination",
        },
        quoted(name),
        TypeDisplay { module, ty }
    )
}

fn fmt_scope(scope: &Scope, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
    let pad = pad(indent);
    write!(f, "{pad}scope sc{} parent ", scope.id.index())?;
    match scope.parent {
        Some(parent) => write!(f, "sc{}", parent.index())?,
        None => write!(f, "none")?,
    }
    write!(f, " comment ")?;
    match &scope.comment {
        Some(comment) => writeln!(f, "{}", quoted(comment)),
        None => writeln!(f, "none"),
    }
}

fn fmt_block(
    module: &Module,
    block: &Block,
    f: &mut fmt::Formatter<'_>,
    indent: usize,
) -> fmt::Result {
    writeln!(f, "@sc{} {{", block.scope.index())?;
    for stmt in &block.statements {
        fmt_stmt(module, stmt, f, indent + 1)?;
    }
    write!(f, "{}}}", pad(indent))
}

fn fmt_stmt(
    module: &Module,
    stmt: &Stmt,
    f: &mut fmt::Formatter<'_>,
    indent: usize,
) -> fmt::Result {
    let pad = pad(indent);
    write!(f, "{pad}stmt{}: ", stmt.id.index())?;
    match &stmt.kind {
        StmtKind::Init { place, value } => writeln!(
            f,
            "init {} = {}",
            PlaceDisplay { module, place },
            ExprDisplay {
                module,
                expr: value
            }
        ),
        StmtKind::Assign { place, value } => writeln!(
            f,
            "assign {} = {}",
            PlaceDisplay { module, place },
            ExprDisplay {
                module,
                expr: value
            }
        ),
        StmtKind::Store { addr, width, value } => writeln!(
            f,
            "store {} {} = {}",
            MemoryWidthDisplay { width: *width },
            ExprDisplay { module, expr: addr },
            ExprDisplay {
                module,
                expr: value
            }
        ),
        StmtKind::Expr(expr) => writeln!(f, "expr {}", ExprDisplay { module, expr }),
        StmtKind::If {
            condition,
            then_block,
            else_block,
        } => {
            write!(
                f,
                "if {} ",
                ExprDisplay {
                    module,
                    expr: condition
                }
            )?;
            fmt_block(module, then_block, f, indent)?;
            if let Some(else_block) = else_block {
                write!(f, " else ")?;
                fmt_block(module, else_block, f, indent)?;
            }
            writeln!(f)
        }
        StmtKind::Loop { body } => {
            write!(f, "loop ")?;
            fmt_block(module, body, f, indent)?;
            writeln!(f)
        }
        StmtKind::Match { scrutinee, arms } => {
            writeln!(
                f,
                "match {} {{",
                ExprDisplay {
                    module,
                    expr: scrutinee
                }
            )?;
            for arm in arms {
                fmt_match_arm(module, arm, f, indent + 1)?;
            }
            writeln!(f, "{}}}", pad)
        }
        StmtKind::Break => writeln!(f, "break"),
        StmtKind::Continue => writeln!(f, "continue"),
        StmtKind::Return(None) => writeln!(f, "return"),
        StmtKind::Return(Some(expr)) => writeln!(f, "return {}", ExprDisplay { module, expr }),
    }
}

fn fmt_match_arm(
    module: &Module,
    arm: &MatchArm,
    f: &mut fmt::Formatter<'_>,
    indent: usize,
) -> fmt::Result {
    let pad = pad(indent);
    write!(
        f,
        "{pad}arm {} ",
        PatternDisplay {
            pattern: &arm.pattern
        }
    )?;
    fmt_block(module, &arm.body, f, indent)?;
    writeln!(f)
}

struct TypeDisplay<'a> {
    module: &'a Module,
    ty: &'a Type,
}

struct MemoryWidthDisplay {
    width: crate::MemoryWidth,
}

impl fmt::Display for MemoryWidthDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.width {
            crate::MemoryWidth::W1 => write!(f, "w1"),
            crate::MemoryWidth::W2 => write!(f, "w2"),
            crate::MemoryWidth::W4 => write!(f, "w4"),
            crate::MemoryWidth::W8 => write!(f, "w8"),
        }
    }
}

impl fmt::Display for TypeDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ty {
            Type::Unit => write!(f, "unit"),
            Type::Bool => write!(f, "bool"),
            Type::Integer(int) => match int.signedness {
                crate::Signedness::Signed => write!(f, "i{}", int.bits),
                crate::Signedness::Unsigned => write!(f, "u{}", int.bits),
            },
            Type::Address { domain } => match domain {
                AllocationDomain::Transient => write!(f, "addr<transient>"),
                AllocationDomain::Persistent => write!(f, "addr<persistent>"),
            },
            Type::Array { element, len } => write!(
                f,
                "Array<{}, {}>",
                TypeDisplay {
                    module: self.module,
                    ty: element
                },
                len
            ),
            Type::Named { def, args } => {
                write!(f, "t{}", def.index())?;
                if !args.is_empty() {
                    write!(f, "<")?;
                    for (index, arg) in args.iter().enumerate() {
                        if index > 0 {
                            write!(f, ", ")?;
                        }
                        match arg {
                            GenericArg::Type(ty) => write!(
                                f,
                                "{}",
                                TypeDisplay {
                                    module: self.module,
                                    ty
                                }
                            )?,
                            GenericArg::Region(region) => write!(f, "r{}", region.index())?,
                            GenericArg::Store(store) => write!(f, "store{}", store.index())?,
                        }
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            Type::Slice { region, element } => write!(
                f,
                "Slice<r{}, {}>",
                region.index(),
                TypeDisplay {
                    module: self.module,
                    ty: element
                }
            ),
            Type::Str { region } => write!(f, "str<r{}>", region.index()),
            Type::Handle { store, value } => write!(
                f,
                "Handle<store{}, {}>",
                store.index(),
                TypeDisplay {
                    module: self.module,
                    ty: value
                }
            ),
        }
    }
}

struct PlaceDisplay<'a> {
    module: &'a Module,
    place: &'a Place,
}

impl fmt::Display for PlaceDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.place {
            Place::Local(local) => write!(f, "l{}", local.index()),
            Place::Field { base, field } => {
                write!(
                    f,
                    "field({}, {})",
                    PlaceDisplay {
                        module: self.module,
                        place: base
                    },
                    quoted(field)
                )
            }
            Place::Index { base, index } => write!(
                f,
                "index({}, {})",
                PlaceDisplay {
                    module: self.module,
                    place: base
                },
                ExprDisplay {
                    module: self.module,
                    expr: index
                }
            ),
        }
    }
}

struct ExprDisplay<'a> {
    module: &'a Module,
    expr: &'a Expr,
}

impl fmt::Display for ExprDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.expr {
            Expr::Literal(lit) => match lit {
                Literal::Unit => write!(f, "()"),
                Literal::Bool(value) => write!(f, "{value}"),
                Literal::Integer(value) => write!(f, "{value:#x}"),
                Literal::String(value) => write!(f, "{}", quoted(value)),
            },
            Expr::Local(local) => write!(f, "l{}", local.index()),
            Expr::Load { addr, width } => write!(
                f,
                "load {}({})",
                MemoryWidthDisplay { width: *width },
                ExprDisplay {
                    module: self.module,
                    expr: addr
                }
            ),
            Expr::Field { base, field } => write!(
                f,
                "field({}, {})",
                ExprDisplay {
                    module: self.module,
                    expr: base
                },
                quoted(field)
            ),
            Expr::Index { base, index } => write!(
                f,
                "index({}, {})",
                ExprDisplay {
                    module: self.module,
                    expr: base
                },
                ExprDisplay {
                    module: self.module,
                    expr: index
                }
            ),
            Expr::Struct { def, fields } => {
                write!(f, "struct t{} {{", def.index())?;
                for (index, (name, expr)) in fields.iter().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    write!(
                        f,
                        "{} = {}",
                        quoted(name),
                        ExprDisplay {
                            module: self.module,
                            expr
                        }
                    )?;
                }
                write!(f, "}}")
            }
            Expr::Variant {
                def,
                variant,
                fields,
            } => {
                write!(f, "variant t{}::{} {{", def.index(), quoted(variant))?;
                for (index, (name, expr)) in fields.iter().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    write!(
                        f,
                        "{} = {}",
                        quoted(name),
                        ExprDisplay {
                            module: self.module,
                            expr
                        }
                    )?;
                }
                write!(f, "}}")
            }
            Expr::Unary { op, value } => write!(
                f,
                "unary {}({})",
                match op {
                    UnaryOp::Not => "not",
                    UnaryOp::Neg => "neg",
                },
                ExprDisplay {
                    module: self.module,
                    expr: value
                }
            ),
            Expr::Binary { op, lhs, rhs } => write!(
                f,
                "binary {}({}, {})",
                match op {
                    BinaryOp::Add => "add",
                    BinaryOp::Sub => "sub",
                    BinaryOp::Mul => "mul",
                    BinaryOp::Div => "div",
                    BinaryOp::BitAnd => "bitand",
                    BinaryOp::BitOr => "bitor",
                    BinaryOp::Xor => "xor",
                    BinaryOp::Shl => "shl",
                    BinaryOp::Shr => "shr",
                    BinaryOp::Eq => "eq",
                    BinaryOp::Ne => "ne",
                    BinaryOp::Lt => "lt",
                    BinaryOp::Le => "le",
                    BinaryOp::Gt => "gt",
                    BinaryOp::Ge => "ge",
                    BinaryOp::And => "and",
                    BinaryOp::Or => "or",
                },
                ExprDisplay {
                    module: self.module,
                    expr: lhs
                },
                ExprDisplay {
                    module: self.module,
                    expr: rhs
                }
            ),
            Expr::Call(call) => fmt_call(self.module, call, f),
        }
    }
}

fn fmt_call(module: &Module, call: &CallExpr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match call.target {
        CallTarget::Callable(callable) => write!(f, "call c{}(", callable.index())?,
    }
    for (index, arg) in call.args.iter().enumerate() {
        if index > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", ExprDisplay { module, expr: arg })?;
    }
    write!(f, ")")
}

struct PatternDisplay<'a> {
    pattern: &'a Pattern,
}

impl fmt::Display for PatternDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.pattern {
            Pattern::Wildcard => write!(f, "_"),
            Pattern::Bool(value) => write!(f, "{value}"),
            Pattern::Integer(value) => write!(f, "{value:#x}"),
            Pattern::Variant { name, fields } => {
                write!(f, "variant {} ", quoted(name))?;
                if fields.is_empty() {
                    write!(f, "{{}}")
                } else {
                    write!(f, "{{")?;
                    for (index, field) in fields.iter().enumerate() {
                        if index > 0 {
                            write!(f, ", ")?;
                        }
                        match field {
                            PatternField::Bind { field, local } => {
                                write!(f, "{} = l{}", quoted(field), local.index())?
                            }
                            PatternField::Wildcard { field } => write!(f, "{} = _", quoted(field))?,
                        }
                    }
                    write!(f, "}}")
                }
            }
        }
    }
}

fn fmt_type_list(module: &Module, types: &[Type]) -> String {
    let mut out = String::from("[");
    for (index, ty) in types.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&TypeDisplay { module, ty }.to_string());
    }
    out.push(']');
    out
}
