#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpec {
    pub name: &'static str,
    pub kind: &'static str,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuleSpec {
    pub name: &'static str,
    pub kind: &'static str,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeUseSpec {
    pub name: &'static str,
    pub kind: &'static str,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeSpec {
    pub name: &'static str,
    pub kind: &'static str,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrintSpec {
    pub name: &'static str,
    pub template: &'static str,
}
pub const REPR_NAME: &str = "HIR";
pub const REPR_FILE_EXT: &str = ".vixen-hir";
pub const REPR_PURPOSE: &str = "Human-semantic structured IR";
pub const REPR_ROUND_TRIP: &str = "canonical-print";
pub const REPR_PROVENANCE: &str = "required";
pub static TOKENS: &[TokenSpec] = &[
    TokenSpec {
        name: "ident",
        kind: "regex",
    },
    TokenSpec {
        name: "int",
        kind: "regex",
    },
    TokenSpec {
        name: "symbol",
        kind: "regex",
    },
];
pub static RULES: &[RuleSpec] = &[
    RuleSpec {
        name: "Block",
        kind: "seq",
    },
    RuleSpec {
        name: "Expr",
        kind: "choice",
    },
    RuleSpec {
        name: "Function",
        kind: "seq",
    },
    RuleSpec {
        name: "Module",
        kind: "seq",
    },
    RuleSpec {
        name: "Param",
        kind: "seq",
    },
    RuleSpec {
        name: "Stmt",
        kind: "choice",
    },
];
pub static COMMON_TYPES: &[TypeUseSpec] = &[
    TypeUseSpec {
        name: "docs",
        kind: "DocBlock",
    },
    TypeUseSpec {
        name: "provenance",
        kind: "Prov",
    },
    TypeUseSpec {
        name: "symbol",
        kind: "Symbol",
    },
];
pub static NODES: &[NodeSpec] = &[
    NodeSpec {
        name: "Block",
        kind: "node",
    },
    NodeSpec {
        name: "Expr",
        kind: "enum",
    },
    NodeSpec {
        name: "Function",
        kind: "node",
    },
    NodeSpec {
        name: "Local",
        kind: "node",
    },
    NodeSpec {
        name: "Module",
        kind: "node",
    },
    NodeSpec {
        name: "Param",
        kind: "node",
    },
    NodeSpec {
        name: "Place",
        kind: "enum",
    },
    NodeSpec {
        name: "Stmt",
        kind: "enum",
    },
    NodeSpec {
        name: "TypeDef",
        kind: "node",
    },
];
pub static CANONICAL_PRINT: &[PrintSpec] = &[
    PrintSpec {
        name: "Block",
        template: "{\n{statements}\n}",
    },
    PrintSpec {
        name: "Expr.Call",
        template: "call {callee}({args:, })",
    },
    PrintSpec {
        name: "Expr.Literal",
        template: "{value}",
    },
    PrintSpec {
        name: "Expr.Local",
        template: "{name}",
    },
    PrintSpec {
        name: "Function",
        template: "fn {name}({params:, }) -> {return_type} {body}",
    },
    PrintSpec {
        name: "Module",
        template: "module {\n{functions}\n}",
    },
    PrintSpec {
        name: "Param",
        template: "{name}: {ty}",
    },
    PrintSpec {
        name: "Stmt.Expr",
        template: "{value}",
    },
    PrintSpec {
        name: "Stmt.Return",
        template: "return{value? : {value}}",
    },
];
