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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryOp;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocBlock;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenericParam;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Literal;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalKind;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prov;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Type;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDefKind;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub prov: Prov,
    pub statements: Vec<Stmt>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Binary { lhs: Box<Expr>, op: BinaryOp, prov: Prov, rhs: Box<Expr> },
    Call { args: Vec<Expr>, callee: Symbol, prov: Prov },
    Field { base: Box<Expr>, field: Symbol, prov: Prov },
    Literal { prov: Prov, value: Literal },
    Local { name: Symbol, prov: Prov },
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub body: Box<Block>,
    pub docs: Option<DocBlock>,
    pub locals: Vec<Local>,
    pub name: Symbol,
    pub params: Vec<Param>,
    pub prov: Prov,
    pub return_type: Type,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local {
    pub kind: LocalKind,
    pub name: Symbol,
    pub prov: Prov,
    pub ty: Type,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub docs: Option<DocBlock>,
    pub functions: Vec<Function>,
    pub type_defs: Vec<TypeDef>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: Symbol,
    pub prov: Prov,
    pub ty: Type,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Place {
    Field { base: Box<Place>, field: Symbol, prov: Prov },
    Local { name: Symbol, prov: Prov },
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Assign { place: Box<Place>, prov: Prov, value: Box<Expr> },
    Expr { prov: Prov, value: Box<Expr> },
    If {
        condition: Box<Expr>,
        r#else: Option<Box<Block>>,
        prov: Prov,
        then: Box<Block>,
    },
    Init { place: Box<Place>, prov: Prov, value: Box<Expr> },
    Return { prov: Prov, value: Option<Box<Expr>> },
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDef {
    pub docs: Option<DocBlock>,
    pub kind: TypeDefKind,
    pub name: Symbol,
    pub params: Vec<GenericParam>,
    pub prov: Prov,
}
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
