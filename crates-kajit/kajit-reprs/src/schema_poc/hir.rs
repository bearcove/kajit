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
pub trait HasProvenance {
    fn provenance(&self) -> Option<&Prov>;
}
pub trait Visit {
    fn visit_block(&mut self, node: &Block) {
        walk_block(self, node);
    }
    fn visit_expr(&mut self, node: &Expr) {
        walk_expr(self, node);
    }
    fn visit_function(&mut self, node: &Function) {
        walk_function(self, node);
    }
    fn visit_local(&mut self, node: &Local) {
        walk_local(self, node);
    }
    fn visit_module(&mut self, node: &Module) {
        walk_module(self, node);
    }
    fn visit_param(&mut self, node: &Param) {
        walk_param(self, node);
    }
    fn visit_place(&mut self, node: &Place) {
        walk_place(self, node);
    }
    fn visit_stmt(&mut self, node: &Stmt) {
        walk_stmt(self, node);
    }
    fn visit_type_def(&mut self, node: &TypeDef) {
        walk_type_def(self, node);
    }
}
pub trait VisitMut {
    fn visit_block_mut(&mut self, node: &mut Block) {
        walk_block_mut(self, node);
    }
    fn visit_expr_mut(&mut self, node: &mut Expr) {
        walk_expr_mut(self, node);
    }
    fn visit_function_mut(&mut self, node: &mut Function) {
        walk_function_mut(self, node);
    }
    fn visit_local_mut(&mut self, node: &mut Local) {
        walk_local_mut(self, node);
    }
    fn visit_module_mut(&mut self, node: &mut Module) {
        walk_module_mut(self, node);
    }
    fn visit_param_mut(&mut self, node: &mut Param) {
        walk_param_mut(self, node);
    }
    fn visit_place_mut(&mut self, node: &mut Place) {
        walk_place_mut(self, node);
    }
    fn visit_stmt_mut(&mut self, node: &mut Stmt) {
        walk_stmt_mut(self, node);
    }
    fn visit_type_def_mut(&mut self, node: &mut TypeDef) {
        walk_type_def_mut(self, node);
    }
}
pub const REPR_NAME: &str = "HIR";
pub const REPR_FILE_EXT: &str = ".vixen-hir";
pub const REPR_PURPOSE: &str = "Human-semantic structured IR";
pub const REPR_ROUND_TRIP: &str = "canonical-print";
pub const REPR_PROVENANCE: &str = "required";
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryOp;
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DocBlock(pub Vec<String>);
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenericParam;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Literal;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalKind;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Prov {
    pub file_id: Option<u32>,
    pub span: Option<Span>,
}
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Symbol(pub String);
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
impl HasProvenance for Expr {
    fn provenance(&self) -> Option<&Prov> {
        match self {
            Self::Binary { prov, .. } => Some(prov),
            Self::Call { prov, .. } => Some(prov),
            Self::Field { prov, .. } => Some(prov),
            Self::Literal { prov, .. } => Some(prov),
            Self::Local { prov, .. } => Some(prov),
        }
    }
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
impl HasProvenance for Place {
    fn provenance(&self) -> Option<&Prov> {
        match self {
            Self::Field { prov, .. } => Some(prov),
            Self::Local { prov, .. } => Some(prov),
        }
    }
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
impl HasProvenance for Stmt {
    fn provenance(&self) -> Option<&Prov> {
        match self {
            Self::Assign { prov, .. } => Some(prov),
            Self::Expr { prov, .. } => Some(prov),
            Self::If { prov, .. } => Some(prov),
            Self::Init { prov, .. } => Some(prov),
            Self::Return { prov, .. } => Some(prov),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDef {
    pub docs: Option<DocBlock>,
    pub kind: TypeDefKind,
    pub name: Symbol,
    pub params: Vec<GenericParam>,
    pub prov: Prov,
}
impl HasProvenance for Block {
    fn provenance(&self) -> Option<&Prov> {
        Some(&self.prov)
    }
}
impl HasProvenance for Function {
    fn provenance(&self) -> Option<&Prov> {
        Some(&self.prov)
    }
}
impl HasProvenance for Local {
    fn provenance(&self) -> Option<&Prov> {
        Some(&self.prov)
    }
}
impl HasProvenance for Param {
    fn provenance(&self) -> Option<&Prov> {
        Some(&self.prov)
    }
}
impl HasProvenance for TypeDef {
    fn provenance(&self) -> Option<&Prov> {
        Some(&self.prov)
    }
}
pub fn walk_block<V: ?Sized + Visit>(v: &mut V, node: &Block) {
    for value in node.statements.iter() {
        v.visit_stmt(value);
    }
}
pub fn walk_expr<V: ?Sized + Visit>(v: &mut V, node: &Expr) {
    match node {
        Expr::Binary { lhs, rhs, .. } => {
            v.visit_expr(lhs);
            v.visit_expr(rhs);
        }
        Expr::Call { args, .. } => {
            for value in args.iter() {
                v.visit_expr(value);
            }
        }
        Expr::Field { base, .. } => {
            v.visit_expr(base);
        }
        Expr::Literal { .. } => {}
        Expr::Local { .. } => {}
    }
}
pub fn walk_function<V: ?Sized + Visit>(v: &mut V, node: &Function) {
    v.visit_block(&node.body);
    for value in node.locals.iter() {
        v.visit_local(value);
    }
    for value in node.params.iter() {
        v.visit_param(value);
    }
}
pub fn walk_local<V: ?Sized + Visit>(_v: &mut V, _node: &Local) {}
pub fn walk_module<V: ?Sized + Visit>(v: &mut V, node: &Module) {
    for value in node.functions.iter() {
        v.visit_function(value);
    }
    for value in node.type_defs.iter() {
        v.visit_type_def(value);
    }
}
pub fn walk_param<V: ?Sized + Visit>(_v: &mut V, _node: &Param) {}
pub fn walk_place<V: ?Sized + Visit>(v: &mut V, node: &Place) {
    match node {
        Place::Field { base, .. } => {
            v.visit_place(base);
        }
        Place::Local { .. } => {}
    }
}
pub fn walk_stmt<V: ?Sized + Visit>(v: &mut V, node: &Stmt) {
    match node {
        Stmt::Assign { place, value, .. } => {
            v.visit_place(place);
            v.visit_expr(value);
        }
        Stmt::Expr { value, .. } => {
            v.visit_expr(value);
        }
        Stmt::If { condition, r#else, then, .. } => {
            v.visit_expr(condition);
            if let Some(value) = r#else {
                v.visit_block(value);
            }
            v.visit_block(then);
        }
        Stmt::Init { place, value, .. } => {
            v.visit_place(place);
            v.visit_expr(value);
        }
        Stmt::Return { value, .. } => {
            if let Some(value) = value {
                v.visit_expr(value);
            }
        }
    }
}
pub fn walk_type_def<V: ?Sized + Visit>(_v: &mut V, _node: &TypeDef) {}
pub fn walk_block_mut<V: ?Sized + VisitMut>(v: &mut V, node: &mut Block) {
    for value in node.statements.iter_mut() {
        v.visit_stmt_mut(value);
    }
}
pub fn walk_expr_mut<V: ?Sized + VisitMut>(v: &mut V, node: &mut Expr) {
    match node {
        Expr::Binary { lhs, rhs, .. } => {
            v.visit_expr_mut(lhs);
            v.visit_expr_mut(rhs);
        }
        Expr::Call { args, .. } => {
            for value in args.iter_mut() {
                v.visit_expr_mut(value);
            }
        }
        Expr::Field { base, .. } => {
            v.visit_expr_mut(base);
        }
        Expr::Literal { .. } => {}
        Expr::Local { .. } => {}
    }
}
pub fn walk_function_mut<V: ?Sized + VisitMut>(v: &mut V, node: &mut Function) {
    v.visit_block_mut(&mut node.body);
    for value in node.locals.iter_mut() {
        v.visit_local_mut(value);
    }
    for value in node.params.iter_mut() {
        v.visit_param_mut(value);
    }
}
pub fn walk_local_mut<V: ?Sized + VisitMut>(_v: &mut V, _node: &mut Local) {}
pub fn walk_module_mut<V: ?Sized + VisitMut>(v: &mut V, node: &mut Module) {
    for value in node.functions.iter_mut() {
        v.visit_function_mut(value);
    }
    for value in node.type_defs.iter_mut() {
        v.visit_type_def_mut(value);
    }
}
pub fn walk_param_mut<V: ?Sized + VisitMut>(_v: &mut V, _node: &mut Param) {}
pub fn walk_place_mut<V: ?Sized + VisitMut>(v: &mut V, node: &mut Place) {
    match node {
        Place::Field { base, .. } => {
            v.visit_place_mut(base);
        }
        Place::Local { .. } => {}
    }
}
pub fn walk_stmt_mut<V: ?Sized + VisitMut>(v: &mut V, node: &mut Stmt) {
    match node {
        Stmt::Assign { place, value, .. } => {
            v.visit_place_mut(place);
            v.visit_expr_mut(value);
        }
        Stmt::Expr { value, .. } => {
            v.visit_expr_mut(value);
        }
        Stmt::If { condition, r#else, then, .. } => {
            v.visit_expr_mut(condition);
            if let Some(value) = r#else {
                v.visit_block_mut(value);
            }
            v.visit_block_mut(then);
        }
        Stmt::Init { place, value, .. } => {
            v.visit_place_mut(place);
            v.visit_expr_mut(value);
        }
        Stmt::Return { value, .. } => {
            if let Some(value) = value {
                v.visit_expr_mut(value);
            }
        }
    }
}
pub fn walk_type_def_mut<V: ?Sized + VisitMut>(_v: &mut V, _node: &mut TypeDef) {}
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
