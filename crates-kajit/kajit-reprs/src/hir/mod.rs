use std::fmt;
use std::marker::PhantomData;

pub mod display;
pub mod lexer;
pub mod parse;
pub mod token_parser;

pub struct Id<T> {
    index: u32,
    _phantom: PhantomData<T>,
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index.cmp(&other.index)
    }
}

impl<T> std::hash::Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.index)
    }
}

impl<T> Id<T> {
    pub const fn new(index: u32) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    pub const fn index(self) -> usize {
        self.index as usize
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn push(&mut self, item: T) -> Id<T> {
        let id = Id::new(self.items.len() as u32);
        self.items.push(item);
        id
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Id<T>, &T)> {
        self.items
            .iter()
            .enumerate()
            .map(|(index, item)| (Id::new(index as u32), item))
    }
}

impl<T> std::ops::Index<Id<T>> for Arena<T> {
    type Output = T;

    fn index(&self, id: Id<T>) -> &Self::Output {
        &self.items[id.index()]
    }
}

impl<T> std::ops::IndexMut<Id<T>> for Arena<T> {
    fn index_mut(&mut self, id: Id<T>) -> &mut Self::Output {
        &mut self.items[id.index()]
    }
}

pub type RegionId = Id<RegionParam>;
pub type StoreId = Id<StoreParam>;
pub type TypeDefId = Id<TypeDef>;
pub type FunctionId = Id<Function>;

pub struct ScopeMarker;
pub type ScopeId = Id<ScopeMarker>;
pub struct LocalMarker;
pub type LocalId = Id<LocalMarker>;
pub struct StmtMarker;
pub type StmtId = Id<StmtMarker>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub regions: Arena<RegionParam>,
    pub stores: Arena<StoreParam>,
    pub type_defs: Arena<TypeDef>,
    pub functions: Arena<Function>,
}

impl Default for Module {
    fn default() -> Self {
        Self::new()
    }
}

impl Module {
    pub fn new() -> Self {
        Self {
            regions: Arena::new(),
            stores: Arena::new(),
            type_defs: Arena::new(),
            functions: Arena::new(),
        }
    }

    pub fn add_region(&mut self, name: impl Into<String>) -> RegionId {
        self.regions.push(RegionParam { name: name.into() })
    }

    pub fn add_store(&mut self, name: impl Into<String>) -> StoreId {
        self.stores.push(StoreParam { name: name.into() })
    }

    pub fn add_type_def(&mut self, type_def: TypeDef) -> TypeDefId {
        self.type_defs.push(type_def)
    }

    pub fn add_function(&mut self, function: Function) -> FunctionId {
        self.functions.push(function)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionParam {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreParam {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenericParam {
    Type { name: String },
    Region { name: String },
    Store { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDef {
    pub name: String,
    pub generic_params: Vec<GenericParam>,
    pub kind: TypeDefKind,
    /// Total byte size of this type's in-memory representation.
    /// Populated by the frontend from facet Shape layout info.
    pub size: Option<u32>,
    /// Whether this type is a transparent newtype wrapper.
    pub transparent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeDefKind {
    Struct {
        fields: Vec<FieldDef>,
    },
    Enum {
        variants: Vec<VariantDef>,
        /// Byte width of the discriminant field (1, 2, 4, or 8).
        discriminant_width: Option<u32>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDef {
    pub name: String,
    pub ty: Type,
    /// Byte offset of this field within the parent struct/variant.
    pub offset: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
    /// Rust discriminant value for this variant.
    pub discriminant: Option<i64>,
    /// Runtime initialization function pointer for this variant.
    /// Used by Option-like enums: None variant carries init_none, Some carries init_some.
    pub init_fn: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signedness {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegerType {
    pub signedness: Signedness,
    pub bits: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationDomain {
    Transient,
    Persistent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenericArg {
    Type(Type),
    Region(RegionId),
    Store(StoreId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Unit,
    Bool,
    Integer(IntegerType),
    Ref {
        mutable: bool,
        pointee: Box<Type>,
    },
    Address {
        domain: AllocationDomain,
    },
    Array {
        element: Box<Type>,
        len: usize,
    },
    Named {
        def: TypeDefId,
        args: Vec<GenericArg>,
    },
    Slice {
        region: RegionId,
        element: Box<Type>,
    },
    Str {
        region: RegionId,
    },
    Handle {
        store: StoreId,
        value: Box<Type>,
    },
}

impl Type {
    pub const fn unit() -> Self {
        Self::Unit
    }

    pub const fn bool() -> Self {
        Self::Bool
    }

    pub fn r#ref(pointee: Type) -> Self {
        Self::Ref {
            mutable: false,
            pointee: Box::new(pointee),
        }
    }

    pub fn mut_ref(pointee: Type) -> Self {
        Self::Ref {
            mutable: true,
            pointee: Box::new(pointee),
        }
    }

    pub const fn address(domain: AllocationDomain) -> Self {
        Self::Address { domain }
    }

    pub const fn transient_addr() -> Self {
        Self::address(AllocationDomain::Transient)
    }

    pub const fn persistent_addr() -> Self {
        Self::address(AllocationDomain::Persistent)
    }

    pub const fn u(bits: u16) -> Self {
        Self::Integer(IntegerType {
            signedness: Signedness::Unsigned,
            bits,
        })
    }

    pub const fn i(bits: u16) -> Self {
        Self::Integer(IntegerType {
            signedness: Signedness::Signed,
            bits,
        })
    }

    pub fn array(element: Type, len: usize) -> Self {
        Self::Array {
            element: Box::new(element),
            len,
        }
    }

    pub fn named(def: TypeDefId, args: impl Into<Vec<GenericArg>>) -> Self {
        Self::Named {
            def,
            args: args.into(),
        }
    }

    pub fn slice(region: RegionId, element: Type) -> Self {
        Self::Slice {
            region,
            element: Box::new(element),
        }
    }

    pub const fn str(region: RegionId) -> Self {
        Self::Str { region }
    }

    pub fn handle(store: StoreId, value: Type) -> Self {
        Self::Handle {
            store,
            value: Box::new(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub region_params: Vec<RegionId>,
    pub store_params: Vec<StoreId>,
    pub params: Vec<Parameter>,
    pub locals: Vec<LocalDecl>,
    pub return_type: Type,
    pub scopes: Vec<Scope>,
    pub body: Block,
}

impl Function {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    pub id: ScopeId,
    pub parent: Option<ScopeId>,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    pub local: LocalId,
    pub name: String,
    pub ty: Type,
    pub kind: LocalKind,
}

impl Parameter {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalDecl {
    pub local: LocalId,
    pub name: String,
    pub ty: Type,
    pub kind: LocalKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalKind {
    Param,
    Let,
    Temp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub scope: ScopeId,
    pub statements: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stmt {
    pub id: StmtId,
    pub kind: StmtKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StmtKind {
    Init {
        place: Place,
        value: Expr,
    },
    Assign {
        place: Place,
        value: Expr,
    },
    Store {
        addr: Expr,
        width: MemoryWidth,
        value: Expr,
    },
    Expr(Expr),
    If {
        condition: Expr,
        then_block: Block,
        else_block: Option<Block>,
    },
    Loop {
        body: Block,
        /// Optional upper bound on iteration count for bounded loops.
        /// Enables the RVSDG unrolling pass to convert to straight-line code.
        max_iterations: Option<u32>,
    },
    Match {
        scrutinee: Expr,
        arms: Vec<MatchArm>,
    },
    Break,
    Continue,
    Return(Option<Expr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternField {
    Bind { field: String, local: LocalId },
    Wildcard { field: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pattern {
    Wildcard,
    Bool(bool),
    Integer(u64),
    Variant {
        name: String,
        fields: Vec<PatternField>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Place {
    Local(LocalId),
    Deref { base: Box<Expr> },
    Field { base: Box<Place>, field: String },
    Index { base: Box<Place>, index: Box<Expr> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryWidth {
    W1,
    W2,
    W4,
    W8,
}

impl MemoryWidth {
    pub const fn bytes(self) -> u16 {
        match self {
            Self::W1 => 1,
            Self::W2 => 2,
            Self::W4 => 4,
            Self::W8 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Unit,
    Bool(bool),
    Integer(u64),
    String(String),
    /// Address of an external symbol (vtable function pointer etc.).
    /// The runtime address is resolved from a symbol table at emit time.
    ExternAddr {
        symbol: kajit_types::SymbolName,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    BitAnd,
    BitOr,
    Xor,
    Shl,
    Shr,
    Sar, // Arithmetic shift right (sign-extending)
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallExpr {
    pub callee: kajit_types::SymbolName,
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Literal(Literal),
    Local(LocalId),
    Deref(Box<Expr>),
    Load {
        addr: Box<Expr>,
        width: MemoryWidth,
    },
    SliceData {
        value: Box<Expr>,
    },
    SliceLen {
        value: Box<Expr>,
    },
    Str {
        data: Box<Expr>,
        len: Box<Expr>,
    },
    Field {
        base: Box<Expr>,
        field: String,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    AddrOf(Box<Place>),
    Struct {
        def: TypeDefId,
        fields: Vec<(String, Expr)>,
    },
    Variant {
        def: TypeDefId,
        variant: String,
        fields: Vec<(String, Expr)>,
    },
    Unary {
        op: UnaryOp,
        value: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Call(CallExpr),
}

#[cfg(test)]
mod tests {}
