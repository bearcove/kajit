use std::collections::BTreeMap;
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
pub type CallableId = Id<CallableSpec>;

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
    pub callables: Arena<CallableSpec>,
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
            callables: Arena::new(),
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

    pub fn add_callable(&mut self, callable: CallableSpec) -> CallableId {
        self.callables.push(callable)
    }

    pub fn callable_named(&self, name: &str) -> Option<CallableId> {
        self.callables
            .iter()
            .find_map(|(id, callable)| (callable.name == name).then_some(id))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectClass {
    Pure,
    Reads,
    Mutates,
    Barrier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainAccess {
    Read,
    Mutate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainEffect {
    pub domain: String,
    pub access: DomainAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlTransfer {
    Returns,
    MayFail,
    NeverReturns,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallSafety {
    SafeCore,
    OpaqueHost,
    UnsafeInterop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSignature {
    pub params: Vec<Type>,
    pub returns: Vec<Type>,
    pub effect_class: EffectClass,
    pub domain_effects: Vec<DomainEffect>,
    pub control: ControlTransfer,
    pub capabilities: Vec<String>,
    pub safety: CallSafety,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallableKind {
    Builtin,
    Host,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeIntrinsic {
    OptionInitNone,
    OptionInitSome,
    AllocTransient,
    AllocPersistent,
    VecFromRawParts,
    ValidateUtf8Range,
    StringValidateAllocCopy,
    Memcpy,
    FreeTransient,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallableSpec {
    pub kind: CallableKind,
    pub name: String,
    pub intrinsic: Option<RuntimeIntrinsic>,
    pub signature: CallSignature,
    pub docs: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallTarget {
    Callable(CallableId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallExpr {
    pub target: CallTarget,
    pub args: Vec<Expr>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_known_len_persistent_vec_kernel_module() -> Module {
        let mut module = Module::new();
        let callables = module.install_runtime_memory_callables();
        module.add_function(Function {
            name: "build_vec_u32_2".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![],
            locals: vec![
                LocalDecl {
                    local: LocalId::new(0),
                    name: "len".to_owned(),
                    ty: Type::u(64),
                    kind: LocalKind::Temp,
                },
                LocalDecl {
                    local: LocalId::new(1),
                    name: "bytes".to_owned(),
                    ty: Type::u(64),
                    kind: LocalKind::Temp,
                },
                LocalDecl {
                    local: LocalId::new(2),
                    name: "ptr".to_owned(),
                    ty: Type::persistent_addr(),
                    kind: LocalKind::Temp,
                },
            ],
            return_type: Type::u(64),
            scopes: vec![Scope {
                id: ScopeId::new(0),
                parent: None,
                comment: Some("Known-length persistent vec kernel".to_owned()),
            }],
            body: Block {
                scope: ScopeId::new(0),
                statements: vec![
                    Stmt {
                        id: StmtId::new(0),
                        kind: StmtKind::Init {
                            place: Place::Local(LocalId::new(0)),
                            value: Expr::Literal(Literal::Integer(2)),
                        },
                    },
                    Stmt {
                        id: StmtId::new(1),
                        kind: StmtKind::Init {
                            place: Place::Local(LocalId::new(1)),
                            value: Expr::Binary {
                                op: BinaryOp::Mul,
                                lhs: Box::new(Expr::Local(LocalId::new(0))),
                                rhs: Box::new(Expr::Literal(Literal::Integer(4))),
                            },
                        },
                    },
                    Stmt {
                        id: StmtId::new(2),
                        kind: StmtKind::Init {
                            place: Place::Local(LocalId::new(2)),
                            value: Expr::Call(CallExpr {
                                target: CallTarget::Callable(callables.alloc_persistent),
                                args: vec![
                                    Expr::Local(LocalId::new(1)),
                                    Expr::Literal(Literal::Integer(4)),
                                ],
                            }),
                        },
                    },
                    Stmt {
                        id: StmtId::new(3),
                        kind: StmtKind::Store {
                            addr: Expr::Local(LocalId::new(2)),
                            width: MemoryWidth::W4,
                            value: Expr::Literal(Literal::Integer(10)),
                        },
                    },
                    Stmt {
                        id: StmtId::new(4),
                        kind: StmtKind::Store {
                            addr: Expr::Binary {
                                op: BinaryOp::Add,
                                lhs: Box::new(Expr::Local(LocalId::new(2))),
                                rhs: Box::new(Expr::Literal(Literal::Integer(4))),
                            },
                            width: MemoryWidth::W4,
                            value: Expr::Literal(Literal::Integer(20)),
                        },
                    },
                    Stmt {
                        id: StmtId::new(5),
                        kind: StmtKind::Return(Some(Expr::Call(CallExpr {
                            target: CallTarget::Callable(callables.vec_from_raw_parts),
                            args: vec![
                                Expr::Local(LocalId::new(2)),
                                Expr::Local(LocalId::new(0)),
                                Expr::Local(LocalId::new(0)),
                                Expr::Literal(Literal::Integer(4)),
                            ],
                        }))),
                    },
                ],
            },
        });
        module
    }

    #[test]
    fn named_types_distinguish_region_arguments() {
        let mut module = Module::new();
        let r_input = module.add_region("input");
        let r_tmp = module.add_region("tmp");
        let header = module.add_type_def(TypeDef {
            name: "Header".to_owned(),
            generic_params: vec![GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "name".to_owned(),
                    ty: Type::str(r_input),
                    offset: None,
                }],
            },
            size: None,
            transparent: false,
        });

        let header_input = Type::named(header, vec![GenericArg::Region(r_input)]);
        let header_tmp = Type::named(header, vec![GenericArg::Region(r_tmp)]);

        assert_ne!(header_input, header_tmp);
    }

    #[test]
    fn function_can_model_borrowed_output_destination() {
        let mut module = Module::new();
        let r_input = module.add_region("input");
        let cursor = module.add_type_def(TypeDef {
            name: "Cursor".to_owned(),
            generic_params: vec![GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: TypeDefKind::Struct {
                fields: vec![
                    FieldDef {
                        name: "bytes".to_owned(),
                        ty: Type::slice(r_input, Type::u(8)),
                        offset: None,
                    },
                    FieldDef {
                        name: "pos".to_owned(),
                        ty: Type::u(64),
                        offset: None,
                    },
                ],
            },
            size: None,
            transparent: false,
        });
        let header = module.add_type_def(TypeDef {
            name: "Header".to_owned(),
            generic_params: vec![GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "name".to_owned(),
                    ty: Type::str(r_input),
                    offset: None,
                }],
            },
            size: None,
            transparent: false,
        });

        let function = Function {
            name: "decode_header".to_owned(),
            region_params: vec![r_input],
            store_params: vec![],
            params: vec![
                Parameter {
                    local: LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: Type::named(cursor, vec![GenericArg::Region(r_input)]),
                    kind: LocalKind::Param,
                },
                Parameter {
                    local: LocalId::new(1),
                    name: "out".to_owned(),
                    ty: Type::named(header, vec![GenericArg::Region(r_input)]),
                    kind: LocalKind::Param,
                },
            ],
            locals: vec![],
            return_type: Type::unit(),
            scopes: vec![Scope {
                id: ScopeId::new(0),
                parent: None,
                comment: Some("decode borrowed header".to_owned()),
            }],
            body: Block {
                scope: ScopeId::new(0),
                statements: vec![Stmt {
                    id: StmtId::new(0),
                    kind: StmtKind::Return(None),
                }],
            },
        };

        assert_eq!(function.region_params, vec![r_input]);
        let out_param = &function.params[1];
        assert!(matches!(out_param.ty, Type::Named { .. }));
        assert!(function.locals.is_empty());
    }

    #[test]
    fn address_types_distinguish_allocation_domains() {
        assert_ne!(Type::transient_addr(), Type::persistent_addr());
        assert_eq!(
            Type::address(AllocationDomain::Transient),
            Type::transient_addr()
        );
        assert_eq!(Type::mut_ref(Type::u(64)), Type::mut_ref(Type::u(64)));
    }

    #[test]
    fn resolved_callables_track_effect_domains_and_control_transfer() {
        let mut module = Module::new();
        let r_tmp = module.add_region("tmp");
        let node = module.add_type_def(TypeDef {
            name: "Node".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "label".to_owned(),
                    ty: Type::slice(r_tmp, Type::u(8)),
                    offset: None,
                }],
            },
            size: None,
            transparent: false,
        });

        let decode_header = module.add_callable(CallableSpec {
            kind: CallableKind::Builtin,
            name: "decode.header".to_owned(),
            intrinsic: None,
            signature: CallSignature {
                params: vec![Type::u(64)],
                returns: vec![Type::bool()],
                effect_class: EffectClass::Mutates,
                domain_effects: vec![
                    DomainEffect {
                        domain: "cursor".to_owned(),
                        access: DomainAccess::Read,
                    },
                    DomainEffect {
                        domain: "output".to_owned(),
                        access: DomainAccess::Mutate,
                    },
                ],
                control: ControlTransfer::MayFail,
                capabilities: vec!["decode.header".to_owned()],
                safety: CallSafety::SafeCore,
            },
            docs: Some("Decode a header into the current destination.".to_owned()),
        });

        let emit_node = module.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "emit.node".to_owned(),
            intrinsic: None,
            signature: CallSignature {
                params: vec![Type::named(node, Vec::new())],
                returns: vec![Type::unit()],
                effect_class: EffectClass::Barrier,
                domain_effects: vec![DomainEffect {
                    domain: "plan".to_owned(),
                    access: DomainAccess::Mutate,
                }],
                control: ControlTransfer::Returns,
                capabilities: vec!["emit.graph".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some("Append a node to the host RulePlan.".to_owned()),
        });

        assert_eq!(module.callable_named("decode.header"), Some(decode_header));
        assert_eq!(module.callable_named("emit.node"), Some(emit_node));

        let call = Expr::Call(CallExpr {
            target: CallTarget::Callable(decode_header),
            args: vec![Expr::Literal(Literal::Integer(4))],
        });

        let Expr::Call(call) = call else {
            panic!("expected call expression");
        };

        let CallTarget::Callable(target) = call.target;
        let callable = &module.callables[target];

        assert_eq!(callable.kind, CallableKind::Builtin);
        assert_eq!(callable.signature.effect_class, EffectClass::Mutates);
        assert_eq!(callable.signature.control, ControlTransfer::MayFail);
        assert_eq!(
            callable.signature.domain_effects,
            vec![
                DomainEffect {
                    domain: "cursor".to_owned(),
                    access: DomainAccess::Read,
                },
                DomainEffect {
                    domain: "output".to_owned(),
                    access: DomainAccess::Mutate,
                },
            ]
        );

        let host = &module.callables[emit_node];
        assert_eq!(host.kind, CallableKind::Host);
        assert_eq!(host.signature.effect_class, EffectClass::Barrier);
        assert_eq!(host.signature.control, ControlTransfer::Returns);
    }

    #[test]
    fn function_can_model_result_style_early_return() {
        let mut module = Module::new();
        let parse_error = module.add_type_def(TypeDef {
            name: "ParseError".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "code".to_owned(),
                    ty: Type::u(32),
                    offset: None,
                }],
            },
            size: None,
            transparent: false,
        });
        let result_u32 = module.add_type_def(TypeDef {
            name: "ResultU32".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Enum {
                variants: vec![
                    VariantDef {
                        name: "Ok".to_owned(),
                        fields: vec![FieldDef {
                            name: "value".to_owned(),
                            ty: Type::u(32),
                            offset: None,
                        }],
                        discriminant: None,
                        init_fn: None,
                    },
                    VariantDef {
                        name: "Err".to_owned(),
                        fields: vec![FieldDef {
                            name: "error".to_owned(),
                            ty: Type::named(parse_error, Vec::new()),
                            offset: None,
                        }],
                        discriminant: None,
                        init_fn: None,
                    },
                ],
                discriminant_width: None,
            },
            size: None,
            transparent: false,
        });

        let function = Function {
            name: "parse_with_try_shape".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![Parameter {
                local: LocalId::new(0),
                name: "result".to_owned(),
                ty: Type::named(result_u32, Vec::new()),
                kind: LocalKind::Param,
            }],
            locals: vec![
                LocalDecl {
                    local: LocalId::new(1),
                    name: "value".to_owned(),
                    ty: Type::u(32),
                    kind: LocalKind::Let,
                },
                LocalDecl {
                    local: LocalId::new(2),
                    name: "error".to_owned(),
                    ty: Type::named(parse_error, Vec::new()),
                    kind: LocalKind::Let,
                },
            ],
            return_type: Type::named(result_u32, Vec::new()),
            scopes: vec![Scope {
                id: ScopeId::new(0),
                parent: None,
                comment: Some("Result-style early return".to_owned()),
            }],
            body: Block {
                scope: ScopeId::new(0),
                statements: vec![Stmt {
                    id: StmtId::new(0),
                    kind: StmtKind::Match {
                        scrutinee: Expr::Local(LocalId::new(0)),
                        arms: vec![
                            MatchArm {
                                pattern: Pattern::Variant {
                                    name: "Ok".to_owned(),
                                    fields: vec![PatternField::Bind {
                                        field: "value".to_owned(),
                                        local: LocalId::new(1),
                                    }],
                                },
                                body: Block {
                                    scope: ScopeId::new(0),
                                    statements: vec![Stmt {
                                        id: StmtId::new(1),
                                        kind: StmtKind::Return(Some(Expr::Variant {
                                            def: result_u32,
                                            variant: "Ok".to_owned(),
                                            fields: vec![(
                                                "value".to_owned(),
                                                Expr::Local(LocalId::new(1)),
                                            )],
                                        })),
                                    }],
                                },
                            },
                            MatchArm {
                                pattern: Pattern::Variant {
                                    name: "Err".to_owned(),
                                    fields: vec![PatternField::Bind {
                                        field: "error".to_owned(),
                                        local: LocalId::new(2),
                                    }],
                                },
                                body: Block {
                                    scope: ScopeId::new(0),
                                    statements: vec![Stmt {
                                        id: StmtId::new(2),
                                        kind: StmtKind::Return(Some(Expr::Variant {
                                            def: result_u32,
                                            variant: "Err".to_owned(),
                                            fields: vec![(
                                                "error".to_owned(),
                                                Expr::Local(LocalId::new(2)),
                                            )],
                                        })),
                                    }],
                                },
                            },
                        ],
                    },
                }],
            },
        };

        let StmtKind::Match { arms, .. } = &function.body.statements[0].kind else {
            panic!("expected top-level match");
        };
        assert_eq!(arms.len(), 2);
        assert!(matches!(
            arms[0].pattern,
            Pattern::Variant { ref name, ref fields }
                if name == "Ok"
                    && fields == &vec![PatternField::Bind {
                        field: "value".to_owned(),
                        local: LocalId::new(1),
                    }]
        ));
        assert!(matches!(
            arms[1].pattern,
            Pattern::Variant { ref name, ref fields }
                if name == "Err"
                    && fields == &vec![PatternField::Bind {
                        field: "error".to_owned(),
                        local: LocalId::new(2),
                    }]
        ));
    }

    #[test]
    fn host_callable_can_carry_capability_and_safety_contract() {
        let signature = CallSignature {
            params: vec![Type::u(64)],
            returns: vec![Type::bool()],
            effect_class: EffectClass::Mutates,
            domain_effects: vec![DomainEffect {
                domain: "env".to_owned(),
                access: DomainAccess::Read,
            }],
            control: ControlTransfer::MayFail,
            capabilities: vec!["env.read".to_owned()],
            safety: CallSafety::OpaqueHost,
        };

        assert_eq!(signature.capabilities, vec!["env.read".to_owned()]);
        assert_eq!(signature.safety, CallSafety::OpaqueHost);
    }

    #[test]
    fn call_signatures_can_name_transient_and_persistent_addresses() {
        let mut module = Module::new();
        let callables = module.install_runtime_memory_callables();

        let signature = &module.callables[callables.vec_from_raw_parts].signature;
        assert_eq!(signature.params[0], Type::persistent_addr());
        assert_eq!(signature.effect_class, EffectClass::Barrier);
    }

    #[test]
    fn installs_runtime_memory_callable_table() {
        let mut module = Module::new();
        let callables = module.install_runtime_memory_callables();

        assert_eq!(
            module.callable_named("runtime.alloc_transient"),
            Some(callables.alloc_transient)
        );
        assert_eq!(
            module.callables[callables.alloc_transient]
                .signature
                .returns,
            vec![Type::transient_addr()]
        );
        assert_eq!(
            module.callables[callables.alloc_persistent]
                .signature
                .returns,
            vec![Type::persistent_addr()]
        );
        assert_eq!(
            module.callables[callables.string_validate_alloc_copy]
                .signature
                .returns,
            vec![Type::persistent_addr()]
        );
        assert_eq!(
            module.callables[callables.vec_from_chunks].signature.params[0],
            Type::transient_addr()
        );
    }

    #[test]
    fn known_len_persistent_vec_kernel_uses_low_level_memory_ops() {
        let module = build_known_len_persistent_vec_kernel_module();
        let function = &module.functions[FunctionId::new(0)];

        assert_eq!(function.return_type, Type::u(64));
        assert_eq!(function.locals[2].ty, Type::persistent_addr());

        let StmtKind::Store { width, .. } = &function.body.statements[3].kind else {
            panic!("expected first store");
        };
        assert_eq!(*width, MemoryWidth::W4);

        let StmtKind::Return(Some(Expr::Call(call))) = &function.body.statements[5].kind else {
            panic!("expected final vec materialization call");
        };
        let CallTarget::Callable(target) = call.target;
        assert_eq!(module.callables[target].name, "runtime.vec_from_raw_parts");

        let text = module.to_string();
        assert!(text.contains("function f0 \"build_vec_u32_2\""));
        assert!(text.contains("store w4"));
        assert!(text.contains("runtime.vec_from_raw_parts"));
    }

    #[test]
    fn load_expressions_model_typed_memory_reads() {
        let mut module = Module::new();
        module.add_function(Function {
            name: "load_demo".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![Parameter {
                local: LocalId::new(0),
                name: "addr".to_owned(),
                ty: Type::persistent_addr(),
                kind: LocalKind::Param,
            }],
            locals: vec![LocalDecl {
                local: LocalId::new(1),
                name: "word".to_owned(),
                ty: Type::u(32),
                kind: LocalKind::Temp,
            }],
            return_type: Type::unit(),
            scopes: vec![Scope {
                id: ScopeId::new(0),
                parent: None,
                comment: None,
            }],
            body: Block {
                scope: ScopeId::new(0),
                statements: vec![
                    Stmt {
                        id: StmtId::new(0),
                        kind: StmtKind::Init {
                            place: Place::Local(LocalId::new(1)),
                            value: Expr::Load {
                                addr: Box::new(Expr::Local(LocalId::new(0))),
                                width: MemoryWidth::W4,
                            },
                        },
                    },
                    Stmt {
                        id: StmtId::new(1),
                        kind: StmtKind::Return(None),
                    },
                ],
            },
        });

        let text = module.to_string();
        assert!(text.contains("load w4(l0)"));
    }

    #[test]
    fn ref_types_and_deref_places_render_in_text() {
        let mut module = Module::new();
        let cursor = module.add_type_def(TypeDef {
            name: "Cursor".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "pos".to_owned(),
                    ty: Type::u(64),
                    offset: Some(0),
                }],
            },
            size: Some(8),
            transparent: false,
        });
        module.add_function(Function {
            name: "ref_demo".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![Parameter {
                local: LocalId::new(0),
                name: "cursor".to_owned(),
                ty: Type::mut_ref(Type::named(cursor, Vec::new())),
                kind: LocalKind::Param,
            }],
            locals: vec![],
            return_type: Type::unit(),
            scopes: vec![Scope {
                id: ScopeId::new(0),
                parent: None,
                comment: None,
            }],
            body: Block {
                scope: ScopeId::new(0),
                statements: vec![
                    Stmt {
                        id: StmtId::new(0),
                        kind: StmtKind::Assign {
                            place: Place::Field {
                                base: Box::new(Place::Deref {
                                    base: Box::new(Expr::Local(LocalId::new(0))),
                                }),
                                field: "pos".to_owned(),
                            },
                            value: Expr::Literal(Literal::Integer(1)),
                        },
                    },
                    Stmt {
                        id: StmtId::new(1),
                        kind: StmtKind::Return(None),
                    },
                ],
            },
        });

        let text = module.to_string();
        assert!(text.contains("&mut t0"));
        assert!(text.contains("field(deref(l0), \"pos\")"));
    }

    #[test]
    fn slice_view_and_fail_render_in_text() {
        let mut module = Module::new();
        let r0 = module.add_region("input");
        let cursor = module.add_type_def(TypeDef {
            name: "Cursor".to_owned(),
            generic_params: vec![GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: TypeDefKind::Struct {
                fields: vec![
                    FieldDef {
                        name: "bytes".to_owned(),
                        ty: Type::slice(r0, Type::u(8)),
                        offset: None,
                    },
                    FieldDef {
                        name: "pos".to_owned(),
                        ty: Type::u(64),
                        offset: None,
                    },
                ],
            },
            size: None,
            transparent: false,
        });
        module.add_function(Function {
            name: "slice_demo".to_owned(),
            region_params: vec![r0],
            store_params: vec![],
            params: vec![Parameter {
                local: LocalId::new(0),
                name: "cursor".to_owned(),
                ty: Type::named(cursor, vec![GenericArg::Region(r0)]),
                kind: LocalKind::Param,
            }],
            locals: vec![],
            return_type: Type::unit(),
            scopes: vec![Scope {
                id: ScopeId::new(0),
                parent: None,
                comment: None,
            }],
            body: Block {
                scope: ScopeId::new(0),
                statements: vec![
                    Stmt {
                        id: StmtId::new(0),
                        kind: StmtKind::Expr(Expr::SliceData {
                            value: Box::new(Expr::Field {
                                base: Box::new(Expr::Local(LocalId::new(0))),
                                field: "bytes".to_owned(),
                            }),
                        }),
                    },
                    Stmt {
                        id: StmtId::new(1),
                        kind: StmtKind::Expr(Expr::SliceLen {
                            value: Box::new(Expr::Field {
                                base: Box::new(Expr::Local(LocalId::new(0))),
                                field: "bytes".to_owned(),
                            }),
                        }),
                    },
                    Stmt {
                        id: StmtId::new(2),
                        kind: StmtKind::Return(None),
                    },
                ],
            },
        });

        let text = module.to_string();
        assert!(text.contains("slice_data("));
        assert!(text.contains("slice_len("));
        assert!(text.contains("return"));
    }
}
