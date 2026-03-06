use std::fmt;
use std::marker::PhantomData;

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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeDefKind {
    Struct { fields: Vec<FieldDef> },
    Enum { variants: Vec<VariantDef> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDef {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
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
    Place(Box<Type>),
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

    pub fn place(inner: Type) -> Self {
        Self::Place(Box::new(inner))
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
    pub return_type: Type,
    pub scopes: Vec<Scope>,
    pub body: Block,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalKind {
    Param,
    Let,
    Temp,
    Destination,
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
    Expr(Expr),
    If {
        condition: Expr,
        then_block: Block,
        else_block: Option<Block>,
    },
    Loop {
        body: Block,
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
pub enum Pattern {
    Wildcard,
    Bool(bool),
    Integer(u64),
    Variant { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Place {
    Local(LocalId),
    Field { base: Box<Place>, field: String },
    Index { base: Box<Place>, index: Box<Expr> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Unit,
    Bool(bool),
    Integer(u64),
    String(String),
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
    Field {
        base: Box<Expr>,
        field: String,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallableSpec {
    pub kind: CallableKind,
    pub name: String,
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
                }],
            },
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
                    },
                    FieldDef {
                        name: "pos".to_owned(),
                        ty: Type::u(64),
                    },
                ],
            },
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
                }],
            },
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
                    ty: Type::place(Type::named(header, vec![GenericArg::Region(r_input)])),
                    kind: LocalKind::Destination,
                },
            ],
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
        assert!(matches!(function.params[1].ty, Type::Place(_)));
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
                }],
            },
        });

        let decode_header = module.add_callable(CallableSpec {
            kind: CallableKind::Builtin,
            name: "decode.header".to_owned(),
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
}
