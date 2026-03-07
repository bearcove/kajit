use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;

mod text;

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
pub struct VixenCoreTypes {
    pub string: Type,
    pub node: Type,
    pub edge: Type,
    pub fact: Type,
    pub crate_graph: Type,
    pub crate_node: Type,
    pub crate_id: Type,
    pub crate_type: Type,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VixenCoreCallables {
    pub emit_node: CallableId,
    pub emit_edge: CallableId,
    pub emit_fact: CallableId,
    pub rust_crate_graph: CallableId,
    pub rust_root: CallableId,
    pub graph_lookup_crate: CallableId,
    pub cargo_registry_package_exists: CallableId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeMemoryCallables {
    pub alloc_transient: CallableId,
    pub alloc_persistent: CallableId,
    pub vec_from_raw_parts: CallableId,
    pub vec_from_chunks: CallableId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VixenBuiltin {
    EmitNode,
    EmitEdge,
    EmitFact,
    RustCrateGraph,
    RustRoot,
    GraphLookupCrate,
    CargoRegistryPackageExists,
}

impl VixenBuiltin {
    pub const fn callable_name(self) -> &'static str {
        match self {
            Self::EmitNode => "emit.node",
            Self::EmitEdge => "emit.edge",
            Self::EmitFact => "emit.fact",
            Self::RustCrateGraph => "rust.crate_graph",
            Self::RustRoot => "rust.root",
            Self::GraphLookupCrate => "graph.lookup_crate",
            Self::CargoRegistryPackageExists => "cargo.registry_package_exists",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VixenCallableRef {
    Builtin(VixenBuiltin),
    Named(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VixenTypedExpr {
    Literal(Literal),
    Local(LocalId),
    Field {
        base: Box<VixenTypedExpr>,
        field: String,
    },
    Struct {
        def: TypeDefId,
        fields: Vec<(String, VixenTypedExpr)>,
    },
    Variant {
        def: TypeDefId,
        variant: String,
        fields: Vec<(String, VixenTypedExpr)>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<VixenTypedExpr>,
        rhs: Box<VixenTypedExpr>,
    },
    Call {
        callee: VixenCallableRef,
        args: Vec<VixenTypedExpr>,
    },
    MethodCall {
        receiver: Box<VixenTypedExpr>,
        method: String,
        args: Vec<VixenTypedExpr>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VixenTypedParam {
    pub local: LocalId,
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VixenTypedLocal {
    pub local: LocalId,
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VixenTypedStmt {
    Let {
        local: LocalId,
        value: VixenTypedExpr,
    },
    Expr(VixenTypedExpr),
    If {
        condition: VixenTypedExpr,
        then_body: Vec<VixenTypedStmt>,
        else_body: Vec<VixenTypedStmt>,
    },
    Return(Option<VixenTypedExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VixenTypedFunction {
    pub name: String,
    pub params: Vec<VixenTypedParam>,
    pub locals: Vec<VixenTypedLocal>,
    pub return_type: Type,
    pub body: Vec<VixenTypedStmt>,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VixenLoweringError {
    MissingCallable {
        builtin: VixenBuiltin,
    },
    MissingNamedCallable {
        name: String,
    },
    MissingMethod {
        method: String,
        receiver_ty: Type,
    },
    AmbiguousMethod {
        method: String,
        receiver_ty: Type,
        candidates: Vec<String>,
    },
    UnknownLocalType {
        local: LocalId,
    },
    UnknownFieldType {
        field: String,
        base: Type,
    },
    CannotInferExprType {
        expr: &'static str,
    },
}

impl Module {
    pub fn install_runtime_memory_callables(&mut self) -> RuntimeMemoryCallables {
        let alloc_transient = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "runtime.alloc_transient".to_owned(),
            signature: CallSignature {
                params: vec![Type::u(64), Type::u(64)],
                returns: vec![Type::transient_addr()],
                effect_class: EffectClass::Mutates,
                domain_effects: vec![DomainEffect {
                    domain: "transient_heap".to_owned(),
                    access: DomainAccess::Mutate,
                }],
                control: ControlTransfer::MayFail,
                capabilities: vec!["runtime.alloc".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some("Allocate transient decode-time memory.".to_owned()),
        });
        let alloc_persistent = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "runtime.alloc_persistent".to_owned(),
            signature: CallSignature {
                params: vec![Type::u(64), Type::u(64)],
                returns: vec![Type::persistent_addr()],
                effect_class: EffectClass::Mutates,
                domain_effects: vec![DomainEffect {
                    domain: "persistent_heap".to_owned(),
                    access: DomainAccess::Mutate,
                }],
                control: ControlTransfer::MayFail,
                capabilities: vec!["runtime.alloc".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some("Allocate persistent memory that may escape in the result.".to_owned()),
        });
        let vec_from_raw_parts = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "runtime.vec_from_raw_parts".to_owned(),
            signature: CallSignature {
                params: vec![
                    Type::persistent_addr(),
                    Type::u(64),
                    Type::u(64),
                    Type::u(64),
                ],
                returns: vec![Type::u(64)],
                effect_class: EffectClass::Barrier,
                domain_effects: vec![DomainEffect {
                    domain: "persistent_heap".to_owned(),
                    access: DomainAccess::Mutate,
                }],
                control: ControlTransfer::MayFail,
                capabilities: vec!["runtime.alloc".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some("Materialize a Vec-like host value from persistent raw parts.".to_owned()),
        });
        let vec_from_chunks = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "runtime.vec_from_chunks".to_owned(),
            signature: CallSignature {
                params: vec![Type::transient_addr(), Type::u(64), Type::u(64)],
                returns: vec![Type::u(64)],
                effect_class: EffectClass::Barrier,
                domain_effects: vec![
                    DomainEffect {
                        domain: "transient_heap".to_owned(),
                        access: DomainAccess::Read,
                    },
                    DomainEffect {
                        domain: "persistent_heap".to_owned(),
                        access: DomainAccess::Mutate,
                    },
                ],
                control: ControlTransfer::MayFail,
                capabilities: vec!["runtime.alloc".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some(
                "Materialize a Vec-like host value from transient chunk storage.".to_owned(),
            ),
        });

        RuntimeMemoryCallables {
            alloc_transient,
            alloc_persistent,
            vec_from_raw_parts,
            vec_from_chunks,
        }
    }

    pub fn install_vixen_core_callables(&mut self, types: &VixenCoreTypes) -> VixenCoreCallables {
        let emit_node = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "emit.node".to_owned(),
            signature: CallSignature {
                params: vec![types.node.clone()],
                returns: vec![Type::unit()],
                effect_class: EffectClass::Mutates,
                domain_effects: vec![DomainEffect {
                    domain: "ruleplan".to_owned(),
                    access: DomainAccess::Mutate,
                }],
                control: ControlTransfer::Returns,
                capabilities: vec!["emit.graph".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some("Append a typed node to the host RulePlan.".to_owned()),
        });
        let emit_edge = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "emit.edge".to_owned(),
            signature: CallSignature {
                params: vec![types.edge.clone()],
                returns: vec![Type::unit()],
                effect_class: EffectClass::Mutates,
                domain_effects: vec![DomainEffect {
                    domain: "ruleplan".to_owned(),
                    access: DomainAccess::Mutate,
                }],
                control: ControlTransfer::Returns,
                capabilities: vec!["emit.graph".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some("Append a typed edge to the host RulePlan.".to_owned()),
        });
        let emit_fact = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "emit.fact".to_owned(),
            signature: CallSignature {
                params: vec![types.fact.clone()],
                returns: vec![Type::unit()],
                effect_class: EffectClass::Mutates,
                domain_effects: vec![DomainEffect {
                    domain: "ruleplan".to_owned(),
                    access: DomainAccess::Mutate,
                }],
                control: ControlTransfer::Returns,
                capabilities: vec!["emit.graph".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some("Append a typed fact to the host RulePlan.".to_owned()),
        });
        let rust_crate_graph = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "rust.crate_graph".to_owned(),
            signature: CallSignature {
                params: vec![],
                returns: vec![types.crate_graph.clone()],
                effect_class: EffectClass::Reads,
                domain_effects: vec![DomainEffect {
                    domain: "workspace".to_owned(),
                    access: DomainAccess::Read,
                }],
                control: ControlTransfer::MayFail,
                capabilities: vec!["env.read".to_owned(), "rust.graph".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some("Load the workspace crate graph from the host environment.".to_owned()),
        });
        let rust_root = self.add_callable(CallableSpec {
            kind: CallableKind::Builtin,
            name: "rust.root".to_owned(),
            signature: CallSignature {
                params: vec![types.crate_graph.clone()],
                returns: vec![types.crate_node.clone()],
                effect_class: EffectClass::Pure,
                domain_effects: vec![],
                control: ControlTransfer::Returns,
                capabilities: vec!["transform".to_owned()],
                safety: CallSafety::SafeCore,
            },
            docs: Some("Return the root crate from a typed crate graph value.".to_owned()),
        });
        let graph_lookup_crate = self.add_callable(CallableSpec {
            kind: CallableKind::Builtin,
            name: "graph.lookup_crate".to_owned(),
            signature: CallSignature {
                params: vec![types.crate_graph.clone(), types.crate_id.clone()],
                returns: vec![types.crate_node.clone()],
                effect_class: EffectClass::Pure,
                domain_effects: vec![],
                control: ControlTransfer::MayFail,
                capabilities: vec!["transform".to_owned()],
                safety: CallSafety::SafeCore,
            },
            docs: Some("Look up one crate node by id and fail if it is missing.".to_owned()),
        });
        let cargo_registry_package_exists = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "cargo.registry_package_exists".to_owned(),
            signature: CallSignature {
                params: vec![types.string.clone(), types.string.clone()],
                returns: vec![Type::bool()],
                effect_class: EffectClass::Reads,
                domain_effects: vec![DomainEffect {
                    domain: "cargo_registry".to_owned(),
                    access: DomainAccess::Read,
                }],
                control: ControlTransfer::MayFail,
                capabilities: vec!["env.read".to_owned(), "cargo.registry".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some(
                "Check whether a registry package exists in the current Cargo environment."
                    .to_owned(),
            ),
        });

        VixenCoreCallables {
            emit_node,
            emit_edge,
            emit_fact,
            rust_crate_graph,
            rust_root,
            graph_lookup_crate,
            cargo_registry_package_exists,
        }
    }

    pub fn lower_vixen_typed_expr(
        &self,
        expr: &VixenTypedExpr,
    ) -> Result<Expr, VixenLoweringError> {
        self.lower_vixen_typed_expr_with_locals(expr, &BTreeMap::new())
    }

    fn lower_vixen_typed_expr_with_locals(
        &self,
        expr: &VixenTypedExpr,
        local_types: &BTreeMap<LocalId, Type>,
    ) -> Result<Expr, VixenLoweringError> {
        match expr {
            VixenTypedExpr::Literal(literal) => Ok(Expr::Literal(literal.clone())),
            VixenTypedExpr::Local(local) => Ok(Expr::Local(*local)),
            VixenTypedExpr::Field { base, field } => Ok(Expr::Field {
                base: Box::new(self.lower_vixen_typed_expr_with_locals(base, local_types)?),
                field: field.clone(),
            }),
            VixenTypedExpr::Struct { def, fields } => Ok(Expr::Struct {
                def: *def,
                fields: fields
                    .iter()
                    .map(|(field, expr)| {
                        Ok((
                            field.clone(),
                            self.lower_vixen_typed_expr_with_locals(expr, local_types)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, VixenLoweringError>>()?,
            }),
            VixenTypedExpr::Variant {
                def,
                variant,
                fields,
            } => Ok(Expr::Variant {
                def: *def,
                variant: variant.clone(),
                fields: fields
                    .iter()
                    .map(|(field, expr)| {
                        Ok((
                            field.clone(),
                            self.lower_vixen_typed_expr_with_locals(expr, local_types)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, VixenLoweringError>>()?,
            }),
            VixenTypedExpr::Binary { op, lhs, rhs } => Ok(Expr::Binary {
                op: *op,
                lhs: Box::new(self.lower_vixen_typed_expr_with_locals(lhs, local_types)?),
                rhs: Box::new(self.lower_vixen_typed_expr_with_locals(rhs, local_types)?),
            }),
            VixenTypedExpr::Call { callee, args } => {
                let callable = self.resolve_vixen_callable(callee)?;
                Ok(Expr::Call(CallExpr {
                    target: CallTarget::Callable(callable),
                    args: args
                        .iter()
                        .map(|arg| self.lower_vixen_typed_expr_with_locals(arg, local_types))
                        .collect::<Result<Vec<_>, VixenLoweringError>>()?,
                }))
            }
            VixenTypedExpr::MethodCall {
                receiver,
                method,
                args,
            } => {
                let receiver_ty = self.vixen_expr_type(receiver, local_types)?;
                let callable = self.resolve_vixen_method(&receiver_ty, method)?;
                let receiver = self.lower_vixen_typed_expr_with_locals(receiver, local_types)?;
                let mut lowered_args = Vec::with_capacity(args.len() + 1);
                lowered_args.push(receiver);
                lowered_args.extend(
                    args.iter()
                        .map(|arg| self.lower_vixen_typed_expr_with_locals(arg, local_types))
                        .collect::<Result<Vec<_>, VixenLoweringError>>()?,
                );
                Ok(Expr::Call(CallExpr {
                    target: CallTarget::Callable(callable),
                    args: lowered_args,
                }))
            }
        }
    }

    pub fn lower_vixen_typed_function(
        &self,
        function: &VixenTypedFunction,
    ) -> Result<Function, VixenLoweringError> {
        let scope = ScopeId::new(0);
        let mut next_stmt = 0u32;
        let local_types = function
            .params
            .iter()
            .map(|param| (param.local, param.ty.clone()))
            .chain(
                function
                    .locals
                    .iter()
                    .map(|local| (local.local, local.ty.clone())),
            )
            .collect::<BTreeMap<_, _>>();
        let body =
            self.lower_vixen_typed_block(&function.body, scope, &mut next_stmt, &local_types)?;

        Ok(Function {
            name: function.name.clone(),
            region_params: Vec::new(),
            store_params: Vec::new(),
            params: function
                .params
                .iter()
                .map(|param| Parameter {
                    local: param.local,
                    name: param.name.clone(),
                    ty: param.ty.clone(),
                    kind: LocalKind::Param,
                })
                .collect(),
            locals: function
                .locals
                .iter()
                .map(|local| LocalDecl {
                    local: local.local,
                    name: local.name.clone(),
                    ty: local.ty.clone(),
                    kind: LocalKind::Let,
                })
                .collect(),
            return_type: function.return_type.clone(),
            scopes: vec![Scope {
                id: scope,
                parent: None,
                comment: function.comment.clone(),
            }],
            body,
        })
    }

    pub fn lower_vixen_typed_function_into_module(
        &self,
        function: &VixenTypedFunction,
    ) -> Result<Module, VixenLoweringError> {
        let mut module = self.clone();
        let function = module.lower_vixen_typed_function(function)?;
        module.add_function(function);
        Ok(module)
    }

    pub fn debug_vixen_typed_function_text(
        &self,
        function: &VixenTypedFunction,
    ) -> Result<String, VixenLoweringError> {
        self.lower_vixen_typed_function_into_module(function)
            .map(|module| module.to_string())
    }

    fn lower_vixen_typed_block(
        &self,
        body: &[VixenTypedStmt],
        scope: ScopeId,
        next_stmt: &mut u32,
        local_types: &BTreeMap<LocalId, Type>,
    ) -> Result<Block, VixenLoweringError> {
        let statements = body
            .iter()
            .map(|stmt| self.lower_vixen_typed_stmt(stmt, scope, next_stmt, local_types))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Block { scope, statements })
    }

    fn lower_vixen_typed_stmt(
        &self,
        stmt: &VixenTypedStmt,
        scope: ScopeId,
        next_stmt: &mut u32,
        local_types: &BTreeMap<LocalId, Type>,
    ) -> Result<Stmt, VixenLoweringError> {
        let id = StmtId::new(*next_stmt);
        *next_stmt += 1;
        let kind = match stmt {
            VixenTypedStmt::Let { local, value } => StmtKind::Init {
                place: Place::Local(*local),
                value: self.lower_vixen_typed_expr_with_locals(value, local_types)?,
            },
            VixenTypedStmt::Expr(expr) => {
                StmtKind::Expr(self.lower_vixen_typed_expr_with_locals(expr, local_types)?)
            }
            VixenTypedStmt::If {
                condition,
                then_body,
                else_body,
            } => StmtKind::If {
                condition: self.lower_vixen_typed_expr_with_locals(condition, local_types)?,
                then_block: self.lower_vixen_typed_block(
                    then_body,
                    scope,
                    next_stmt,
                    local_types,
                )?,
                else_block: Some(self.lower_vixen_typed_block(
                    else_body,
                    scope,
                    next_stmt,
                    local_types,
                )?),
            },
            VixenTypedStmt::Return(expr) => StmtKind::Return(
                expr.as_ref()
                    .map(|expr| self.lower_vixen_typed_expr_with_locals(expr, local_types))
                    .transpose()?,
            ),
        };
        Ok(Stmt { id, kind })
    }

    fn resolve_vixen_callable(
        &self,
        callee: &VixenCallableRef,
    ) -> Result<CallableId, VixenLoweringError> {
        match callee {
            VixenCallableRef::Builtin(builtin) => self
                .callable_named(builtin.callable_name())
                .ok_or(VixenLoweringError::MissingCallable { builtin: *builtin }),
            VixenCallableRef::Named(name) => self
                .callable_named(name)
                .ok_or_else(|| VixenLoweringError::MissingNamedCallable { name: name.clone() }),
        }
    }

    fn resolve_vixen_method(
        &self,
        receiver_ty: &Type,
        method: &str,
    ) -> Result<CallableId, VixenLoweringError> {
        let mut matches = self
            .callables
            .iter()
            .filter_map(|(id, callable)| {
                let suffix = callable.name.rsplit('.').next().unwrap_or(&callable.name);
                let first_param = callable.signature.params.first()?;
                (suffix == method && first_param == receiver_ty)
                    .then_some((id, callable.name.clone()))
            })
            .collect::<Vec<_>>();

        match matches.len() {
            0 => Err(VixenLoweringError::MissingMethod {
                method: method.to_owned(),
                receiver_ty: receiver_ty.clone(),
            }),
            1 => Ok(matches.pop().unwrap().0),
            _ => Err(VixenLoweringError::AmbiguousMethod {
                method: method.to_owned(),
                receiver_ty: receiver_ty.clone(),
                candidates: matches.into_iter().map(|(_, name)| name).collect(),
            }),
        }
    }

    fn vixen_expr_type(
        &self,
        expr: &VixenTypedExpr,
        local_types: &BTreeMap<LocalId, Type>,
    ) -> Result<Type, VixenLoweringError> {
        match expr {
            VixenTypedExpr::Literal(Literal::Unit) => Ok(Type::unit()),
            VixenTypedExpr::Literal(Literal::Bool(_)) => Ok(Type::bool()),
            VixenTypedExpr::Literal(Literal::Integer(_)) => Ok(Type::u(64)),
            VixenTypedExpr::Literal(Literal::String(_)) => {
                Err(VixenLoweringError::CannotInferExprType {
                    expr: "string literal",
                })
            }
            VixenTypedExpr::Local(local) => local_types
                .get(local)
                .cloned()
                .ok_or(VixenLoweringError::UnknownLocalType { local: *local }),
            VixenTypedExpr::Field { base, field } => {
                let base_ty = self.vixen_expr_type(base, local_types)?;
                self.field_type(&base_ty, field)
            }
            VixenTypedExpr::Struct { def, .. } => Ok(Type::named(*def, Vec::new())),
            VixenTypedExpr::Variant { def, .. } => Ok(Type::named(*def, Vec::new())),
            VixenTypedExpr::Binary { op, lhs, .. } => match op {
                BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge
                | BinaryOp::And
                | BinaryOp::Or => Ok(Type::bool()),
                BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::BitAnd
                | BinaryOp::BitOr
                | BinaryOp::Xor
                | BinaryOp::Shl
                | BinaryOp::Shr => self.vixen_expr_type(lhs, local_types),
            },
            VixenTypedExpr::Call { callee, .. } => {
                let callable = self.resolve_vixen_callable(callee)?;
                Ok(self.callables[callable]
                    .signature
                    .returns
                    .first()
                    .cloned()
                    .unwrap_or_else(Type::unit))
            }
            VixenTypedExpr::MethodCall {
                receiver, method, ..
            } => {
                let receiver_ty = self.vixen_expr_type(receiver, local_types)?;
                let callable = self.resolve_vixen_method(&receiver_ty, method)?;
                Ok(self.callables[callable]
                    .signature
                    .returns
                    .first()
                    .cloned()
                    .unwrap_or_else(Type::unit))
            }
        }
    }

    fn field_type(&self, base: &Type, field: &str) -> Result<Type, VixenLoweringError> {
        match base {
            Type::Named { def, .. } => match &self.type_defs[*def].kind {
                TypeDefKind::Struct { fields } => fields
                    .iter()
                    .find(|candidate| candidate.name == field)
                    .map(|field| field.ty.clone())
                    .ok_or_else(|| VixenLoweringError::UnknownFieldType {
                        field: field.to_owned(),
                        base: base.clone(),
                    }),
                TypeDefKind::Enum { .. } => Err(VixenLoweringError::UnknownFieldType {
                    field: field.to_owned(),
                    base: base.clone(),
                }),
            },
            _ => Err(VixenLoweringError::UnknownFieldType {
                field: field.to_owned(),
                base: base.clone(),
            }),
        }
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

impl Function {
    pub fn destination_param(&self) -> Option<&Parameter> {
        self.params.iter().find(|param| param.is_destination())
    }
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

impl Parameter {
    pub fn is_destination(&self) -> bool {
        self.kind == LocalKind::Destination
    }
}

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
    Load {
        addr: Box<Expr>,
        width: MemoryWidth,
    },
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
                    ty: Type::named(header, vec![GenericArg::Region(r_input)]),
                    kind: LocalKind::Destination,
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
        let destination = function.destination_param().unwrap();
        assert!(matches!(destination.ty, Type::Named { .. }));
        assert!(function.locals.is_empty());
    }

    #[test]
    fn address_types_distinguish_allocation_domains() {
        assert_ne!(Type::transient_addr(), Type::persistent_addr());
        assert_eq!(
            Type::address(AllocationDomain::Transient),
            Type::transient_addr()
        );
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
    fn function_can_model_result_style_early_return() {
        let mut module = Module::new();
        let parse_error = module.add_type_def(TypeDef {
            name: "ParseError".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "code".to_owned(),
                    ty: Type::u(32),
                }],
            },
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
                        }],
                    },
                    VariantDef {
                        name: "Err".to_owned(),
                        fields: vec![FieldDef {
                            name: "error".to_owned(),
                            ty: Type::named(parse_error, Vec::new()),
                        }],
                    },
                ],
            },
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
    fn installs_vixen_core_callable_table() {
        let mut module = Module::new();
        let string = module.add_type_def(TypeDef {
            name: "String".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let node = module.add_type_def(TypeDef {
            name: "Node".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let edge = module.add_type_def(TypeDef {
            name: "Edge".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let fact = module.add_type_def(TypeDef {
            name: "Fact".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_graph = module.add_type_def(TypeDef {
            name: "CrateGraph".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_node = module.add_type_def(TypeDef {
            name: "CrateNode".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_id = module.add_type_def(TypeDef {
            name: "CrateId".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_type = module.add_type_def(TypeDef {
            name: "CrateType".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Enum {
                variants: vec![
                    VariantDef {
                        name: "Lib".to_owned(),
                        fields: vec![],
                    },
                    VariantDef {
                        name: "Bin".to_owned(),
                        fields: vec![],
                    },
                ],
            },
        });

        let callables = module.install_vixen_core_callables(&VixenCoreTypes {
            string: Type::named(string, Vec::new()),
            node: Type::named(node, Vec::new()),
            edge: Type::named(edge, Vec::new()),
            fact: Type::named(fact, Vec::new()),
            crate_graph: Type::named(crate_graph, Vec::new()),
            crate_node: Type::named(crate_node, Vec::new()),
            crate_id: Type::named(crate_id, Vec::new()),
            crate_type: Type::named(crate_type, Vec::new()),
        });

        assert_eq!(
            module.callable_named("emit.node"),
            Some(callables.emit_node)
        );
        assert_eq!(
            module.callable_named("graph.lookup_crate"),
            Some(callables.graph_lookup_crate)
        );

        let emit_node = &module.callables[callables.emit_node];
        assert_eq!(emit_node.kind, CallableKind::Host);
        assert_eq!(emit_node.signature.effect_class, EffectClass::Mutates);
        assert_eq!(
            emit_node.signature.domain_effects,
            vec![DomainEffect {
                domain: "ruleplan".to_owned(),
                access: DomainAccess::Mutate,
            }]
        );
        assert_eq!(
            emit_node.signature.capabilities,
            vec!["emit.graph".to_owned()]
        );

        let rust_crate_graph = &module.callables[callables.rust_crate_graph];
        assert_eq!(rust_crate_graph.kind, CallableKind::Host);
        assert_eq!(rust_crate_graph.signature.effect_class, EffectClass::Reads);
        assert_eq!(rust_crate_graph.signature.control, ControlTransfer::MayFail);
        assert_eq!(
            rust_crate_graph.signature.returns,
            vec![Type::named(crate_graph, Vec::new())]
        );

        let lookup = &module.callables[callables.graph_lookup_crate];
        assert_eq!(lookup.kind, CallableKind::Builtin);
        assert_eq!(lookup.signature.effect_class, EffectClass::Pure);
        assert_eq!(lookup.signature.control, ControlTransfer::MayFail);
        assert_eq!(
            lookup.signature.params,
            vec![
                Type::named(crate_graph, Vec::new()),
                Type::named(crate_id, Vec::new()),
            ]
        );
        assert_eq!(
            lookup.signature.returns,
            vec![Type::named(crate_node, Vec::new())]
        );

        let registry_exists = &module.callables[callables.cargo_registry_package_exists];
        assert_eq!(registry_exists.kind, CallableKind::Host);
        assert_eq!(registry_exists.signature.effect_class, EffectClass::Reads);
        assert_eq!(
            registry_exists.signature.domain_effects,
            vec![DomainEffect {
                domain: "cargo_registry".to_owned(),
                access: DomainAccess::Read,
            }]
        );
        assert_eq!(
            registry_exists.signature.params,
            vec![
                Type::named(string, Vec::new()),
                Type::named(string, Vec::new())
            ]
        );
    }

    #[test]
    fn lowers_vixen_typed_exprs_into_hir_calls() {
        let mut module = Module::new();
        let string = module.add_type_def(TypeDef {
            name: "String".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let node = module.add_type_def(TypeDef {
            name: "Node".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "label".to_owned(),
                    ty: Type::named(string, Vec::new()),
                }],
            },
        });
        let edge = module.add_type_def(TypeDef {
            name: "Edge".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let fact = module.add_type_def(TypeDef {
            name: "Fact".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_graph = module.add_type_def(TypeDef {
            name: "CrateGraph".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_node = module.add_type_def(TypeDef {
            name: "CrateNode".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_id = module.add_type_def(TypeDef {
            name: "CrateId".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "value".to_owned(),
                    ty: Type::named(string, Vec::new()),
                }],
            },
        });
        let crate_type = module.add_type_def(TypeDef {
            name: "CrateType".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Enum {
                variants: vec![
                    VariantDef {
                        name: "Lib".to_owned(),
                        fields: vec![],
                    },
                    VariantDef {
                        name: "Bin".to_owned(),
                        fields: vec![],
                    },
                ],
            },
        });

        let callables = module.install_vixen_core_callables(&VixenCoreTypes {
            string: Type::named(string, Vec::new()),
            node: Type::named(node, Vec::new()),
            edge: Type::named(edge, Vec::new()),
            fact: Type::named(fact, Vec::new()),
            crate_graph: Type::named(crate_graph, Vec::new()),
            crate_node: Type::named(crate_node, Vec::new()),
            crate_id: Type::named(crate_id, Vec::new()),
            crate_type: Type::named(crate_type, Vec::new()),
        });

        let emit_expr = module
            .lower_vixen_typed_expr(&VixenTypedExpr::Call {
                callee: VixenCallableRef::Named("emit.node".to_owned()),
                args: vec![VixenTypedExpr::Struct {
                    def: node,
                    fields: vec![(
                        "label".to_owned(),
                        VixenTypedExpr::Literal(Literal::String("compile app".to_owned())),
                    )],
                }],
            })
            .expect("emit.node should lower");

        let Expr::Call(call) = emit_expr else {
            panic!("expected lowered call");
        };
        assert_eq!(call.target, CallTarget::Callable(callables.emit_node));
        assert_eq!(call.args.len(), 1);

        let lookup_expr = module
            .lower_vixen_typed_expr(&VixenTypedExpr::Call {
                callee: VixenCallableRef::Named("graph.lookup_crate".to_owned()),
                args: vec![
                    VixenTypedExpr::Call {
                        callee: VixenCallableRef::Named("rust.crate_graph".to_owned()),
                        args: vec![],
                    },
                    VixenTypedExpr::Struct {
                        def: crate_id,
                        fields: vec![(
                            "value".to_owned(),
                            VixenTypedExpr::Literal(Literal::String("root".to_owned())),
                        )],
                    },
                ],
            })
            .expect("graph.lookup_crate should lower");

        let Expr::Call(lookup_call) = lookup_expr else {
            panic!("expected lookup call");
        };
        assert_eq!(
            lookup_call.target,
            CallTarget::Callable(callables.graph_lookup_crate)
        );
        let Expr::Call(graph_call) = &lookup_call.args[0] else {
            panic!("expected nested rust.crate_graph call");
        };
        assert_eq!(
            graph_call.target,
            CallTarget::Callable(callables.rust_crate_graph)
        );
    }

    #[test]
    fn lowering_vixen_typed_expr_requires_installed_callables() {
        let module = Module::new();
        let err = module
            .lower_vixen_typed_expr(&VixenTypedExpr::Call {
                callee: VixenCallableRef::Builtin(VixenBuiltin::EmitFact),
                args: vec![],
            })
            .expect_err("missing callable table should fail");

        assert_eq!(
            err,
            VixenLoweringError::MissingCallable {
                builtin: VixenBuiltin::EmitFact,
            }
        );
    }

    #[test]
    fn lowers_vixen_typed_function_into_hir_function() {
        let mut module = Module::new();
        let string = module.add_type_def(TypeDef {
            name: "String".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let node = module.add_type_def(TypeDef {
            name: "Node".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "label".to_owned(),
                    ty: Type::named(string, Vec::new()),
                }],
            },
        });
        let edge = module.add_type_def(TypeDef {
            name: "Edge".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let fact = module.add_type_def(TypeDef {
            name: "Fact".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_graph = module.add_type_def(TypeDef {
            name: "CrateGraph".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_node = module.add_type_def(TypeDef {
            name: "CrateNode".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_id = module.add_type_def(TypeDef {
            name: "CrateId".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "value".to_owned(),
                    ty: Type::named(string, Vec::new()),
                }],
            },
        });
        let crate_type = module.add_type_def(TypeDef {
            name: "CrateType".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Enum {
                variants: vec![
                    VariantDef {
                        name: "Lib".to_owned(),
                        fields: vec![],
                    },
                    VariantDef {
                        name: "Bin".to_owned(),
                        fields: vec![],
                    },
                ],
            },
        });

        let callables = module.install_vixen_core_callables(&VixenCoreTypes {
            string: Type::named(string, Vec::new()),
            node: Type::named(node, Vec::new()),
            edge: Type::named(edge, Vec::new()),
            fact: Type::named(fact, Vec::new()),
            crate_graph: Type::named(crate_graph, Vec::new()),
            crate_node: Type::named(crate_node, Vec::new()),
            crate_id: Type::named(crate_id, Vec::new()),
            crate_type: Type::named(crate_type, Vec::new()),
        });

        let function = module
            .lower_vixen_typed_function(&VixenTypedFunction {
                name: "plan_compile".to_owned(),
                params: vec![VixenTypedParam {
                    local: LocalId::new(0),
                    name: "graph".to_owned(),
                    ty: Type::named(crate_graph, Vec::new()),
                }],
                locals: vec![
                    VixenTypedLocal {
                        local: LocalId::new(1),
                        name: "node".to_owned(),
                        ty: Type::named(node, Vec::new()),
                    },
                    VixenTypedLocal {
                        local: LocalId::new(2),
                        name: "emit_enabled".to_owned(),
                        ty: Type::bool(),
                    },
                ],
                return_type: Type::unit(),
                body: vec![
                    VixenTypedStmt::Let {
                        local: LocalId::new(2),
                        value: VixenTypedExpr::Literal(Literal::Bool(true)),
                    },
                    VixenTypedStmt::If {
                        condition: VixenTypedExpr::Local(LocalId::new(2)),
                        then_body: vec![
                            VixenTypedStmt::Let {
                                local: LocalId::new(1),
                                value: VixenTypedExpr::Struct {
                                    def: node,
                                    fields: vec![(
                                        "label".to_owned(),
                                        VixenTypedExpr::Literal(Literal::String(
                                            "compile".to_owned(),
                                        )),
                                    )],
                                },
                            },
                            VixenTypedStmt::Expr(VixenTypedExpr::Call {
                                callee: VixenCallableRef::Named("emit.node".to_owned()),
                                args: vec![VixenTypedExpr::Local(LocalId::new(1))],
                            }),
                        ],
                        else_body: vec![],
                    },
                    VixenTypedStmt::Return(None),
                ],
                comment: Some("lowered from typed Vixen stub".to_owned()),
            })
            .expect("typed Vixen function should lower");

        assert_eq!(function.name, "plan_compile");
        assert_eq!(function.params.len(), 1);
        assert_eq!(function.locals.len(), 2);
        assert_eq!(function.scopes.len(), 1);
        assert_eq!(
            function.scopes[0].comment,
            Some("lowered from typed Vixen stub".to_owned())
        );
        assert_eq!(function.body.statements.len(), 3);

        let StmtKind::Init { place, .. } = &function.body.statements[0].kind else {
            panic!("expected first local init");
        };
        assert_eq!(*place, Place::Local(LocalId::new(2)));

        let StmtKind::If {
            then_block,
            else_block,
            ..
        } = &function.body.statements[1].kind
        else {
            panic!("expected lowered if");
        };
        assert_eq!(then_block.statements.len(), 2);
        let Some(else_block) = else_block else {
            panic!("expected explicit else block");
        };
        assert!(else_block.statements.is_empty());

        let StmtKind::Expr(Expr::Call(call)) = &then_block.statements[1].kind else {
            panic!("expected emit.node call inside then block");
        };
        assert_eq!(call.target, CallTarget::Callable(callables.emit_node));

        assert!(matches!(
            function.body.statements[2].kind,
            StmtKind::Return(None)
        ));
    }

    #[test]
    fn lowers_vixen_method_calls_into_callable_resolution() {
        let mut module = Module::new();
        let string = module.add_type_def(TypeDef {
            name: "String".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let node = module.add_type_def(TypeDef {
            name: "Node".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let edge = module.add_type_def(TypeDef {
            name: "Edge".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let fact = module.add_type_def(TypeDef {
            name: "Fact".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_graph = module.add_type_def(TypeDef {
            name: "CrateGraph".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_node = module.add_type_def(TypeDef {
            name: "CrateNode".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_id = module.add_type_def(TypeDef {
            name: "CrateId".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_type = module.add_type_def(TypeDef {
            name: "CrateType".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Enum {
                variants: vec![
                    VariantDef {
                        name: "Lib".to_owned(),
                        fields: vec![],
                    },
                    VariantDef {
                        name: "Bin".to_owned(),
                        fields: vec![],
                    },
                ],
            },
        });

        let callables = module.install_vixen_core_callables(&VixenCoreTypes {
            string: Type::named(string, Vec::new()),
            node: Type::named(node, Vec::new()),
            edge: Type::named(edge, Vec::new()),
            fact: Type::named(fact, Vec::new()),
            crate_graph: Type::named(crate_graph, Vec::new()),
            crate_node: Type::named(crate_node, Vec::new()),
            crate_id: Type::named(crate_id, Vec::new()),
            crate_type: Type::named(crate_type, Vec::new()),
        });

        let lowered = module
            .lower_vixen_typed_function(&VixenTypedFunction {
                name: "plan_dep".to_owned(),
                params: vec![
                    VixenTypedParam {
                        local: LocalId::new(0),
                        name: "graph".to_owned(),
                        ty: Type::named(crate_graph, Vec::new()),
                    },
                    VixenTypedParam {
                        local: LocalId::new(1),
                        name: "crate_id".to_owned(),
                        ty: Type::named(crate_id, Vec::new()),
                    },
                ],
                locals: vec![],
                return_type: Type::named(crate_node, Vec::new()),
                body: vec![VixenTypedStmt::Return(Some(VixenTypedExpr::MethodCall {
                    receiver: Box::new(VixenTypedExpr::Local(LocalId::new(0))),
                    method: "lookup_crate".to_owned(),
                    args: vec![VixenTypedExpr::Local(LocalId::new(1))],
                }))],
                comment: Some("method call lowers through callable table".to_owned()),
            })
            .expect("method call should lower");

        let StmtKind::Return(Some(Expr::Call(call))) = &lowered.body.statements[0].kind else {
            panic!("expected method call return");
        };
        assert_eq!(
            call.target,
            CallTarget::Callable(callables.graph_lookup_crate)
        );
        assert_eq!(
            call.args,
            vec![Expr::Local(LocalId::new(0)), Expr::Local(LocalId::new(1))]
        );
    }

    #[test]
    fn lowering_vixen_named_expr_requires_installed_callables() {
        let module = Module::new();
        let err = module
            .lower_vixen_typed_expr(&VixenTypedExpr::Call {
                callee: VixenCallableRef::Named("emit.node".to_owned()),
                args: vec![],
            })
            .expect_err("missing named callable should fail");

        assert_eq!(
            err,
            VixenLoweringError::MissingNamedCallable {
                name: "emit.node".to_owned(),
            }
        );
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
}
