use std::collections::BTreeMap;

use kajit_reprs::hir::*;

pub trait VixenExt {
    fn install_runtime_memory_callables(&mut self) -> RuntimeMemoryCallables;
    fn install_vixen_core_callables(&mut self, types: &VixenCoreTypes) -> VixenCoreCallables;
    fn lower_vixen_typed_expr(&self, expr: &VixenTypedExpr) -> Result<Expr, VixenLoweringError>;
    fn lower_vixen_typed_expr_with_locals(
        &self,
        expr: &VixenTypedExpr,
        local_types: &BTreeMap<LocalId, Type>,
    ) -> Result<Expr, VixenLoweringError>;
    fn lower_vixen_typed_function(
        &self,
        function: &VixenTypedFunction,
    ) -> Result<Function, VixenLoweringError>;
    fn lower_vixen_typed_function_into_module(
        &self,
        function: &VixenTypedFunction,
    ) -> Result<Module, VixenLoweringError>;
    fn debug_vixen_typed_function_text(
        &self,
        function: &VixenTypedFunction,
    ) -> Result<String, VixenLoweringError>;
    fn lower_vixen_typed_block(
        &self,
        body: &[VixenTypedStmt],
        scope: ScopeId,
        next_stmt: &mut u32,
        local_types: &BTreeMap<LocalId, Type>,
    ) -> Result<Block, VixenLoweringError>;
    fn distribute_field_into_if(expr: &VixenTypedExpr) -> Option<VixenTypedExpr>
    where
        Self: Sized;
    fn desugar_if_expr(stmt: &VixenTypedStmt) -> Option<VixenTypedStmt>
    where
        Self: Sized;
    fn lower_vixen_typed_stmt(
        &self,
        stmt: &VixenTypedStmt,
        scope: ScopeId,
        next_stmt: &mut u32,
        local_types: &BTreeMap<LocalId, Type>,
    ) -> Result<Stmt, VixenLoweringError>;
    fn resolve_vixen_callable(
        &self,
        callee: &VixenCallableRef,
    ) -> Result<CallableId, VixenLoweringError>;
    fn resolve_vixen_method(
        &self,
        receiver_ty: &Type,
        method: &str,
    ) -> Result<CallableId, VixenLoweringError>;
    fn vixen_expr_type(
        &self,
        expr: &VixenTypedExpr,
        local_types: &BTreeMap<LocalId, Type>,
    ) -> Result<Type, VixenLoweringError>;
    fn field_type(&self, base: &Type, field: &str) -> Result<Type, VixenLoweringError>;
}

impl VixenExt for Module {
    fn install_runtime_memory_callables(&mut self) -> RuntimeMemoryCallables {
        let alloc_transient = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "runtime.alloc_transient".to_owned(),
            intrinsic: Some(RuntimeIntrinsic::AllocTransient),
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
            intrinsic: Some(RuntimeIntrinsic::AllocPersistent),
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
            intrinsic: Some(RuntimeIntrinsic::VecFromRawParts),
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
        let validate_utf8_range = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "runtime.validate_utf8_range".to_owned(),
            intrinsic: Some(RuntimeIntrinsic::ValidateUtf8Range),
            signature: CallSignature {
                params: vec![Type::u(64), Type::u(32)],
                returns: vec![],
                effect_class: EffectClass::Reads,
                domain_effects: vec![DomainEffect {
                    domain: "input".to_owned(),
                    access: DomainAccess::Read,
                }],
                control: ControlTransfer::MayFail,
                capabilities: vec!["runtime.utf8".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some("Validate that a borrowed byte range is UTF-8.".to_owned()),
        });
        let string_validate_alloc_copy = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "runtime.string_validate_alloc_copy".to_owned(),
            intrinsic: Some(RuntimeIntrinsic::StringValidateAllocCopy),
            signature: CallSignature {
                params: vec![Type::u(64), Type::u(32)],
                returns: vec![Type::persistent_addr()],
                effect_class: EffectClass::Barrier,
                domain_effects: vec![
                    DomainEffect {
                        domain: "input".to_owned(),
                        access: DomainAccess::Read,
                    },
                    DomainEffect {
                        domain: "persistent_heap".to_owned(),
                        access: DomainAccess::Mutate,
                    },
                ],
                control: ControlTransfer::MayFail,
                capabilities: vec!["runtime.alloc".to_owned(), "runtime.utf8".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some(
                "Validate a UTF-8 range, allocate persistent bytes, and copy them.".to_owned(),
            ),
        });
        let vec_from_chunks = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "runtime.vec_from_chunks".to_owned(),
            intrinsic: None,
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

        let memcpy = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "runtime.memcpy".to_owned(),
            intrinsic: Some(RuntimeIntrinsic::Memcpy),
            signature: CallSignature {
                params: vec![Type::u(64), Type::u(64), Type::u(64)],
                returns: vec![Type::u(64)],
                effect_class: EffectClass::Mutates,
                domain_effects: vec![DomainEffect {
                    domain: "transient_heap".to_owned(),
                    access: DomainAccess::Mutate,
                }],
                control: ControlTransfer::Returns,
                capabilities: vec!["runtime.memcpy".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some(
                "Copy len bytes from src to dst (non-overlapping). Returns dst + len.".to_owned(),
            ),
        });
        let free_transient = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "runtime.free_transient".to_owned(),
            intrinsic: Some(RuntimeIntrinsic::FreeTransient),
            signature: CallSignature {
                params: vec![Type::u(64), Type::u(64), Type::u(64)],
                returns: vec![],
                effect_class: EffectClass::Mutates,
                domain_effects: vec![DomainEffect {
                    domain: "transient_heap".to_owned(),
                    access: DomainAccess::Mutate,
                }],
                control: ControlTransfer::Returns,
                capabilities: vec!["runtime.alloc".to_owned()],
                safety: CallSafety::OpaqueHost,
            },
            docs: Some("Free heap memory allocated by runtime.alloc_transient.".to_owned()),
        });

        RuntimeMemoryCallables {
            alloc_transient,
            alloc_persistent,
            validate_utf8_range,
            string_validate_alloc_copy,
            vec_from_raw_parts,
            vec_from_chunks,
            memcpy,
            free_transient,
        }
    }

    fn install_vixen_core_callables(&mut self, types: &VixenCoreTypes) -> VixenCoreCallables {
        let emit_node = self.add_callable(CallableSpec {
            kind: CallableKind::Host,
            name: "emit.node".to_owned(),
            intrinsic: None,
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
            intrinsic: None,
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
            intrinsic: None,
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
            intrinsic: None,
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
            intrinsic: None,
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
            intrinsic: None,
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
            intrinsic: None,
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

    fn lower_vixen_typed_expr(&self, expr: &VixenTypedExpr) -> Result<Expr, VixenLoweringError> {
        self.lower_vixen_typed_expr_with_locals(expr, &BTreeMap::new())
    }

    fn lower_vixen_typed_expr_with_locals(
        &self,
        expr: &VixenTypedExpr,
        local_types: &BTreeMap<LocalId, Type>,
    ) -> Result<Expr, VixenLoweringError> {
        match expr {
            VixenTypedExpr::Literal(literal) => Ok(Expr::Literal(literal.clone())),
            VixenTypedExpr::TypedLiteral { literal, .. } => Ok(Expr::Literal(literal.clone())),
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
            VixenTypedExpr::If { .. } => Err(VixenLoweringError::NonStatementIfExpr),
        }
    }

    fn lower_vixen_typed_function(
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

    fn lower_vixen_typed_function_into_module(
        &self,
        function: &VixenTypedFunction,
    ) -> Result<Module, VixenLoweringError> {
        let mut module = self.clone();
        let function = module.lower_vixen_typed_function(function)?;
        module.add_function(function);
        Ok(module)
    }

    fn debug_vixen_typed_function_text(
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

    /// Desugar a statement containing a `VixenTypedExpr::If` in value position
    /// into a `VixenTypedStmt::If` with the enclosing statement pushed into both branches.
    ///
    /// - `return if c { a } else { b }` → `if c { return a } else { return b }`
    /// - `(if c { a } else { b })` as expr-stmt → `if c { a } else { b }` as expr-stmts
    ///
    /// NOTE: `let x = if c { a } else { b }` is NOT supported — the scalar HIR→IR
    /// lowerer cannot flow Init values out of gamma branches. Use statement-form `If`
    /// with explicit writes instead. The `Let` case falls through to
    /// `lower_vixen_typed_expr_with_locals` which returns `NonStatementIfExpr`.
    /// Recursively distribute Field projections into If branches and then
    /// desugar the resulting If. Returns the simplified expression if any
    /// Field-of-If distribution was applied.
    fn distribute_field_into_if(expr: &VixenTypedExpr) -> Option<VixenTypedExpr> {
        match expr {
            VixenTypedExpr::Field { base, field } => {
                // First, recursively distribute inside the base.
                let base = if let Some(simplified) = Self::distribute_field_into_if(base) {
                    simplified
                } else {
                    *base.clone()
                };
                // If the (possibly simplified) base is an If, distribute the field.
                if let VixenTypedExpr::If {
                    condition,
                    then_expr,
                    else_expr,
                } = &base
                {
                    Some(VixenTypedExpr::If {
                        condition: condition.clone(),
                        then_expr: Box::new(VixenTypedExpr::Field {
                            base: then_expr.clone(),
                            field: field.clone(),
                        }),
                        else_expr: Box::new(VixenTypedExpr::Field {
                            base: else_expr.clone(),
                            field: field.clone(),
                        }),
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn desugar_if_expr(stmt: &VixenTypedStmt) -> Option<VixenTypedStmt> {
        match stmt {
            // return <expr-with-if> → distribute Field-of-If, then desugar
            VixenTypedStmt::Return(Some(expr)) => {
                // Try to distribute Field-of-If in the return expression.
                let expr = if let Some(distributed) = Self::distribute_field_into_if(expr) {
                    distributed
                } else {
                    expr.clone()
                };
                // Now desugar the (possibly rewritten) expression.
                if let VixenTypedExpr::If {
                    condition,
                    then_expr,
                    else_expr,
                } = &expr
                {
                    Some(VixenTypedStmt::If {
                        condition: *condition.clone(),
                        then_body: vec![VixenTypedStmt::Return(Some(*then_expr.clone()))],
                        else_body: vec![VixenTypedStmt::Return(Some(*else_expr.clone()))],
                    })
                } else {
                    None
                }
            }
            VixenTypedStmt::Expr(VixenTypedExpr::If {
                condition,
                then_expr,
                else_expr,
            }) => Some(VixenTypedStmt::If {
                condition: *condition.clone(),
                then_body: vec![VixenTypedStmt::Expr(*then_expr.clone())],
                else_body: vec![VixenTypedStmt::Expr(*else_expr.clone())],
            }),
            _ => None,
        }
    }

    fn lower_vixen_typed_stmt(
        &self,
        stmt: &VixenTypedStmt,
        scope: ScopeId,
        next_stmt: &mut u32,
        local_types: &BTreeMap<LocalId, Type>,
    ) -> Result<Stmt, VixenLoweringError> {
        // Desugar expression-form If in statement positions before lowering.
        if let Some(desugared) = Self::desugar_if_expr(stmt) {
            return self.lower_vixen_typed_stmt(&desugared, scope, next_stmt, local_types);
        }

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
            VixenTypedExpr::Literal(Literal::Integer(_))
            | VixenTypedExpr::Literal(Literal::ExternAddr { .. }) => Ok(Type::u(64)),
            VixenTypedExpr::Literal(Literal::String(_)) => {
                Err(VixenLoweringError::CannotInferExprType {
                    expr: "string literal",
                })
            }
            VixenTypedExpr::TypedLiteral { ty, .. } => Ok(ty.clone()),
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
                | BinaryOp::Shr
                | BinaryOp::Sar => self.vixen_expr_type(lhs, local_types),
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
            VixenTypedExpr::If {
                then_expr,
                else_expr,
                ..
            } => {
                let then_ty = self.vixen_expr_type(then_expr, local_types)?;
                let else_ty = self.vixen_expr_type(else_expr, local_types)?;
                if then_ty != else_ty {
                    return Err(VixenLoweringError::IfBranchTypeMismatch { then_ty, else_ty });
                }
                Ok(then_ty)
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
