//! Shape/field collection utilities for the compiler.

use super::*;

pub(super) fn collect_fields(shape: &'static Shape) -> (Vec<FieldEmitInfo>, Vec<SkippedFieldInfo>) {
    let mut out = Vec::new();
    let mut skipped = Vec::new();
    let container_has_default = shape.has_default_attr();
    collect_fields_recursive(shape, 0, container_has_default, &mut out, &mut skipped);
    check_field_name_collisions(&out);
    (out, skipped)
}

// r[impl deser.default]
// r[impl deser.default.fn-ptr]
// r[impl deser.skip]
// r[impl deser.skip.filter]
fn collect_fields_recursive(
    shape: &'static Shape,
    base_offset: usize,
    container_has_default: bool,
    out: &mut Vec<FieldEmitInfo>,
    skipped: &mut Vec<SkippedFieldInfo>,
) {
    use crate::format::DefaultInfo;
    use facet::DefaultSource;

    let st = match &shape.ty {
        Type::User(UserType::Struct(st)) => st,
        _ => panic!("unsupported shape: {}", shape.type_identifier),
    };
    for f in st.fields {
        if f.is_flattened() {
            collect_fields_recursive(
                f.shape(),
                base_offset + f.offset,
                container_has_default,
                out,
                skipped,
            );
            continue;
        }

        // Resolve default information for this field.
        let default = match f.default {
            Some(DefaultSource::Custom(custom_fn)) => {
                // Custom default expression: #[facet(default = expr)]
                Some(DefaultInfo {
                    trampoline: crate::intrinsics::kajit_field_default_custom as *const u8,
                    fn_ptr: custom_fn as *const u8,
                    shape: None,
                })
            }
            Some(DefaultSource::FromTrait) => {
                // Field-level #[facet(default)] — use the field type's Default impl.
                resolve_trait_default(f.shape())
            }
            None if container_has_default => {
                // Container-level #[facet(default)] — all fields get Default.
                resolve_trait_default(f.shape())
            }
            None => None,
        };

        // r[impl deser.skip.default-required]
        if f.should_skip_deserializing() {
            // Skipped fields are excluded from the dispatch list but need default init.
            let default = default.unwrap_or_else(|| {
                panic!(
                    "field \"{}\" on {} is skipped but has no default — \
                     add #[facet(default)] or impl Default",
                    f.effective_name(),
                    shape.type_identifier,
                )
            });
            skipped.push(SkippedFieldInfo {
                offset: base_offset + f.offset,
                default,
            });
            continue;
        }

        out.push(FieldEmitInfo {
            offset: base_offset + f.offset,
            shape: f.shape(),
            name: f.effective_name(),
            required_index: out.len(),
            default,
        });
    }
}

/// Resolve a trait-based default for a field type.
/// Returns the DefaultInfo if the type has a Default impl via its shape vtable.
pub(super) fn resolve_trait_default(shape: &'static Shape) -> Option<crate::format::DefaultInfo> {
    use crate::format::DefaultInfo;

    // Get the default_in_place function from the shape's TypeOps.
    let type_ops = shape.type_ops?;
    match type_ops {
        facet::TypeOps::Direct(ops) => {
            let default_fn = ops.default_in_place?;
            Some(DefaultInfo {
                trampoline: crate::intrinsics::kajit_field_default_trait as *const u8,
                fn_ptr: default_fn as *const u8,
                shape: None,
            })
        }
        facet::TypeOps::Indirect(ops) => {
            let default_fn = ops.default_in_place?;
            Some(DefaultInfo {
                trampoline: crate::intrinsics::kajit_field_default_indirect as *const u8,
                fn_ptr: default_fn as *const u8,
                shape: Some(shape),
            })
        }
    }
}

// r[impl deser.flatten.conflict]
pub(super) fn check_field_name_collisions(fields: &[FieldEmitInfo]) {
    let mut seen = std::collections::HashSet::new();
    for f in fields {
        if !seen.insert(f.name) {
            panic!(
                "field name collision: \"{}\" (possibly from #[facet(flatten)])",
                f.name
            );
        }
    }
}

// r[impl deser.enum.variant-kinds]

pub(super) fn collect_variants(enum_type: &'static facet::EnumType) -> Vec<VariantEmitInfo> {
    enum_type
        .variants
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let kind = VariantKind::from_struct_type(&v.data);
            let mut fields = Vec::new();
            let mut skipped = Vec::new();
            for f in v.data.fields {
                if f.is_flattened() {
                    collect_fields_recursive(f.shape(), f.offset, false, &mut fields, &mut skipped);
                } else {
                    fields.push(FieldEmitInfo {
                        offset: f.offset,
                        shape: f.shape(),
                        name: f.effective_name(),
                        required_index: fields.len(),
                        default: None,
                    });
                }
            }
            // Note: skipped fields in enum variants are not yet supported.
            // If needed, we'd emit default calls in the variant body.
            VariantEmitInfo {
                index: i,
                name: v.effective_name(),
                rust_discriminant: v.discriminant.expect(
                    "enum variant must have a known discriminant (use #[repr(u8)] or similar)",
                ),
                fields,
                kind,
            }
        })
        .collect()
}

/// Get the discriminant storage size in bytes from an EnumRepr.
pub(super) fn discriminant_size(repr: EnumRepr) -> u32 {
    match repr {
        EnumRepr::U8 | EnumRepr::I8 => 1,
        EnumRepr::U16 | EnumRepr::I16 => 2,
        EnumRepr::U32 | EnumRepr::I32 => 4,
        EnumRepr::U64 | EnumRepr::I64 | EnumRepr::USize | EnumRepr::ISize => 8,
        EnumRepr::Rust | EnumRepr::RustNPO => {
            panic!("cannot JIT-compile enums with #[repr(Rust)] — use #[repr(u8)] or similar")
        }
    }
}

/// Returns the OptionDef if this shape is an Option type.
pub(super) fn get_option_def(shape: &'static Shape) -> Option<&'static OptionDef> {
    match &shape.def {
        Def::Option(opt_def) => Some(opt_def),
        _ => None,
    }
}

pub(crate) fn symbol_registry_for_shape(shape: &'static Shape) -> crate::ir::IntrinsicRegistry {
    let mut registry = crate::ir::IntrinsicRegistry::empty();
    for (name, func) in crate::intrinsics::known_intrinsics() {
        registry.register(name, func);
    }
    for (name, func) in crate::json_intrinsics::known_intrinsics() {
        registry.register(name, func);
    }

    let mut seen = HashSet::new();
    collect_shape_symbols(shape, &mut seen, &mut registry);
    registry
}

fn collect_shape_symbols(
    shape: &'static Shape,
    seen: &mut HashSet<usize>,
    registry: &mut crate::ir::IntrinsicRegistry,
) {
    let shape_key = shape as *const Shape as usize;
    if !seen.insert(shape_key) {
        return;
    }

    if let Type::User(UserType::Struct(st)) = &shape.ty {
        for field in st.fields {
            if field.is_flattened() {
                collect_shape_symbols(field.shape(), seen, registry);
                continue;
            }

            let name = field.effective_name();
            registry.register_const(
                format!("json_key_ptr.{}", encode_symbol_bytes(name)),
                name.as_ptr() as u64,
            );
            collect_shape_symbols(field.shape(), seen, registry);
        }
    } else if let Type::User(UserType::Enum(enum_type)) = &shape.ty {
        for variant in enum_type.variants {
            for field in variant.data.fields {
                collect_shape_symbols(field.shape(), seen, registry);
            }
        }
    }

    match &shape.def {
        Def::Map(map_def) => {
            collect_shape_symbols(map_def.k, seen, registry);
            collect_shape_symbols(map_def.v, seen, registry);
        }
        Def::Set(set_def) => collect_shape_symbols(set_def.t, seen, registry),
        Def::List(list_def) => collect_shape_symbols(list_def.t, seen, registry),
        Def::Array(array_def) => collect_shape_symbols(array_def.t, seen, registry),
        Def::NdArray(ndarray_def) => collect_shape_symbols(ndarray_def.t, seen, registry),
        Def::Slice(slice_def) => collect_shape_symbols(slice_def.t, seen, registry),
        Def::Option(opt_def) => {
            let type_id = instantiated_shape_symbol_key(opt_def.t);
            registry.register_const(
                format!("option_init_none.{type_id}"),
                opt_def.vtable.init_none as *const () as usize as u64,
            );
            registry.register_const(
                format!("option_init_some.{type_id}"),
                opt_def.vtable.init_some as *const () as usize as u64,
            );
            collect_shape_symbols(opt_def.t, seen, registry);
        }
        Def::Result(result_def) => {
            collect_shape_symbols(result_def.t, seen, registry);
            collect_shape_symbols(result_def.e, seen, registry);
        }
        Def::Pointer(pointer_def) => {
            if let Some(pointee) = pointer_def.pointee {
                collect_shape_symbols(pointee, seen, registry);
            }
        }
        Def::Undefined | Def::Scalar | Def::DynamicValue(_) => {}
        _ => {}
    }
}

fn encode_symbol_bytes(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    for byte in text.as_bytes() {
        use core::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String should not fail");
    }
    out
}

pub(super) fn instantiated_shape_symbol_key(shape: &'static Shape) -> String {
    let mut out = String::new();
    append_instantiated_shape_symbol_key(shape, &mut out);
    out
}

fn append_instantiated_shape_symbol_key(shape: &'static Shape, out: &mut String) {
    use core::fmt::Write as _;

    write!(out, "d{:032x}", shape.decl_id.0).expect("writing to String should not fail");

    if !shape.type_params.is_empty() {
        out.push_str("__t");
        for (index, param) in shape.type_params.iter().enumerate() {
            write!(out, "_{index}_").expect("writing to String should not fail");
            append_instantiated_shape_symbol_key(param.shape(), out);
        }
    }

    if !shape.const_params.is_empty() {
        out.push_str("__c");
        for (index, param) in shape.const_params.iter().enumerate() {
            write!(out, "_{index}_").expect("writing to String should not fail");
            out.push(const_param_kind_symbol(param.kind));
            write!(out, "{:x}", param.value).expect("writing to String should not fail");
        }
    }
}

fn const_param_kind_symbol(kind: ConstParamKind) -> char {
    match kind {
        ConstParamKind::Bool => 'b',
        ConstParamKind::Char => 'c',
        ConstParamKind::U8 => 'h',
        ConstParamKind::U16 => 't',
        ConstParamKind::U32 => 'j',
        ConstParamKind::U64 => 'm',
        ConstParamKind::Usize => 'u',
        ConstParamKind::I8 => 'a',
        ConstParamKind::I16 => 's',
        ConstParamKind::I32 => 'i',
        ConstParamKind::I64 => 'l',
        ConstParamKind::Isize => 'n',
    }
}

// r[impl deser.pointer]

/// Returns the PointerDef if this shape is a supported smart pointer (Box, Arc, Rc).
pub(super) fn get_pointer_def(shape: &'static Shape) -> Option<&'static PointerDef> {
    match &shape.def {
        Def::Pointer(ptr_def)
            if matches!(
                ptr_def.known,
                Some(KnownPointer::Box | KnownPointer::Arc | KnownPointer::Rc)
            ) =>
        {
            Some(ptr_def)
        }
        _ => None,
    }
}

// r[impl deser.pointer.nesting]

/// Returns true if the shape is a struct type.
/// Emit a default initialization call for a field.
///
/// Direct types use a 2-arg call (trampoline, fn_ptr, offset).
/// Indirect types (generic containers) use a 3-arg call that also passes the shape.
pub fn emit_default_init(ectx: &mut EmitCtx, default: &crate::format::DefaultInfo, offset: u32) {
    if let Some(shape) = default.shape {
        ectx.emit_call_trampoline_3(
            default.trampoline,
            default.fn_ptr,
            offset,
            shape as *const _ as *const u8,
        );
    } else {
        ectx.emit_call_option_init_none(default.trampoline, default.fn_ptr, offset);
    }
}

pub(super) fn is_unit(shape: &'static Shape) -> bool {
    shape.scalar_type() == Some(ScalarType::Unit)
}

pub(super) fn is_string_like_scalar(scalar_type: ScalarType) -> bool {
    matches!(
        scalar_type,
        ScalarType::String | ScalarType::Str | ScalarType::CowStr
    )
}

/// Emit code for a single field, dispatching to nested struct calls, inline
/// expansion, scalar intrinsics, or Option handling.
pub(super) fn ir_width_from_disc_size(size: u32) -> IrWidth {
    match size {
        1 => IrWidth::W1,
        2 => IrWidth::W2,
        4 => IrWidth::W4,
        8 => IrWidth::W8,
        _ => panic!("unsupported discriminant size: {size}"),
    }
}

// (Dead IR-direct lowering code removed — all paths go through HIR now.)
