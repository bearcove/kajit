//! Shape/field collection utilities for the compiler.
//!
//! Re-exports from kajit-format and provides kajit-specific wrappers that
//! supply the intrinsic trampolines.

use super::*;

// Re-export everything from kajit-format's shape_utils.
pub(super) use kajit_format::{
    collect_variants, discriminant_size, get_option_def, get_pointer_def,
    instantiated_shape_symbol_key, is_string_like_scalar, is_unit,
};

/// Collect fields using kajit's intrinsic trampolines for default resolution.
pub(super) fn collect_fields(shape: &'static Shape) -> (Vec<FieldEmitInfo>, Vec<SkippedFieldInfo>) {
    kajit_format::collect_fields(shape, resolve_trait_default, resolve_custom_default)
}

/// Resolve a trait-based default for a field type.
/// Returns the DefaultInfo if the type has a Default impl via its shape vtable.
fn resolve_trait_default(shape: &'static Shape) -> Option<crate::format::DefaultInfo> {
    use crate::format::DefaultInfo;

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

/// Resolve a custom default expression for a field.
fn resolve_custom_default(custom_fn: facet::DefaultInPlaceFn) -> crate::format::DefaultInfo {
    crate::format::DefaultInfo {
        trampoline: crate::intrinsics::kajit_field_default_custom as *const u8,
        fn_ptr: custom_fn as *const u8,
        shape: None,
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

/// Width of a discriminant field in IR representation.
pub(super) fn ir_width_from_disc_size(size: u32) -> IrWidth {
    match size {
        1 => IrWidth::W1,
        2 => IrWidth::W2,
        4 => IrWidth::W4,
        8 => IrWidth::W8,
        _ => panic!("unsupported discriminant size: {size}"),
    }
}
