//! Shared HIR generation helpers for kajit frontends.
//!
//! These helpers generate common HIR structures (Cursor type, type defs from
//! facet Shapes) with layout annotations populated, so frontends don't
//! duplicate this boilerplate.

use kajit_hir as hir;

/// Add the standard DeserContext type definition used by all deserialization frontends.
///
/// Returns the `TypeDefId` for the repr(C) context struct:
/// ```text
/// struct DeserContext {
///     input_ptr: addr<transient> @0
///     input_end: addr<transient> @8
///     error_code: u32 @16
///     error_offset: u32 @20
/// }
/// ```
pub fn add_deser_context_type(module: &mut hir::Module) -> hir::TypeDefId {
    module.add_type_def(hir::TypeDef {
        name: "DeserContext".to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "input_ptr".to_owned(),
                    ty: hir::Type::transient_addr(),
                    offset: Some(0),
                },
                hir::FieldDef {
                    name: "input_end".to_owned(),
                    ty: hir::Type::transient_addr(),
                    offset: Some(8),
                },
                hir::FieldDef {
                    name: "error_code".to_owned(),
                    ty: hir::Type::u(32),
                    offset: Some(16),
                },
                hir::FieldDef {
                    name: "error_offset".to_owned(),
                    ty: hir::Type::u(32),
                    offset: Some(20),
                },
            ],
        },
        size: Some(24),
        transparent: false,
    })
}

/// Add the standard Cursor type definition used by all frontends.
///
/// Returns the `TypeDefId` for `Cursor<'r_input> { bytes: &[u8], pos: u64 }`.
pub fn add_cursor_type(module: &mut hir::Module, input_region: hir::RegionId) -> hir::TypeDefId {
    module.add_type_def(hir::TypeDef {
        name: "Cursor".to_owned(),
        generic_params: vec![hir::GenericParam::Region {
            name: "r_input".to_owned(),
        }],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "bytes".to_owned(),
                    ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    offset: None,
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    })
}
