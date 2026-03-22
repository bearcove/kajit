//! Shared HIR generation helpers for kajit frontends.
//!
//! These helpers generate common HIR structures (Cursor type, type defs from
//! facet Shapes) with layout annotations populated, so frontends don't
//! duplicate this boilerplate.

use kajit_hir as hir;

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
