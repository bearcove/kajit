// Re-export all types from kajit-format.
pub use kajit_format::*;

use facet::{MapDef, ScalarType};

use crate::ir::{RegionBuilder, SlotId};

/// Information about a struct field needed during IR lowering.
///
/// IR-side equivalent of [`FieldEmitInfo`]. No function pointers — defaults
/// are represented as IR nodes rather than raw trampolines.
pub struct FieldLowerInfo {
    /// Byte offset of this field within the output struct.
    pub offset: usize,
    /// The facet shape of this field.
    pub shape: &'static facet::Shape,
    /// The field name (for formats that use named fields).
    pub name: &'static str,
    /// Index of this field for required-field bitset tracking.
    pub required_index: usize,
    /// Whether this field has a default value (details resolved during lowering).
    pub has_default: bool,
}

/// Information about an enum variant needed during IR lowering.
///
/// IR-side equivalent of [`VariantEmitInfo`].
pub struct VariantLowerInfo {
    /// Variant index (0-based, used as wire discriminant for postcard).
    pub index: usize,
    /// Variant name (for JSON key matching).
    pub name: &'static str,
    /// Rust discriminant value to write to the tag slot.
    pub rust_discriminant: i64,
    /// Fields of this variant.
    pub fields: Vec<FieldLowerInfo>,
    /// Variant kind.
    pub kind: VariantKind,
}

// r[impl ir.format-trait]
// r[impl ir.format-trait.stateless]

/// A wire format that lowers deserialization logic into RVSDG nodes.
///
/// Implementations are stateless at JIT-compile time: they produce IR nodes
/// but hold no mutable state between calls. Runtime state lives in the
/// `format_state` pointer inside `DeserContext`.
pub trait Decoder {
    fn hir_lowering_kind(&self) -> Option<HIRLoweringKind> {
        None
    }

    // r[impl ir.format-trait.lower]

    /// Whether this format can safely treat string slices as trusted UTF-8 input.
    fn supports_trusted_utf8_input(&self) -> bool {
        false
    }

    /// Lower struct field iteration into RVSDG nodes.
    fn lower_struct_fields(
        &self,
        builder: &mut RegionBuilder<'_>,
        fields: &[FieldLowerInfo],
        deny_unknown_fields: bool,
        lower_field: &mut dyn FnMut(&mut RegionBuilder<'_>, &FieldLowerInfo),
    );

    /// Lower a scalar read into RVSDG nodes.
    fn lower_read_scalar(
        &self,
        builder: &mut RegionBuilder<'_>,
        offset: usize,
        scalar_type: ScalarType,
    );

    /// Lower a string read into RVSDG nodes.
    fn lower_read_string(
        &self,
        builder: &mut RegionBuilder<'_>,
        offset: usize,
        scalar_type: ScalarType,
    );

    /// Lower positional (tuple / fixed-size array) field reads into RVSDG nodes.
    fn lower_positional_fields(
        &self,
        builder: &mut RegionBuilder<'_>,
        fields: &[FieldLowerInfo],
        lower_field: &mut dyn FnMut(&mut RegionBuilder<'_>, &FieldLowerInfo),
    ) {
        for field in fields {
            lower_field(builder, field);
        }
    }

    /// Lower enum deserialization into RVSDG nodes.
    fn lower_enum(
        &self,
        _builder: &mut RegionBuilder<'_>,
        _variants: &[VariantLowerInfo],
        _lower_variant_body: &mut dyn FnMut(&mut RegionBuilder<'_>, &VariantLowerInfo),
    ) {
        panic!("enum lowering not yet implemented for this format");
    }

    /// Lower Option deserialization into RVSDG nodes.
    fn lower_option(
        &self,
        _builder: &mut RegionBuilder<'_>,
        _offset: usize,
        _init_none_fn: *const u8,
        _init_some_fn: *const u8,
        _scratch_slot: SlotId,
        _lower_inner: &mut dyn FnMut(&mut RegionBuilder<'_>),
    ) {
        panic!("option lowering not yet implemented for this format");
    }

    /// Lower Vec deserialization into RVSDG nodes.
    fn lower_vec(
        &self,
        _builder: &mut RegionBuilder<'_>,
        _offset: usize,
        _elem_shape: &'static facet::Shape,
        _vec_offsets: &crate::malum::VecOffsets,
        _lower_elem: &mut dyn FnMut(&mut RegionBuilder<'_>),
    ) {
        panic!("vec lowering not yet implemented for this format");
    }

    /// Lower Map deserialization into RVSDG nodes.
    fn lower_map(
        &self,
        _builder: &mut RegionBuilder<'_>,
        _offset: usize,
        _map_def: &'static MapDef,
        _lower_key: &mut dyn FnMut(&mut RegionBuilder<'_>),
        _lower_value: &mut dyn FnMut(&mut RegionBuilder<'_>),
    ) {
        panic!("map lowering not yet implemented for this format");
    }
}
