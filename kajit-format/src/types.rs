//! Core data types for format-specific code emission and lowering.

use facet::StructKind;

/// Wire format kind — determines which HIR frontend and runtime behavior to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderKind {
    Json,
    Postcard,
}

impl DecoderKind {
    /// Whether `from_str` can safely enable trusted UTF-8 mode for this format.
    pub fn supports_trusted_utf8_input(self) -> bool {
        matches!(self, DecoderKind::Json)
    }
}

/// Resolved default information for a field.
#[derive(Clone, Copy)]
pub struct DefaultInfo {
    /// Pointer to the intrinsic trampoline (kajit_field_default_trait or kajit_field_default_custom).
    pub trampoline: *const u8,
    /// Pointer to the actual default function (from TypeOps or custom expression).
    pub fn_ptr: *const u8,
    /// For indirect types (generic containers), the shape needed to construct OxPtrUninit.
    /// When Some, the 3-argument trampoline `kajit_field_default_indirect` is used.
    pub shape: Option<&'static facet::Shape>,
}

/// Information about a struct field needed during code emission.
pub struct FieldEmitInfo {
    /// Byte offset of this field within the output struct.
    pub offset: usize,
    /// The facet shape of this field.
    pub shape: &'static facet::Shape,
    /// The field name (for formats that use named fields).
    pub name: &'static str,
    /// Index of this field for required-field bitset tracking.
    pub required_index: usize,
    /// If set, this field has a default value and is optional in JSON.
    pub default: Option<DefaultInfo>,
}

// r[impl deser.skip]

/// Information about a skipped field that needs default initialization.
pub struct SkippedFieldInfo {
    /// Byte offset of this field within the output struct.
    pub offset: usize,
    /// Default trampoline + function pointer for initializing this field.
    pub default: DefaultInfo,
}

// r[impl deser.enum.variant-kinds]

/// The kind of an enum variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariantKind {
    Unit,
    Tuple,
    Struct,
}

impl VariantKind {
    pub fn from_struct_type(st: &facet::StructType) -> Self {
        match st.kind {
            StructKind::Unit => VariantKind::Unit,
            StructKind::Struct => VariantKind::Struct,
            StructKind::TupleStruct | StructKind::Tuple => VariantKind::Tuple,
        }
    }
}

/// Information about an enum variant needed during code emission.
pub struct VariantEmitInfo {
    /// Variant index (0-based, used as wire discriminant for postcard).
    pub index: usize,
    /// Variant name (for JSON key matching).
    pub name: &'static str,
    /// Rust discriminant value to write to the tag slot.
    pub rust_discriminant: i64,
    /// Fields of this variant (offsets are absolute from enum base).
    pub fields: Vec<FieldEmitInfo>,
    /// Variant kind.
    pub kind: VariantKind,
}
