use std::collections::HashMap;

use facet::{ScalarType, Type, UserType};

use crate::format::VariantEmitInfo;

// r[impl deser.json.enum.untagged.object-solver]
// r[impl deser.json.enum.untagged.value-type]
// r[impl deser.json.enum.untagged.nested-key]

/// JSON value type classification for solver evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JsonValueType {
    Number,
    String,
    Bool,
    Object,
    Array,
    Null,
}

impl JsonValueType {
    /// Classify a facet Shape into its expected JSON value type.
    pub fn of_shape(shape: &'static facet::Shape) -> Self {
        match &shape.ty {
            Type::User(UserType::Struct(_)) => JsonValueType::Object,
            Type::User(UserType::Enum(_)) => JsonValueType::Object,
            _ => match shape.scalar_type() {
                Some(ScalarType::Char) => JsonValueType::String,
                Some(ScalarType::String | ScalarType::Str | ScalarType::CowStr) => {
                    JsonValueType::String
                }
                Some(ScalarType::Bool) => JsonValueType::Bool,
                Some(
                    ScalarType::U8
                    | ScalarType::U16
                    | ScalarType::U32
                    | ScalarType::U64
                    | ScalarType::U128
                    | ScalarType::USize
                    | ScalarType::I8
                    | ScalarType::I16
                    | ScalarType::I32
                    | ScalarType::I64
                    | ScalarType::I128
                    | ScalarType::ISize
                    | ScalarType::F32
                    | ScalarType::F64,
                ) => JsonValueType::Number,
                _ => panic!(
                    "cannot classify shape {} for JSON value-type evidence",
                    shape.type_identifier
                ),
            },
        }
    }
}

/// A sub-solver for nested key evidence.
///
/// When multiple object-bucket candidates have the same top-level key mapping
/// to different struct types, the sub-solver maps inner field names to outer
/// candidate bitmasks.
pub struct SubSolver {
    /// (inner_field_name, outer_candidate_mask).
    /// Bit i set means outer candidate i has this sub-field in its nested struct.
    pub inner_key_masks: Vec<(&'static str, u64)>,
}

impl SubSolver {
    /// Build a sub-solver for a specific key that maps to struct-typed fields
    /// in multiple candidates.
    ///
    /// Returns `None` if the inner keys can't disambiguate (all shared).
    fn build(
        key_name: &'static str,
        key_mask: u64,
        object_variants: &[(usize, &VariantEmitInfo)],
    ) -> Option<Self> {
        let mut all_inner_names: Vec<&'static str> = Vec::new();

        for (bit, (_, variant)) in object_variants.iter().enumerate() {
            if key_mask & (1u64 << bit) == 0 {
                continue;
            }
            let field = variant.fields.iter().find(|f| f.name == key_name).unwrap();
            match &field.shape.ty {
                Type::User(UserType::Struct(st)) => {
                    for inner_f in st.fields {
                        let name = inner_f.effective_name();
                        if !all_inner_names.contains(&name) {
                            all_inner_names.push(name);
                        }
                    }
                }
                _ => return None,
            }
        }

        all_inner_names.sort();

        let inner_key_masks: Vec<(&'static str, u64)> = all_inner_names
            .into_iter()
            .map(|inner_name| {
                let mut mask = 0u64;
                for (bit, (_, variant)) in object_variants.iter().enumerate() {
                    if key_mask & (1u64 << bit) == 0 {
                        continue;
                    }
                    let field = variant.fields.iter().find(|f| f.name == key_name).unwrap();
                    if let Type::User(UserType::Struct(st)) = &field.shape.ty
                        && st.fields.iter().any(|f| f.effective_name() == inner_name)
                    {
                        mask |= 1u64 << bit;
                    }
                }
                (inner_name, mask)
            })
            .collect();

        // Check if sub-solver can disambiguate: at least one inner key
        // has a different mask than the outer key_mask (i.e., not shared by all).
        let can_disambiguate = inner_key_masks.iter().any(|&(_, m)| m != key_mask);
        if !can_disambiguate {
            return None;
        }

        Some(SubSolver { inner_key_masks })
    }
}

/// A solver state machine lowered for JIT emission.
///
/// Built at JIT-compile time from the struct variants in an untagged enum's
/// object bucket. The solver narrows a candidate bitmask using three kinds
/// of evidence:
///
/// 1. **Key presence**: AND a per-key mask (bit i set if candidate i has that field)
/// 2. **Value type**: peek at the value's first byte and AND a per-type mask
/// 3. **Nested keys**: sub-scan a nested object's keys and AND per-inner-key masks
///
/// At runtime this is just a u64 in a stack slot, narrowed by AND with
/// baked-in immediates. No heap, no function calls.
pub struct LoweredSolver {
    /// Number of candidates (object-bucket variants). Max 64.
    pub num_candidates: usize,
    /// Initial candidate bitmask (all candidates set).
    pub initial_mask: u64,
    /// Per-key masks: (field_name, bitmask).
    /// Bit i is set if candidate i has this field.
    /// Unknown keys (not in this list) don't narrow the candidates.
    pub key_masks: Vec<(&'static str, u64)>,
    /// Maps candidate bit position → index into the original variants slice.
    pub candidate_to_variant: Vec<usize>,

    /// Per-key value-type masks (same index as key_masks).
    /// Empty = no value-type evidence for this key (either unique or all same type).
    /// Non-empty = peek at value byte, AND the matching type's mask.
    pub value_type_masks: Vec<Vec<(JsonValueType, u64)>>,

    /// Per-key sub-solver (same index as key_masks).
    /// Present when all candidates with this key have Object-typed fields
    /// and the nested structs have distinguishing sub-keys.
    pub sub_solvers: Vec<Option<SubSolver>>,
}

impl LoweredSolver {
    /// Build a solver from the struct variants in the object bucket.
    ///
    /// `object_variants` is a list of (original_variant_index, variant_info) pairs.
    /// The bit position in the bitmask corresponds to position in this slice.
    pub fn build(object_variants: &[(usize, &VariantEmitInfo)]) -> Self {
        assert!(
            object_variants.len() <= 64,
            "untagged enum object bucket has {} variants, max 64",
            object_variants.len()
        );

        let num_candidates = object_variants.len();
        let initial_mask = if num_candidates == 64 {
            !0u64
        } else {
            (1u64 << num_candidates) - 1
        };

        // Collect all unique field names across all candidates.
        let mut all_names: Vec<&'static str> = Vec::new();
        for (_, variant) in object_variants {
            for field in &variant.fields {
                if !all_names.contains(&field.name) {
                    all_names.push(field.name);
                }
            }
        }
        all_names.sort();

        // Build inverted index: for each field name, which candidates have it.
        let key_masks: Vec<(&'static str, u64)> = all_names
            .into_iter()
            .map(|name| {
                let mut mask = 0u64;
                for (bit, (_, variant)) in object_variants.iter().enumerate() {
                    if variant.fields.iter().any(|f| f.name == name) {
                        mask |= 1u64 << bit;
                    }
                }
                (name, mask)
            })
            .collect();

        let candidate_to_variant: Vec<usize> =
            object_variants.iter().map(|(idx, _)| *idx).collect();

        // Build extended evidence for ambiguous keys.
        let mut value_type_masks: Vec<Vec<(JsonValueType, u64)>> = Vec::new();
        let mut sub_solvers: Vec<Option<SubSolver>> = Vec::new();

        for &(key_name, key_mask) in &key_masks {
            if key_mask.count_ones() <= 1 {
                // Key is unique to one candidate — no further evidence needed.
                value_type_masks.push(Vec::new());
                sub_solvers.push(None);
                continue;
            }

            // Classify the field type for each candidate that has this key.
            let mut type_map: HashMap<JsonValueType, u64> = HashMap::new();
            for (bit, (_, variant)) in object_variants.iter().enumerate() {
                if key_mask & (1u64 << bit) == 0 {
                    continue;
                }
                if let Some(field) = variant.fields.iter().find(|f| f.name == key_name) {
                    let vtype = JsonValueType::of_shape(field.shape);
                    *type_map.entry(vtype).or_insert(0) |= 1u64 << bit;
                }
            }

            let vt_masks: Vec<(JsonValueType, u64)> =
                type_map.iter().map(|(&k, &v)| (k, v)).collect();
            let useful_vt = vt_masks.len() > 1;

            // Build sub-solver if all candidates have Object type for this key.
            let all_objects = vt_masks.len() == 1 && vt_masks[0].0 == JsonValueType::Object;
            let sub_solver = if all_objects {
                SubSolver::build(key_name, key_mask, object_variants)
            } else {
                None
            };

            value_type_masks.push(if useful_vt { vt_masks } else { Vec::new() });
            sub_solvers.push(sub_solver);
        }

        let solver = LoweredSolver {
            num_candidates,
            initial_mask,
            key_masks,
            candidate_to_variant,
            value_type_masks,
            sub_solvers,
        };

        // Validate: simulate worst-case resolution.
        solver.validate(object_variants);

        solver
    }

    // r[impl deser.json.enum.untagged.ambiguity-error]

    /// Simulate worst-case resolution and panic if variants are unresolvable.
    fn validate(&self, object_variants: &[(usize, &VariantEmitInfo)]) {
        // Apply all key-presence masks.
        let mut candidates = self.initial_mask;
        for &(_, mask) in &self.key_masks {
            candidates &= mask;
        }

        if candidates.count_ones() <= 1 {
            return; // Key presence alone resolves.
        }

        // For keys with value-type evidence, check if types can split remaining.
        // We check per-type: if any type mask isolates a single candidate, it resolves.
        for vt_masks in &self.value_type_masks {
            if vt_masks.is_empty() {
                continue;
            }
            // If there are multiple types, each type's mask is a subset of candidates.
            // Each candidate belongs to exactly one type, so the types partition them.
            let all_singleton = vt_masks.iter().all(|&(_, m)| m.count_ones() <= 1);
            if all_singleton {
                return; // Value-type evidence fully resolves.
            }
        }

        // For keys with sub-solvers, check if inner keys can split remaining.
        for sub in self.sub_solvers.iter().flatten() {
            // Simulate: apply all inner key masks to see if they resolve.
            let mut inner_candidates = candidates;
            for &(_, inner_mask) in &sub.inner_key_masks {
                inner_candidates &= inner_mask;
            }
            if inner_candidates.count_ones() <= 1 {
                return; // Nested key evidence resolves.
            }
        }

        // Still ambiguous — collect names for the error message.
        let mut ambiguous_names = Vec::new();
        for (bit, (_, variant)) in object_variants.iter().enumerate().take(self.num_candidates) {
            if candidates & (1u64 << bit) != 0 {
                ambiguous_names.push(variant.name);
            }
        }
        panic!(
            "untagged enum: variants [{}] are indistinguishable — \
             same key sets, same value types, same nested structure",
            ambiguous_names.join(", ")
        );
    }
}
