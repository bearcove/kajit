use std::collections::HashMap;

use dynasmrt::{AssemblyOffset, DynamicLabel};
use facet::{
    ArrayDef, Def, EnumRepr, KnownPointer, ListDef, MapDef, OptionDef, PointerDef, ScalarType,
    Shape, StructKind, Type, UserType,
};

use crate::arch::EmitCtx;
use crate::format::{
    Decoder, Encoder, FieldEmitInfo, FieldEncodeInfo, FieldLowerInfo, IrDecoder, SkippedFieldInfo,
    VariantEmitInfo, VariantEncodeInfo, VariantKind, VariantLowerInfo,
};
use crate::ir::{LambdaId, RegionBuilder, Width as IrWidth};
use crate::malum::StringOffsets;

/// A compiled deserializer. Owns the executable buffer containing JIT'd machine code.
pub struct CompiledDecoder {
    buf: dynasmrt::ExecutableBuffer,
    entry: AssemblyOffset,
    func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext),
    trusted_utf8_input: bool,
    _jit_registration: Option<crate::jit_debug::JitRegistration>,
}

impl CompiledDecoder {
    pub(crate) fn func(&self) -> unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) {
        self.func
    }

    /// The raw executable code buffer.
    pub fn code(&self) -> &[u8] {
        &self.buf
    }

    /// Byte offset of the entry point within the code buffer.
    pub fn entry_offset(&self) -> usize {
        self.entry.0
    }

    /// Whether `from_str` can safely enable trusted UTF-8 mode for this format.
    pub fn supports_trusted_utf8_input(&self) -> bool {
        self.trusted_utf8_input
    }
}

// r[impl compiler.walk]
// r[impl compiler.recursive]
// r[impl compiler.recursive.one-func-per-shape]

/// Per-shape compilation result: the DynamicLabel for inter-function calls,
/// and the AssemblyOffset for resolving the final function pointer.
struct ShapeEntry {
    /// Label bound at the function entry — used by `bl =>label` / `call =>label`.
    label: DynamicLabel,
    /// Assembly offset of the function entry — used to get the pointer from the buffer.
    offset: Option<AssemblyOffset>,
}

/// Compiler state for emitting one function per shape into a shared buffer.
struct DecoderCompiler<'fmt> {
    ectx: EmitCtx,
    decoder: &'fmt dyn Decoder,
    /// All shapes we've started or finished compiling.
    /// If offset is Some, the function is fully emitted.
    /// If offset is None, the function is in progress (label allocated but not yet bound).
    shapes: HashMap<*const Shape, ShapeEntry>,
    /// Stack offset where Option inner values can be temporarily deserialized.
    /// Zero if no Options are present.
    option_scratch_offset: u32,
    /// Discovered String layout offsets (ptr, len, cap).
    string_offsets: StringOffsets,
}

impl<'fmt> DecoderCompiler<'fmt> {
    fn new(
        extra_stack: u32,
        option_scratch_offset: u32,
        string_offsets: StringOffsets,
        decoder: &'fmt dyn Decoder,
    ) -> Self {
        DecoderCompiler {
            ectx: EmitCtx::new(extra_stack),
            decoder,
            shapes: HashMap::new(),
            option_scratch_offset,
            string_offsets,
        }
    }

    /// Compile a deserializer for `shape`. Returns the entry label and offset.
    ///
    /// - If already compiled, returns the cached entry immediately.
    /// - If currently in progress (recursive back-edge), returns the pre-allocated
    ///   label (not yet bound) — dynasmrt patches it as a forward reference.
    /// - Otherwise, emits the function depth-first.
    fn compile_shape(&mut self, shape: &'static Shape) -> DynamicLabel {
        let key = shape as *const Shape;

        if let Some(entry) = self.shapes.get(&key) {
            return entry.label;
        }

        // Pre-allocate the entry label so recursive calls can target it.
        let entry_label = self.ectx.new_label();
        self.shapes.insert(
            key,
            ShapeEntry {
                label: entry_label,
                offset: None, // not yet emitted
            },
        );

        // r[impl deser.transparent]
        // Check for transparent wrappers first — deserialize as the inner type.
        if shape.is_transparent() {
            self.compile_transparent(shape, entry_label);
            return entry_label;
        }

        // Check for List (Vec<T>) first — it's detected by Def, not Type.
        // Vec<T> has Type::User(UserType::Opaque) + Def::List.
        if let Def::List(list_def) = &shape.def {
            self.compile_vec(shape, list_def, entry_label);
            return entry_label;
        }

        // Check for Map (HashMap<K,V> / BTreeMap<K,V>) — detected by Def::Map.
        if let Def::Map(map_def) = &shape.def {
            self.compile_map(shape, map_def, entry_label);
            return entry_label;
        }

        // Check for fixed-size arrays ([T; N]) — detected by Def::Array.
        if let Def::Array(array_def) = &shape.def {
            self.compile_array(shape, array_def, entry_label);
            return entry_label;
        }

        // r[impl deser.pointer]
        // Check for smart pointers (Box<T>, Arc<T>, Rc<T>) — detected by Def::Pointer.
        if let Some(ptr_def) = get_pointer_def(shape) {
            self.compile_pointer(shape, ptr_def, entry_label);
            return entry_label;
        }

        match &shape.ty {
            Type::User(UserType::Struct(_)) => {
                self.compile_struct(shape, entry_label);
            }
            Type::User(UserType::Enum(enum_type)) => {
                self.compile_enum(shape, enum_type, entry_label);
            }
            _ => match shape.scalar_type() {
                Some(_) => self.compile_root_scalar(shape, entry_label),
                None => panic!("unsupported shape: {}", shape.type_identifier),
            },
        }

        entry_label
    }

    /// Compile a scalar root shape into a function.
    fn compile_root_scalar(&mut self, shape: &'static Shape, entry_label: DynamicLabel) {
        let key = shape as *const Shape;
        let scalar_type = shape
            .scalar_type()
            .unwrap_or_else(|| panic!("expected scalar root shape: {}", shape.type_identifier));

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_func();

        if is_string_like_scalar(scalar_type) {
            self.decoder
                .emit_read_string(&mut self.ectx, 0, scalar_type, &self.string_offsets);
        } else {
            self.decoder
                .emit_read_scalar(&mut self.ectx, 0, scalar_type);
        }

        self.ectx.end_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    /// Compile a struct shape into a function.
    fn compile_struct(&mut self, shape: &'static Shape, entry_label: DynamicLabel) {
        let key = shape as *const Shape;
        let (fields, skipped_fields) = collect_fields(shape);
        let deny_unknown_fields = shape.has_deny_unknown_fields_attr();
        let inline = self.decoder.supports_inline_nested();

        // Detect positional (tuple / tuple struct) — use positional field emission.
        let is_positional = matches!(
            &shape.ty,
            Type::User(UserType::Struct(st))
                if matches!(st.kind, StructKind::Tuple | StructKind::TupleStruct)
        );

        // For non-inlining formats, depth-first compile all nested composite fields
        // (structs and enums) so they're available as call targets.
        // Enum fields always need pre-compilation (even in inline formats) since
        // they can't be inlined — they need their own discriminant dispatch.
        // Option fields: if the inner T is composite, pre-compile it too.
        let nested: Vec<Option<DynamicLabel>> = if inline {
            fields
                .iter()
                .map(|f| {
                    let target = unwrap_inner_or_self(f.shape);
                    // Lists (Vec<T>) and Maps always need pre-compilation.
                    // Enums always need pre-compilation (can't be inlined).
                    if matches!(&target.ty, Type::User(UserType::Enum(_)))
                        || matches!(&target.def, Def::List(_) | Def::Map(_))
                    {
                        Some(self.compile_shape(target))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            fields
                .iter()
                .map(|f| {
                    let target = unwrap_inner_or_self(f.shape);
                    if is_composite(target) {
                        Some(self.compile_shape(target))
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Bind the entry label at the function start, then emit prologue.
        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_func();

        // r[impl deser.skip]
        // Emit default initialization for skipped fields before the key-dispatch loop.
        for sf in &skipped_fields {
            emit_default_init(&mut self.ectx, &sf.default, sf.offset as u32);
        }

        let format = self.decoder;
        let option_scratch_offset = self.option_scratch_offset;
        let string_offsets = &self.string_offsets;
        let ectx = &mut self.ectx;

        if is_positional {
            format.emit_positional_fields(ectx, &fields, &mut |ectx, field| {
                emit_field(
                    ectx,
                    format,
                    field,
                    &fields,
                    &nested,
                    option_scratch_offset,
                    string_offsets,
                );
            });
        } else {
            format.emit_struct_fields(ectx, &fields, deny_unknown_fields, &mut |ectx, field| {
                emit_field(
                    ectx,
                    format,
                    field,
                    &fields,
                    &nested,
                    option_scratch_offset,
                    string_offsets,
                );
            });
        }

        self.ectx.end_func(error_exit);

        // Mark as finished with the resolved offset.
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    // r[impl deser.transparent]
    // r[impl deser.transparent.forwarding]
    // r[impl deser.transparent.composite]

    /// Compile a transparent wrapper: deserialize as the inner type directly.
    fn compile_transparent(&mut self, shape: &'static Shape, entry_label: DynamicLabel) {
        let key = shape as *const Shape;

        // Get the inner shape from the transparent wrapper.
        let inner_shape = shape.inner.unwrap_or_else(|| {
            panic!(
                "transparent shape {} has no inner shape",
                shape.type_identifier
            )
        });

        // Get the field offset — transparent structs have exactly one field.
        let field_offset = match &shape.ty {
            Type::User(UserType::Struct(st)) => {
                assert!(
                    st.fields.len() == 1,
                    "transparent struct {} has {} fields, expected 1",
                    shape.type_identifier,
                    st.fields.len()
                );
                st.fields[0].offset
            }
            _ => 0, // Non-struct transparent (e.g. newtype via repr(transparent))
        };

        // Pre-compile the inner shape if it's composite.
        let inner_label = if needs_precompilation(inner_shape) {
            Some(self.compile_shape(inner_shape))
        } else {
            None
        };

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_func();

        // Emit deserialization of the inner type at the field's offset.
        if let Some(label) = inner_label {
            self.ectx.emit_call_emitted_func(label, field_offset as u32);
        } else {
            match inner_shape.scalar_type() {
                Some(st) if is_string_like_scalar(st) => {
                    self.decoder.emit_read_string(
                        &mut self.ectx,
                        field_offset,
                        st,
                        &self.string_offsets,
                    );
                }
                Some(st) => {
                    self.decoder
                        .emit_read_scalar(&mut self.ectx, field_offset, st);
                }
                None => panic!(
                    "unsupported transparent inner type: {}",
                    inner_shape.type_identifier,
                ),
            }
        }

        self.ectx.end_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    // r[impl seq.malum]

    /// Compile a Vec<T> shape into a function.
    fn compile_vec(
        &mut self,
        shape: &'static Shape,
        list_def: &'static ListDef,
        entry_label: DynamicLabel,
    ) {
        let key = shape as *const Shape;

        // Discover Vec<T> layout at JIT-compile time using vtable probing.
        let vec_offsets = crate::malum::discover_vec_offsets(list_def, shape);

        let elem_shape = list_def.t;

        // Pre-compile element shape if it's composite (struct/enum/vec).
        let elem_label = if needs_precompilation(elem_shape) {
            Some(self.compile_shape(elem_shape))
        } else {
            None
        };

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_func();

        let format = self.decoder;
        let option_scratch_offset = self.option_scratch_offset;
        let string_offsets = &self.string_offsets;

        format.emit_vec(
            &mut self.ectx,
            0, // offset=0: function receives out pointing to the Vec slot
            elem_shape,
            elem_label,
            &vec_offsets,
            option_scratch_offset,
            &mut |ectx| {
                // Emit element deserialization. Out is already redirected to the slot.
                emit_elem(
                    ectx,
                    format,
                    elem_shape,
                    elem_label,
                    option_scratch_offset,
                    string_offsets,
                );
            },
        );

        self.ectx.end_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    /// Compile a fixed-size array ([T; N]) shape into a function.
    ///
    /// Arrays have N elements of the same type at contiguous offsets.
    /// Wire format: positional (no count prefix for postcard, `[...]` for JSON).
    fn compile_array(
        &mut self,
        shape: &'static Shape,
        array_def: &'static ArrayDef,
        entry_label: DynamicLabel,
    ) {
        let key = shape as *const Shape;

        let elem_shape = array_def.t;
        let n = array_def.n;
        let elem_size = elem_shape
            .layout
            .sized_layout()
            .expect("array element must be Sized")
            .size();

        // Pre-compile element shape if it's composite.
        let elem_label = if needs_precompilation(elem_shape) {
            Some(self.compile_shape(elem_shape))
        } else {
            None
        };

        // Build fake FieldEmitInfo for each element: offset = i * elem_size.
        let fake_fields: Vec<FieldEmitInfo> = (0..n)
            .map(|i| FieldEmitInfo {
                offset: i * elem_size,
                shape: elem_shape,
                name: "",
                required_index: i,
                default: None,
            })
            .collect();
        let nested: Vec<Option<DynamicLabel>> = vec![elem_label; n];

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_func();

        let format = self.decoder;
        let option_scratch_offset = self.option_scratch_offset;
        let string_offsets = &self.string_offsets;
        let ectx = &mut self.ectx;

        format.emit_positional_fields(ectx, &fake_fields, &mut |ectx, field| {
            emit_field(
                ectx,
                format,
                field,
                &fake_fields,
                &nested,
                option_scratch_offset,
                string_offsets,
            );
        });

        self.ectx.end_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    /// Compile a Map<K, V> shape into a function using the two-pass approach:
    /// deserialize (K, V) pairs into a temp buffer, then call from_pair_slice.
    fn compile_map(
        &mut self,
        shape: &'static Shape,
        map_def: &'static MapDef,
        entry_label: DynamicLabel,
    ) {
        let key = shape as *const Shape;

        let k_shape = map_def.k;
        let v_shape = map_def.v;

        // Pre-compile key and value shapes if they are composite.
        let k_label = if needs_precompilation(k_shape) {
            Some(self.compile_shape(k_shape))
        } else {
            None
        };
        let v_label = if needs_precompilation(v_shape) {
            Some(self.compile_shape(v_shape))
        } else {
            None
        };

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_func();

        let format = self.decoder;
        let option_scratch_offset = self.option_scratch_offset;
        let string_offsets = &self.string_offsets;

        format.emit_map(
            &mut self.ectx,
            0, // offset=0: function receives out pointing to the Map slot
            map_def,
            k_shape,
            v_shape,
            k_label,
            v_label,
            option_scratch_offset,
            &mut |ectx| {
                emit_elem(
                    ectx,
                    format,
                    k_shape,
                    k_label,
                    option_scratch_offset,
                    string_offsets,
                );
            },
            &mut |ectx| {
                emit_elem(
                    ectx,
                    format,
                    v_shape,
                    v_label,
                    option_scratch_offset,
                    string_offsets,
                );
            },
        );

        self.ectx.end_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    // r[impl deser.pointer]
    // r[impl deser.pointer.scratch]
    // r[impl deser.pointer.format-transparent]

    /// Compile a smart pointer (Box<T>, Arc<T>, Rc<T>) into a function.
    ///
    /// Smart pointers are wire-transparent: the wire format contains just T.
    /// We deserialize T into the scratch area, then call `new_into_fn` to
    /// wrap it into the heap-allocated pointer at `out + 0`.
    fn compile_pointer(
        &mut self,
        shape: &'static Shape,
        ptr_def: &'static PointerDef,
        entry_label: DynamicLabel,
    ) {
        let key = shape as *const Shape;

        let pointee = ptr_def
            .pointee
            .unwrap_or_else(|| panic!("pointer {} has no pointee", shape.type_identifier));
        let new_into_fn = ptr_def
            .vtable
            .new_into_fn
            .unwrap_or_else(|| panic!("pointer {} has no new_into_fn", shape.type_identifier));

        // Pre-compile inner type if composite.
        let inner_label = if needs_precompilation(pointee) {
            Some(self.compile_shape(pointee))
        } else {
            None
        };

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_func();

        let option_scratch_offset = self.option_scratch_offset;
        let string_offsets = &self.string_offsets;

        // Redirect out to scratch area.
        self.ectx.emit_redirect_out_to_stack(option_scratch_offset);

        // Deserialize inner T at offset 0 (scratch area).
        if let Some(label) = inner_label {
            self.ectx.emit_call_emitted_func(label, 0);
        } else if is_struct(pointee) && self.decoder.supports_inline_nested() {
            emit_inline_struct(
                &mut self.ectx,
                self.decoder,
                pointee,
                0,
                option_scratch_offset,
                string_offsets,
            );
        } else {
            match pointee.scalar_type() {
                Some(st) if is_string_like_scalar(st) => {
                    self.decoder
                        .emit_read_string(&mut self.ectx, 0, st, string_offsets);
                }
                Some(st) => {
                    self.decoder.emit_read_scalar(&mut self.ectx, 0, st);
                }
                None => panic!(
                    "unsupported pointer inner type: {}",
                    pointee.type_identifier,
                ),
            }
        }

        // Restore out, then call new_into_fn to wrap T into the pointer.
        self.ectx.emit_restore_out(option_scratch_offset);
        self.ectx.emit_call_option_init_some(
            crate::intrinsics::kajit_pointer_new_into as *const u8,
            new_into_fn as *const u8,
            0, // offset=0: function receives out pointing to the pointer slot
            option_scratch_offset,
        );

        self.ectx.end_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    // r[impl deser.postcard.enum]
    // r[impl deser.json.enum.external]

    /// Compile an enum shape into a function.
    fn compile_enum(
        &mut self,
        shape: &'static Shape,
        enum_type: &'static facet::EnumType,
        entry_label: DynamicLabel,
    ) {
        let key = shape as *const Shape;
        let variants = collect_variants(enum_type);
        let disc_size = discriminant_size(enum_type.enum_repr);
        let inline = self.decoder.supports_inline_nested();

        // Depth-first compile all nested composite types (structs and enums)
        // found in variant fields so they're available as call targets.
        // Enum fields always need pre-compilation even in inline formats.
        // Option fields: if the inner T is composite, pre-compile it too.
        let nested_labels: Vec<Vec<Option<DynamicLabel>>> = if inline {
            variants
                .iter()
                .map(|v| {
                    v.fields
                        .iter()
                        .map(|f| {
                            let target = unwrap_inner_or_self(f.shape);
                            if matches!(&target.ty, Type::User(UserType::Enum(_)))
                                || matches!(&target.def, Def::List(_) | Def::Map(_))
                            {
                                Some(self.compile_shape(target))
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .collect()
        } else {
            variants
                .iter()
                .map(|v| {
                    v.fields
                        .iter()
                        .map(|f| {
                            let target = unwrap_inner_or_self(f.shape);
                            if is_composite(target) {
                                Some(self.compile_shape(target))
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .collect()
        };

        // Bind the entry label, emit prologue.
        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_func();

        let format = self.decoder;
        let option_scratch_offset = self.option_scratch_offset;
        let string_offsets = &self.string_offsets;
        let ectx = &mut self.ectx;

        // Detect tagging mode from shape metadata.
        let tag_key = shape.get_tag_attr();
        let content_key = shape.get_content_attr();

        // Closure for the standard variant body: writes discriminant + deserializes fields.
        let mut emit_standard_variant_body = |ectx: &mut EmitCtx, variant: &VariantEmitInfo| {
            ectx.emit_write_discriminant(variant.rust_discriminant, disc_size);

            if variant.kind == VariantKind::Unit {
                return;
            }

            let nested = &nested_labels[variant.index];

            if variant.kind == VariantKind::Struct && !inline {
                format.emit_struct_fields(ectx, &variant.fields, false, &mut |ectx, field| {
                    emit_field(
                        ectx,
                        format,
                        field,
                        &variant.fields,
                        nested,
                        option_scratch_offset,
                        string_offsets,
                    );
                });
                return;
            }

            for field in &variant.fields {
                emit_field(
                    ectx,
                    format,
                    field,
                    &variant.fields,
                    nested,
                    option_scratch_offset,
                    string_offsets,
                );
            }
        };

        match (tag_key, content_key, shape.is_untagged()) {
            // Externally tagged (default)
            (None, None, false) => {
                format.emit_enum(ectx, &variants, &mut emit_standard_variant_body);
            }

            // r[impl deser.json.enum.adjacent]
            // Adjacently tagged: { "tag_key": "Variant", "content_key": value }
            (Some(tk), Some(ck), false) => {
                format.emit_enum_adjacent(ectx, &variants, tk, ck, &mut emit_standard_variant_body);
            }

            // r[impl deser.json.enum.internal]
            // Internally tagged: { "tag_key": "Variant", ...variant_fields... }
            (Some(tk), None, false) => {
                format.emit_enum_internal(
                    ectx,
                    &variants,
                    tk,
                    &mut |ectx, variant| {
                        ectx.emit_write_discriminant(variant.rust_discriminant, disc_size);
                    },
                    &mut |ectx, variant, field| {
                        let nested = &nested_labels[variant.index];
                        emit_field(
                            ectx,
                            format,
                            field,
                            &variant.fields,
                            nested,
                            option_scratch_offset,
                            string_offsets,
                        );
                    },
                );
            }

            // r[impl deser.json.enum.untagged]
            // Untagged: value-type bucketing + peek dispatch + solver
            (_, _, true) => {
                format.emit_enum_untagged(ectx, &variants, &mut emit_standard_variant_body);
            }

            // Invalid: content without tag
            (None, Some(_), _) => {
                panic!("content attribute without tag attribute is invalid");
            }
        }

        self.ectx.end_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }
}

// r[impl deser.flatten]
// r[impl deser.flatten.offset-accumulation]
// r[impl deser.flatten.inline]

fn collect_fields(shape: &'static Shape) -> (Vec<FieldEmitInfo>, Vec<SkippedFieldInfo>) {
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
fn resolve_trait_default(shape: &'static Shape) -> Option<crate::format::DefaultInfo> {
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
fn check_field_name_collisions(fields: &[FieldEmitInfo]) {
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

fn collect_variants(enum_type: &'static facet::EnumType) -> Vec<VariantEmitInfo> {
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
fn discriminant_size(repr: EnumRepr) -> u32 {
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
fn get_option_def(shape: &'static Shape) -> Option<&'static OptionDef> {
    match &shape.def {
        Def::Option(opt_def) => Some(opt_def),
        _ => None,
    }
}

// r[impl deser.pointer]

/// Returns the PointerDef if this shape is a supported smart pointer (Box, Arc, Rc).
fn get_pointer_def(shape: &'static Shape) -> Option<&'static PointerDef> {
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

/// If the shape is an Option or smart pointer, return the inner T's shape;
/// otherwise return the shape itself.
/// Used for nested label pre-compilation — we need to compile the inner T,
/// not Option<T> or Box<T>.
fn unwrap_inner_or_self(shape: &'static Shape) -> &'static Shape {
    if let Some(opt_def) = get_option_def(shape) {
        return opt_def.t;
    }
    if let Some(ptr_def) = get_pointer_def(shape)
        && let Some(pointee) = ptr_def.pointee
    {
        return pointee;
    }
    shape
}

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

fn is_unit(shape: &'static Shape) -> bool {
    shape.scalar_type() == Some(ScalarType::Unit)
}

fn is_struct(shape: &'static Shape) -> bool {
    matches!(&shape.ty, Type::User(UserType::Struct(_))) && !is_unit(shape)
}

/// Returns true if the shape needs its own compiled function
/// (struct, enum, vec, map, or smart pointer).
fn is_composite(shape: &'static Shape) -> bool {
    if is_unit(shape) {
        return false;
    }
    matches!(
        &shape.ty,
        Type::User(UserType::Struct(_) | UserType::Enum(_))
    ) || matches!(&shape.def, Def::List(_) | Def::Map(_) | Def::Array(_))
        || get_pointer_def(shape).is_some()
}

/// Returns true if the shape needs pre-compilation as a separate function.
fn needs_precompilation(shape: &'static Shape) -> bool {
    is_composite(shape)
}

fn is_string_like_scalar(scalar_type: ScalarType) -> bool {
    matches!(
        scalar_type,
        ScalarType::String | ScalarType::Str | ScalarType::CowStr
    )
}

/// Emit code for a single field, dispatching to nested struct calls, inline
/// expansion, scalar intrinsics, or Option handling.
fn emit_field(
    ectx: &mut EmitCtx,
    decoder: &dyn Decoder,
    field: &FieldEmitInfo,
    all_fields: &[FieldEmitInfo],
    nested: &[Option<DynamicLabel>],
    option_scratch_offset: u32,
    string_offsets: &StringOffsets,
) {
    // Find this field's index by matching offset.
    let idx = all_fields
        .iter()
        .position(|f| f.offset == field.offset)
        .expect("field not found in all_fields");

    // r[impl deser.option]
    // Check if this field is an Option<T>.
    if let Some(opt_def) = get_option_def(field.shape) {
        let init_none_fn = opt_def.vtable.init_none as *const u8;
        let init_some_fn = opt_def.vtable.init_some as *const u8;
        let inner_shape = opt_def.t;

        decoder.emit_option(
            ectx,
            field.offset,
            init_none_fn,
            init_some_fn,
            option_scratch_offset,
            &mut |ectx| {
                // Emit inner T deserialization into the scratch area (out is already redirected).
                // The inner T's offset is 0 because out now points to the scratch area.
                if let Some(label) = nested[idx] {
                    ectx.emit_call_emitted_func(label, 0);
                } else if is_struct(inner_shape) && decoder.supports_inline_nested() {
                    emit_inline_struct(
                        ectx,
                        decoder,
                        inner_shape,
                        0,
                        option_scratch_offset,
                        string_offsets,
                    );
                } else {
                    match inner_shape.scalar_type() {
                        Some(st) if is_string_like_scalar(st) => {
                            decoder.emit_read_string(ectx, 0, st, string_offsets);
                        }
                        Some(st) => {
                            decoder.emit_read_scalar(ectx, 0, st);
                        }
                        None => panic!(
                            "unsupported Option inner type: {}",
                            inner_shape.type_identifier,
                        ),
                    }
                }
            },
        );
        return;
    }

    // r[impl deser.pointer]
    // r[impl deser.pointer.scratch]
    // Check if this field is a smart pointer (Box<T>, Arc<T>, Rc<T>).
    if let Some(ptr_def) = get_pointer_def(field.shape) {
        let pointee = ptr_def
            .pointee
            .unwrap_or_else(|| panic!("pointer {} has no pointee", field.shape.type_identifier));
        let new_into_fn = ptr_def.vtable.new_into_fn.unwrap_or_else(|| {
            panic!("pointer {} has no new_into_fn", field.shape.type_identifier)
        });

        // Redirect out to scratch area, deserialize inner T, then wrap.
        ectx.emit_redirect_out_to_stack(option_scratch_offset);

        // Deserialize inner T at offset 0 (scratch area).
        if let Some(label) = nested[idx] {
            ectx.emit_call_emitted_func(label, 0);
        } else if is_struct(pointee) && decoder.supports_inline_nested() {
            emit_inline_struct(
                ectx,
                decoder,
                pointee,
                0,
                option_scratch_offset,
                string_offsets,
            );
        } else {
            match pointee.scalar_type() {
                Some(st) if is_string_like_scalar(st) => {
                    decoder.emit_read_string(ectx, 0, st, string_offsets);
                }
                Some(st) => {
                    decoder.emit_read_scalar(ectx, 0, st);
                }
                None => panic!(
                    "unsupported pointer inner type: {}",
                    pointee.type_identifier,
                ),
            }
        }

        // Restore out, then call new_into_fn to wrap T into the pointer.
        ectx.emit_restore_out(option_scratch_offset);
        ectx.emit_call_option_init_some(
            crate::intrinsics::kajit_pointer_new_into as *const u8,
            new_into_fn as *const u8,
            field.offset as u32,
            option_scratch_offset,
        );
        return;
    }

    if let Some(label) = nested[idx] {
        // r[impl deser.nested-struct]
        // r[impl deser.nested-struct.offset]
        ectx.emit_call_emitted_func(label, field.offset as u32);
        return;
    }

    // r[impl deser.nested-struct]
    // r[impl deser.nested-struct.offset]
    if is_struct(field.shape) && decoder.supports_inline_nested() {
        emit_inline_struct(
            ectx,
            decoder,
            field.shape,
            field.offset,
            option_scratch_offset,
            string_offsets,
        );
        return;
    }

    match field.shape.scalar_type() {
        Some(st) if is_string_like_scalar(st) => {
            decoder.emit_read_string(ectx, field.offset, st, string_offsets);
        }
        Some(st) => {
            decoder.emit_read_scalar(ectx, field.offset, st);
        }
        None => panic!(
            "unsupported field type: {} (scalar_type={:?})",
            field.shape.type_identifier,
            field.shape.scalar_type()
        ),
    }
}

/// Emit deserialization code for a single element (used by Vec loop).
/// Out pointer has already been redirected to the element slot at offset 0.
fn emit_elem(
    ectx: &mut EmitCtx,
    decoder: &dyn Decoder,
    elem_shape: &'static Shape,
    elem_label: Option<DynamicLabel>,
    option_scratch_offset: u32,
    string_offsets: &StringOffsets,
) {
    if let Some(label) = elem_label {
        ectx.emit_call_emitted_func(label, 0);
    } else if is_struct(elem_shape) && decoder.supports_inline_nested() {
        emit_inline_struct(
            ectx,
            decoder,
            elem_shape,
            0,
            option_scratch_offset,
            string_offsets,
        );
    } else {
        match elem_shape.scalar_type() {
            Some(st) if is_string_like_scalar(st) => {
                decoder.emit_read_string(ectx, 0, st, string_offsets);
            }
            Some(st) => {
                decoder.emit_read_scalar(ectx, 0, st);
            }
            None => panic!(
                "unsupported Vec element type: {}",
                elem_shape.type_identifier,
            ),
        }
    }
}

/// Inline a nested struct's fields into the parent function.
///
/// Instead of emitting a function call, we collect the nested struct's fields,
/// adjust their offsets by `base_offset`, and emit them directly via the format's
/// `emit_struct_fields`. This recurses for deeply nested structs.
fn emit_inline_struct(
    ectx: &mut EmitCtx,
    decoder: &dyn Decoder,
    shape: &'static Shape,
    base_offset: usize,
    option_scratch_offset: u32,
    string_offsets: &StringOffsets,
) {
    let (inner_fields, skipped_fields) = collect_fields(shape);

    // Emit default initialization for skipped fields in the inlined struct.
    for sf in &skipped_fields {
        emit_default_init(ectx, &sf.default, (base_offset + sf.offset) as u32);
    }

    let adjusted: Vec<FieldEmitInfo> = inner_fields
        .into_iter()
        .map(|f| FieldEmitInfo {
            offset: base_offset + f.offset,
            shape: f.shape,
            name: f.name,
            required_index: f.required_index,
            default: f.default,
        })
        .collect();

    // No nested labels — all struct fields will recurse into emit_inline_struct.
    let no_nested = vec![None; adjusted.len()];

    // Inline nested structs inherit the parent's deny_unknown_fields behavior,
    // but currently inline is only used by postcard (positional, no unknown fields possible).
    decoder.emit_struct_fields(ectx, &adjusted, false, &mut |ectx, field| {
        emit_field(
            ectx,
            decoder,
            field,
            &adjusted,
            &no_nested,
            option_scratch_offset,
            string_offsets,
        );
    });
}

// r[impl deser.pointer.scratch]

/// Compute the maximum scratch inner type size across a shape tree.
/// Both Option<T> and smart pointers (Box<T>, Arc<T>, Rc<T>) need a scratch
/// area to deserialize the inner T before wrapping. Returns 0 if none found.
fn max_scratch_inner_size(
    shape: &'static Shape,
    visited: &mut std::collections::HashSet<*const Shape>,
) -> usize {
    let key = shape as *const Shape;
    if !visited.insert(key) {
        return 0;
    }

    let mut max_size = 0usize;

    // Walk through List element types and Map key/value types.
    if let Def::List(list_def) = &shape.def {
        max_size = max_size.max(max_scratch_inner_size(list_def.t, visited));
    }
    if let Def::Map(map_def) = &shape.def {
        max_size = max_size.max(max_scratch_inner_size(map_def.k, visited));
        max_size = max_size.max(max_scratch_inner_size(map_def.v, visited));
    }

    // Check if shape itself is a pointer type (e.g. Vec<Box<T>> element)
    if let Some(ptr_def) = get_pointer_def(shape)
        && let Some(pointee) = ptr_def.pointee
    {
        let inner_size = pointee
            .layout
            .sized_layout()
            .expect("Pointer inner type must be Sized")
            .size();
        max_size = max_size.max(inner_size);
        max_size = max_size.max(max_scratch_inner_size(pointee, visited));
    }

    match &shape.ty {
        Type::User(UserType::Struct(st)) => {
            for f in st.fields {
                let field_shape = f.shape();
                if let Some(opt_def) = get_option_def(field_shape) {
                    let inner_size = opt_def
                        .t
                        .layout
                        .sized_layout()
                        .expect("Option inner type must be Sized")
                        .size();
                    max_size = max_size.max(inner_size);
                    max_size = max_size.max(max_scratch_inner_size(opt_def.t, visited));
                }
                if let Some(ptr_def) = get_pointer_def(field_shape)
                    && let Some(pointee) = ptr_def.pointee
                {
                    let inner_size = pointee
                        .layout
                        .sized_layout()
                        .expect("Pointer inner type must be Sized")
                        .size();
                    max_size = max_size.max(inner_size);
                    max_size = max_size.max(max_scratch_inner_size(pointee, visited));
                }
                max_size = max_size.max(max_scratch_inner_size(field_shape, visited));
            }
        }
        Type::User(UserType::Enum(enum_type)) => {
            for v in enum_type.variants {
                for f in v.data.fields {
                    let field_shape = f.shape();
                    if let Some(opt_def) = get_option_def(field_shape) {
                        let inner_size = opt_def
                            .t
                            .layout
                            .sized_layout()
                            .expect("Option inner type must be Sized")
                            .size();
                        max_size = max_size.max(inner_size);
                        max_size = max_size.max(max_scratch_inner_size(opt_def.t, visited));
                    }
                    if let Some(ptr_def) = get_pointer_def(field_shape)
                        && let Some(pointee) = ptr_def.pointee
                    {
                        let inner_size = pointee
                            .layout
                            .sized_layout()
                            .expect("Pointer inner type must be Sized")
                            .size();
                        max_size = max_size.max(inner_size);
                        max_size = max_size.max(max_scratch_inner_size(pointee, visited));
                    }
                    max_size = max_size.max(max_scratch_inner_size(field_shape, visited));
                }
            }
        }
        _ => {}
    }

    max_size
}

/// Check if a shape tree contains any List (Vec<T>) or Map types that need
/// the sequence buffer stack layout.
fn has_seq_or_map_in_tree(
    shape: &'static Shape,
    visited: &mut std::collections::HashSet<*const Shape>,
) -> bool {
    let key = shape as *const Shape;
    if !visited.insert(key) {
        return false;
    }

    if matches!(&shape.def, Def::List(_) | Def::Map(_)) {
        return true;
    }

    // Recurse into map key/value types.
    if let Def::Map(map_def) = &shape.def {
        if has_seq_or_map_in_tree(map_def.k, visited) {
            return true;
        }
        if has_seq_or_map_in_tree(map_def.v, visited) {
            return true;
        }
    }

    // Recurse through smart pointer inner types.
    if let Some(ptr_def) = get_pointer_def(shape)
        && let Some(pointee) = ptr_def.pointee
        && has_seq_or_map_in_tree(pointee, visited)
    {
        return true;
    }

    match &shape.ty {
        Type::User(UserType::Struct(st)) => {
            for f in st.fields {
                let target = unwrap_inner_or_self(f.shape());
                if has_seq_or_map_in_tree(target, visited) {
                    return true;
                }
            }
        }
        Type::User(UserType::Enum(enum_type)) => {
            for v in enum_type.variants {
                for f in v.data.fields {
                    let target = unwrap_inner_or_self(f.shape());
                    if has_seq_or_map_in_tree(target, visited) {
                        return true;
                    }
                }
            }
        }
        _ => {}
    }

    false
}

/// Compile a deserializer for the given shape and format.
pub fn compile_decoder(shape: &'static Shape, decoder: &dyn Decoder) -> CompiledDecoder {
    // Compute extra stack space. For structs, pass fields. For enums, we need
    // the max across all variant bodies. Use empty fields for enums since JSON
    // enum handling computes its own stack needs.
    let mut format_extra = if matches!(&shape.def, Def::List(_)) {
        // Vec<T> as root shape — format needs stack space for vec loop state.
        decoder.extra_stack_space(&[])
    } else if matches!(&shape.def, Def::Map(_)) {
        // Map<K,V> as root shape — format needs stack space for map loop state.
        decoder.extra_stack_space(&[])
    } else if matches!(&shape.def, Def::Array(_)) {
        // [T; N] as root shape — positional, no format-level key-matching state.
        decoder.extra_stack_space(&[])
    } else if get_pointer_def(shape).is_some() {
        // Smart pointer as root shape — no format-specific stack needed at this level,
        // but inner type may need stack (handled by recursive compilation).
        decoder.extra_stack_space(&[])
    } else {
        match &shape.ty {
            Type::User(UserType::Struct(_)) => {
                let (fields, _) = collect_fields(shape);
                decoder.extra_stack_space(&fields)
            }
            Type::User(UserType::Enum(enum_type)) => {
                // For enums, the format might need extra stack for variant body
                // deserialization (e.g., JSON struct variant key matching).
                // Compute the max across all variants.
                let mut max_extra = 0u32;
                for v in enum_type.variants {
                    let fields: Vec<FieldEmitInfo> = v
                        .data
                        .fields
                        .iter()
                        .enumerate()
                        .map(|(j, f)| FieldEmitInfo {
                            offset: f.offset,
                            shape: f.shape(),
                            name: f.name,
                            required_index: j,
                            default: None,
                        })
                        .collect();
                    max_extra = max_extra.max(decoder.extra_stack_space(&fields));
                }
                max_extra
            }
            _ => decoder.extra_stack_space(&[]),
        }
    };

    // If the shape tree contains any Vec or Map types, ensure we have enough stack
    // for sequence buffer state (buf_ptr, count/cap, counter, saved_out).
    if has_seq_or_map_in_tree(shape, &mut std::collections::HashSet::new()) {
        format_extra = format_extra.max(decoder.vec_extra_stack_space());
        format_extra = format_extra.max(decoder.map_extra_stack_space());
    }

    // Compute Option scratch space: 8 bytes for saved out pointer + max inner T size.
    // The scratch area lives after the base frame and format's extra stack.
    // Stack layout from sp:
    //   [0..BASE_FRAME)         callee-saved registers
    //   [BASE_FRAME..BASE_FRAME+format_extra)  format-specific data (bitset, key_ptr, etc.)
    //   [BASE_FRAME+format_extra..BASE_FRAME+format_extra+8)  saved out pointer
    //   [BASE_FRAME+format_extra+8..BASE_FRAME+format_extra+8+max_inner)  inner T scratch
    //
    // option_scratch_offset is the absolute sp offset where inner T is written.
    // The emit_redirect_out_to_stack method saves old out at scratch_offset-8.
    let max_inner = max_scratch_inner_size(shape, &mut std::collections::HashSet::new());
    let (option_scratch_offset, extra_stack) = if max_inner > 0 {
        let option_extra = 8 + max_inner as u32;
        let base_frame = crate::arch::BASE_FRAME;
        let scratch_offset = base_frame + format_extra + 8;
        (scratch_offset, format_extra + option_extra)
    } else {
        (0, format_extra)
    };

    // Discover String layout offsets (cached after first call).
    let string_offsets = crate::malum::discover_string_offsets();

    let mut compiler =
        DecoderCompiler::new(extra_stack, option_scratch_offset, string_offsets, decoder);
    compiler.compile_shape(shape);

    // Get the entry offset for the root shape.
    let key = shape as *const Shape;
    let entry_offset = compiler.shapes[&key]
        .offset
        .expect("root shape was not fully compiled");

    let buf = compiler.ectx.finalize();
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.ptr(entry_offset)) };

    let registration = register_jit_symbols(&compiler.shapes, &buf, "decode");

    CompiledDecoder {
        buf,
        entry: entry_offset,
        func,
        trusted_utf8_input: decoder.supports_trusted_utf8_input(),
        _jit_registration: Some(registration),
    }
}

fn ir_width_from_disc_size(size: u32) -> IrWidth {
    match size {
        1 => IrWidth::W1,
        2 => IrWidth::W2,
        4 => IrWidth::W4,
        8 => IrWidth::W8,
        _ => panic!("unsupported discriminant size: {size}"),
    }
}

fn lower_fields_for_ir(
    shape: &'static Shape,
    base_offset: usize,
) -> (Vec<FieldLowerInfo>, Vec<SkippedFieldInfo>) {
    let (fields, skipped) = collect_fields(shape);
    let lowered = fields
        .into_iter()
        .map(|f| FieldLowerInfo {
            offset: base_offset + f.offset,
            shape: f.shape,
            name: f.name,
            required_index: f.required_index,
            has_default: f.default.is_some(),
        })
        .collect();
    (lowered, skipped)
}

fn lower_variants_for_ir(
    enum_type: &'static facet::EnumType,
    base_offset: usize,
) -> Vec<VariantLowerInfo> {
    collect_variants(enum_type)
        .into_iter()
        .map(|v| VariantLowerInfo {
            index: v.index,
            name: v.name,
            rust_discriminant: v.rust_discriminant,
            kind: v.kind,
            fields: v
                .fields
                .into_iter()
                .map(|f| FieldLowerInfo {
                    offset: base_offset + f.offset,
                    shape: f.shape,
                    name: f.name,
                    required_index: f.required_index,
                    has_default: f.default.is_some(),
                })
                .collect(),
        })
        .collect()
}

fn ir_shape_needs_lambda(shape: &'static Shape) -> bool {
    if is_unit(shape) {
        return false;
    }

    if shape.is_transparent() || get_option_def(shape).is_some() {
        return true;
    }

    if matches!(&shape.def, Def::List(_) | Def::Map(_) | Def::Array(_)) {
        return true;
    }

    if get_pointer_def(shape).is_some() {
        return true;
    }

    matches!(
        &shape.ty,
        Type::User(UserType::Struct(_) | UserType::Enum(_))
    )
}

fn collect_ir_lambda_shapes(
    shape: &'static Shape,
    seen: &mut std::collections::HashSet<*const Shape>,
    out: &mut Vec<&'static Shape>,
) {
    if !ir_shape_needs_lambda(shape) {
        return;
    }

    let key = shape as *const Shape;
    if !seen.insert(key) {
        return;
    }
    out.push(shape);

    if shape.is_transparent() {
        let (fields, _) = collect_fields(shape);
        for field in &fields {
            collect_ir_lambda_shapes(field.shape, seen, out);
        }
        return;
    }

    if let Some(opt_def) = get_option_def(shape) {
        collect_ir_lambda_shapes(opt_def.t, seen, out);
        return;
    }

    if let Some(ptr_def) = get_pointer_def(shape) {
        if let Some(pointee) = ptr_def.pointee {
            collect_ir_lambda_shapes(pointee, seen, out);
        }
        return;
    }

    match &shape.def {
        Def::List(list_def) => {
            collect_ir_lambda_shapes(list_def.t, seen, out);
            return;
        }
        Def::Map(map_def) => {
            collect_ir_lambda_shapes(map_def.k, seen, out);
            collect_ir_lambda_shapes(map_def.v, seen, out);
            return;
        }
        Def::Array(array_def) => {
            collect_ir_lambda_shapes(array_def.t, seen, out);
            return;
        }
        _ => {}
    }

    match &shape.ty {
        Type::User(UserType::Struct(_)) => {
            let (fields, _) = collect_fields(shape);
            for field in &fields {
                collect_ir_lambda_shapes(field.shape, seen, out);
            }
        }
        Type::User(UserType::Enum(enum_type)) => {
            let variants = collect_variants(enum_type);
            for variant in &variants {
                for field in &variant.fields {
                    collect_ir_lambda_shapes(field.shape, seen, out);
                }
            }
        }
        _ => {}
    }
}

struct IrShapeLowerer<'a> {
    decoder: &'a dyn IrDecoder,
    lambda_by_shape: HashMap<*const Shape, LambdaId>,
}

impl<'a> IrShapeLowerer<'a> {
    fn new(decoder: &'a dyn IrDecoder, lambda_by_shape: HashMap<*const Shape, LambdaId>) -> Self {
        Self {
            decoder,
            lambda_by_shape,
        }
    }

    fn lambda_for_shape(&self, shape: &'static Shape) -> LambdaId {
        *self
            .lambda_by_shape
            .get(&(shape as *const Shape))
            .unwrap_or_else(|| {
                panic!(
                    "missing lambda for composite shape in IR lowering: {}",
                    shape.type_identifier
                )
            })
    }

    fn emit_apply_with_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        shape: &'static Shape,
        offset: usize,
    ) {
        let lambda = self.lambda_for_shape(shape);
        if offset == 0 {
            let _ = rb.apply(lambda, &[], 0);
            return;
        }

        let saved_out = rb.save_out_ptr();
        let off = rb.const_val(offset as u64);
        let adjusted_out = rb.binop(crate::ir::IrOp::Add, saved_out, off);
        rb.set_out_ptr(adjusted_out);
        let _ = rb.apply(lambda, &[], 0);
        rb.set_out_ptr(saved_out);
    }

    fn lower_value(&self, rb: &mut RegionBuilder<'_>, shape: &'static Shape, offset: usize) {
        if is_unit(shape) {
            return;
        }

        if ir_shape_needs_lambda(shape) {
            self.emit_apply_with_offset(rb, shape, offset);
            return;
        }

        match shape.scalar_type() {
            Some(st) if is_string_like_scalar(st) => {
                self.decoder.lower_read_string(rb, offset, st);
            }
            Some(st) => {
                self.decoder.lower_read_scalar(rb, offset, st);
            }
            None => panic!("unsupported IR-lowered shape: {}", shape.type_identifier),
        }
    }

    fn lower_shape_body(
        &self,
        rb: &mut RegionBuilder<'_>,
        shape: &'static Shape,
        base_offset: usize,
    ) {
        if is_unit(shape) {
            return;
        }

        if shape.is_transparent() {
            let (fields, skipped) = lower_fields_for_ir(shape, base_offset);
            if !skipped.is_empty() {
                panic!(
                    "IR path does not support transparent wrappers with skipped/defaulted fields: {}",
                    shape.type_identifier
                );
            }
            if fields.len() != 1 {
                panic!(
                    "transparent wrapper must lower to exactly one field: {}",
                    shape.type_identifier
                );
            }
            self.lower_value(rb, fields[0].shape, fields[0].offset);
            return;
        }

        if let Some(opt_def) = get_option_def(shape) {
            let init_none_fn = opt_def.vtable.init_none as *const u8;
            let init_some_fn = opt_def.vtable.init_some as *const u8;
            let inner_layout = opt_def
                .t
                .layout
                .sized_layout()
                .expect("Option inner type must be Sized");
            let bytes = inner_layout.size().max(1);
            let slots = bytes.div_ceil(8);
            let scratch_slot = rb.alloc_slot();
            for _ in 1..slots {
                let _ = rb.alloc_slot();
            }
            self.decoder.lower_option(
                rb,
                base_offset,
                init_none_fn,
                init_some_fn,
                scratch_slot,
                &mut |inner_rb| {
                    self.lower_value(inner_rb, opt_def.t, 0);
                },
            );
            return;
        }

        if get_pointer_def(shape).is_some() {
            panic!(
                "IR path does not support pointer lowering yet: {}",
                shape.type_identifier
            );
        }

        if let Def::List(list_def) = &shape.def {
            let vec_offsets = crate::malum::discover_vec_offsets(list_def, shape);
            self.decoder
                .lower_vec(rb, base_offset, list_def.t, &vec_offsets, &mut |inner_rb| {
                    self.lower_value(inner_rb, list_def.t, 0);
                });
            return;
        }

        if let Def::Map(map_def) = &shape.def {
            let _ = (rb, map_def, base_offset);
            panic!(
                "IR path does not support Map lowering yet: {}",
                shape.type_identifier
            );
        }

        if let Def::Array(array_def) = &shape.def {
            let elem_layout = array_def
                .t
                .layout
                .sized_layout()
                .expect("array element must be Sized");
            let stride = elem_layout.size();
            for i in 0..array_def.n {
                self.lower_value(rb, array_def.t, base_offset + i * stride);
            }
            return;
        }

        match &shape.ty {
            Type::User(UserType::Struct(st)) => {
                let (fields, skipped) = lower_fields_for_ir(shape, base_offset);
                if !skipped.is_empty() {
                    panic!(
                        "IR path does not support skipped/defaulted fields yet: {}",
                        shape.type_identifier
                    );
                }
                let deny_unknown_fields = shape.has_deny_unknown_fields_attr();
                let is_positional = matches!(st.kind, StructKind::Tuple | StructKind::TupleStruct);
                if is_positional {
                    self.decoder
                        .lower_positional_fields(rb, &fields, &mut |inner_rb, field| {
                            self.lower_value(inner_rb, field.shape, field.offset);
                        });
                } else {
                    self.decoder.lower_struct_fields(
                        rb,
                        &fields,
                        deny_unknown_fields,
                        &mut |inner_rb, field| {
                            self.lower_value(inner_rb, field.shape, field.offset);
                        },
                    );
                }
            }
            Type::User(UserType::Enum(enum_type)) => {
                let disc_size = discriminant_size(enum_type.enum_repr);
                let disc_width = ir_width_from_disc_size(disc_size);
                let variants = lower_variants_for_ir(enum_type, base_offset);
                self.decoder
                    .lower_enum(rb, &variants, &mut |inner_rb, variant| {
                        let disc = inner_rb.const_val(variant.rust_discriminant as u64);
                        inner_rb.write_to_field(disc, base_offset as u32, disc_width);
                        for field in &variant.fields {
                            self.lower_value(inner_rb, field.shape, field.offset);
                        }
                    });
            }
            _ => match shape.scalar_type() {
                Some(st) if is_string_like_scalar(st) => {
                    self.decoder.lower_read_string(rb, base_offset, st);
                }
                Some(st) => {
                    self.decoder.lower_read_scalar(rb, base_offset, st);
                }
                None => panic!("unsupported IR-lowered shape: {}", shape.type_identifier),
            },
        }
    }
}

/// Compile a deserializer through RVSDG + linearization + backend adapter.
///
/// This path coexists with the legacy direct dynasm emission path for A/B benchmarking.
pub fn compile_decoder_via_ir<F: Decoder + IrDecoder>(
    shape: &'static Shape,
    decoder: &F,
) -> CompiledDecoder {
    compile_decoder_via_ir_dyn(shape, decoder, decoder)
}

/// Compile a deserializer through RVSDG + linearization + backend adapter
/// using separate trait-object views for legacy and IR decode traits.
pub fn compile_decoder_via_ir_dyn(
    shape: &'static Shape,
    decoder: &dyn Decoder,
    ir_decoder: &dyn IrDecoder,
) -> CompiledDecoder {
    let mut func = build_decoder_ir(shape, ir_decoder);
    let linear = crate::linearize::linearize(&mut func);
    compile_linear_ir_decoder(&linear, decoder.supports_trusted_utf8_input())
}

// r[impl ir.regalloc.regressions]
/// Build IR + linear form and run regalloc over it, returning total edit count.
pub fn regalloc_edit_count_via_ir(shape: &'static Shape, ir_decoder: &dyn IrDecoder) -> usize {
    let mut func = build_decoder_ir(shape, ir_decoder);
    crate::ir_passes::run_default_passes(&mut func);
    let linear = crate::linearize::linearize(&mut func);
    let alloc = crate::regalloc_engine::allocate_linear_ir(&linear)
        .unwrap_or_else(|err| panic!("regalloc2 allocation failed while counting edits: {err}"));
    alloc.functions.iter().map(|f| f.edits.len()).sum()
}

pub(crate) fn build_decoder_ir(
    shape: &'static Shape,
    ir_decoder: &dyn IrDecoder,
) -> crate::ir::IrFunc {
    let mut builder = crate::ir::IrBuilder::new(shape);
    let mut lambda_shapes = Vec::new();
    collect_ir_lambda_shapes(
        shape,
        &mut std::collections::HashSet::new(),
        &mut lambda_shapes,
    );

    let mut lambda_by_shape = HashMap::new();
    lambda_by_shape.insert(shape as *const Shape, LambdaId::new(0));
    for lambda_shape in lambda_shapes.iter().copied().skip(1) {
        let lambda = builder.create_lambda(lambda_shape);
        lambda_by_shape.insert(lambda_shape as *const Shape, lambda);
    }

    let ir_lowerer = IrShapeLowerer::new(ir_decoder, lambda_by_shape);

    {
        let mut rb = builder.root_region();
        ir_lowerer.lower_shape_body(&mut rb, shape, 0);
        rb.set_results(&[]);
    }

    for lambda_shape in lambda_shapes.iter().copied().skip(1) {
        let lambda_id = ir_lowerer.lambda_for_shape(lambda_shape);
        let mut rb = builder.lambda_region(lambda_id);
        ir_lowerer.lower_shape_body(&mut rb, lambda_shape, 0);
        rb.set_results(&[]);
    }

    builder.finish()
}

/// Compile a deserializer from already-linearized IR.
///
/// This is the first backend-adapter entrypoint used by the IR migration.
pub fn compile_linear_ir_decoder(
    ir: &crate::linearize::LinearIr,
    trusted_utf8_input: bool,
) -> CompiledDecoder {
    // r[impl ir.regalloc.ra-mir]
    // Build allocator-oriented CFG IR before machine emission.
    let ra_mir = crate::regalloc_mir::lower_linear_ir(ir);
    // r[impl ir.regalloc.engine]
    // Run regalloc2 over RA-MIR and thread allocation artifacts into emission.
    let regalloc_alloc = crate::regalloc_engine::allocate_program(&ra_mir)
        .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));

    let crate::ir_backend::LinearBackendResult { buf, entry } =
        crate::ir_backend::compile_linear_ir_with_alloc(ir, &regalloc_alloc);
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.ptr(entry)) };
    let root_name = ir
        .ops
        .iter()
        .find_map(|op| match op {
            crate::linearize::LinearOp::FuncStart {
                lambda_id, shape, ..
            } if lambda_id.index() == 0 => Some(format!("fad::decode::{}", shape.type_identifier)),
            _ => None,
        })
        .unwrap_or_else(|| "fad::decode::<ir-root>".to_string());
    let registration = crate::jit_debug::register_jit_code(
        buf.as_ptr(),
        buf.len(),
        &[crate::jit_debug::JitSymbolEntry {
            name: root_name,
            offset: entry.0,
            size: buf.len().saturating_sub(entry.0),
        }],
    );

    CompiledDecoder {
        buf,
        entry,
        func,
        trusted_utf8_input,
        _jit_registration: Some(registration),
    }
}

// =============================================================================
// Encoder compiler — serialization direction
// =============================================================================

/// A compiled encoder. Owns the executable buffer containing JIT'd machine code.
pub struct CompiledEncoder {
    buf: dynasmrt::ExecutableBuffer,
    entry: AssemblyOffset,
    func: unsafe extern "C" fn(*const u8, *mut crate::context::EncodeContext),
    _jit_registration: Option<crate::jit_debug::JitRegistration>,
}

impl CompiledEncoder {
    pub(crate) fn func(
        &self,
    ) -> unsafe extern "C" fn(*const u8, *mut crate::context::EncodeContext) {
        self.func
    }

    /// The raw executable code buffer.
    pub fn code(&self) -> &[u8] {
        &self.buf
    }

    /// Byte offset of the entry point within the code buffer.
    pub fn entry_offset(&self) -> usize {
        self.entry.0
    }
}

/// Collect fields for encode — produces lean FieldEncodeInfo (no defaults, no required_index).
fn collect_encode_fields(shape: &'static Shape) -> Vec<FieldEncodeInfo> {
    let mut out = Vec::new();
    collect_encode_fields_recursive(shape, 0, &mut out);
    out
}

fn collect_encode_fields_recursive(
    shape: &'static Shape,
    base_offset: usize,
    out: &mut Vec<FieldEncodeInfo>,
) {
    let st = match &shape.ty {
        Type::User(UserType::Struct(st)) => st,
        _ => panic!("unsupported shape for encode: {}", shape.type_identifier),
    };
    for f in st.fields {
        if f.is_flattened() {
            collect_encode_fields_recursive(f.shape(), base_offset + f.offset, out);
            continue;
        }
        // Skip fields marked skip_serializing
        if f.should_skip_serializing_unconditional() {
            continue;
        }
        out.push(FieldEncodeInfo {
            offset: base_offset + f.offset,
            shape: f.shape(),
            name: f.effective_name(),
        });
    }
}

/// Collect variants for encode — produces VariantEncodeInfo.
fn collect_encode_variants(enum_type: &'static facet::EnumType) -> Vec<VariantEncodeInfo> {
    enum_type
        .variants
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let kind = VariantKind::from_struct_type(&v.data);
            let mut fields = Vec::new();
            for f in v.data.fields {
                if f.is_flattened() {
                    collect_encode_fields_recursive(f.shape(), f.offset, &mut fields);
                } else if f.should_skip_serializing_unconditional() {
                    continue;
                } else {
                    fields.push(FieldEncodeInfo {
                        offset: f.offset,
                        shape: f.shape(),
                        name: f.effective_name(),
                    });
                }
            }
            VariantEncodeInfo {
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

/// Compiler state for emitting encode functions (one per shape).
struct EncoderCompiler<'fmt> {
    ectx: EmitCtx,
    encoder: &'fmt dyn Encoder,
    shapes: HashMap<*const Shape, ShapeEntry>,
    string_offsets: StringOffsets,
}

impl<'fmt> EncoderCompiler<'fmt> {
    fn new(extra_stack: u32, string_offsets: StringOffsets, encoder: &'fmt dyn Encoder) -> Self {
        EncoderCompiler {
            ectx: EmitCtx::new(extra_stack),
            encoder,
            shapes: HashMap::new(),
            string_offsets,
        }
    }

    /// Compile an encoder for `shape`. Returns the entry label.
    fn compile_shape(&mut self, shape: &'static Shape) -> DynamicLabel {
        let key = shape as *const Shape;

        if let Some(entry) = self.shapes.get(&key) {
            return entry.label;
        }

        let entry_label = self.ectx.new_label();
        self.shapes.insert(
            key,
            ShapeEntry {
                label: entry_label,
                offset: None,
            },
        );

        // Transparent wrappers — serialize as the inner type.
        if shape.is_transparent() {
            self.compile_transparent(shape, entry_label);
            return entry_label;
        }

        // Vec<T>
        if let Def::List(list_def) = &shape.def {
            self.compile_vec(shape, list_def, entry_label);
            return entry_label;
        }

        // Map<K,V>
        if let Def::Map(map_def) = &shape.def {
            self.compile_map(shape, map_def, entry_label);
            return entry_label;
        }

        // Smart pointers (Box<T>, Arc<T>, Rc<T>)
        if let Some(ptr_def) = get_pointer_def(shape) {
            self.compile_pointer(shape, ptr_def, entry_label);
            return entry_label;
        }

        match &shape.ty {
            Type::User(UserType::Struct(_)) => {
                self.compile_struct(shape, entry_label);
            }
            Type::User(UserType::Enum(enum_type)) => {
                self.compile_enum(shape, enum_type, entry_label);
            }
            _ => match shape.scalar_type() {
                Some(_) => self.compile_root_scalar(shape, entry_label),
                None => panic!("unsupported shape for encode: {}", shape.type_identifier),
            },
        }

        entry_label
    }

    fn compile_root_scalar(&mut self, shape: &'static Shape, entry_label: DynamicLabel) {
        let key = shape as *const Shape;
        let scalar_type = shape.scalar_type().unwrap_or_else(|| {
            panic!(
                "expected scalar root shape for encode: {}",
                shape.type_identifier
            )
        });

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_encode_func();

        if is_string_like_scalar(scalar_type) {
            self.encoder
                .emit_encode_string(&mut self.ectx, 0, scalar_type, &self.string_offsets);
        } else {
            self.encoder
                .emit_encode_scalar(&mut self.ectx, 0, scalar_type);
        }

        self.ectx.end_encode_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    fn compile_struct(&mut self, shape: &'static Shape, entry_label: DynamicLabel) {
        let key = shape as *const Shape;
        let fields = collect_encode_fields(shape);
        let inline = self.encoder.supports_inline_nested();

        // Pre-compile nested composite fields.
        let nested: Vec<Option<DynamicLabel>> = if inline {
            fields
                .iter()
                .map(|f| {
                    let target = unwrap_inner_or_self(f.shape);
                    if matches!(&target.ty, Type::User(UserType::Enum(_)))
                        || matches!(&target.def, Def::List(_) | Def::Map(_))
                    {
                        Some(self.compile_shape(target))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            fields
                .iter()
                .map(|f| {
                    let target = unwrap_inner_or_self(f.shape);
                    if is_composite(target) {
                        Some(self.compile_shape(target))
                    } else {
                        None
                    }
                })
                .collect()
        };

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_encode_func();

        let encoder = self.encoder;
        let string_offsets = &self.string_offsets;
        let ectx = &mut self.ectx;

        encoder.emit_encode_struct_fields(ectx, &fields, &mut |ectx, field| {
            emit_encode_field(ectx, encoder, field, &fields, &nested, string_offsets);
        });

        self.ectx.end_encode_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    fn compile_transparent(&mut self, shape: &'static Shape, entry_label: DynamicLabel) {
        let key = shape as *const Shape;

        let inner_shape = shape.inner.unwrap_or_else(|| {
            panic!(
                "transparent shape {} has no inner shape",
                shape.type_identifier
            )
        });

        let field_offset = match &shape.ty {
            Type::User(UserType::Struct(st)) => {
                assert!(
                    st.fields.len() == 1,
                    "transparent struct {} has {} fields, expected 1",
                    shape.type_identifier,
                    st.fields.len()
                );
                st.fields[0].offset
            }
            _ => 0,
        };

        let inner_label = if needs_precompilation(inner_shape) {
            Some(self.compile_shape(inner_shape))
        } else {
            None
        };

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_encode_func();

        if let Some(label) = inner_label {
            self.ectx
                .emit_enc_call_emitted_func(label, field_offset as u32);
        } else {
            match inner_shape.scalar_type() {
                Some(st) if is_string_like_scalar(st) => {
                    self.encoder.emit_encode_string(
                        &mut self.ectx,
                        field_offset,
                        st,
                        &self.string_offsets,
                    );
                }
                Some(st) => {
                    self.encoder
                        .emit_encode_scalar(&mut self.ectx, field_offset, st);
                }
                None => panic!(
                    "unsupported transparent inner type for encode: {}",
                    inner_shape.type_identifier,
                ),
            }
        }

        self.ectx.end_encode_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    fn compile_vec(
        &mut self,
        shape: &'static Shape,
        list_def: &'static ListDef,
        entry_label: DynamicLabel,
    ) {
        let key = shape as *const Shape;
        let vec_offsets = crate::malum::discover_vec_offsets(list_def, shape);
        let elem_shape = list_def.t;

        let elem_label = if needs_precompilation(elem_shape) {
            Some(self.compile_shape(elem_shape))
        } else {
            None
        };

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_encode_func();

        let encoder = self.encoder;
        let string_offsets = &self.string_offsets;

        encoder.emit_encode_vec(
            &mut self.ectx,
            0,
            elem_shape,
            elem_label,
            &vec_offsets,
            &mut |ectx| {
                emit_encode_elem(ectx, encoder, elem_shape, elem_label, string_offsets);
            },
        );

        self.ectx.end_encode_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    fn compile_map(
        &mut self,
        shape: &'static Shape,
        map_def: &'static MapDef,
        entry_label: DynamicLabel,
    ) {
        let key = shape as *const Shape;
        let k_shape = map_def.k;
        let v_shape = map_def.v;

        let k_label = if needs_precompilation(k_shape) {
            Some(self.compile_shape(k_shape))
        } else {
            None
        };
        let v_label = if needs_precompilation(v_shape) {
            Some(self.compile_shape(v_shape))
        } else {
            None
        };

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_encode_func();

        let encoder = self.encoder;
        let string_offsets = &self.string_offsets;

        encoder.emit_encode_map(
            &mut self.ectx,
            0,
            map_def,
            k_shape,
            v_shape,
            k_label,
            v_label,
            &mut |ectx| {
                emit_encode_elem(ectx, encoder, k_shape, k_label, string_offsets);
            },
            &mut |ectx| {
                emit_encode_elem(ectx, encoder, v_shape, v_label, string_offsets);
            },
        );

        self.ectx.end_encode_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    fn compile_pointer(
        &mut self,
        shape: &'static Shape,
        ptr_def: &'static PointerDef,
        entry_label: DynamicLabel,
    ) {
        let key = shape as *const Shape;

        let pointee = ptr_def
            .pointee
            .unwrap_or_else(|| panic!("pointer {} has no pointee", shape.type_identifier));

        // Pre-compile inner type if composite.
        let inner_label = if needs_precompilation(pointee) {
            Some(self.compile_shape(pointee))
        } else {
            None
        };

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_encode_func();

        // For encoding, the pointer field at [input + 0] contains a raw pointer
        // to the pointee. Load that pointer, then serialize the pointee through it.
        // The Encoder trait's emit_encode_* methods will handle dereferencing.
        //
        // For now, smart pointer encoding follows the same pattern as transparent:
        // call through to the inner type's encoder with the dereferenced pointer.
        // The actual dereference intrinsic will be added when we implement a concrete
        // encoder format.
        if let Some(label) = inner_label {
            self.ectx.emit_enc_call_emitted_func(label, 0);
        } else {
            match pointee.scalar_type() {
                Some(st) if is_string_like_scalar(st) => {
                    self.encoder
                        .emit_encode_string(&mut self.ectx, 0, st, &self.string_offsets);
                }
                Some(st) => {
                    self.encoder.emit_encode_scalar(&mut self.ectx, 0, st);
                }
                None => panic!(
                    "unsupported pointer inner type for encode: {}",
                    pointee.type_identifier,
                ),
            }
        }

        self.ectx.end_encode_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }

    fn compile_enum(
        &mut self,
        shape: &'static Shape,
        enum_type: &'static facet::EnumType,
        entry_label: DynamicLabel,
    ) {
        let key = shape as *const Shape;
        let variants = collect_encode_variants(enum_type);
        let inline = self.encoder.supports_inline_nested();

        // Pre-compile nested composite types in variant fields.
        let nested_labels: Vec<Vec<Option<DynamicLabel>>> = if inline {
            variants
                .iter()
                .map(|v| {
                    v.fields
                        .iter()
                        .map(|f| {
                            let target = unwrap_inner_or_self(f.shape);
                            if matches!(&target.ty, Type::User(UserType::Enum(_)))
                                || matches!(&target.def, Def::List(_) | Def::Map(_))
                            {
                                Some(self.compile_shape(target))
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .collect()
        } else {
            variants
                .iter()
                .map(|v| {
                    v.fields
                        .iter()
                        .map(|f| {
                            let target = unwrap_inner_or_self(f.shape);
                            if is_composite(target) {
                                Some(self.compile_shape(target))
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .collect()
        };

        self.ectx.bind_label(entry_label);
        let (entry_offset, error_exit) = self.ectx.begin_encode_func();

        let encoder = self.encoder;
        let string_offsets = &self.string_offsets;
        let ectx = &mut self.ectx;

        let tag_key = shape.get_tag_attr();
        let content_key = shape.get_content_attr();

        let mut emit_variant_body = |ectx: &mut EmitCtx, variant: &VariantEncodeInfo| {
            if variant.kind == VariantKind::Unit {
                return;
            }

            let nested = &nested_labels[variant.index];

            for (fi, field) in variant.fields.iter().enumerate() {
                if let Some(label) = nested[fi] {
                    ectx.emit_enc_call_emitted_func(label, field.offset as u32);
                } else {
                    match field.shape.scalar_type() {
                        Some(st) if is_string_like_scalar(st) => {
                            encoder.emit_encode_string(ectx, field.offset, st, string_offsets);
                        }
                        Some(st) => {
                            encoder.emit_encode_scalar(ectx, field.offset, st);
                        }
                        None if is_struct(field.shape) && encoder.supports_inline_nested() => {
                            emit_inline_encode_struct(
                                ectx,
                                encoder,
                                field.shape,
                                field.offset,
                                string_offsets,
                            );
                        }
                        None => panic!(
                            "unsupported variant field type for encode: {}",
                            field.shape.type_identifier,
                        ),
                    }
                }
            }
        };

        match (tag_key, content_key, shape.is_untagged()) {
            (None, None, false) => {
                encoder.emit_encode_enum(ectx, &variants, &mut emit_variant_body);
            }
            (Some(tk), Some(ck), false) => {
                encoder.emit_encode_enum_adjacent(ectx, &variants, tk, ck, &mut emit_variant_body);
            }
            (Some(tk), None, false) => {
                encoder.emit_encode_enum_internal(ectx, &variants, tk, &mut emit_variant_body);
            }
            (_, _, true) => {
                encoder.emit_encode_enum_untagged(ectx, &variants, &mut emit_variant_body);
            }
            (None, Some(_), _) => {
                panic!("content attribute without tag attribute is invalid");
            }
        }

        self.ectx.end_encode_func(error_exit);
        self.shapes.get_mut(&key).unwrap().offset = Some(entry_offset);
    }
}

/// Emit code for a single field during encode.
fn emit_encode_field(
    ectx: &mut EmitCtx,
    encoder: &dyn Encoder,
    field: &FieldEncodeInfo,
    all_fields: &[FieldEncodeInfo],
    nested: &[Option<DynamicLabel>],
    string_offsets: &StringOffsets,
) {
    let idx = all_fields
        .iter()
        .position(|f| f.offset == field.offset)
        .expect("field not found in all_fields");

    // Option<T>
    if let Some(opt_def) = get_option_def(field.shape) {
        let inner_shape = opt_def.t;

        encoder.emit_encode_option(ectx, field.offset, &mut |ectx| {
            // Emit inner T serialization. The encoder has already set up
            // the input pointer to point at the Some payload.
            if let Some(label) = nested[idx] {
                ectx.emit_enc_call_emitted_func(label, 0);
            } else if is_struct(inner_shape) && encoder.supports_inline_nested() {
                emit_inline_encode_struct(ectx, encoder, inner_shape, 0, string_offsets);
            } else {
                match inner_shape.scalar_type() {
                    Some(st) if is_string_like_scalar(st) => {
                        encoder.emit_encode_string(ectx, 0, st, string_offsets);
                    }
                    Some(st) => {
                        encoder.emit_encode_scalar(ectx, 0, st);
                    }
                    None => panic!(
                        "unsupported Option inner type for encode: {}",
                        inner_shape.type_identifier,
                    ),
                }
            }
        });
        return;
    }

    // Smart pointers (Box<T>, Arc<T>, Rc<T>)
    if let Some(_ptr_def) = get_pointer_def(field.shape) {
        // For now, pointer encoding will be handled by the format's intrinsics
        // when a concrete encoder is implemented. The compiler just calls through
        // to the inner type's label if pre-compiled.
        if let Some(label) = nested[idx] {
            ectx.emit_enc_call_emitted_func(label, field.offset as u32);
        } else {
            panic!(
                "smart pointer field encode not yet fully supported: {}",
                field.shape.type_identifier,
            );
        }
        return;
    }

    if let Some(label) = nested[idx] {
        ectx.emit_enc_call_emitted_func(label, field.offset as u32);
        return;
    }

    if is_struct(field.shape) && encoder.supports_inline_nested() {
        emit_inline_encode_struct(ectx, encoder, field.shape, field.offset, string_offsets);
        return;
    }

    match field.shape.scalar_type() {
        Some(st) if is_string_like_scalar(st) => {
            encoder.emit_encode_string(ectx, field.offset, st, string_offsets);
        }
        Some(st) => {
            encoder.emit_encode_scalar(ectx, field.offset, st);
        }
        None => panic!(
            "unsupported field type for encode: {} (scalar_type={:?})",
            field.shape.type_identifier,
            field.shape.scalar_type()
        ),
    }
}

/// Emit encode code for a single element (used by Vec/Map loops).
fn emit_encode_elem(
    ectx: &mut EmitCtx,
    encoder: &dyn Encoder,
    elem_shape: &'static Shape,
    elem_label: Option<DynamicLabel>,
    string_offsets: &StringOffsets,
) {
    if let Some(label) = elem_label {
        ectx.emit_enc_call_emitted_func(label, 0);
    } else if is_struct(elem_shape) && encoder.supports_inline_nested() {
        emit_inline_encode_struct(ectx, encoder, elem_shape, 0, string_offsets);
    } else {
        match elem_shape.scalar_type() {
            Some(st) if is_string_like_scalar(st) => {
                encoder.emit_encode_string(ectx, 0, st, string_offsets);
            }
            Some(st) => {
                encoder.emit_encode_scalar(ectx, 0, st);
            }
            None => panic!(
                "unsupported Vec/Map element type for encode: {}",
                elem_shape.type_identifier,
            ),
        }
    }
}

/// Inline a nested struct's fields into the parent encode function.
fn emit_inline_encode_struct(
    ectx: &mut EmitCtx,
    encoder: &dyn Encoder,
    shape: &'static Shape,
    base_offset: usize,
    string_offsets: &StringOffsets,
) {
    let inner_fields = collect_encode_fields(shape);

    let adjusted: Vec<FieldEncodeInfo> = inner_fields
        .into_iter()
        .map(|f| FieldEncodeInfo {
            offset: base_offset + f.offset,
            shape: f.shape,
            name: f.name,
        })
        .collect();

    let no_nested = vec![None; adjusted.len()];

    encoder.emit_encode_struct_fields(ectx, &adjusted, &mut |ectx, field| {
        emit_encode_field(ectx, encoder, field, &adjusted, &no_nested, string_offsets);
    });
}

/// Compile an encoder for the given shape and format.
pub fn compile_encoder(shape: &'static Shape, encoder: &dyn Encoder) -> CompiledEncoder {
    let format_extra = match &shape.ty {
        Type::User(UserType::Struct(_)) => {
            let fields = collect_encode_fields(shape);
            encoder.extra_stack_space(&fields)
        }
        Type::User(UserType::Enum(enum_type)) => {
            let mut max_extra = 0u32;
            for v in enum_type.variants {
                let fields: Vec<FieldEncodeInfo> = v
                    .data
                    .fields
                    .iter()
                    .filter(|f| !f.should_skip_serializing_unconditional())
                    .map(|f| FieldEncodeInfo {
                        offset: f.offset,
                        shape: f.shape(),
                        name: f.effective_name(),
                    })
                    .collect();
                max_extra = max_extra.max(encoder.extra_stack_space(&fields));
            }
            max_extra
        }
        _ => encoder.extra_stack_space(&[]),
    };

    let string_offsets = crate::malum::discover_string_offsets();

    let mut compiler = EncoderCompiler::new(format_extra, string_offsets, encoder);
    compiler.compile_shape(shape);

    let key = shape as *const Shape;
    let entry_offset = compiler.shapes[&key]
        .offset
        .expect("root shape was not fully compiled");

    let buf = compiler.ectx.finalize();
    let func: unsafe extern "C" fn(*const u8, *mut crate::context::EncodeContext) =
        unsafe { core::mem::transmute(buf.ptr(entry_offset)) };

    let registration = register_jit_symbols(&compiler.shapes, &buf, "encode");

    CompiledEncoder {
        buf,
        entry: entry_offset,
        func,
        _jit_registration: Some(registration),
    }
}

/// Collect symbols from the compiler's shape map and register them with the
/// GDB JIT interface so debugger backtraces show function names.
fn register_jit_symbols(
    shapes: &HashMap<*const Shape, ShapeEntry>,
    buf: &dynasmrt::ExecutableBuffer,
    direction: &str,
) -> crate::jit_debug::JitRegistration {
    let buf_len = buf.len();
    let buf_base = buf.as_ptr();

    // Collect (name, offset) pairs, sorted by offset.
    let mut entries: Vec<(String, usize)> = shapes
        .iter()
        .filter_map(|(&shape_ptr, entry)| {
            entry.offset.map(|off| {
                let shape = unsafe { &*shape_ptr };
                (
                    format!("kajit::{direction}::{}", shape.type_identifier),
                    off.0,
                )
            })
        })
        .collect();
    entries.sort_by_key(|&(_, off)| off);

    // Compute sizes from gaps between consecutive offsets.
    let symbols: Vec<crate::jit_debug::JitSymbolEntry> = entries
        .iter()
        .enumerate()
        .map(|(i, (name, offset))| {
            let next_offset = entries.get(i + 1).map_or(buf_len, |e| e.1);
            crate::jit_debug::JitSymbolEntry {
                name: name.clone(),
                offset: *offset,
                size: next_offset.saturating_sub(*offset),
            }
        })
        .collect();

    crate::jit_debug::register_jit_code(buf_base, buf_len, &symbols)
}
