use dynasmrt::DynamicLabel;
use facet::{MapDef, ScalarType, Type, UserType};

use crate::arch::{BASE_FRAME, EmitCtx};
use crate::context::ErrorCode;
use crate::format::{
    Decoder, Encoder, FieldEmitInfo, FieldEncodeInfo, FieldLowerInfo, IrDecoder, VariantEmitInfo,
    VariantKind,
};
use crate::intrinsics;
use crate::ir::{IntrinsicFn, RegionBuilder};
use crate::json_intrinsics;
use crate::malum::VecOffsets;
use crate::solver::{JsonValueType, LoweredSolver};

// r[impl deser.json.struct]
// r[impl deser.json.struct.unknown-keys]

// Stack layout (offsets from sp):
//   [sp+0..BASE_FRAME)                   callee-saved registers (+ shadow space on Windows)
//
// Format base slots (extra_stack = 48 for structs, 80 for vec/map):
//   [sp+BASE_FRAME+0..BASE_FRAME+8)      bitset (u64) — tracks which required fields
//   [sp+BASE_FRAME+8..BASE_FRAME+16)     key_ptr (*const u8) — borrowed pointer into input
//   [sp+BASE_FRAME+16..BASE_FRAME+24)    key_len (usize)
//   [sp+BASE_FRAME+24..BASE_FRAME+32)    comma_or_end result (u8 + padding)
//   [sp+BASE_FRAME+32..BASE_FRAME+40)    saved_cursor — saved input_ptr for solver/string scan
//   [sp+BASE_FRAME+40..BASE_FRAME+48)    candidates — solver bitmask / string scan temp
//
// Vec/map loop slots (must not overlap base slots — string scan writes to +32/+40):
//   [sp+BASE_FRAME+48..BASE_FRAME+56)    saved_out — original output pointer
//   [sp+BASE_FRAME+56..BASE_FRAME+64)    buf — heap buffer pointer
//   [sp+BASE_FRAME+64..BASE_FRAME+72)    len — current element count
//   [sp+BASE_FRAME+72..BASE_FRAME+80)    cap — current capacity
pub(crate) const BITSET_OFFSET: u32 = BASE_FRAME;
pub(crate) const KEY_PTR_OFFSET: u32 = BASE_FRAME + 8;
pub(crate) const KEY_LEN_OFFSET: u32 = BASE_FRAME + 16;
pub(crate) const RESULT_BYTE_OFFSET: u32 = BASE_FRAME + 24;
pub(crate) const SAVED_CURSOR_OFFSET: u32 = BASE_FRAME + 32;
pub(crate) const CANDIDATES_OFFSET: u32 = BASE_FRAME + 40;

fn ifn(f: *const u8) -> IntrinsicFn {
    IntrinsicFn(f as usize)
}

/// Emit an inline key read using the vectorized string scan.
///
/// After this, KEY_PTR_OFFSET and KEY_LEN_OFFSET on the stack contain the
/// key pointer and length. The cursor is past the closing `"`.
///
/// Fast path (no escapes): entirely JIT'd, no intrinsic calls.
/// Slow path (escapes): calls kajit_json_key_slow_from_jit.
fn emit_read_key_inline(ectx: &mut EmitCtx) {
    let found_quote = ectx.new_label();
    let found_escape = ectx.new_label();
    let unterminated = ectx.new_label();
    let done = ectx.new_label();

    // 1. Skip whitespace, expect and consume opening '"'
    ectx.emit_json_expect_quote_after_ws(json_intrinsics::kajit_json_skip_ws as *const u8);

    // 2. Save start position to KEY_PTR_OFFSET
    //    (for the fast path, this IS the key pointer — zero-copy)
    ectx.emit_save_cursor_to_stack(KEY_PTR_OFFSET);

    // 3. Vectorized scan for '"' or '\'
    ectx.emit_json_string_scan(found_quote, found_escape, unterminated);

    // 4. Fast path: found closing '"' — entirely JIT'd
    ectx.bind_label(found_quote);
    ectx.emit_compute_key_len_and_advance(KEY_PTR_OFFSET, KEY_LEN_OFFSET);
    ectx.emit_branch(done);

    // 5. Slow path: found '\' — call key unescape intrinsic
    ectx.bind_label(found_escape);
    ectx.emit_call_key_slow_from_jit(
        json_intrinsics::kajit_json_key_slow_from_jit as *const u8,
        KEY_PTR_OFFSET,
        KEY_PTR_OFFSET,
        KEY_LEN_OFFSET,
    );
    ectx.emit_branch(done);

    // 6. Unterminated string
    ectx.bind_label(unterminated);
    ectx.emit_set_error(ErrorCode::UnterminatedString);

    ectx.bind_label(done);
}

/// JSON wire format — key-value pairs, linear key dispatch.
pub struct KajitJson;

impl Decoder for KajitJson {
    fn extra_stack_space(&self, _fields: &[FieldEmitInfo]) -> u32 {
        48
    }

    fn supports_trusted_utf8_input(&self) -> bool {
        true
    }

    fn as_ir_decoder(&self) -> Option<&dyn IrDecoder> {
        Some(self)
    }

    // r[impl deser.deny-unknown-fields.json]
    fn emit_struct_fields(
        &self,
        ectx: &mut EmitCtx,
        fields: &[FieldEmitInfo],
        deny_unknown_fields: bool,
        emit_field: &mut dyn FnMut(&mut EmitCtx, &FieldEmitInfo),
    ) {
        let after_loop = ectx.new_label();
        let loop_top = ectx.new_label();
        let empty_object = ectx.new_label();

        // Zero the bitset
        ectx.emit_zero_stack_slot(BITSET_OFFSET);

        // expect '{'
        ectx.emit_expect_byte_after_ws(b'{', crate::context::ErrorCode::ExpectedObjectStart);

        // peek after whitespace — check for empty object
        ectx.emit_call_intrinsic_ctx_and_stack_out(
            json_intrinsics::kajit_json_peek_after_ws as *const u8,
            RESULT_BYTE_OFFSET,
        );
        // If peek == '}', branch to empty_object handler
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'}', empty_object);

        // === loop_top ===
        ectx.bind_label(loop_top);

        // Read key → (key_ptr, key_len) on stack
        emit_read_key_inline(ectx);

        // expect ':'
        ectx.emit_expect_byte_after_ws(b':', crate::context::ErrorCode::ExpectedColon);

        // Linear key dispatch chain
        let after_dispatch = ectx.new_label();
        let match_labels: Vec<_> = fields.iter().map(|_| ectx.new_label()).collect();
        let unknown_key = ectx.new_label();

        // Emit comparisons: for each field, compare key and branch if match
        for (i, field) in fields.iter().enumerate() {
            let name_bytes = field.name.as_bytes();
            ectx.emit_call_pure_4arg(
                json_intrinsics::kajit_json_key_equals as *const u8,
                KEY_PTR_OFFSET,
                KEY_LEN_OFFSET,
                name_bytes.as_ptr(),
                name_bytes.len() as u32,
            );
            ectx.emit_cbnz_x0(match_labels[i]);
        }

        // No match → unknown key: skip value
        ectx.emit_branch(unknown_key);

        // === match handlers ===
        for (i, field) in fields.iter().enumerate() {
            ectx.bind_label(match_labels[i]);
            emit_field(ectx, field);
            ectx.emit_set_bit_on_stack(BITSET_OFFSET, field.required_index as u32);
            ectx.emit_branch(after_dispatch);
        }

        // === unknown key handler ===
        ectx.bind_label(unknown_key);
        if deny_unknown_fields {
            ectx.emit_error(ErrorCode::UnknownField);
        } else {
            ectx.emit_call_intrinsic_ctx_only(json_intrinsics::kajit_json_skip_value as *const u8);
        }

        // === after_dispatch ===
        ectx.bind_label(after_dispatch);

        // comma_or_end → result byte: 0 = comma (continue), 1 = '}' (done)
        ectx.emit_inline_comma_or_end_object(RESULT_BYTE_OFFSET);
        // If result == 1 ('}'), jump to after_loop
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, 1, after_loop);
        // Otherwise, loop back
        ectx.emit_branch(loop_top);

        // === empty_object: advance past '}' and fall through to after_loop ===
        ectx.bind_label(empty_object);
        ectx.emit_advance_cursor_by(1);

        // === after_loop ===
        ectx.bind_label(after_loop);

        // r[impl deser.default]
        // For fields with defaults: if bit is NOT set, call the default function.
        // For fields without defaults: they are required — check with bitset mask.
        emit_default_or_required_check(ectx, fields);
    }

    // r[impl deser.json.tuple]
    fn emit_positional_fields(
        &self,
        ectx: &mut EmitCtx,
        fields: &[crate::format::FieldEmitInfo],
        emit_field: &mut dyn FnMut(&mut EmitCtx, &crate::format::FieldEmitInfo),
    ) {
        // Expect '['
        ectx.emit_call_intrinsic_ctx_only(
            json_intrinsics::kajit_json_expect_array_start as *const u8,
        );

        if fields.is_empty() {
            ectx.emit_call_intrinsic_ctx_only(
                json_intrinsics::kajit_json_expect_array_end as *const u8,
            );
            return;
        }

        for (i, field) in fields.iter().enumerate() {
            emit_field(ectx, field);
            if i + 1 < fields.len() {
                ectx.emit_call_intrinsic_ctx_only(
                    json_intrinsics::kajit_json_expect_comma as *const u8,
                );
            } else {
                ectx.emit_call_intrinsic_ctx_only(
                    json_intrinsics::kajit_json_expect_array_end as *const u8,
                );
            }
        }
    }

    // r[impl deser.json.option]
    fn emit_option(
        &self,
        ectx: &mut EmitCtx,
        offset: usize,
        init_none_fn: *const u8,
        init_some_fn: *const u8,
        scratch_offset: u32,
        emit_inner: &mut dyn FnMut(&mut EmitCtx),
    ) {
        // JSON Option: null = None, value = Some(T)
        let none_label = ectx.new_label();
        let done_label = ectx.new_label();

        // Peek at first non-whitespace byte
        ectx.emit_call_intrinsic_ctx_and_stack_out(
            json_intrinsics::kajit_json_peek_after_ws as *const u8,
            RESULT_BYTE_OFFSET,
        );

        // If peek == 'n', it's null → None
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'n', none_label);

        // === Some path ===
        // Redirect out to scratch area and deserialize inner T
        ectx.emit_redirect_out_to_stack(scratch_offset);
        emit_inner(ectx);
        ectx.emit_restore_out(scratch_offset);

        // Call init_some(init_some_fn, out + offset, scratch)
        ectx.emit_call_option_init_some(
            crate::intrinsics::kajit_option_init_some as *const u8,
            init_some_fn,
            offset as u32,
            scratch_offset,
        );
        ectx.emit_branch(done_label);

        // === None path ===
        ectx.bind_label(none_label);
        // Consume the "null" literal
        ectx.emit_call_intrinsic_ctx_only(json_intrinsics::kajit_json_skip_value as *const u8);
        ectx.emit_call_option_init_none(
            crate::intrinsics::kajit_option_init_none as *const u8,
            init_none_fn,
            offset as u32,
        );

        ectx.bind_label(done_label);
    }

    fn vec_extra_stack_space(&self) -> u32 {
        // JSON Vec needs 48 bytes of base format slots (bitset, key_ptr, key_len,
        // result_byte, saved_cursor, candidates) + saved_out + buf + len + cap = 32 bytes.
        // Total: 80 bytes
        80
    }

    fn map_extra_stack_space(&self) -> u32 {
        // Same layout as vec: 48 bytes base format + 32 bytes (saved_out + buf + len + cap).
        80
    }

    // r[impl deser.json.seq]
    // r[impl seq.malum.json]
    #[allow(clippy::too_many_arguments)]
    fn emit_vec(
        &self,
        ectx: &mut EmitCtx,
        offset: usize,
        elem_shape: &'static facet::Shape,
        _elem_label: Option<DynamicLabel>,
        vec_offsets: &VecOffsets,
        _option_scratch_offset: u32,
        emit_elem: &mut dyn FnMut(&mut EmitCtx),
    ) {
        let elem_layout = elem_shape
            .layout
            .sized_layout()
            .expect("Vec element must be Sized");
        let elem_size = elem_layout.size() as u32;
        let elem_align = elem_layout.align() as u32;

        // Stack slot offsets — after the 48 bytes of base format slots
        let saved_out_slot = BASE_FRAME + 48;
        let buf_slot = BASE_FRAME + 56;
        let len_slot = BASE_FRAME + 64;
        let cap_slot = BASE_FRAME + 72;

        let empty_label = ectx.new_label();
        let done_label = ectx.new_label();
        let loop_label = ectx.new_label();
        let grow_label = ectx.new_label();
        let after_grow = ectx.new_label();
        let write_vec_label = ectx.new_label();
        let error_cleanup = ectx.new_label();

        // Expect '['
        ectx.emit_call_intrinsic_ctx_only(
            json_intrinsics::kajit_json_expect_array_start as *const u8,
        );

        // Peek: check for empty array ']'
        ectx.emit_call_intrinsic_ctx_and_stack_out(
            json_intrinsics::kajit_json_peek_after_ws as *const u8,
            RESULT_BYTE_OFFSET,
        );
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b']', empty_label);

        // Initial allocation: cap=4
        let initial_cap: u32 = 4;
        ectx.emit_call_json_vec_initial_alloc(
            intrinsics::kajit_vec_alloc as *const u8,
            initial_cap,
            elem_size,
            elem_align,
        );

        // Save loop state: saved_out, buf (rax/x0), len=0, cap=initial_cap
        ectx.emit_json_vec_loop_init(saved_out_slot, buf_slot, len_slot, cap_slot, initial_cap);

        // === Loop ===
        ectx.bind_label(loop_label);

        // Check if len == cap → need to grow
        ectx.emit_cmp_stack_slots_branch_eq(len_slot, cap_slot, grow_label);
        ectx.bind_label(after_grow);

        // Compute slot = buf + len * elem_size, redirect out
        ectx.emit_vec_loop_slot(buf_slot, len_slot, elem_size);

        // Deserialize one element
        emit_elem(ectx);

        // Check error from element deserialization
        ectx.emit_check_error_branch(error_cleanup);

        // len += 1
        ectx.emit_inc_stack_slot(len_slot);

        // comma_or_end_array (inlined)
        ectx.emit_inline_comma_or_end_array(RESULT_BYTE_OFFSET);
        // If result == 1 (']'), we're done
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, 1, write_vec_label);
        // Otherwise comma → loop back
        ectx.emit_branch(loop_label);

        // === Growth path ===
        ectx.bind_label(grow_label);
        ectx.emit_call_vec_grow(
            intrinsics::kajit_vec_grow as *const u8,
            buf_slot,
            len_slot,
            cap_slot,
            elem_size,
            elem_align,
        );
        ectx.emit_branch(after_grow);

        // === Write Vec to output ===
        ectx.bind_label(write_vec_label);
        ectx.emit_vec_store(
            offset as u32,
            saved_out_slot,
            buf_slot,
            len_slot,
            cap_slot,
            vec_offsets,
        );
        ectx.emit_branch(done_label);

        // === Empty path ===
        ectx.bind_label(empty_label);
        // Consume the ']'
        ectx.emit_advance_cursor_by(1);
        ectx.emit_vec_store_empty_with_align(offset as u32, elem_align, vec_offsets);
        ectx.emit_branch(done_label);

        // === Error cleanup ===
        ectx.bind_label(error_cleanup);
        ectx.emit_vec_error_cleanup(
            intrinsics::kajit_vec_free as *const u8,
            saved_out_slot,
            buf_slot,
            cap_slot,
            elem_size,
            elem_align,
        );

        ectx.bind_label(done_label);
    }

    // r[impl deser.json.map]
    #[allow(clippy::too_many_arguments)]
    fn emit_map(
        &self,
        ectx: &mut EmitCtx,
        _offset: usize,
        map_def: &'static MapDef,
        k_shape: &'static facet::Shape,
        _v_shape: &'static facet::Shape,
        _k_label: Option<DynamicLabel>,
        _v_label: Option<DynamicLabel>,
        _option_scratch_offset: u32,
        emit_key: &mut dyn FnMut(&mut EmitCtx),
        emit_value: &mut dyn FnMut(&mut EmitCtx),
    ) {
        // JSON map keys must be strings — JSON object keys are always quoted strings.
        assert!(
            matches!(
                k_shape.scalar_type(),
                Some(ScalarType::String | ScalarType::Str | ScalarType::CowStr)
            ),
            "JSON map deserialization only supports string-like keys, got: {}",
            k_shape.type_identifier,
        );

        let from_pair_slice_fn = map_def
            .vtable
            .from_pair_slice
            .expect("MapVTable must have from_pair_slice for JIT deserialization")
            as *const u8;

        let pair_stride = map_def.vtable.pair_stride as u32;
        let value_offset = map_def.vtable.value_offset_in_pair as u32;

        let k_align = map_def
            .k
            .layout
            .sized_layout()
            .expect("Map key must be Sized")
            .align() as u32;
        let v_align = map_def
            .v
            .layout
            .sized_layout()
            .expect("Map value must be Sized")
            .align() as u32;
        let pair_align = k_align.max(v_align);

        // Stack slot offsets — after the 48 bytes of base format slots
        let saved_out_slot = BASE_FRAME + 48;
        let buf_slot = BASE_FRAME + 56;
        let len_slot = BASE_FRAME + 64;
        let cap_slot = BASE_FRAME + 72;

        let empty_label = ectx.new_label();
        let done_label = ectx.new_label();
        let loop_label = ectx.new_label();
        let grow_label = ectx.new_label();
        let after_grow = ectx.new_label();
        let write_map_label = ectx.new_label();
        let error_cleanup = ectx.new_label();

        // Expect '{'
        ectx.emit_expect_byte_after_ws(b'{', crate::context::ErrorCode::ExpectedObjectStart);

        // Peek: check for empty object '}'
        ectx.emit_call_intrinsic_ctx_and_stack_out(
            json_intrinsics::kajit_json_peek_after_ws as *const u8,
            RESULT_BYTE_OFFSET,
        );
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'}', empty_label);

        // Initial allocation: cap=4
        let initial_cap: u32 = 4;
        ectx.emit_call_json_vec_initial_alloc(
            intrinsics::kajit_vec_alloc as *const u8,
            initial_cap,
            pair_stride,
            pair_align,
        );

        // Save loop state: saved_out, buf, len=0, cap=initial_cap
        ectx.emit_json_vec_loop_init(saved_out_slot, buf_slot, len_slot, cap_slot, initial_cap);

        // === Loop ===
        ectx.bind_label(loop_label);

        // Check if len == cap → grow
        ectx.emit_cmp_stack_slots_branch_eq(len_slot, cap_slot, grow_label);
        ectx.bind_label(after_grow);

        // Compute pair slot: out = buf + len * pair_stride (points to key at offset 0)
        ectx.emit_vec_loop_slot(buf_slot, len_slot, pair_stride);

        // Emit key deserialization at offset 0 (JSON string read)
        emit_key(ectx);

        // Expect ':' separating key from value
        ectx.emit_expect_byte_after_ws(b':', crate::context::ErrorCode::ExpectedColon);

        // Advance out to value position: out = pair_base + value_offset
        ectx.emit_advance_out_by(value_offset);

        // Emit value deserialization at offset 0
        emit_value(ectx);

        // len += 1
        ectx.emit_inc_stack_slot(len_slot);

        // comma_or_end_object: 0=comma (more entries), 1='}' (done)
        ectx.emit_inline_comma_or_end_object(RESULT_BYTE_OFFSET);
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, 1, write_map_label);
        ectx.emit_branch(loop_label);

        // === Growth path ===
        ectx.bind_label(grow_label);
        ectx.emit_call_vec_grow(
            intrinsics::kajit_vec_grow as *const u8,
            buf_slot,
            len_slot,
            cap_slot,
            pair_stride,
            pair_align,
        );
        ectx.emit_branch(after_grow);

        // === Build map from pairs buffer ===
        ectx.bind_label(write_map_label);
        ectx.emit_call_map_from_pairs(from_pair_slice_fn, saved_out_slot, buf_slot, len_slot);
        // Free the pairs buffer (using cap, which may be > len after growth)
        ectx.emit_call_pairs_free(
            intrinsics::kajit_vec_free as *const u8,
            buf_slot,
            cap_slot,
            pair_stride,
            pair_align,
        );
        ectx.emit_branch(done_label);

        // === Empty path: consume '}', build empty map ===
        ectx.bind_label(empty_label);
        ectx.emit_advance_cursor_by(1); // consume '}'
        ectx.emit_call_map_from_pairs_empty(from_pair_slice_fn);
        ectx.emit_branch(done_label);

        // === Error cleanup: free pairs buffer, branch to error exit ===
        ectx.bind_label(error_cleanup);
        ectx.emit_vec_error_cleanup(
            intrinsics::kajit_vec_free as *const u8,
            saved_out_slot,
            buf_slot,
            cap_slot,
            pair_stride,
            pair_align,
        );

        ectx.bind_label(done_label);
    }

    // r[impl deser.json.scalar.integer]
    // r[impl deser.json.scalar.float]
    // r[impl deser.json.scalar.bool]
    fn emit_read_scalar(&self, ectx: &mut EmitCtx, offset: usize, scalar_type: ScalarType) {
        // Unit type: skip the JSON value (always null), write nothing (ZST).
        if scalar_type == ScalarType::Unit {
            ectx.emit_call_intrinsic_ctx_only(json_intrinsics::kajit_json_skip_value as *const u8);
            return;
        }
        if scalar_type == ScalarType::F64 {
            ectx.emit_jit_f64_parse(offset as u32);
            return;
        }
        let fn_ptr: *const u8 = match scalar_type {
            ScalarType::U8 => json_intrinsics::kajit_json_read_u8 as _,
            ScalarType::U16 => json_intrinsics::kajit_json_read_u16 as _,
            ScalarType::U32 => json_intrinsics::kajit_json_read_u32 as _,
            ScalarType::U64 => json_intrinsics::kajit_json_read_u64 as _,
            ScalarType::U128 => json_intrinsics::kajit_json_read_u128 as _,
            ScalarType::USize => json_intrinsics::kajit_json_read_usize as _,
            ScalarType::I8 => json_intrinsics::kajit_json_read_i8 as _,
            ScalarType::I16 => json_intrinsics::kajit_json_read_i16 as _,
            ScalarType::I32 => json_intrinsics::kajit_json_read_i32 as _,
            ScalarType::I64 => json_intrinsics::kajit_json_read_i64 as _,
            ScalarType::I128 => json_intrinsics::kajit_json_read_i128 as _,
            ScalarType::ISize => json_intrinsics::kajit_json_read_isize as _,
            ScalarType::F32 => json_intrinsics::kajit_json_read_f32 as _,
            ScalarType::F64 => unreachable!(),
            ScalarType::Bool => json_intrinsics::kajit_json_read_bool as _,
            ScalarType::Char => json_intrinsics::kajit_json_read_char as _,
            _ => panic!("unsupported JSON scalar: {:?}", scalar_type),
        };
        ectx.emit_call_intrinsic(fn_ptr, offset as u32);
    }

    fn emit_read_string(
        &self,
        ectx: &mut EmitCtx,
        offset: usize,
        scalar_type: ScalarType,
        string_offsets: &crate::malum::StringOffsets,
    ) {
        let found_quote = ectx.new_label();
        let found_escape = ectx.new_label();
        let unterminated = ectx.new_label();
        let done = ectx.new_label();

        // 1. Skip whitespace, expect and consume opening '"'
        ectx.emit_json_expect_quote_after_ws(json_intrinsics::kajit_json_skip_ws as *const u8);

        // 2. Save start position to stack
        ectx.emit_save_cursor_to_stack(SAVED_CURSOR_OFFSET);

        // 3. Vectorized scan for '"' or '\'
        ectx.emit_json_string_scan(found_quote, found_escape, unterminated);

        // 4. Fast path: found closing '"' (no escapes)
        ectx.bind_label(found_quote);
        match scalar_type {
            ScalarType::String => {
                // Malum path: validate + alloc + copy, then write ptr/len/cap directly
                ectx.emit_call_validate_alloc_copy_from_scan(
                    intrinsics::kajit_string_validate_alloc_copy as *const u8,
                    SAVED_CURSOR_OFFSET,
                    CANDIDATES_OFFSET, // reuse as temp for len
                );
                ectx.emit_write_malum_string_and_advance(
                    offset as u32,
                    string_offsets,
                    CANDIDATES_OFFSET,
                );
            }
            ScalarType::Str => {
                ectx.emit_call_string_finish(
                    json_intrinsics::kajit_json_finish_str_fast as *const u8,
                    offset as u32,
                    SAVED_CURSOR_OFFSET,
                );
            }
            ScalarType::CowStr => {
                ectx.emit_call_string_finish(
                    json_intrinsics::kajit_json_finish_cow_str_fast as *const u8,
                    offset as u32,
                    SAVED_CURSOR_OFFSET,
                );
            }
            _ => panic!("unsupported JSON string scalar: {:?}", scalar_type),
        }
        ectx.emit_branch(done);

        // 5. Slow path: found '\' (escapes)
        ectx.bind_label(found_escape);
        match scalar_type {
            ScalarType::String => {
                ectx.emit_call_string_escape(
                    json_intrinsics::kajit_json_string_with_escapes as *const u8,
                    offset as u32,
                    SAVED_CURSOR_OFFSET,
                );
            }
            ScalarType::Str => {
                // &str cannot represent escaped strings — error
                ectx.emit_set_error(ErrorCode::InvalidEscapeSequence);
            }
            ScalarType::CowStr => {
                ectx.emit_call_string_escape(
                    json_intrinsics::kajit_json_cow_str_with_escapes as *const u8,
                    offset as u32,
                    SAVED_CURSOR_OFFSET,
                );
            }
            _ => panic!("unsupported JSON string scalar: {:?}", scalar_type),
        }
        ectx.emit_branch(done);

        // 6. Unterminated string
        ectx.bind_label(unterminated);
        ectx.emit_set_error(ErrorCode::UnterminatedString);

        ectx.bind_label(done);
    }

    // r[impl deser.json.enum.external]
    // r[impl deser.json.enum.external.unit-as-string]
    // r[impl deser.json.enum.external.struct-variant]
    // r[impl deser.json.enum.external.tuple-variant]
    fn emit_enum(
        &self,
        ectx: &mut EmitCtx,
        variants: &[VariantEmitInfo],
        emit_variant_body: &mut dyn FnMut(&mut EmitCtx, &VariantEmitInfo),
    ) {
        let done_label = ectx.new_label();
        let object_path = ectx.new_label();

        // Peek at first non-whitespace byte to determine path.
        ectx.emit_call_intrinsic_ctx_and_stack_out(
            json_intrinsics::kajit_json_peek_after_ws as *const u8,
            RESULT_BYTE_OFFSET,
        );

        // If peek == '"', it's a bare string (unit variant).
        // Otherwise, expect an object.
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'{', object_path);

        // ══════════════════════════════════════════════════════════════════
        // Bare string path: read the quoted string, match unit variants.
        // ══════════════════════════════════════════════════════════════════
        emit_read_key_inline(ectx);

        // Compare against unit variant names.
        let unit_labels: Vec<_> = variants
            .iter()
            .filter(|v| v.kind == VariantKind::Unit)
            .map(|v| (v, ectx.new_label()))
            .collect();

        for (variant, label) in &unit_labels {
            let name_bytes = variant.name.as_bytes();
            ectx.emit_call_pure_4arg(
                json_intrinsics::kajit_json_key_equals as *const u8,
                KEY_PTR_OFFSET,
                KEY_LEN_OFFSET,
                name_bytes.as_ptr(),
                name_bytes.len() as u32,
            );
            ectx.emit_cbnz_x0(*label);
        }

        // No match → unknown variant.
        ectx.emit_unknown_variant_error();

        // Unit variant handlers.
        for (variant, label) in &unit_labels {
            ectx.bind_label(*label);
            emit_variant_body(ectx, variant);
            ectx.emit_branch(done_label);
        }

        // ══════════════════════════════════════════════════════════════════
        // Object path: { "VariantName": value }
        // ══════════════════════════════════════════════════════════════════
        ectx.bind_label(object_path);

        // Consume '{'
        ectx.emit_expect_byte_after_ws(b'{', crate::context::ErrorCode::ExpectedObjectStart);

        // Read the variant name key.
        emit_read_key_inline(ectx);

        // Consume ':'
        ectx.emit_expect_byte_after_ws(b':', crate::context::ErrorCode::ExpectedColon);

        // Compare against all variant names.
        let variant_labels: Vec<_> = variants.iter().map(|_| ectx.new_label()).collect();

        for (i, variant) in variants.iter().enumerate() {
            let name_bytes = variant.name.as_bytes();
            ectx.emit_call_pure_4arg(
                json_intrinsics::kajit_json_key_equals as *const u8,
                KEY_PTR_OFFSET,
                KEY_LEN_OFFSET,
                name_bytes.as_ptr(),
                name_bytes.len() as u32,
            );
            ectx.emit_cbnz_x0(variant_labels[i]);
        }

        // No match → unknown variant.
        ectx.emit_unknown_variant_error();

        // Variant handlers.
        let expect_end_label = ectx.new_label();
        for (i, variant) in variants.iter().enumerate() {
            ectx.bind_label(variant_labels[i]);
            match variant.kind {
                VariantKind::Unit => {
                    // Unit variant inside object: { "Cat": null } — skip the value.
                    emit_variant_body(ectx, variant);
                    ectx.emit_call_intrinsic_ctx_only(
                        json_intrinsics::kajit_json_skip_value as *const u8,
                    );
                }
                VariantKind::Struct | VariantKind::Tuple => {
                    // Struct variant: { "Dog": { "name": "Rex", ... } }
                    // Tuple variant (newtype): { "Parrot": "Polly" }
                    // emit_variant_body handles both — for struct it calls
                    // emit_struct_fields, for tuple it emits the single field.
                    emit_variant_body(ectx, variant);
                }
            }
            ectx.emit_branch(expect_end_label);
        }

        // After variant body: expect closing '}'
        ectx.bind_label(expect_end_label);
        ectx.emit_call_intrinsic_ctx_only(
            json_intrinsics::kajit_json_expect_object_end as *const u8,
        );

        ectx.bind_label(done_label);
    }

    // r[impl deser.json.enum.adjacent]
    // r[impl deser.json.enum.adjacent.key-order]
    // r[impl deser.json.enum.adjacent.unit-variant]
    // r[impl deser.json.enum.adjacent.tuple-variant]
    fn emit_enum_adjacent(
        &self,
        ectx: &mut EmitCtx,
        variants: &[VariantEmitInfo],
        tag_key: &'static str,
        content_key: &'static str,
        emit_variant_body: &mut dyn FnMut(&mut EmitCtx, &VariantEmitInfo),
    ) {
        let done_label = ectx.new_label();

        // expect '{'
        ectx.emit_expect_byte_after_ws(b'{', crate::context::ErrorCode::ExpectedObjectStart);

        // Read first key
        emit_read_key_inline(ectx);

        // Verify it equals tag_key, error if not
        let tag_key_bytes = tag_key.as_bytes();
        ectx.emit_call_pure_4arg(
            json_intrinsics::kajit_json_key_equals as *const u8,
            KEY_PTR_OFFSET,
            KEY_LEN_OFFSET,
            tag_key_bytes.as_ptr(),
            tag_key_bytes.len() as u32,
        );
        let tag_ok = ectx.new_label();
        ectx.emit_cbnz_x0(tag_ok);
        ectx.emit_call_intrinsic_ctx_only(
            json_intrinsics::kajit_json_error_expected_tag_key as *const u8,
        );
        ectx.bind_label(tag_ok);

        // expect ':'
        ectx.emit_expect_byte_after_ws(b':', crate::context::ErrorCode::ExpectedColon);

        // Read variant name string
        emit_read_key_inline(ectx);

        // Variant name dispatch (linear compare chain)
        let variant_labels: Vec<_> = variants.iter().map(|_| ectx.new_label()).collect();
        for (i, variant) in variants.iter().enumerate() {
            let name_bytes = variant.name.as_bytes();
            ectx.emit_call_pure_4arg(
                json_intrinsics::kajit_json_key_equals as *const u8,
                KEY_PTR_OFFSET,
                KEY_LEN_OFFSET,
                name_bytes.as_ptr(),
                name_bytes.len() as u32,
            );
            ectx.emit_cbnz_x0(variant_labels[i]);
        }
        ectx.emit_unknown_variant_error();

        // Variant handlers
        for (i, variant) in variants.iter().enumerate() {
            ectx.bind_label(variant_labels[i]);
            match variant.kind {
                VariantKind::Unit => {
                    // Unit variant: discriminant only, content key is optional.
                    emit_variant_body(ectx, variant);

                    // comma_or_end: '}' means done, ',' means content key follows
                    ectx.emit_call_intrinsic_ctx_and_stack_out(
                        json_intrinsics::kajit_json_comma_or_end_object as *const u8,
                        RESULT_BYTE_OFFSET,
                    );
                    let unit_done = ectx.new_label();
                    ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, 1, unit_done);

                    // Had comma → read content_key, ':', skip value (typically null), expect '}'
                    ectx.emit_call_intrinsic_ctx_and_two_stack_outs(
                        json_intrinsics::kajit_json_read_key as *const u8,
                        KEY_PTR_OFFSET,
                        KEY_LEN_OFFSET,
                    );
                    ectx.emit_expect_byte_after_ws(b':', crate::context::ErrorCode::ExpectedColon);
                    ectx.emit_call_intrinsic_ctx_only(
                        json_intrinsics::kajit_json_skip_value as *const u8,
                    );
                    ectx.emit_call_intrinsic_ctx_only(
                        json_intrinsics::kajit_json_expect_object_end as *const u8,
                    );

                    ectx.bind_label(unit_done);
                }
                VariantKind::Struct | VariantKind::Tuple => {
                    // Non-unit: expect comma, content key, colon, then variant body, then '}'
                    // comma_or_end → expect ','
                    ectx.emit_call_intrinsic_ctx_and_stack_out(
                        json_intrinsics::kajit_json_comma_or_end_object as *const u8,
                        RESULT_BYTE_OFFSET,
                    );
                    // (If we got '}' here, the next read_key will error — that's fine)

                    // Read content key
                    ectx.emit_call_intrinsic_ctx_and_two_stack_outs(
                        json_intrinsics::kajit_json_read_key as *const u8,
                        KEY_PTR_OFFSET,
                        KEY_LEN_OFFSET,
                    );

                    // Verify it equals content_key (optional check for better errors)
                    let ck_bytes = content_key.as_bytes();
                    ectx.emit_call_pure_4arg(
                        json_intrinsics::kajit_json_key_equals as *const u8,
                        KEY_PTR_OFFSET,
                        KEY_LEN_OFFSET,
                        ck_bytes.as_ptr(),
                        ck_bytes.len() as u32,
                    );
                    // We don't error on mismatch for now — just proceed.
                    // TODO: add ExpectedContentKey error for stricter validation.

                    // expect ':'
                    ectx.emit_expect_byte_after_ws(b':', crate::context::ErrorCode::ExpectedColon);

                    // Dispatch variant body (writes discriminant + deserializes struct/tuple)
                    emit_variant_body(ectx, variant);

                    // expect '}'
                    ectx.emit_call_intrinsic_ctx_only(
                        json_intrinsics::kajit_json_expect_object_end as *const u8,
                    );
                }
            }
            ectx.emit_branch(done_label);
        }

        ectx.bind_label(done_label);
    }

    // r[impl deser.json.enum.internal]
    // r[impl deser.json.enum.internal.struct-only]
    // r[impl deser.json.enum.internal.unit-variant]
    fn emit_enum_internal(
        &self,
        ectx: &mut EmitCtx,
        variants: &[VariantEmitInfo],
        tag_key: &'static str,
        emit_variant_discriminant: &mut dyn FnMut(&mut EmitCtx, &VariantEmitInfo),
        emit_variant_field: &mut dyn FnMut(&mut EmitCtx, &VariantEmitInfo, &FieldEmitInfo),
    ) {
        let done_label = ectx.new_label();

        // expect '{'
        ectx.emit_expect_byte_after_ws(b'{', crate::context::ErrorCode::ExpectedObjectStart);

        // Read first key
        emit_read_key_inline(ectx);

        // Verify it equals tag_key
        let tag_key_bytes = tag_key.as_bytes();
        ectx.emit_call_pure_4arg(
            json_intrinsics::kajit_json_key_equals as *const u8,
            KEY_PTR_OFFSET,
            KEY_LEN_OFFSET,
            tag_key_bytes.as_ptr(),
            tag_key_bytes.len() as u32,
        );
        let tag_ok = ectx.new_label();
        ectx.emit_cbnz_x0(tag_ok);
        ectx.emit_call_intrinsic_ctx_only(
            json_intrinsics::kajit_json_error_expected_tag_key as *const u8,
        );
        ectx.bind_label(tag_ok);

        // expect ':'
        ectx.emit_expect_byte_after_ws(b':', crate::context::ErrorCode::ExpectedColon);

        // Read variant name string
        emit_read_key_inline(ectx);

        // Variant name dispatch
        let variant_labels: Vec<_> = variants.iter().map(|_| ectx.new_label()).collect();
        for (i, variant) in variants.iter().enumerate() {
            let name_bytes = variant.name.as_bytes();
            ectx.emit_call_pure_4arg(
                json_intrinsics::kajit_json_key_equals as *const u8,
                KEY_PTR_OFFSET,
                KEY_LEN_OFFSET,
                name_bytes.as_ptr(),
                name_bytes.len() as u32,
            );
            ectx.emit_cbnz_x0(variant_labels[i]);
        }
        ectx.emit_unknown_variant_error();

        // Variant handlers
        for (i, variant) in variants.iter().enumerate() {
            ectx.bind_label(variant_labels[i]);
            match variant.kind {
                VariantKind::Unit => {
                    emit_variant_discriminant(ectx, variant);
                    // Expect closing '}' (possibly after comma + unknown keys)
                    ectx.emit_call_intrinsic_ctx_only(
                        json_intrinsics::kajit_json_expect_object_end as *const u8,
                    );
                }
                VariantKind::Struct => {
                    emit_variant_discriminant(ectx, variant);
                    // Remaining keys are the variant's fields — use continuation
                    self.emit_struct_fields_continuation(
                        ectx,
                        &variant.fields,
                        false,
                        &mut |ectx, field| {
                            emit_variant_field(ectx, variant, field);
                        },
                    );
                }
                VariantKind::Tuple => {
                    // r[impl deser.json.enum.internal.struct-only]
                    panic!(
                        "internally tagged enums do not support tuple variants \
                         (variant \"{}\")",
                        variant.name
                    );
                }
            }
            ectx.emit_branch(done_label);
        }

        ectx.bind_label(done_label);
    }

    // r[impl deser.json.enum.untagged]
    // r[impl deser.json.enum.untagged.bucket]
    // r[impl deser.json.enum.untagged.peek]
    // r[impl deser.json.enum.untagged.object-solver]
    // r[impl deser.json.enum.untagged.string-trie]
    // r[impl deser.json.enum.untagged.scalar-unique]
    fn emit_enum_untagged(
        &self,
        ectx: &mut EmitCtx,
        variants: &[VariantEmitInfo],
        emit_variant_body: &mut dyn FnMut(&mut EmitCtx, &VariantEmitInfo),
    ) {
        // ══════════════════════════════════════════════════════════════════
        // Layer 1: Bucket variants by JSON value type (JIT-compile time)
        // ══════════════════════════════════════════════════════════════════

        let mut object_variants: Vec<usize> = Vec::new();
        let mut string_variants: Vec<usize> = Vec::new();
        let mut bool_variants: Vec<usize> = Vec::new();
        let mut number_variants: Vec<usize> = Vec::new();
        let null_variants: Vec<usize> = Vec::new();

        for (i, variant) in variants.iter().enumerate() {
            match variant.kind {
                VariantKind::Unit => {
                    string_variants.push(i);
                }
                VariantKind::Struct => {
                    object_variants.push(i);
                }
                VariantKind::Tuple => {
                    assert!(
                        variant.fields.len() == 1,
                        "untagged tuple variant {} has {} fields, expected 1",
                        variant.name,
                        variant.fields.len()
                    );
                    let inner_shape = variant.fields[0].shape;
                    match &inner_shape.ty {
                        Type::User(UserType::Struct(_)) => {
                            object_variants.push(i);
                        }
                        _ => match inner_shape.scalar_type() {
                            Some(ScalarType::Char) => string_variants.push(i),
                            Some(ScalarType::String | ScalarType::Str | ScalarType::CowStr) => {
                                string_variants.push(i)
                            }
                            Some(ScalarType::Bool) => bool_variants.push(i),
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
                            ) => number_variants.push(i),
                            _ => panic!(
                                "unsupported inner type for untagged tuple variant {}: {}",
                                variant.name, inner_shape.type_identifier
                            ),
                        },
                    }
                }
            }
        }

        // Validate scalar bucket uniqueness
        assert!(
            bool_variants.len() <= 1,
            "untagged enum has {} bool variants — at most 1 allowed",
            bool_variants.len()
        );
        assert!(
            number_variants.len() <= 1,
            "untagged enum has {} number variants — at most 1 allowed",
            number_variants.len()
        );
        let string_newtype_count = string_variants
            .iter()
            .filter(|&&i| variants[i].kind == VariantKind::Tuple)
            .count();
        assert!(
            string_newtype_count <= 1,
            "untagged enum has {} newtype String variants — at most 1 allowed",
            string_newtype_count
        );

        // ══════════════════════════════════════════════════════════════════
        // Layer 2: Peek dispatch (emitted code)
        // ══════════════════════════════════════════════════════════════════

        let done_label = ectx.new_label();

        // Peek at first non-whitespace byte
        ectx.emit_call_intrinsic_ctx_and_stack_out(
            json_intrinsics::kajit_json_peek_after_ws as *const u8,
            RESULT_BYTE_OFFSET,
        );

        let object_label = ectx.new_label();
        let string_label = ectx.new_label();
        let bool_label = ectx.new_label();
        let number_label = ectx.new_label();
        let null_label = ectx.new_label();

        if !object_variants.is_empty() {
            ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'{', object_label);
        }
        if !string_variants.is_empty() {
            ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'"', string_label);
        }
        if !bool_variants.is_empty() {
            ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b't', bool_label);
            ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'f', bool_label);
        }
        if !null_variants.is_empty() {
            ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'n', null_label);
        }
        if !number_variants.is_empty() {
            ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'-', number_label);
            for d in b'0'..=b'9' {
                ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, d, number_label);
            }
        }

        // No bucket matched
        ectx.emit_unknown_variant_error();

        // ══════════════════════════════════════════════════════════════════
        // Layer 3: Within-bucket disambiguation
        // ══════════════════════════════════════════════════════════════════

        // ── Object bucket ────────────────────────────────────────────────
        if !object_variants.is_empty() {
            ectx.bind_label(object_label);
            if object_variants.len() == 1 {
                emit_variant_body(ectx, &variants[object_variants[0]]);
            } else {
                let solver_variants: Vec<(usize, &VariantEmitInfo)> =
                    object_variants.iter().map(|&i| (i, &variants[i])).collect();
                let solver = LoweredSolver::build(&solver_variants);
                self.emit_object_solver(ectx, &solver, variants, done_label, emit_variant_body);
            }
            ectx.emit_branch(done_label);
        }

        // ── String bucket ────────────────────────────────────────────────
        if !string_variants.is_empty() {
            ectx.bind_label(string_label);

            let has_newtype = string_variants
                .iter()
                .any(|&i| variants[i].kind == VariantKind::Tuple);

            if has_newtype {
                // Save cursor so we can restore for the newtype String fallback.
                // read_key consumes the string, but emit_variant_body for the
                // newtype needs to read it again via read_string_value.
                ectx.emit_save_input_ptr(SAVED_CURSOR_OFFSET);
            }

            // Read the string as a borrowed key (ptr + len)
            ectx.emit_call_intrinsic_ctx_and_two_stack_outs(
                json_intrinsics::kajit_json_read_key as *const u8,
                KEY_PTR_OFFSET,
                KEY_LEN_OFFSET,
            );

            // Try unit variant names
            let unit_labels: Vec<_> = string_variants
                .iter()
                .filter(|&&i| variants[i].kind == VariantKind::Unit)
                .map(|&i| (i, ectx.new_label()))
                .collect();

            for &(vi, label) in &unit_labels {
                let name_bytes = variants[vi].name.as_bytes();
                ectx.emit_call_pure_4arg(
                    json_intrinsics::kajit_json_key_equals as *const u8,
                    KEY_PTR_OFFSET,
                    KEY_LEN_OFFSET,
                    name_bytes.as_ptr(),
                    name_bytes.len() as u32,
                );
                ectx.emit_cbnz_x0(label);
            }

            // No unit name matched
            if let Some(&nt_idx) = string_variants
                .iter()
                .find(|&&i| variants[i].kind == VariantKind::Tuple)
            {
                // Restore cursor to before the string, then call emit_variant_body
                // which will re-read it via read_string_value (allocating a String).
                ectx.emit_restore_input_ptr(SAVED_CURSOR_OFFSET);
                emit_variant_body(ectx, &variants[nt_idx]);
                ectx.emit_branch(done_label);
            } else {
                ectx.emit_unknown_variant_error();
            }

            // Unit variant handlers
            for &(vi, label) in &unit_labels {
                ectx.bind_label(label);
                emit_variant_body(ectx, &variants[vi]);
                ectx.emit_branch(done_label);
            }
        }

        // ── Bool bucket ──────────────────────────────────────────────────
        if !bool_variants.is_empty() {
            ectx.bind_label(bool_label);
            emit_variant_body(ectx, &variants[bool_variants[0]]);
            ectx.emit_branch(done_label);
        }

        // ── Number bucket ────────────────────────────────────────────────
        if !number_variants.is_empty() {
            ectx.bind_label(number_label);
            emit_variant_body(ectx, &variants[number_variants[0]]);
            ectx.emit_branch(done_label);
        }

        // ── Null bucket ──────────────────────────────────────────────────
        if !null_variants.is_empty() {
            ectx.bind_label(null_label);
            emit_variant_body(ectx, &variants[null_variants[0]]);
            ectx.emit_branch(done_label);
        }

        ectx.bind_label(done_label);
    }

    fn emit_struct_fields_continuation(
        &self,
        ectx: &mut EmitCtx,
        fields: &[FieldEmitInfo],
        deny_unknown_fields: bool,
        emit_field: &mut dyn FnMut(&mut EmitCtx, &FieldEmitInfo),
    ) {
        let after_loop = ectx.new_label();
        let loop_top = ectx.new_label();

        // Zero the bitset
        ectx.emit_zero_stack_slot(BITSET_OFFSET);

        // We're already inside the object, right after reading "tag_key": "VariantName".
        // Next is either ',' (more fields) or '}' (no variant fields).
        ectx.emit_inline_comma_or_end_object(RESULT_BYTE_OFFSET);
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, 1, after_loop);

        // === loop_top ===
        ectx.bind_label(loop_top);

        // Read key
        emit_read_key_inline(ectx);

        // expect ':'
        ectx.emit_expect_byte_after_ws(b':', crate::context::ErrorCode::ExpectedColon);

        // Linear key dispatch chain
        let after_dispatch = ectx.new_label();
        let match_labels: Vec<_> = fields.iter().map(|_| ectx.new_label()).collect();
        let unknown_key = ectx.new_label();

        for (i, field) in fields.iter().enumerate() {
            let name_bytes = field.name.as_bytes();
            ectx.emit_call_pure_4arg(
                json_intrinsics::kajit_json_key_equals as *const u8,
                KEY_PTR_OFFSET,
                KEY_LEN_OFFSET,
                name_bytes.as_ptr(),
                name_bytes.len() as u32,
            );
            ectx.emit_cbnz_x0(match_labels[i]);
        }

        ectx.emit_branch(unknown_key);

        // Match handlers
        for (i, field) in fields.iter().enumerate() {
            ectx.bind_label(match_labels[i]);
            emit_field(ectx, field);
            ectx.emit_set_bit_on_stack(BITSET_OFFSET, field.required_index as u32);
            ectx.emit_branch(after_dispatch);
        }

        // Unknown key handler
        ectx.bind_label(unknown_key);
        if deny_unknown_fields {
            ectx.emit_error(ErrorCode::UnknownField);
        } else {
            ectx.emit_call_intrinsic_ctx_only(json_intrinsics::kajit_json_skip_value as *const u8);
        }

        // After dispatch
        ectx.bind_label(after_dispatch);

        // comma_or_end
        ectx.emit_inline_comma_or_end_object(RESULT_BYTE_OFFSET);
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, 1, after_loop);
        ectx.emit_branch(loop_top);

        // === after_loop ===
        ectx.bind_label(after_loop);

        emit_default_or_required_check(ectx, fields);
    }
}

impl IrDecoder for KajitJson {
    fn lower_struct_fields(
        &self,
        builder: &mut RegionBuilder<'_>,
        fields: &[FieldLowerInfo],
        deny_unknown_fields: bool,
        lower_field: &mut dyn FnMut(&mut RegionBuilder<'_>, &FieldLowerInfo),
    ) {
        for field in fields {
            assert!(
                !field.has_default,
                "JSON IR lowering does not support defaulted fields yet"
            );
            assert!(
                field.required_index < 64,
                "JSON IR lowering currently supports at most 64 fields"
            );
        }

        let required_mask = fields
            .iter()
            .fold(0u64, |mask, field| mask | (1u64 << field.required_index));

        builder.call_intrinsic(
            ifn(json_intrinsics::kajit_json_expect_object_start as _),
            &[],
            0,
            false,
        );

        let key_ptr_slot = builder.alloc_slot();
        let key_len_slot = builder.alloc_slot();

        let bitset0 = builder.const_val(0);
        builder.call_intrinsic(ifn(json_intrinsics::kajit_json_skip_ws as _), &[], 0, false);
        let first = builder.peek_byte();
        let close = builder.const_val(b'}' as u64);
        let has_entries = builder.binop(crate::ir::IrOp::CmpNe, first, close);

        let out = builder.gamma(has_entries, &[bitset0], 2, |branch_idx, rb| {
            let args = rb.region_args(1);
            let bitset_in = args[0];
            match branch_idx {
                0 => {
                    rb.advance_cursor(1);
                    rb.set_results(&[bitset_in]);
                }
                1 => {
                    let loop_out = rb.theta(&[bitset_in], |tb| {
                        let args = tb.region_args(1);
                        let mut bitset_iter = args[0];

                        let key_ptr_out_addr = tb.slot_addr(key_ptr_slot);
                        let key_len_out_addr = tb.slot_addr(key_len_slot);
                        tb.call_intrinsic(
                            ifn(json_intrinsics::kajit_json_read_key as _),
                            &[key_ptr_out_addr, key_len_out_addr],
                            0,
                            false,
                        );
                        let key_ptr = tb.read_from_slot(key_ptr_slot);
                        let key_len = tb.read_from_slot(key_len_slot);
                        tb.call_intrinsic(ifn(json_intrinsics::kajit_json_expect_colon as _), &[], 0, false);

                        let mut handled = tb.const_val(0);
                        for field in fields {
                            let expected_ptr = tb.const_val(field.name.as_ptr() as u64);
                            let expected_len = tb.const_val(field.name.len() as u64);
                            let eq = tb.call_pure(
                                ifn(json_intrinsics::kajit_json_key_equals as _),
                                &[key_ptr, key_len, expected_ptr, expected_len],
                            );
                            let one = tb.const_val(1);
                            let not_handled = tb.binop(crate::ir::IrOp::CmpNe, handled, one);
                            let should_handle = tb.binop(crate::ir::IrOp::And, eq, not_handled);

                            let updated = tb.gamma(should_handle, &[bitset_iter, handled], 2, |idx, mb| {
                                let pass = mb.region_args(2);
                                let bitset_in = pass[0];
                                let handled_in = pass[1];
                                match idx {
                                    0 => mb.set_results(&[bitset_in, handled_in]),
                                    1 => {
                                        lower_field(mb, field);
                                        let mask = mb.const_val(1u64 << field.required_index);
                                        let bitset_new =
                                            mb.binop(crate::ir::IrOp::Or, bitset_in, mask);
                                        let handled_yes = mb.const_val(1);
                                        mb.set_results(&[bitset_new, handled_yes]);
                                    }
                                    _ => unreachable!(),
                                }
                            });
                            bitset_iter = updated[0];
                            handled = updated[1];
                        }

                        let after_unknown = tb.gamma(handled, &[bitset_iter], 2, |idx, ub| {
                            let pass = ub.region_args(1);
                            let bitset_in = pass[0];
                            match idx {
                                0 => {
                                    if deny_unknown_fields {
                                        ub.call_intrinsic(
                                            ifn(json_intrinsics::kajit_json_error_unknown_field as _),
                                            &[],
                                            0,
                                            false,
                                        );
                                    } else {
                                        ub.call_intrinsic(
                                            ifn(json_intrinsics::kajit_json_skip_value as _),
                                            &[],
                                            0,
                                            false,
                                        );
                                    }
                                    ub.set_results(&[bitset_in]);
                                }
                                1 => ub.set_results(&[bitset_in]),
                                _ => unreachable!(),
                            }
                        });
                        bitset_iter = after_unknown[0];

                        tb.call_intrinsic(ifn(json_intrinsics::kajit_json_skip_ws as _), &[], 0, false);
                        tb.bounds_check(1);
                        let sep = tb.read_bytes(1);
                        let comma = tb.const_val(b',' as u64);
                        let sep_ne_comma = tb.binop(crate::ir::IrOp::CmpNe, sep, comma);

                        let delim = tb.gamma(sep_ne_comma, &[bitset_iter, sep], 2, |idx, db| {
                            let pass = db.region_args(2);
                            let bitset_in = pass[0];
                            let sep_val = pass[1];
                            match idx {
                                0 => {
                                    let cont = db.const_val(1);
                                    db.set_results(&[cont, bitset_in]);
                                }
                                1 => {
                                    let close = db.const_val(b'}' as u64);
                                    let sep_ne_close =
                                        db.binop(crate::ir::IrOp::CmpNe, sep_val, close);
                                    let close_out =
                                        db.gamma(sep_ne_close, &[bitset_in], 2, |j, cb| {
                                            let pass = cb.region_args(1);
                                            let bitset_in = pass[0];
                                            match j {
                                                0 => {
                                                    let stop = cb.const_val(0);
                                                    cb.set_results(&[stop, bitset_in]);
                                                }
                                                1 => {
                                                    cb.call_intrinsic(
                                                        ifn(
                                                            json_intrinsics::kajit_json_error_unexpected_character
                                                                as _,
                                                        ),
                                                        &[],
                                                        0,
                                                        false,
                                                    );
                                                    let stop = cb.const_val(0);
                                                    cb.set_results(&[stop, bitset_in]);
                                                }
                                                _ => unreachable!(),
                                            }
                                        });
                                    db.set_results(&[close_out[0], close_out[1]]);
                                }
                                _ => unreachable!(),
                            }
                        });

                        tb.set_results(&[delim[0], delim[1]]);
                    });

                    rb.set_results(&[loop_out[0]]);
                }
                _ => unreachable!(),
            }
        });

        let seen_mask = out[0];
        if required_mask != 0 {
            let required = builder.const_val(required_mask);
            let present = builder.binop(crate::ir::IrOp::And, seen_mask, required);
            let missing = builder.binop(crate::ir::IrOp::CmpNe, present, required);
            builder.gamma(missing, &[], 2, |idx, rb| match idx {
                0 => rb.set_results(&[]),
                1 => {
                    rb.error_exit(ErrorCode::MissingRequiredField);
                    rb.set_results(&[]);
                }
                _ => unreachable!(),
            });
        }
    }

    fn lower_read_scalar(
        &self,
        builder: &mut RegionBuilder<'_>,
        offset: usize,
        scalar_type: ScalarType,
    ) {
        let offset = offset as u32;
        if scalar_type == ScalarType::Unit {
            builder.call_intrinsic(
                ifn(json_intrinsics::kajit_json_skip_value as _),
                &[],
                0,
                false,
            );
            return;
        }

        let func = match scalar_type {
            ScalarType::U8 => ifn(json_intrinsics::kajit_json_read_u8 as _),
            ScalarType::U16 => ifn(json_intrinsics::kajit_json_read_u16 as _),
            ScalarType::U32 => ifn(json_intrinsics::kajit_json_read_u32 as _),
            ScalarType::U64 => ifn(json_intrinsics::kajit_json_read_u64 as _),
            ScalarType::U128 => ifn(json_intrinsics::kajit_json_read_u128 as _),
            ScalarType::USize => ifn(json_intrinsics::kajit_json_read_usize as _),
            ScalarType::I8 => ifn(json_intrinsics::kajit_json_read_i8 as _),
            ScalarType::I16 => ifn(json_intrinsics::kajit_json_read_i16 as _),
            ScalarType::I32 => ifn(json_intrinsics::kajit_json_read_i32 as _),
            ScalarType::I64 => ifn(json_intrinsics::kajit_json_read_i64 as _),
            ScalarType::I128 => ifn(json_intrinsics::kajit_json_read_i128 as _),
            ScalarType::ISize => ifn(json_intrinsics::kajit_json_read_isize as _),
            ScalarType::F32 => ifn(json_intrinsics::kajit_json_read_f32 as _),
            ScalarType::F64 => ifn(json_intrinsics::kajit_json_read_f64 as _),
            ScalarType::Bool => ifn(json_intrinsics::kajit_json_read_bool as _),
            ScalarType::Char => ifn(json_intrinsics::kajit_json_read_char as _),
            _ => panic!("unsupported JSON scalar in IR lowering: {:?}", scalar_type),
        };

        builder.call_intrinsic(func, &[], offset, false);
    }

    fn lower_read_string(
        &self,
        builder: &mut RegionBuilder<'_>,
        offset: usize,
        scalar_type: ScalarType,
    ) {
        let func = match scalar_type {
            ScalarType::String => ifn(json_intrinsics::kajit_json_read_string_value as _),
            ScalarType::Str => ifn(json_intrinsics::kajit_json_read_str_value as _),
            ScalarType::CowStr => ifn(json_intrinsics::kajit_json_read_cow_str_value as _),
            _ => panic!(
                "unsupported JSON string scalar in IR lowering: {:?}",
                scalar_type
            ),
        };

        builder.call_intrinsic(func, &[], offset as u32, false);
    }

    fn lower_vec(
        &self,
        builder: &mut RegionBuilder<'_>,
        offset: usize,
        elem_shape: &'static facet::Shape,
        vec_offsets: &crate::malum::VecOffsets,
        lower_elem: &mut dyn FnMut(&mut RegionBuilder<'_>),
    ) {
        let offset = offset as u32;
        let elem_layout = elem_shape
            .layout
            .sized_layout()
            .expect("Vec element must be Sized");
        let elem_size = elem_layout.size() as u64;
        let elem_align = elem_layout.align() as u64;
        let usize_width = if core::mem::size_of::<usize>() == 8 {
            crate::ir::Width::W8
        } else {
            crate::ir::Width::W4
        };

        builder.call_intrinsic(
            ifn(json_intrinsics::kajit_json_expect_array_start as _),
            &[],
            0,
            false,
        );

        let sep_slot = builder.alloc_slot();
        let sep_addr = builder.slot_addr(sep_slot);
        builder.call_intrinsic(
            ifn(json_intrinsics::kajit_json_peek_after_ws as _),
            &[sep_addr],
            0,
            false,
        );
        let first = builder.read_from_slot(sep_slot);
        let close = builder.const_val(b']' as u64);
        let has_entries = builder.binop(crate::ir::IrOp::CmpNe, first, close);

        builder.gamma(has_entries, &[], 2, |branch_idx, rb| match branch_idx {
            0 => {
                rb.advance_cursor(1);
                let ptr = rb.const_val(elem_align);
                let zero = rb.const_val(0);
                rb.write_to_field(ptr, offset + vec_offsets.ptr_offset, usize_width);
                rb.write_to_field(zero, offset + vec_offsets.len_offset, usize_width);
                rb.write_to_field(zero, offset + vec_offsets.cap_offset, usize_width);
                rb.set_results(&[]);
            }
            1 => {
                let initial_cap = rb.const_val(4);
                let size_arg = rb.const_val(elem_size);
                let align_arg = rb.const_val(elem_align);
                let buf = rb
                    .call_intrinsic(
                        ifn(intrinsics::kajit_vec_alloc as _),
                        &[initial_cap, size_arg, align_arg],
                        0,
                        true,
                    )
                    .expect("vec_alloc should produce a result");

                let zero = rb.const_val(0);
                let saved_out = rb.save_out_ptr();
                rb.set_out_ptr(buf);

                let loop_out = rb.theta(&[buf, buf, zero, initial_cap], |tb| {
                    let args = tb.region_args(4);
                    let buf = args[0];
                    let cursor = args[1];
                    let len = args[2];
                    let cap = args[3];

                    let len_ne_cap = tb.binop(crate::ir::IrOp::CmpNe, len, cap);
                    let grown = tb.gamma(len_ne_cap, &[buf, cap], 2, |idx, gb| {
                        let pass = gb.region_args(2);
                        let buf = pass[0];
                        let cap = pass[1];
                        match idx {
                            0 => {
                                let one = gb.const_val(1);
                                let new_cap = gb.binop(crate::ir::IrOp::Shl, cap, one);
                                let size_arg = gb.const_val(elem_size);
                                let align_arg = gb.const_val(elem_align);
                                let grown_buf = gb
                                    .call_intrinsic(
                                        ifn(intrinsics::kajit_vec_grow as _),
                                        &[buf, len, cap, new_cap, size_arg, align_arg],
                                        0,
                                        true,
                                    )
                                    .expect("vec_grow should produce a result");
                                gb.set_results(&[grown_buf, new_cap]);
                            }
                            1 => gb.set_results(&[buf, cap]),
                            _ => unreachable!(),
                        }
                    });
                    let buf = grown[0];
                    let cap = grown[1];

                    tb.set_out_ptr(cursor);
                    lower_elem(tb);

                    let one = tb.const_val(1);
                    let len2 = tb.binop(crate::ir::IrOp::Add, len, one);
                    let elem_size = tb.const_val(elem_size);
                    let cursor2 = tb.binop(crate::ir::IrOp::Add, cursor, elem_size);

                    let sep_addr = tb.slot_addr(sep_slot);
                    tb.call_intrinsic(
                        ifn(json_intrinsics::kajit_json_comma_or_end_array as _),
                        &[sep_addr],
                        0,
                        false,
                    );
                    let sep = tb.read_from_slot(sep_slot);
                    let continue_loop = tb.binop(crate::ir::IrOp::CmpNe, sep, one);
                    tb.set_results(&[continue_loop, buf, cursor2, len2, cap]);
                });

                let buf = loop_out[0];
                let len = loop_out[2];
                let cap = loop_out[3];
                rb.set_out_ptr(saved_out);
                rb.write_to_field(buf, offset + vec_offsets.ptr_offset, usize_width);
                rb.write_to_field(len, offset + vec_offsets.len_offset, usize_width);
                rb.write_to_field(cap, offset + vec_offsets.cap_offset, usize_width);
                rb.set_results(&[]);
            }
            _ => unreachable!(),
        });
    }
}

/// Emit default-or-required field checks after a JSON struct loop.
///
/// For each field with a default: if its bitset bit is NOT set, call the default
/// trampoline to initialize it. Then check that all required (non-default) fields
/// were seen.
fn emit_default_or_required_check(ectx: &mut EmitCtx, fields: &[FieldEmitInfo]) {
    let mut required_mask = 0u64;
    for field in fields {
        if let Some(default_info) = &field.default {
            let after_default = ectx.new_label();
            // If bit IS set, skip the default call.
            ectx.emit_test_bit_branch(BITSET_OFFSET, field.required_index as u32, after_default);
            // Bit not set — call the default trampoline.
            crate::compiler::emit_default_init(ectx, default_info, field.offset as u32);
            ectx.bind_label(after_default);
        } else {
            required_mask |= 1u64 << field.required_index;
        }
    }

    // Check that all required (non-default) fields were seen.
    if required_mask != 0 {
        ectx.emit_check_bitset(BITSET_OFFSET, required_mask);
    }
}

impl KajitJson {
    /// Emit the object-bucket solver for untagged enum disambiguation.
    ///
    /// Two-pass approach:
    /// 1. Save cursor, scan all keys, narrow candidates using key-presence,
    ///    value-type, and nested-key evidence until popcount == 1 or end of object.
    /// 2. Restore cursor, dispatch to the resolved variant's body.
    fn emit_object_solver(
        &self,
        ectx: &mut EmitCtx,
        solver: &LoweredSolver,
        variants: &[VariantEmitInfo],
        done_label: DynamicLabel,
        emit_variant_body: &mut dyn FnMut(&mut EmitCtx, &VariantEmitInfo),
    ) {
        let resolve_label = ectx.new_label();
        let error_no_match = ectx.new_label();
        let scan_loop = ectx.new_label();

        // ── Pass 1: Save cursor, scan keys, narrow candidates ────────

        // Save cursor (input_ptr) before consuming the object
        ectx.emit_save_input_ptr(SAVED_CURSOR_OFFSET);

        // Initialize candidates bitmask to all candidates set
        ectx.emit_store_imm64_to_stack(CANDIDATES_OFFSET, solver.initial_mask);

        // Consume '{'
        ectx.emit_expect_byte_after_ws(b'{', crate::context::ErrorCode::ExpectedObjectStart);

        // Check for empty object → resolve immediately
        ectx.emit_call_intrinsic_ctx_and_stack_out(
            json_intrinsics::kajit_json_peek_after_ws as *const u8,
            RESULT_BYTE_OFFSET,
        );
        let empty_object = ectx.new_label();
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'}', empty_object);

        // ── scan_loop ────────────────────────────────────────────────
        ectx.bind_label(scan_loop);

        // Read key
        emit_read_key_inline(ectx);

        // Consume ':'
        ectx.emit_expect_byte_after_ws(b':', crate::context::ErrorCode::ExpectedColon);

        // NOTE: skip_value is NOT called here — it's in each per-key handler,
        // because some handlers need to peek at or sub-scan the value first.

        // Linear key dispatch: identify which key this is.
        let after_key_dispatch = ectx.new_label();
        let key_labels: Vec<_> = solver.key_masks.iter().map(|_| ectx.new_label()).collect();

        for (i, &(name, _mask)) in solver.key_masks.iter().enumerate() {
            let name_bytes = name.as_bytes();
            ectx.emit_call_pure_4arg(
                json_intrinsics::kajit_json_key_equals as *const u8,
                KEY_PTR_OFFSET,
                KEY_LEN_OFFSET,
                name_bytes.as_ptr(),
                name_bytes.len() as u32,
            );
            ectx.emit_cbnz_x0(key_labels[i]);
        }
        // Unknown key — skip value, don't narrow
        ectx.emit_call_intrinsic_ctx_only(json_intrinsics::kajit_json_skip_value as *const u8);
        ectx.emit_branch(after_key_dispatch);

        // Per-key handlers
        for (i, &(_name, mask)) in solver.key_masks.iter().enumerate() {
            ectx.bind_label(key_labels[i]);

            // Always AND key-presence mask
            ectx.emit_and_imm64_on_stack(CANDIDATES_OFFSET, mask);

            let has_vt = !solver.value_type_masks[i].is_empty();
            let has_sub = solver.sub_solvers[i].is_some();

            if !has_vt && !has_sub {
                // Simple: just skip the value
                ectx.emit_call_intrinsic_ctx_only(
                    json_intrinsics::kajit_json_skip_value as *const u8,
                );
            } else if has_sub {
                // Sub-solver: peek, if '{' run sub-scan, else value-type + skip
                self.emit_sub_scan_or_skip(
                    ectx,
                    &solver.value_type_masks[i],
                    solver.sub_solvers[i].as_ref().unwrap(),
                );
            } else {
                // Value-type evidence only: peek, AND type mask, skip
                self.emit_value_type_peek(ectx, &solver.value_type_masks[i]);
                ectx.emit_call_intrinsic_ctx_only(
                    json_intrinsics::kajit_json_skip_value as *const u8,
                );
            }

            ectx.emit_branch(after_key_dispatch);
        }

        ectx.bind_label(after_key_dispatch);

        // Check candidates: popcount == 1 → resolve early
        ectx.emit_popcount_eq1_branch(CANDIDATES_OFFSET, resolve_label);
        // popcount == 0 → no match
        ectx.emit_stack_zero_branch(CANDIDATES_OFFSET, error_no_match);

        // comma_or_end: '}' → resolve, ',' → continue scanning
        ectx.emit_inline_comma_or_end_object(RESULT_BYTE_OFFSET);
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, 1, resolve_label);
        ectx.emit_branch(scan_loop);

        // Empty object → resolve with whatever candidates remain
        ectx.bind_label(empty_object);
        ectx.emit_advance_cursor_by(1); // consume '}'
        // Fall through to resolve

        // ── Pass 2: Resolve — restore cursor, dispatch to variant ────
        ectx.bind_label(resolve_label);

        // Restore cursor to before the '{'
        ectx.emit_restore_input_ptr(SAVED_CURSOR_OFFSET);

        // For each candidate bit position, check if that bit is set and
        // dispatch to the variant body.
        for (bit, &orig_variant_idx) in solver.candidate_to_variant.iter().enumerate() {
            let variant_label = ectx.new_label();
            ectx.emit_test_bit_branch(CANDIDATES_OFFSET, bit as u32, variant_label);
            // Not this candidate, try next
            let skip_label = ectx.new_label();
            ectx.emit_branch(skip_label);

            ectx.bind_label(variant_label);
            emit_variant_body(ectx, &variants[orig_variant_idx]);
            ectx.emit_branch(done_label);

            ectx.bind_label(skip_label);
        }

        // If we get here, no candidate bit was set — shouldn't happen but be safe
        ectx.emit_branch(error_no_match);

        // ── Error paths ──────────────────────────────────────────────
        ectx.bind_label(error_no_match);
        ectx.emit_error(ErrorCode::UnknownVariant);
    }

    /// Emit a value-type peek: read the value's first byte and AND the matching
    /// type mask into CANDIDATES.
    fn emit_value_type_peek(&self, ectx: &mut EmitCtx, vt_masks: &[(JsonValueType, u64)]) {
        let after_vt = ectx.new_label();

        // Peek at the value's first byte
        ectx.emit_call_intrinsic_ctx_and_stack_out(
            json_intrinsics::kajit_json_peek_after_ws as *const u8,
            RESULT_BYTE_OFFSET,
        );

        Self::emit_value_type_dispatch(ectx, vt_masks, after_vt);

        ectx.bind_label(after_vt);
    }

    /// Emit value-type dispatch using an already-loaded byte in RESULT_BYTE_OFFSET.
    fn emit_value_type_dispatch(
        ectx: &mut EmitCtx,
        vt_masks: &[(JsonValueType, u64)],
        after_label: DynamicLabel,
    ) {
        // For each value type, check peek byte(s) and branch to handler.
        let mut handlers: Vec<(DynamicLabel, u64)> = Vec::new();

        for &(vtype, mask) in vt_masks {
            let handler = ectx.new_label();
            match vtype {
                JsonValueType::Object => {
                    ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'{', handler);
                }
                JsonValueType::Array => {
                    ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'[', handler);
                }
                JsonValueType::String => {
                    ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'"', handler);
                }
                JsonValueType::Bool => {
                    ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b't', handler);
                    ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'f', handler);
                }
                JsonValueType::Null => {
                    ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'n', handler);
                }
                JsonValueType::Number => {
                    ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'-', handler);
                    for d in b'0'..=b'9' {
                        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, d, handler);
                    }
                }
            }
            handlers.push((handler, mask));
        }

        // No type matched — don't narrow (shouldn't happen for valid JSON)
        ectx.emit_branch(after_label);

        // Emit handlers: AND mask and branch to after
        for (handler, mask) in handlers {
            ectx.bind_label(handler);
            ectx.emit_and_imm64_on_stack(CANDIDATES_OFFSET, mask);
            ectx.emit_branch(after_label);
        }
    }

    /// Emit code for a key with a sub-solver: peek at value, if '{' run sub-scan
    /// of the nested object's keys, otherwise apply value-type mask + skip_value.
    fn emit_sub_scan_or_skip(
        &self,
        ectx: &mut EmitCtx,
        vt_masks: &[(JsonValueType, u64)],
        sub_solver: &crate::solver::SubSolver,
    ) {
        let after_all = ectx.new_label();
        let sub_scan_path = ectx.new_label();

        // Peek at value to determine if it's an object
        ectx.emit_call_intrinsic_ctx_and_stack_out(
            json_intrinsics::kajit_json_peek_after_ws as *const u8,
            RESULT_BYTE_OFFSET,
        );
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'{', sub_scan_path);

        // Not an object: apply value-type mask if available, then skip_value
        if !vt_masks.is_empty() {
            // Byte is already loaded in RESULT_BYTE_OFFSET — dispatch on it
            // (filter out Object since we already handled that branch)
            let non_object: Vec<_> = vt_masks
                .iter()
                .filter(|&&(vt, _)| vt != JsonValueType::Object)
                .copied()
                .collect();
            if !non_object.is_empty() {
                let after_vt = ectx.new_label();
                Self::emit_value_type_dispatch(ectx, &non_object, after_vt);
                ectx.bind_label(after_vt);
            }
        }
        ectx.emit_call_intrinsic_ctx_only(json_intrinsics::kajit_json_skip_value as *const u8);
        ectx.emit_branch(after_all);

        // ── Object path: run sub-scan ────────────────────────────────
        ectx.bind_label(sub_scan_path);

        // Consume '{'
        ectx.emit_expect_byte_after_ws(b'{', crate::context::ErrorCode::ExpectedObjectStart);

        // Check for empty inner object
        let sub_scan_loop = ectx.new_label();
        let after_sub_scan = ectx.new_label();

        ectx.emit_call_intrinsic_ctx_and_stack_out(
            json_intrinsics::kajit_json_peek_after_ws as *const u8,
            RESULT_BYTE_OFFSET,
        );
        let empty_inner = ectx.new_label();
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, b'}', empty_inner);

        // Sub-scan loop
        ectx.bind_label(sub_scan_loop);

        // Read inner key
        emit_read_key_inline(ectx);
        ectx.emit_expect_byte_after_ws(b':', crate::context::ErrorCode::ExpectedColon);
        ectx.emit_call_intrinsic_ctx_only(json_intrinsics::kajit_json_skip_value as *const u8);

        // Inner key dispatch: for each inner key, AND outer mask
        let after_inner_dispatch = ectx.new_label();
        let inner_labels: Vec<_> = sub_solver
            .inner_key_masks
            .iter()
            .map(|_| ectx.new_label())
            .collect();

        for (j, &(inner_name, _)) in sub_solver.inner_key_masks.iter().enumerate() {
            let name_bytes = inner_name.as_bytes();
            ectx.emit_call_pure_4arg(
                json_intrinsics::kajit_json_key_equals as *const u8,
                KEY_PTR_OFFSET,
                KEY_LEN_OFFSET,
                name_bytes.as_ptr(),
                name_bytes.len() as u32,
            );
            ectx.emit_cbnz_x0(inner_labels[j]);
        }
        // Unknown inner key — don't narrow
        ectx.emit_branch(after_inner_dispatch);

        // Inner key handlers
        for (j, &(_, outer_mask)) in sub_solver.inner_key_masks.iter().enumerate() {
            ectx.bind_label(inner_labels[j]);
            ectx.emit_and_imm64_on_stack(CANDIDATES_OFFSET, outer_mask);
            ectx.emit_branch(after_inner_dispatch);
        }

        ectx.bind_label(after_inner_dispatch);

        // Early exit if resolved
        ectx.emit_popcount_eq1_branch(CANDIDATES_OFFSET, after_sub_scan);

        // comma_or_end for inner object
        ectx.emit_inline_comma_or_end_object(RESULT_BYTE_OFFSET);
        ectx.emit_stack_byte_cmp_branch(RESULT_BYTE_OFFSET, 1, after_sub_scan);
        ectx.emit_branch(sub_scan_loop);

        // Empty inner object — consume '}'
        ectx.bind_label(empty_inner);
        ectx.emit_advance_cursor_by(1);

        ectx.bind_label(after_sub_scan);
        // Sub-scan consumed the entire nested object, no skip_value needed.
        ectx.emit_branch(after_all);

        ectx.bind_label(after_all);
    }
}

// ── JSON Encoder ────────────────────────────────────────────────────

/// Compact JSON encoder — no whitespace between tokens.
pub struct KajitJsonEncoder;

impl Encoder for KajitJsonEncoder {
    fn supports_inline_nested(&self) -> bool {
        false // JSON needs `{`/`}` framing per struct level
    }

    fn emit_encode_struct_fields(
        &self,
        ectx: &mut EmitCtx,
        fields: &[FieldEncodeInfo],
        emit_field: &mut dyn FnMut(&mut EmitCtx, &FieldEncodeInfo),
    ) {
        for (i, field) in fields.iter().enumerate() {
            // Build the literal prefix for this field:
            // First field:  {"fieldname":
            // Subsequent:   ,"fieldname":
            let mut prefix = Vec::new();
            if i == 0 {
                prefix.push(b'{');
            } else {
                prefix.push(b',');
            }
            prefix.push(b'"');
            prefix.extend_from_slice(field.name.as_bytes());
            prefix.push(b'"');
            prefix.push(b':');

            ectx.emit_recipe(&crate::recipe::write_literal(&prefix));
            emit_field(ectx, field);
        }

        if fields.is_empty() {
            // Empty struct: just "{}"
            ectx.emit_recipe(&crate::recipe::write_literal(b"{}"));
        } else {
            ectx.emit_recipe(&crate::recipe::write_literal(b"}"));
        }
    }

    fn emit_encode_scalar(&self, ectx: &mut EmitCtx, offset: usize, scalar_type: ScalarType) {
        let fn_ptr: *const u8 = match scalar_type {
            ScalarType::U8 => json_intrinsics::kajit_json_write_u8 as _,
            ScalarType::U16 => json_intrinsics::kajit_json_write_u16 as _,
            ScalarType::U32 => json_intrinsics::kajit_json_write_u32 as _,
            ScalarType::U64 => json_intrinsics::kajit_json_write_u64 as _,
            ScalarType::U128 => json_intrinsics::kajit_json_write_u128 as _,
            ScalarType::USize => json_intrinsics::kajit_json_write_usize as _,
            ScalarType::I8 => json_intrinsics::kajit_json_write_i8 as _,
            ScalarType::I16 => json_intrinsics::kajit_json_write_i16 as _,
            ScalarType::I32 => json_intrinsics::kajit_json_write_i32 as _,
            ScalarType::I64 => json_intrinsics::kajit_json_write_i64 as _,
            ScalarType::I128 => json_intrinsics::kajit_json_write_i128 as _,
            ScalarType::ISize => json_intrinsics::kajit_json_write_isize as _,
            ScalarType::F32 => json_intrinsics::kajit_json_write_f32 as _,
            ScalarType::F64 => json_intrinsics::kajit_json_write_f64 as _,
            ScalarType::Bool => json_intrinsics::kajit_json_write_bool as _,
            ScalarType::Char => json_intrinsics::kajit_json_write_char as _,
            ScalarType::Unit => json_intrinsics::kajit_json_write_unit as _,
            _ => panic!("unsupported JSON scalar for encode: {:?}", scalar_type),
        };
        ectx.emit_enc_call_intrinsic_with_input(fn_ptr, offset as u32);
    }

    fn emit_encode_string(
        &self,
        ectx: &mut EmitCtx,
        offset: usize,
        _scalar_type: ScalarType,
        _string_offsets: &crate::malum::StringOffsets,
    ) {
        ectx.emit_enc_call_intrinsic_with_input(
            json_intrinsics::kajit_json_write_string as *const u8,
            offset as u32,
        );
    }
}
