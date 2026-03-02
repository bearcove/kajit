use dynasmrt::DynamicLabel;
use facet::{MapDef, ScalarType};

use crate::arch::{BASE_FRAME, EmitCtx};
use crate::format::{
    Decoder, Encoder, FieldEmitInfo, FieldEncodeInfo, FieldLowerInfo, IrDecoder, VariantEmitInfo,
};
use crate::intrinsics;
use crate::ir::{IntrinsicFn, IrOp, PortSource, RegionBuilder, SlotId};
use crate::malum::VecOffsets;
use crate::recipe::{self, Width};

// r[impl deser.postcard.struct]

/// Postcard wire format — fields in declaration order, varint-encoded integers,
/// length-prefixed strings.
pub struct KajitPostcard;

impl Decoder for KajitPostcard {
    fn extra_stack_space(&self, _fields: &[FieldEmitInfo]) -> u32 {
        // We need at least 8 bytes of scratch space on the stack for:
        // - varint slow path temp (used by emit_read_postcard_discriminant
        //   and emit_inline_postcard_string)
        8
    }

    fn supports_inline_nested(&self) -> bool {
        true
    }

    fn as_ir_decoder(&self) -> Option<&dyn IrDecoder> {
        Some(self)
    }

    fn emit_struct_fields(
        &self,
        ectx: &mut EmitCtx,
        fields: &[FieldEmitInfo],
        _deny_unknown_fields: bool,
        emit_field: &mut dyn FnMut(&mut EmitCtx, &FieldEmitInfo),
    ) {
        // Postcard: fields are serialized in declaration order, no keys.
        for field in fields {
            emit_field(ectx, field);
        }
    }

    // r[impl deser.postcard.scalar.varint]
    // r[impl deser.postcard.scalar.float]
    // r[impl deser.postcard.scalar.bool]
    fn emit_read_scalar(&self, ectx: &mut EmitCtx, offset: usize, scalar_type: ScalarType) {
        let offset = offset as u32;
        match scalar_type {
            // Tier 1: fixed-size, fully inlined
            ScalarType::U8 | ScalarType::I8 => ectx.emit_inline_read_byte(offset),
            ScalarType::Bool => ectx.emit_inline_read_bool(offset),
            ScalarType::F32 => ectx.emit_inline_read_f32(offset),
            ScalarType::F64 => ectx.emit_inline_read_f64(offset),

            // Tier 2: varint fast path + intrinsic slow path
            ScalarType::U16 => {
                ectx.emit_inline_varint_fast_path(offset, 2, false, intrinsics::kajit_read_u16 as _)
            }
            ScalarType::U32 => ectx.emit_inline_varint_fast_path(
                offset,
                4,
                false,
                intrinsics::kajit_read_varint_u32 as _,
            ),
            ScalarType::U64 => {
                ectx.emit_inline_varint_fast_path(offset, 8, false, intrinsics::kajit_read_u64 as _)
            }
            ScalarType::I16 => {
                ectx.emit_inline_varint_fast_path(offset, 2, true, intrinsics::kajit_read_i16 as _)
            }
            ScalarType::I32 => {
                ectx.emit_inline_varint_fast_path(offset, 4, true, intrinsics::kajit_read_i32 as _)
            }
            ScalarType::I64 => {
                ectx.emit_inline_varint_fast_path(offset, 8, true, intrinsics::kajit_read_i64 as _)
            }
            ScalarType::U128 => ectx.emit_call_intrinsic(intrinsics::kajit_read_u128 as _, offset),
            ScalarType::I128 => ectx.emit_call_intrinsic(intrinsics::kajit_read_i128 as _, offset),
            ScalarType::USize => {
                ectx.emit_call_intrinsic(intrinsics::kajit_read_usize as _, offset)
            }
            ScalarType::ISize => {
                ectx.emit_call_intrinsic(intrinsics::kajit_read_isize as _, offset)
            }
            ScalarType::Char => ectx.emit_call_intrinsic(intrinsics::kajit_read_char as _, offset),
            _ => panic!("unsupported postcard scalar: {:?}", scalar_type),
        }
    }

    fn emit_read_string(
        &self,
        ectx: &mut EmitCtx,
        offset: usize,
        scalar_type: ScalarType,
        string_offsets: &crate::malum::StringOffsets,
    ) {
        match scalar_type {
            ScalarType::String => {
                ectx.emit_inline_postcard_string_malum(
                    offset as u32,
                    string_offsets,
                    intrinsics::kajit_read_varint_u32 as *const u8,
                    intrinsics::kajit_string_validate_alloc_copy as *const u8,
                );
            }
            ScalarType::Str => {
                ectx.emit_call_intrinsic(
                    intrinsics::kajit_read_postcard_str as *const u8,
                    offset as u32,
                );
            }
            ScalarType::CowStr => {
                ectx.emit_call_intrinsic(
                    intrinsics::kajit_read_postcard_cow_str as *const u8,
                    offset as u32,
                );
            }
            _ => panic!("unsupported postcard string scalar: {:?}", scalar_type),
        }
    }

    // r[impl deser.postcard.option]
    fn emit_option(
        &self,
        ectx: &mut EmitCtx,
        offset: usize,
        init_none_fn: *const u8,
        init_some_fn: *const u8,
        scratch_offset: u32,
        emit_inner: &mut dyn FnMut(&mut EmitCtx),
    ) {
        // Postcard Option: 0x00 = None, 0x01 = Some(T)
        let none_label = ectx.new_label();
        let done_label = ectx.new_label();

        // Read tag byte from input into a stack slot (the save slot area).
        let tag_slot = scratch_offset - 8;
        ectx.emit_inline_read_byte_to_stack(tag_slot);

        // Compare: if tag == 0, branch to None
        ectx.emit_stack_byte_cmp_branch(tag_slot, 0, none_label);

        // === Some path ===
        // Redirect out to the scratch area and deserialize inner T
        ectx.emit_redirect_out_to_stack(scratch_offset);
        emit_inner(ectx);
        ectx.emit_restore_out(scratch_offset);

        // Call init_some(init_some_fn, out + offset, scratch)
        ectx.emit_call_option_init_some(
            intrinsics::kajit_option_init_some as *const u8,
            init_some_fn,
            offset as u32,
            scratch_offset,
        );
        ectx.emit_branch(done_label);

        // === None path ===
        ectx.bind_label(none_label);
        ectx.emit_call_option_init_none(
            intrinsics::kajit_option_init_none as *const u8,
            init_none_fn,
            offset as u32,
        );

        ectx.bind_label(done_label);
    }

    fn vec_extra_stack_space(&self) -> u32 {
        // varint scratch (8) + saved_out (8) + buf (8) + count (8) + cursor (8) + end (8) = 48
        48
    }

    fn map_extra_stack_space(&self) -> u32 {
        // Same layout as vec: saved_out (8) + buf (8) + count (8) + cursor (8) + end (8) = 48
        // Plus varint scratch (8) = 48 (shared with vec_extra_stack_space base)
        48
    }

    // r[impl deser.postcard.seq]
    // r[impl seq.malum.postcard]
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

        // Stack slot offsets (relative to sp)
        let saved_out_slot = BASE_FRAME + 8;
        let buf_slot = BASE_FRAME + 16;
        let count_slot = BASE_FRAME + 24;
        let save_x23_slot = BASE_FRAME + 32; // save/restore callee-saved x23 (cursor)
        let save_x24_slot = BASE_FRAME + 40; // save/restore callee-saved x24 (end)

        let empty_label = ectx.new_label();
        let done_label = ectx.new_label();
        let loop_done_label = ectx.new_label(); // loop finished → write Vec to output
        let loop_label = ectx.new_label();
        let error_cleanup = ectx.new_label();

        // Read element count (varint → w9)
        ectx.emit_read_postcard_discriminant(intrinsics::kajit_read_varint_u32 as *const u8);

        // r[impl seq.malum.empty]
        // If count == 0, write empty Vec and skip
        ectx.emit_cbz_count(empty_label);

        // Save count to stack before alloc call (caller-saved reg, clobbered by call)
        ectx.emit_save_count_to_stack(count_slot);

        // Allocate: kajit_vec_alloc(ctx, count, elem_size, elem_align)
        // count is in w9, result in x0
        ectx.emit_call_vec_alloc(
            intrinsics::kajit_vec_alloc as *const u8,
            elem_size,
            elem_align,
        );

        // Init loop: save x23/x24, set x23=cursor=buf, x24=end
        ectx.emit_vec_loop_init_cursor(
            saved_out_slot,
            buf_slot,
            count_slot,
            save_x23_slot,
            save_x24_slot,
            elem_size,
        );

        // === Loop body ===
        ectx.bind_label(loop_label);

        // Check if we can use the tight varint loop (writes directly to cursor
        // register, no mov to out, slow path out-of-line).
        let varint_info = elem_shape.scalar_type().and_then(|st| match st {
            ScalarType::U16 => Some((2, false, intrinsics::kajit_read_u16 as *const u8)),
            ScalarType::U32 => Some((4, false, intrinsics::kajit_read_varint_u32 as *const u8)),
            ScalarType::U64 => Some((8, false, intrinsics::kajit_read_u64 as *const u8)),
            ScalarType::I16 => Some((2, true, intrinsics::kajit_read_i16 as *const u8)),
            ScalarType::I32 => Some((4, true, intrinsics::kajit_read_i32 as *const u8)),
            ScalarType::I64 => Some((8, true, intrinsics::kajit_read_i64 as *const u8)),
            ScalarType::USize => Some((
                core::mem::size_of::<usize>() as u32,
                false,
                intrinsics::kajit_read_usize as *const u8,
            )),
            ScalarType::ISize => Some((
                core::mem::size_of::<isize>() as u32,
                true,
                intrinsics::kajit_read_isize as *const u8,
            )),
            _ => None,
        });

        if let Some((store_width, zigzag, intrinsic_fn_ptr)) = varint_info {
            // Tight varint loop: writes directly to cursor register,
            // no mov x21/r14, slow path placed out-of-line.
            ectx.emit_vec_varint_loop(
                store_width,
                zigzag,
                intrinsic_fn_ptr,
                elem_size,
                save_x24_slot,
                loop_label,
                loop_done_label,
                error_cleanup,
            );
        } else {
            // Generic path for non-varint elements (strings, structs, etc.)
            ectx.emit_vec_loop_load_cursor(save_x23_slot);

            // Redirect error_exit to error_cleanup during element emission.
            let saved_error_exit = ectx.error_exit;
            ectx.error_exit = error_cleanup;

            emit_elem(ectx);

            ectx.error_exit = saved_error_exit;

            ectx.emit_vec_loop_advance_no_error_check(save_x24_slot, elem_size, loop_label);
        }

        // === Write Vec to output (reached after loop completes) ===
        ectx.bind_label(loop_done_label);
        // For postcard, len == cap == count (exact allocation)
        ectx.emit_vec_restore_callee_saved(save_x23_slot, save_x24_slot);
        ectx.emit_vec_store(
            offset as u32,
            saved_out_slot,
            buf_slot,
            count_slot, // len = count
            count_slot, // cap = count
            vec_offsets,
        );
        ectx.emit_branch(done_label);

        // === Empty path (x23/x24 were never modified) ===
        ectx.bind_label(empty_label);
        ectx.emit_vec_store_empty_with_align(offset as u32, elem_align, vec_offsets);
        ectx.emit_branch(done_label);

        // === Error cleanup ===
        ectx.bind_label(error_cleanup);
        ectx.emit_vec_restore_callee_saved(save_x23_slot, save_x24_slot);
        ectx.emit_vec_error_cleanup(
            intrinsics::kajit_vec_free as *const u8,
            saved_out_slot,
            buf_slot,
            count_slot,
            elem_size,
            elem_align,
        );

        ectx.bind_label(done_label);
    }

    // r[impl deser.postcard.map]
    #[allow(clippy::too_many_arguments)]
    fn emit_map(
        &self,
        ectx: &mut EmitCtx,
        offset: usize,
        map_def: &'static MapDef,
        _k_shape: &'static facet::Shape,
        _v_shape: &'static facet::Shape,
        _k_label: Option<DynamicLabel>,
        _v_label: Option<DynamicLabel>,
        _option_scratch_offset: u32,
        emit_key: &mut dyn FnMut(&mut EmitCtx),
        emit_value: &mut dyn FnMut(&mut EmitCtx),
    ) {
        let from_pair_slice_fn = map_def
            .vtable
            .from_pair_slice
            .expect("MapVTable must have from_pair_slice for JIT deserialization")
            as *const u8;

        let pair_stride = map_def.vtable.pair_stride as u32;
        let value_offset = map_def.vtable.value_offset_in_pair as u32;

        // Compute pair alignment: max of key and value alignment.
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

        // Stack slot offsets (relative to sp). Same layout as postcard Vec.
        let saved_out_slot = BASE_FRAME + 8;
        let buf_slot = BASE_FRAME + 16;
        let count_slot = BASE_FRAME + 24;
        let save_x23_slot = BASE_FRAME + 32;
        let save_x24_slot = BASE_FRAME + 40;

        let empty_label = ectx.new_label();
        let done_label = ectx.new_label();
        let loop_done_label = ectx.new_label();
        let loop_label = ectx.new_label();
        let error_cleanup = ectx.new_label();

        // Read pair count (varint → w9/r10d)
        ectx.emit_read_postcard_discriminant(intrinsics::kajit_read_varint_u32 as *const u8);

        // If count == 0, emit empty map directly
        ectx.emit_cbz_count(empty_label);

        // Save count before alloc call (caller-saved reg, clobbered by call)
        ectx.emit_save_count_to_stack(count_slot);

        // Allocate pairs buffer: kajit_vec_alloc(ctx, count, pair_stride, pair_align)
        ectx.emit_call_vec_alloc(
            intrinsics::kajit_vec_alloc as *const u8,
            pair_stride,
            pair_align,
        );

        // Init cursor loop: save x23/x24, set x23=buf, x24=end
        ectx.emit_vec_loop_init_cursor(
            saved_out_slot,
            buf_slot,
            count_slot,
            save_x23_slot,
            save_x24_slot,
            pair_stride,
        );

        // === Loop body ===
        ectx.bind_label(loop_label);

        // Redirect errors to our cleanup label
        let saved_error_exit = ectx.error_exit;
        ectx.error_exit = error_cleanup;

        // Load cursor for key: out = x23 (pair_base)
        ectx.emit_vec_loop_load_cursor(save_x23_slot);

        // Emit key deserialization at offset 0 from out
        emit_key(ectx);

        // Advance out to value position: out = x23 + value_offset
        ectx.emit_advance_out_by(value_offset);

        // Emit value deserialization at offset 0 from out
        emit_value(ectx);

        ectx.error_exit = saved_error_exit;

        // Advance cursor by pair_stride; loop back if more pairs remain
        ectx.emit_vec_loop_advance_no_error_check(save_x24_slot, pair_stride, loop_label);

        // === Loop done: build the final map from the pairs buffer ===
        ectx.bind_label(loop_done_label);
        ectx.emit_vec_restore_callee_saved(save_x23_slot, save_x24_slot);
        // Call from_pair_slice(saved_out + offset, pairs_buf, count)
        // Note: offset is 0 when called as a standalone function (out already points to map slot)
        let _ = offset; // offset=0 always for standalone map function
        ectx.emit_call_map_from_pairs(from_pair_slice_fn, saved_out_slot, buf_slot, count_slot);
        // Free the pairs buffer (count == cap for postcard's exact allocation)
        ectx.emit_call_pairs_free(
            intrinsics::kajit_vec_free as *const u8,
            buf_slot,
            count_slot, // cap == count for postcard
            pair_stride,
            pair_align,
        );
        ectx.emit_branch(done_label);

        // === Empty path: build empty map without allocating a pairs buffer ===
        ectx.bind_label(empty_label);
        ectx.emit_call_map_from_pairs_empty(from_pair_slice_fn);
        ectx.emit_branch(done_label);

        // === Error cleanup: restore callee-saved, free pairs buffer, exit ===
        ectx.bind_label(error_cleanup);
        ectx.emit_vec_restore_callee_saved(save_x23_slot, save_x24_slot);
        ectx.emit_vec_error_cleanup(
            intrinsics::kajit_vec_free as *const u8,
            saved_out_slot,
            buf_slot,
            count_slot, // cap == count for postcard
            pair_stride,
            pair_align,
        );

        ectx.bind_label(done_label);
    }

    // r[impl deser.postcard.enum]
    // r[impl deser.postcard.enum.dispatch]
    // r[impl deser.postcard.enum.unit]
    fn emit_enum(
        &self,
        ectx: &mut EmitCtx,
        variants: &[VariantEmitInfo],
        emit_variant_body: &mut dyn FnMut(&mut EmitCtx, &VariantEmitInfo),
    ) {
        // Read varint discriminant into scratch register (w9 on aarch64, r10d on x64).
        // Uses kajit_read_varint_u32 as slow path intrinsic.
        ectx.emit_read_postcard_discriminant(intrinsics::kajit_read_varint_u32 as *const u8);

        let done_label = ectx.new_label();
        let variant_labels: Vec<_> = variants.iter().map(|_| ectx.new_label()).collect();

        // Linear compare chain: cmp discriminant, N; branch if equal to variant N.
        for (i, variant) in variants.iter().enumerate() {
            ectx.emit_cmp_imm_branch_eq(variant.index as u32, variant_labels[i]);
        }

        // Default: unknown discriminant.
        ectx.emit_unknown_variant_error();

        // Variant handlers.
        for (i, variant) in variants.iter().enumerate() {
            ectx.bind_label(variant_labels[i]);
            emit_variant_body(ectx, variant);
            ectx.emit_branch(done_label);
        }

        ectx.bind_label(done_label);
    }
}

// ── Encoder implementation ──────────────────────────────────────────

impl Encoder for KajitPostcard {
    fn supports_inline_nested(&self) -> bool {
        true
    }

    fn emit_encode_struct_fields(
        &self,
        ectx: &mut EmitCtx,
        fields: &[FieldEncodeInfo],
        emit_field: &mut dyn FnMut(&mut EmitCtx, &FieldEncodeInfo),
    ) {
        // Postcard: fields in declaration order, no framing.
        for field in fields {
            emit_field(ectx, field);
        }
    }

    fn emit_encode_scalar(&self, ectx: &mut EmitCtx, offset: usize, scalar_type: ScalarType) {
        let offset = offset as u32;
        match scalar_type {
            // Fixed-size: raw bytes, no varint.
            ScalarType::U8 | ScalarType::I8 | ScalarType::Bool => {
                ectx.emit_recipe(&recipe::encode_raw(offset, Width::W1));
            }
            ScalarType::F32 => {
                ectx.emit_recipe(&recipe::encode_raw(offset, Width::W4));
            }
            ScalarType::F64 => {
                ectx.emit_recipe(&recipe::encode_raw(offset, Width::W8));
            }

            // Varint-encoded unsigned integers.
            ScalarType::U16 => {
                ectx.emit_recipe(&recipe::encode_varint(offset, Width::W2, false));
            }
            ScalarType::U32 => {
                ectx.emit_recipe(&recipe::encode_varint(offset, Width::W4, false));
            }
            ScalarType::U64 => {
                ectx.emit_recipe(&recipe::encode_varint(offset, Width::W8, false));
            }
            ScalarType::USize => {
                ectx.emit_recipe(&recipe::encode_varint(offset, Width::W8, false));
            }

            // Zigzag + varint signed integers.
            ScalarType::I16 => {
                ectx.emit_recipe(&recipe::encode_varint(offset, Width::W2, true));
            }
            ScalarType::I32 => {
                ectx.emit_recipe(&recipe::encode_varint(offset, Width::W4, true));
            }
            ScalarType::I64 => {
                ectx.emit_recipe(&recipe::encode_varint(offset, Width::W8, true));
            }
            ScalarType::ISize => {
                ectx.emit_recipe(&recipe::encode_varint(offset, Width::W8, true));
            }

            // Multi-byte varints — intrinsic calls.
            ScalarType::U128 => {
                ectx.emit_enc_call_intrinsic_with_input(
                    intrinsics::kajit_encode_u128 as *const u8,
                    offset,
                );
            }
            ScalarType::I128 => {
                ectx.emit_enc_call_intrinsic_with_input(
                    intrinsics::kajit_encode_i128 as *const u8,
                    offset,
                );
            }
            ScalarType::Char => {
                ectx.emit_enc_call_intrinsic_with_input(
                    intrinsics::kajit_encode_char as *const u8,
                    offset,
                );
            }

            _ => panic!("unsupported postcard encode scalar: {:?}", scalar_type),
        }
    }

    fn emit_encode_string(
        &self,
        ectx: &mut EmitCtx,
        offset: usize,
        _scalar_type: ScalarType,
        _string_offsets: &crate::malum::StringOffsets,
    ) {
        // All string-like types (String, &str, Cow<str>) share the same (ptr, len)
        // layout. The intrinsic reads ptr and len using discovered offsets.
        ectx.emit_enc_call_intrinsic_with_input(
            intrinsics::kajit_encode_postcard_string as *const u8,
            offset as u32,
        );
    }
}

// =============================================================================
// IR lowering — produce RVSDG nodes instead of machine code
// =============================================================================

/// Wrap a function pointer (as `*const u8`) into an `IntrinsicFn`.
fn ifn(f: *const u8) -> IntrinsicFn {
    IntrinsicFn(f as usize)
}

fn lower_nonzero_to_bool(builder: &mut RegionBuilder<'_>, value: PortSource) -> PortSource {
    let zero = builder.const_val(0);
    builder.binop(IrOp::CmpNe, value, zero)
}

fn lower_error_if_nonzero(
    builder: &mut RegionBuilder<'_>,
    value: PortSource,
    code: crate::context::ErrorCode,
) {
    let cond = lower_nonzero_to_bool(builder, value);
    builder.gamma(cond, &[], 2, |branch_idx, rb| {
        if branch_idx == 1 {
            rb.error_exit(code);
        }
        rb.set_results(&[]);
    });
}

// r[impl ir.intrinsics.representable-inline]
// r[impl ir.varint.lowering]
// r[impl ir.varint.errors]
// r[impl deser.postcard.scalar.varint.no-rust-call]
fn lower_postcard_varint_value(builder: &mut RegionBuilder<'_>, bits: u32) -> PortSource {
    let max_bytes = match bits {
        16 => 3_u64,
        32 => 5_u64,
        64 => 10_u64,
        _ => panic!("unsupported varint bit width: {bits}"),
    };
    // One-byte fast path: consume one byte, and only enter theta when continuation is set.
    builder.bounds_check(1);
    let first_byte = builder.read_bytes(1);
    let mask_7f = builder.const_val(0x7f);
    let first_low = builder.binop(IrOp::And, first_byte, mask_7f);
    let mask_80 = builder.const_val(0x80);
    let first_cont = builder.binop(IrOp::And, first_byte, mask_80);
    let first_cont_bool = lower_nonzero_to_bool(builder, first_cont);

    // Outputs: [value, rem, last_low, last_cont]
    let out = builder.gamma(first_cont_bool, &[first_low], 2, |branch_idx, rb| {
        let args = rb.region_args(1);
        let low0 = args[0];

        if branch_idx == 0 {
            let rem0 = rb.const_val(max_bytes - 1);
            let zero = rb.const_val(0);
            rb.set_results(&[low0, rem0, low0, zero]);
            return;
        }

        let shift0 = rb.const_val(7);
        let rem0 = rb.const_val(max_bytes - 1);
        let one = rb.const_val(1);
        let loop_out = rb.theta(&[low0, shift0, rem0, low0, one], |tb| {
            let loop_args = tb.region_args(5);
            let acc = loop_args[0];
            let shift = loop_args[1];
            let rem = loop_args[2];

            tb.bounds_check(1);
            let byte = tb.read_bytes(1);

            let mask_7f = tb.const_val(0x7f);
            let low = tb.binop(IrOp::And, byte, mask_7f);
            let part = tb.binop(IrOp::Shl, low, shift);
            let acc2 = tb.binop(IrOp::Or, acc, part);
            let seven = tb.const_val(7);
            let shift2 = tb.binop(IrOp::Add, shift, seven);
            let one = tb.const_val(1);
            let rem2 = tb.binop(IrOp::Sub, rem, one);

            let mask_80 = tb.const_val(0x80);
            let cont = tb.binop(IrOp::And, byte, mask_80);
            let cont_bool = lower_nonzero_to_bool(tb, cont);
            let rem_bool = lower_nonzero_to_bool(tb, rem2);
            let predicate = tb.binop(IrOp::And, cont_bool, rem_bool);

            tb.set_results(&[predicate, acc2, shift2, rem2, low, cont_bool]);
        });

        let value = loop_out[0];
        let rem = loop_out[2];
        let last_low = loop_out[3];
        let last_cont = loop_out[4];
        rb.set_results(&[value, rem, last_low, last_cont]);
    });

    let value = out[0];
    let rem = out[1];
    let last_low = out[2];
    let last_cont = out[3];

    // Continued after exhausting byte budget => malformed varint.
    lower_error_if_nonzero(builder, last_cont, crate::context::ErrorCode::InvalidVarint);

    if bits == 64 {
        // 10th byte may only carry one payload bit for u64.
        let zero = builder.const_val(0);
        let rem_nonzero = builder.binop(IrOp::CmpNe, rem, zero);
        builder.gamma(rem_nonzero, &[last_low], 2, |branch_idx, rb| {
            let args = rb.region_args(1);
            if branch_idx == 0 {
                let mask_7e = rb.const_val(0x7e);
                let extra = rb.binop(IrOp::And, args[0], mask_7e);
                lower_error_if_nonzero(rb, extra, crate::context::ErrorCode::InvalidVarint);
            }
            rb.set_results(&[]);
        });
    } else {
        let bit_width = builder.const_val(bits as u64);
        let upper = builder.binop(IrOp::Shr, value, bit_width);
        lower_error_if_nonzero(builder, upper, crate::context::ErrorCode::NumberOutOfRange);
    }

    value
}

fn lower_postcard_varint_scalar(
    builder: &mut RegionBuilder<'_>,
    offset: u32,
    width: crate::ir::Width,
    zigzag: bool,
    bits: u32,
) {
    let mut value = lower_postcard_varint_value(builder, bits);
    if zigzag {
        value = builder.unary(IrOp::ZigzagDecode { wide: bits == 64 }, value);
    }
    builder.write_to_field(value, offset, width);
}

impl IrDecoder for KajitPostcard {
    // r[impl ir.format-trait.lower]

    fn lower_struct_fields(
        &self,
        builder: &mut RegionBuilder<'_>,
        fields: &[FieldLowerInfo],
        _deny_unknown_fields: bool,
        lower_field: &mut dyn FnMut(&mut RegionBuilder<'_>, &FieldLowerInfo),
    ) {
        // Postcard: fields in declaration order, no key dispatch.
        for field in fields {
            lower_field(builder, field);
        }
    }

    fn lower_read_scalar(
        &self,
        builder: &mut RegionBuilder<'_>,
        offset: usize,
        scalar_type: ScalarType,
    ) {
        let offset = offset as u32;

        match scalar_type {
            ScalarType::U16 => {
                lower_postcard_varint_scalar(builder, offset, crate::ir::Width::W2, false, 16);
            }
            ScalarType::U32 => {
                lower_postcard_varint_scalar(builder, offset, crate::ir::Width::W4, false, 32);
            }
            ScalarType::U64 => {
                lower_postcard_varint_scalar(builder, offset, crate::ir::Width::W8, false, 64);
            }
            ScalarType::I16 => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_i16 as _), &[], offset, false);
            }
            ScalarType::I32 => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_i32 as _), &[], offset, false);
            }
            ScalarType::I64 => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_i64 as _), &[], offset, false);
            }

            ScalarType::Bool => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_bool as _), &[], offset, false);
            }
            ScalarType::U8 => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_u8 as _), &[], offset, false);
            }
            ScalarType::I8 => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_i8 as _), &[], offset, false);
            }
            ScalarType::F32 => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_f32 as _), &[], offset, false);
            }
            ScalarType::F64 => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_f64 as _), &[], offset, false);
            }
            ScalarType::U128 => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_u128 as _), &[], offset, false);
            }
            ScalarType::I128 => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_i128 as _), &[], offset, false);
            }
            ScalarType::USize => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_usize as _), &[], offset, false);
            }
            ScalarType::ISize => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_isize as _), &[], offset, false);
            }
            ScalarType::Char => {
                builder.call_intrinsic(ifn(intrinsics::kajit_read_char as _), &[], offset, false);
            }
            _ => panic!("unsupported postcard scalar: {:?}", scalar_type),
        }
    }

    fn lower_read_string(
        &self,
        builder: &mut RegionBuilder<'_>,
        offset: usize,
        scalar_type: ScalarType,
    ) {
        let offset = offset as u32;
        let len = lower_postcard_varint_value(builder, 32);

        let func = match scalar_type {
            ScalarType::String => ifn(intrinsics::kajit_read_postcard_string_with_len as _),
            ScalarType::Str => ifn(intrinsics::kajit_read_postcard_str_with_len as _),
            ScalarType::CowStr => ifn(intrinsics::kajit_read_postcard_cow_str_with_len as _),
            _ => panic!("unsupported postcard string scalar: {:?}", scalar_type),
        };

        builder.call_intrinsic(func, &[len], offset, false);
    }

    // r[impl deser.postcard.option]
    fn lower_option(
        &self,
        builder: &mut RegionBuilder<'_>,
        offset: usize,
        init_none_fn: *const u8,
        init_some_fn: *const u8,
        scratch_slot: SlotId,
        lower_inner: &mut dyn FnMut(&mut RegionBuilder<'_>),
    ) {
        let offset = offset as u32;

        // Read the tag byte (0x00 = None, 0x01 = Some).
        builder.bounds_check(1);
        let tag = builder.read_bytes(1);

        // Gamma: branch on tag value.
        // Branch 0 = None, Branch 1 = Some.
        builder.gamma(tag, &[], 2, |branch_idx, rb| {
            match branch_idx {
                0 => {
                    // None path: call init_none(init_none_fn, out + offset).
                    let init_fn = rb.const_val(init_none_fn as u64);
                    rb.call_intrinsic(
                        ifn(intrinsics::kajit_option_init_none_ctx as _),
                        &[init_fn],
                        offset,
                        false,
                    );
                    rb.set_results(&[]);
                }
                1 => {
                    // Some path: deserialize inner T into stack scratch, then init_some.
                    let saved_out = rb.save_out_ptr();
                    let scratch_ptr = rb.slot_addr(scratch_slot);
                    rb.set_out_ptr(scratch_ptr);
                    lower_inner(rb);
                    rb.set_out_ptr(saved_out);
                    let init_fn = rb.const_val(init_some_fn as u64);
                    rb.call_intrinsic(
                        ifn(intrinsics::kajit_option_init_some_ctx as _),
                        &[init_fn, scratch_ptr],
                        offset,
                        false,
                    );
                    rb.set_results(&[]);
                }
                _ => unreachable!(),
            }
        });
    }

    // r[impl deser.postcard.enum]
    // r[impl deser.postcard.enum.dispatch]
    fn lower_enum(
        &self,
        builder: &mut RegionBuilder<'_>,
        variants: &[crate::format::VariantLowerInfo],
        lower_variant_body: &mut dyn FnMut(
            &mut RegionBuilder<'_>,
            &crate::format::VariantLowerInfo,
        ),
    ) {
        // Read varint discriminant.
        let discriminant = lower_postcard_varint_value(builder, 32);

        let n = variants.len();

        // Clamp discriminant to [0, n] — any value >= n maps to the error branch.
        // We do: clamped = if discriminant >= n { n } else { discriminant }
        // This is: clamped = min(discriminant, n)
        // Using: threshold = const(n), ge = (discriminant >= threshold)
        // Then gamma on ge: branch 0 = discriminant is valid, branch 1 = error
        //
        // Actually, simpler: gamma with n+1 branches. Predicate = discriminant.
        // If predicate >= branch_count, RVSDG behavior is undefined, so we need
        // to clamp. Let's use a two-level approach:
        //   1. Compare discriminant >= n, get a 0/1 flag
        //   2. Outer gamma on flag: branch 0 = valid (inner gamma on discriminant),
        //      branch 1 = error
        //
        // This is clean but adds nesting. For now, let's just use n+1 branches
        // and add a bounds check node that clamps the predicate. We can add a
        // Clamp op or just handle it: if discriminant >= n, branch to n (error).

        // For now: use a simple approach with a two-level gamma.
        // Level 1: check if discriminant < n
        // Gamma with n+1 branches: branches 0..n-1 are variant bodies,
        // branch n is the error path for unknown discriminants.
        // The backend/linearizer handles out-of-bounds predicate clamping
        // (e.g., bounds check before jump table).
        builder.gamma(discriminant, &[], n + 1, |branch_idx, rb| {
            if branch_idx < n {
                lower_variant_body(rb, &variants[branch_idx]);
                rb.set_results(&[]);
            } else {
                // Error branch: unknown variant.
                rb.error_exit(crate::context::ErrorCode::UnknownVariant);
                rb.set_results(&[]);
            }
        });
    }

    // r[impl deser.postcard.seq]
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

        // Read element count (varint).
        let count = lower_postcard_varint_value(builder, 32);

        // Branch on count: 0 => empty Vec, nonzero => allocate and fill.
        builder.gamma(count, &[], 2, |branch_idx, rb| {
            match branch_idx {
                0 => {
                    // Empty Vec: ptr = dangling(elem_align), len=0, cap=0.
                    let ptr = rb.const_val(elem_align);
                    let zero = rb.const_val(0);
                    rb.write_to_field(ptr, offset + vec_offsets.ptr_offset, usize_width);
                    rb.write_to_field(zero, offset + vec_offsets.len_offset, usize_width);
                    rb.write_to_field(zero, offset + vec_offsets.cap_offset, usize_width);
                    rb.set_results(&[]);
                }
                1 => {
                    // Allocate buffer: kajit_vec_alloc(ctx, count, elem_size, elem_align) → buf ptr.
                    let size_arg = rb.const_val(elem_size);
                    let align_arg = rb.const_val(elem_align);
                    let buf = rb
                        .call_intrinsic(
                            ifn(intrinsics::kajit_vec_alloc as _),
                            &[count, size_arg, align_arg],
                            0,
                            true,
                        )
                        .expect("vec_alloc should produce a result");

                    // Redirect out pointer to the element cursor while lowering elements.
                    let saved_out = rb.save_out_ptr();
                    rb.set_out_ptr(buf);

                    // Loop: theta over count elements.
                    // Loop vars: [counter, cursor]
                    let _results = rb.theta(&[count, buf], |inner_rb| {
                        let args = inner_rb.region_args(2);
                        let counter = args[0];
                        let cursor = args[1];

                        inner_rb.set_out_ptr(cursor);
                        lower_elem(inner_rb);

                        let one = inner_rb.const_val(1);
                        let new_counter = inner_rb.binop(crate::ir::IrOp::Sub, counter, one);

                        let stride = inner_rb.const_val(elem_size);
                        let new_cursor = inner_rb.binop(crate::ir::IrOp::Add, cursor, stride);

                        // Predicate: new_counter != 0 → continue.
                        inner_rb.set_results(&[new_counter, new_counter, new_cursor]);
                    });

                    // Restore outer out pointer and write Vec fields.
                    rb.set_out_ptr(saved_out);
                    rb.write_to_field(buf, offset + vec_offsets.ptr_offset, usize_width);
                    rb.write_to_field(count, offset + vec_offsets.len_offset, usize_width);
                    rb.write_to_field(count, offset + vec_offsets.cap_offset, usize_width);
                    rb.set_results(&[]);
                }
                _ => unreachable!(),
            }
        });
    }
}

#[cfg(test)]
mod ir_tests {
    use super::*;
    use crate::format::FieldLowerInfo;
    use crate::ir::{IrBuilder, IrOp, NodeKind};

    fn test_shape() -> &'static facet::Shape {
        <u8 as facet::Facet>::SHAPE
    }

    /// Build IR for a flat struct with the given fields using postcard lowering.
    fn build_postcard_struct(
        fields: &[FieldLowerInfo],
        lower_field: &mut dyn FnMut(&mut RegionBuilder<'_>, &FieldLowerInfo),
    ) -> crate::ir::IrFunc {
        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            KajitPostcard.lower_struct_fields(&mut rb, fields, false, lower_field);
            rb.set_results(&[]);
        }
        builder.finish()
    }

    #[test]
    fn postcard_ir_scalar_bool() {
        let fields = vec![FieldLowerInfo {
            offset: 0,
            shape: <bool as facet::Facet>::SHAPE,
            name: "flag",
            required_index: 0,
            has_default: false,
        }];

        let func = build_postcard_struct(&fields, &mut |builder, field| {
            KajitPostcard.lower_read_scalar(builder, field.offset, ScalarType::Bool);
        });

        // One field → one CallIntrinsic node in the root region body.
        let body = func.root_body();
        let nodes = &func.regions[body].nodes;
        assert_eq!(nodes.len(), 1);

        let node = &func.nodes[nodes[0]];
        match &node.kind {
            NodeKind::Simple(IrOp::CallIntrinsic {
                func: f,
                field_offset,
                has_result,
                ..
            }) => {
                assert_eq!(f.0, intrinsics::kajit_read_bool as *const () as usize);
                assert_eq!(*field_offset, 0);
                assert!(!has_result);
            }
            other => panic!("expected CallIntrinsic, got {:?}", other),
        }
    }

    #[test]
    fn postcard_ir_flat_struct_two_fields() {
        let fields = vec![
            FieldLowerInfo {
                offset: 0,
                shape: <u32 as facet::Facet>::SHAPE,
                name: "x",
                required_index: 0,
                has_default: false,
            },
            FieldLowerInfo {
                offset: 4,
                shape: <u8 as facet::Facet>::SHAPE,
                name: "y",
                required_index: 1,
                has_default: false,
            },
        ];

        let func = build_postcard_struct(&fields, &mut |builder, field| match field.name {
            "x" => KajitPostcard.lower_read_scalar(builder, field.offset, ScalarType::U32),
            "y" => KajitPostcard.lower_read_scalar(builder, field.offset, ScalarType::U8),
            _ => panic!("unexpected field"),
        });

        let mut has_u32_intrinsic = false;
        let mut has_u8_intrinsic = false;
        let mut has_theta = false;

        for (_, node) in func.nodes.iter() {
            match &node.kind {
                NodeKind::Simple(IrOp::CallIntrinsic {
                    func: f,
                    field_offset,
                    ..
                }) => {
                    if f.0 == intrinsics::kajit_read_varint_u32 as *const () as usize
                        && *field_offset == 0
                    {
                        has_u32_intrinsic = true;
                    }
                    if f.0 == intrinsics::kajit_read_u8 as *const () as usize && *field_offset == 4
                    {
                        has_u8_intrinsic = true;
                    }
                }
                NodeKind::Theta { .. } => has_theta = true,
                _ => {}
            }
        }

        assert!(
            !has_u32_intrinsic,
            "u32 varint should lower to core IR, not CallIntrinsic"
        );
        assert!(has_u8_intrinsic, "u8 should still use intrinsic path");
        assert!(has_theta, "u32 varint lowering should include a Theta loop");
    }

    #[test]
    fn postcard_ir_string() {
        let fields = vec![FieldLowerInfo {
            offset: 0,
            shape: <String as facet::Facet>::SHAPE,
            name: "name",
            required_index: 0,
            has_default: false,
        }];

        let func = build_postcard_struct(&fields, &mut |builder, field| {
            KajitPostcard.lower_read_string(builder, field.offset, ScalarType::String);
        });

        let has_varint_intrinsic = func.nodes.iter().any(|(_, n)| {
            matches!(
                &n.kind,
                NodeKind::Simple(IrOp::CallIntrinsic { func: f, .. })
                    if f.0 == intrinsics::kajit_read_varint_u32 as *const () as usize
            )
        });
        assert!(
            !has_varint_intrinsic,
            "string length should not use varint intrinsic in IR path"
        );

        let has_string_intrinsic = func.nodes.iter().any(|(_, n)| {
            matches!(
                &n.kind,
                NodeKind::Simple(IrOp::CallIntrinsic { func: f, .. })
                    if f.0 == intrinsics::kajit_read_postcard_string_with_len as *const () as usize
            )
        });
        assert!(
            has_string_intrinsic,
            "string lowering should call with-len string intrinsic"
        );
    }

    #[test]
    fn postcard_ir_state_threading() {
        // Two sequential scalar reads should have chained state edges.
        let fields = vec![
            FieldLowerInfo {
                offset: 0,
                shape: <u8 as facet::Facet>::SHAPE,
                name: "a",
                required_index: 0,
                has_default: false,
            },
            FieldLowerInfo {
                offset: 1,
                shape: <u8 as facet::Facet>::SHAPE,
                name: "b",
                required_index: 1,
                has_default: false,
            },
        ];

        let func = build_postcard_struct(&fields, &mut |builder, field| {
            KajitPostcard.lower_read_scalar(builder, field.offset, ScalarType::U8);
        });

        let body = func.root_body();
        let nodes = &func.regions[body].nodes;
        assert_eq!(nodes.len(), 2);

        // The second node's cursor-state input should reference the first node's
        // cursor-state output — verifying state threading works through lowering.
        let node1 = &func.nodes[nodes[1]];
        let cursor_input = node1
            .inputs
            .iter()
            .find(|i| i.kind == crate::ir::PortKind::StateCursor)
            .expect("second node should have cursor state input");

        // The source should be an OutputRef pointing at nodes[0].
        match cursor_input.source {
            crate::ir::PortSource::Node(crate::ir::OutputRef { node, .. }) => {
                assert_eq!(node, nodes[0], "cursor state should chain from first node");
            }
            other => panic!("expected Node source, got {:?}", other),
        }
    }

    #[test]
    fn postcard_ir_display() {
        // Verify the Display output for a simple struct lowering.
        let fields = vec![FieldLowerInfo {
            offset: 0,
            shape: <bool as facet::Facet>::SHAPE,
            name: "flag",
            required_index: 0,
            has_default: false,
        }];

        let func = build_postcard_struct(&fields, &mut |builder, field| {
            KajitPostcard.lower_read_scalar(builder, field.offset, ScalarType::Bool);
        });

        let display = format!("{}", func);
        // Should contain a CallIntrinsic node with the function address.
        assert!(
            display.contains("CallIntrinsic"),
            "display should show CallIntrinsic: {}",
            display
        );
    }

    #[test]
    fn postcard_ir_option() {
        // Option lowering should produce: bounds_check + read_bytes + gamma(2 branches).
        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            let none_fn = intrinsics::kajit_option_init_none as *const u8;
            let some_fn = intrinsics::kajit_option_init_some as *const u8;
            let scratch = rb.alloc_slot();
            KajitPostcard.lower_option(&mut rb, 0, none_fn, some_fn, scratch, &mut |rb| {
                // Inner: read a u8.
                KajitPostcard.lower_read_scalar(rb, 0, ScalarType::U8);
            });
            rb.set_results(&[]);
        }
        let func = builder.finish();

        let body = func.root_body();
        let nodes = &func.regions[body].nodes;

        // Should have: BoundsCheck, ReadBytes, Gamma.
        assert_eq!(
            nodes.len(),
            3,
            "option should produce 3 nodes in root region"
        );

        match &func.nodes[nodes[0]].kind {
            NodeKind::Simple(IrOp::BoundsCheck { count: 1 }) => {}
            other => panic!("expected BoundsCheck(1), got {:?}", other),
        }

        match &func.nodes[nodes[1]].kind {
            NodeKind::Simple(IrOp::ReadBytes { count: 1 }) => {}
            other => panic!("expected ReadBytes(1), got {:?}", other),
        }

        match &func.nodes[nodes[2]].kind {
            NodeKind::Gamma { regions } => {
                assert_eq!(regions.len(), 2, "option gamma should have 2 branches");

                // Branch 0 (None): should have a Const + CallIntrinsic (init_none).
                let none_nodes = &func.regions[regions[0]].nodes;
                assert_eq!(none_nodes.len(), 2, "None branch: const + call_intrinsic");

                // Branch 1 (Some): save/redirect out ptr, read_u8 in scratch,
                // restore out ptr, then init_some.
                let some_nodes = &func.regions[regions[1]].nodes;
                assert_eq!(
                    some_nodes.len(),
                    7,
                    "Some branch: save/slot_addr/set_out/read_u8/restore/const/call"
                );
            }
            other => panic!("expected Gamma, got {:?}", other),
        }
    }

    #[test]
    fn postcard_ir_enum() {
        use crate::format::{VariantKind, VariantLowerInfo};

        // Enum with 2 variants should produce varint decode core-IR + Gamma(3 branches).
        let variants = vec![
            VariantLowerInfo {
                index: 0,
                name: "A",
                rust_discriminant: 0,
                fields: vec![],
                kind: VariantKind::Unit,
            },
            VariantLowerInfo {
                index: 1,
                name: "B",
                rust_discriminant: 1,
                fields: vec![FieldLowerInfo {
                    offset: 0,
                    shape: <u8 as facet::Facet>::SHAPE,
                    name: "0",
                    required_index: 0,
                    has_default: false,
                }],
                kind: VariantKind::Tuple,
            },
        ];

        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            KajitPostcard.lower_enum(&mut rb, &variants, &mut |rb, variant| {
                // For variant B, read a u8 field.
                for field in &variant.fields {
                    KajitPostcard.lower_read_scalar(rb, field.offset, ScalarType::U8);
                }
            });
            rb.set_results(&[]);
        }
        let func = builder.finish();

        let body = func.root_body();
        let nodes = &func.regions[body].nodes;

        assert!(
            nodes.len() >= 2,
            "enum should produce decode nodes + dispatch gamma"
        );

        let has_varint_intrinsic = func.nodes.iter().any(|(_, n)| {
            matches!(
                &n.kind,
                NodeKind::Simple(IrOp::CallIntrinsic { func: f, .. })
                    if f.0 == intrinsics::kajit_read_varint_u32 as *const () as usize
            )
        });
        assert!(
            !has_varint_intrinsic,
            "enum discriminant should not use varint intrinsic in IR path"
        );

        let dispatch_regions = func.nodes.iter().find_map(|(_, n)| match &n.kind {
            NodeKind::Gamma { regions } if regions.len() == 3 => Some(regions.clone()),
            _ => None,
        });
        let regions = dispatch_regions.expect("enum should have 3-way dispatch gamma");

        // Branch 0 (variant A, unit): no nodes.
        assert_eq!(
            func.regions[regions[0]].nodes.len(),
            0,
            "unit variant should have no nodes"
        );

        // Branch 1 (variant B, tuple with u8): one CallIntrinsic.
        assert_eq!(
            func.regions[regions[1]].nodes.len(),
            1,
            "tuple variant should have 1 node"
        );

        // Branch 2 (error): one ErrorExit.
        let err_nodes = &func.regions[regions[2]].nodes;
        assert_eq!(err_nodes.len(), 1, "error branch should have 1 node");
        match &func.nodes[err_nodes[0]].kind {
            NodeKind::Simple(IrOp::ErrorExit { code }) => {
                assert_eq!(*code, crate::context::ErrorCode::UnknownVariant);
            }
            other => panic!("expected ErrorExit, got {:?}", other),
        }
    }

    #[test]
    fn postcard_ir_vec() {
        // Vec lowering should produce: CallIntrinsic (count) + Const + Const +
        // CallIntrinsic (alloc) + Const + Theta.
        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            let list_shape = <Vec<u8> as facet::Facet>::SHAPE;
            let list_def = match &list_shape.def {
                facet::Def::List(def) => def,
                other => panic!("expected List def, got {other:?}"),
            };
            let vec_offsets = crate::malum::discover_vec_offsets(list_def, list_shape);
            KajitPostcard.lower_vec(
                &mut rb,
                0,
                <u8 as facet::Facet>::SHAPE,
                &vec_offsets,
                &mut |rb| {
                    KajitPostcard.lower_read_scalar(rb, 0, ScalarType::U8);
                },
            );
            rb.set_results(&[]);
        }
        let func = builder.finish();

        // Theta now lives in the non-empty gamma branch body.
        let has_theta = func
            .nodes
            .iter()
            .any(|(_, n)| matches!(&n.kind, NodeKind::Theta { .. }));
        assert!(has_theta, "vec lowering should produce a theta node");

        // Vec count decode should use core IR, not varint intrinsic.
        let has_count_intrinsic = func.nodes.iter().any(|(_, n)| {
            matches!(
                &n.kind,
                NodeKind::Simple(IrOp::CallIntrinsic {
                    func: f,
                    ..
                }) if f.0 == intrinsics::kajit_read_varint_u32 as *const () as usize
            )
        });
        assert!(
            !has_count_intrinsic,
            "vec count should not use varint intrinsic in IR path"
        );

        // Find the alloc call.
        let has_alloc = func.nodes.iter().any(|(_, n)| {
            matches!(
                &n.kind,
                NodeKind::Simple(IrOp::CallIntrinsic {
                    func: f,
                    has_result: true,
                    ..
                }) if f.0 == intrinsics::kajit_vec_alloc as *const () as usize
            )
        });
        assert!(has_alloc, "vec should call kajit_vec_alloc");

        // Verify theta body has the element read.
        let theta_node = func
            .nodes
            .iter()
            .find_map(|(nid, n)| matches!(&n.kind, NodeKind::Theta { .. }).then_some(nid))
            .unwrap();
        if let NodeKind::Theta { body: theta_body } = &func.nodes[theta_node].kind {
            let theta_nodes = &func.regions[*theta_body].nodes;
            // Should have: CallIntrinsic (read u8) + Const(1) + Sub + Const(elem_size) + Add.
            assert!(
                theta_nodes.len() >= 3,
                "theta body should have element read + counter decrement + cursor advance"
            );
        }
    }
}
