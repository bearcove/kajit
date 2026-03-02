use facet::ScalarType;

use crate::arch::EmitCtx;
use crate::context::ErrorCode;
use crate::format::{Decoder, Encoder, FieldEncodeInfo, FieldLowerInfo};
use crate::intrinsics;
use crate::ir::{IntrinsicFn, RegionBuilder};
use crate::json_intrinsics;

// r[impl deser.json.struct]
// r[impl deser.json.struct.unknown-keys]

fn ifn(f: *const u8) -> IntrinsicFn {
    IntrinsicFn(f as usize)
}

/// JSON wire format — key-value pairs, linear key dispatch.
pub struct KajitJson;

impl Decoder for KajitJson {
    fn supports_trusted_utf8_input(&self) -> bool {
        true
    }

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

// ── JSON Encoder// ── JSON Encoder ────────────────────────────────────────────────────

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
