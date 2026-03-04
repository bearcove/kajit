use kajit_emit::aarch64::{self, Width as A64Width};
use kajit_emit::x64::{self, Emitter as X64Emitter};
use proptest::prelude::*;

fn sign_extend_u32(value: u32, bits: u8) -> i32 {
    let shift = 32 - bits;
    ((value << shift) as i32) >> shift
}

proptest! {
    #[test]
    fn aarch64_add_reg_fields_roundtrip(
        is_64 in any::<bool>(),
        rd in 0u8..32,
        rn in 0u8..32,
        rm in 0u8..32,
    ) {
        let width = if is_64 { A64Width::X64 } else { A64Width::W32 };
        let word = aarch64::encode_add_reg(width, rd, rn, rm).unwrap();
        prop_assert_eq!(word & 0x1f, rd as u32);
        prop_assert_eq!((word >> 5) & 0x1f, rn as u32);
        prop_assert_eq!((word >> 16) & 0x1f, rm as u32);
        prop_assert_eq!((word >> 31) & 1, if is_64 { 1 } else { 0 });
    }

    #[test]
    fn aarch64_branch_imm_roundtrip(imm in -33_554_432i32..33_554_432i32) {
        let word = aarch64::encode_b(imm).unwrap();
        let decoded = sign_extend_u32(word & 0x03ff_ffff, 26);
        prop_assert_eq!(decoded, imm);
    }

    #[test]
    fn aarch64_cbz_imm_roundtrip(
        is_64 in any::<bool>(),
        rt in 0u8..32,
        imm in -262_144i32..262_144i32,
    ) {
        let width = if is_64 { A64Width::X64 } else { A64Width::W32 };
        let word = aarch64::encode_cbz(width, rt, imm).unwrap();
        let decoded = sign_extend_u32((word >> 5) & 0x7ffff, 19);
        prop_assert_eq!(decoded, imm);
    }

    #[test]
    fn x64_call_rel32_roundtrip(disp in any::<i32>()) {
        let mut buf = Vec::new();
        x64::encode_call_rel32(&mut buf, disp).unwrap();
        let parsed = i32::from_le_bytes(buf[1..5].try_into().unwrap());
        prop_assert_eq!(parsed, disp);
    }

    #[test]
    fn x64_reg_to_reg_mov_is_compact(
        dst in 0u8..16,
        src in 0u8..16,
    ) {
        let mut buf = Vec::new();
        x64::encode_mov_r64_r64(dst, src, &mut buf).unwrap();
        prop_assert_eq!(buf.len(), 3);
    }

    #[test]
    fn x64_emitter_fixup_forward_delta_matches_padding(padding in 0usize..1024) {
        let mut emitter = X64Emitter::new();
        let target = emitter.new_label();
        emitter.emit_call_label(target).unwrap();
        for _ in 0..padding {
            emitter.emit_bytes(&[0x90]);
        }
        emitter.bind_label(target).unwrap();

        let finalized = emitter.finalize().unwrap();
        let disp = i32::from_le_bytes(finalized.code[1..5].try_into().unwrap());
        prop_assert_eq!(disp, padding as i32);
    }

    #[test]
    fn x64_emitter_fixup_backward_delta_matches_padding(padding in 0usize..1024) {
        let mut emitter = X64Emitter::new();
        let target = emitter.new_label();
        emitter.bind_label(target).unwrap();
        for _ in 0..padding {
            emitter.emit_bytes(&[0x90]);
        }
        emitter.emit_call_label(target).unwrap();

        let finalized = emitter.finalize().unwrap();
        let disp_offset = finalized.code.len() - 4;
        let disp = i32::from_le_bytes(finalized.code[disp_offset..].try_into().unwrap());
        prop_assert_eq!(disp, -((padding as i32) + 5));
    }

    #[test]
    fn x64_invalid_registers_are_rejected(reg in 16u8..=u8::MAX) {
        let mut buf = Vec::new();
        let err = x64::encode_push_r64(reg, &mut buf).unwrap_err();
        let is_invalid = matches!(err, x64::EmitError::InvalidRegister { reg: r } if r == reg);
        prop_assert!(is_invalid);
    }
}
