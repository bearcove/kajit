use kajit_reprs::asm::aarch64_asm::{Instruction, Item, Label, Program};

/// Assemble a Program into executable machine code.
///
/// This function replays the program through an Emitter, handling label
/// binding and resolution automatically.
#[cfg(target_arch = "aarch64")]
pub fn assemble(program: &Program) -> Result<crate::aarch64::FinalizedEmission, String> {
    use crate::aarch64::Emitter;
    use std::collections::HashMap;

    let mut emitter = Emitter::new();
    let mut label_map: HashMap<Label, crate::aarch64::LabelId> = HashMap::new();

    // First pass: collect labels and allocate LabelIds
    for item in &program.items {
        if let Item::Label(label) = item {
            let label_id = emitter.new_label();
            label_map.insert(*label, label_id);
        }
    }

    // Second pass: emit instructions and bind labels
    for item in &program.items {
        match item {
            Item::Label(label) => {
                let label_id = label_map[label];

                emitter.bind_label(label_id);
            }
            Item::Instruction(inst) => {
                emit_instruction(&mut emitter, inst, &label_map)
                    .map_err(|e| format!("Failed to emit instruction: {:?}", e))?;
            }
        }
    }

    emitter
        .finalize()
        .map_err(|e| format!("Failed to finalize: {:?}", e))
}

#[cfg(target_arch = "aarch64")]
fn emit_instruction(
    emitter: &mut crate::aarch64::Emitter,
    inst: &Instruction,
    label_map: &std::collections::HashMap<Label, crate::aarch64::LabelId>,
) -> Result<(), String> {
    let map_err = |e: crate::aarch64::EmitError| format!("{:?}", e);

    match inst {
        Instruction::Ret => emitter.emit_ret().map_err(map_err),
        Instruction::Nop => emitter.emit_nop().map_err(map_err),
        Instruction::MovReg { width, rd, rm } => {
            emitter.emit_mov_reg(*width, *rd, *rm).map_err(map_err)
        }
        Instruction::MovzImm {
            width,
            rd,
            imm,
            shift,
        } => emitter
            .emit_movz_imm(*width, *rd, *imm, *shift)
            .map_err(map_err),
        Instruction::MovkImm {
            width,
            rd,
            imm,
            shift,
        } => emitter
            .emit_movk_imm(*width, *rd, *imm, *shift)
            .map_err(map_err),
        Instruction::AddImm { width, rd, rn, imm } => emitter
            .emit_add_imm(*width, *rd, *rn, *imm as u16, false)
            .map_err(map_err),
        Instruction::SubImm { width, rd, rn, imm } => emitter
            .emit_sub_imm(*width, *rd, *rn, *imm as u16, false)
            .map_err(map_err),
        Instruction::CmpImm { width, rn, imm } => emitter
            .emit_cmp_imm(*width, *rn, *imm as u16)
            .map_err(map_err),
        Instruction::AddReg { width, rd, rn, rm } => {
            emitter.emit_add_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
        Instruction::SubReg { width, rd, rn, rm } => {
            emitter.emit_sub_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
        Instruction::CmpReg { width, rn, rm } => {
            emitter.emit_cmp_reg(*width, *rn, *rm).map_err(map_err)
        }
        Instruction::MulReg { width, rd, rn, rm } => {
            emitter.emit_mul_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
        Instruction::NegReg { width, rd, rm } => {
            emitter.emit_neg_reg(*width, *rd, *rm).map_err(map_err)
        }
        Instruction::AndImm { width, rd, rn, imm } => emitter
            .emit_and_imm(*width, *rd, *rn, *imm)
            .map_err(map_err),
        Instruction::AndReg { width, rd, rn, rm } => {
            emitter.emit_and_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
        Instruction::OrrImm { width, rd, rn, imm } => emitter
            .emit_orr_imm(*width, *rd, *rn, *imm)
            .map_err(map_err),
        Instruction::OrrReg { width, rd, rn, rm } => {
            emitter.emit_orr_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
        Instruction::EorImm { width, rd, rn, imm } => emitter
            .emit_eor_imm(*width, *rd, *rn, *imm)
            .map_err(map_err),
        Instruction::EorReg { width, rd, rn, rm } => {
            emitter.emit_eor_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
        Instruction::Bfi {
            width,
            rd,
            rn,
            lsb,
            bit_width,
        } => emitter
            .emit_bfi(*width, *rd, *rn, *lsb, *bit_width)
            .map_err(map_err),
        Instruction::LslImm {
            width,
            rd,
            rn,
            shift,
        } => emitter
            .emit_lsl_imm(*width, *rd, *rn, *shift)
            .map_err(map_err),
        Instruction::LslReg { width, rd, rn, rm } => {
            emitter.emit_lsl_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
        Instruction::LsrImm {
            width,
            rd,
            rn,
            shift,
        } => emitter
            .emit_lsr_imm(*width, *rd, *rn, *shift)
            .map_err(map_err),
        Instruction::LsrReg { width, rd, rn, rm } => {
            emitter.emit_lsr_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
        Instruction::AsrImm {
            width,
            rd,
            rn,
            shift,
        } => emitter
            .emit_asr_imm(*width, *rd, *rn, *shift)
            .map_err(map_err),
        Instruction::AsrReg { width, rd, rn, rm } => {
            emitter.emit_asr_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
        Instruction::LdrImm {
            width,
            rt,
            rn,
            offset,
        } => emitter
            .emit_ldr_imm(*width, *rt, *rn, *offset)
            .map_err(map_err),
        Instruction::LdrbImm { rt, rn, offset } => {
            emitter.emit_ldrb_imm(*rt, *rn, *offset).map_err(map_err)
        }
        Instruction::LdrhImm { rt, rn, offset } => {
            emitter.emit_ldrh_imm(*rt, *rn, *offset).map_err(map_err)
        }
        Instruction::StrImm {
            width,
            rt,
            rn,
            offset,
        } => emitter
            .emit_str_imm(*width, *rt, *rn, *offset)
            .map_err(map_err),
        Instruction::StrbImm { rt, rn, offset } => {
            emitter.emit_strb_imm(*rt, *rn, *offset).map_err(map_err)
        }
        Instruction::StrhImm { rt, rn, offset } => {
            emitter.emit_strh_imm(*rt, *rn, *offset).map_err(map_err)
        }
        Instruction::Stp {
            width,
            rt1,
            rt2,
            rn,
            offset,
        } => emitter
            .emit_stp(*width, *rt1, *rt2, *rn, *offset)
            .map_err(map_err),
        Instruction::Ldp {
            width,
            rt1,
            rt2,
            rn,
            offset,
        } => emitter
            .emit_ldp(*width, *rt1, *rt2, *rn, *offset)
            .map_err(map_err),
        Instruction::Sxtb { width, rd, rn } => emitter.emit_sxtb(*width, *rd, *rn).map_err(map_err),
        Instruction::Sxth { width, rd, rn } => emitter.emit_sxth(*width, *rd, *rn).map_err(map_err),
        Instruction::Sxtw { rd, rn } => emitter.emit_sxtw(*rd, *rn).map_err(map_err),
        Instruction::Cset { width, rd, cond } => {
            emitter.emit_cset(*width, *rd, *cond).map_err(map_err)
        }
        Instruction::B { target } => {
            let label_id = label_map
                .get(target)
                .copied()
                .ok_or_else(|| format!("Unknown label: {}", target))?;
            emitter.emit_b_label(label_id).map_err(map_err)
        }
        Instruction::BCond { cond, target } => {
            let label_id = label_map
                .get(target)
                .copied()
                .ok_or_else(|| format!("Unknown label: {}", target))?;
            emitter.emit_b_cond_label(*cond, label_id).map_err(map_err)
        }
        Instruction::Bl { target } => {
            let label_id = label_map
                .get(target)
                .copied()
                .ok_or_else(|| format!("Unknown label: {}", target))?;
            emitter.emit_bl_label(label_id).map_err(map_err)
        }
        Instruction::Blr { rn } => emitter.emit_blr(*rn).map_err(map_err),
        Instruction::Cbz { width, rt, target } => {
            let label_id = label_map
                .get(target)
                .copied()
                .ok_or_else(|| format!("Unknown label: {}", target))?;
            emitter
                .emit_cbz_label(*width, *rt, label_id)
                .map_err(map_err)
        }
        Instruction::Cbnz { width, rt, target } => {
            let label_id = label_map
                .get(target)
                .copied()
                .ok_or_else(|| format!("Unknown label: {}", target))?;
            emitter
                .emit_cbnz_label(*width, *rt, label_id)
                .map_err(map_err)
        }
        Instruction::Tbz { rt, bit, target } => {
            let label_id = label_map
                .get(target)
                .copied()
                .ok_or_else(|| format!("Unknown label: {}", target))?;
            emitter.emit_tbz_label(*rt, *bit, label_id).map_err(map_err)
        }
        Instruction::Tbnz { rt, bit, target } => {
            let label_id = label_map
                .get(target)
                .copied()
                .ok_or_else(|| format!("Unknown label: {}", target))?;
            emitter
                .emit_tbnz_label(*rt, *bit, label_id)
                .map_err(map_err)
        }
        Instruction::LoadExtern { .. } => {
            // LoadExtern should have been expanded before assembly
            Err("LoadExtern pseudo-instruction must be expanded before assembly".to_string())
        }
    }
}
