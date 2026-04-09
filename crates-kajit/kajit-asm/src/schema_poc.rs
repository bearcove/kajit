use std::collections::HashMap;

use kajit_reprs::schema_poc::asm::{A64Item, A64Reg, Imm, LabelName, Program, X64Item, X64Reg};
use kajit_types::{Prov, SourceLocation};

pub enum FinalizedSchemaPocEmission {
    AArch64(crate::aarch64::FinalizedEmission),
    X64(crate::x64::FinalizedEmission),
}

impl FinalizedSchemaPocEmission {
    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::AArch64(emission) => &emission.code,
            Self::X64(emission) => emission.exec.as_ref(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::AArch64(emission) => emission.len(),
            Self::X64(emission) => emission.len(),
        }
    }

    pub fn trace_text(&self) -> Result<String, String> {
        match self {
            Self::AArch64(emission) => emission.trace_text().map_err(|e| format!("{e:?}")),
            Self::X64(emission) => emission.trace_text().map_err(|e| format!("{e:?}")),
        }
    }
}

pub fn assemble_schema_poc_program(
    program: &Program,
    source: &str,
) -> Result<FinalizedSchemaPocEmission, String> {
    match program {
        Program::AArch64 { items, .. } => {
            assemble_aarch64(items, source).map(FinalizedSchemaPocEmission::AArch64)
        }
        Program::X86_64 { items, .. } => {
            assemble_x64(items, source).map(FinalizedSchemaPocEmission::X64)
        }
    }
}

fn assemble_aarch64(
    items: &[A64Item],
    source: &str,
) -> Result<crate::aarch64::FinalizedEmission, String> {
    use crate::aarch64::Emitter;

    let mut emitter = Emitter::new();
    let label_map = allocate_a64_labels(&mut emitter, items);

    for item in items {
        emit_a64_item(&mut emitter, item, &label_map, source)?;
    }

    emitter.finalize().map_err(|e| format!("{e:?}"))
}

fn allocate_a64_labels(
    emitter: &mut crate::aarch64::Emitter,
    items: &[A64Item],
) -> HashMap<String, crate::aarch64::LabelId> {
    let mut label_map = HashMap::new();
    for item in items {
        if let A64Item::Label { name, .. } = item {
            label_map
                .entry(name.text.clone())
                .or_insert_with(|| emitter.new_label());
        }
    }
    label_map
}

fn emit_a64_item(
    emitter: &mut crate::aarch64::Emitter,
    item: &A64Item,
    label_map: &HashMap<String, crate::aarch64::LabelId>,
    source: &str,
) -> Result<(), String> {
    use crate::aarch64::Width;

    let map_err = |e| format!("{e:?}");
    match item {
        A64Item::Label { name, prov } => {
            set_source_location(emitter, prov, source);
            let label_id = lookup_a64_label(label_map, name)?;
            emitter.bind_label(label_id).map_err(map_err)
        }
        A64Item::Ret { prov, .. } => {
            set_source_location(emitter, prov, source);
            emitter.emit_ret().map_err(map_err)
        }
        A64Item::Nop { prov, .. } => {
            set_source_location(emitter, prov, source);
            emitter.emit_nop().map_err(map_err)
        }
        A64Item::Movz { rd, imm, prov, .. } => {
            set_source_location(emitter, prov, source);
            emitter
                .emit_movz_imm(Width::X64, lower_a64_reg(rd), parse_imm_u16(imm)?, 0)
                .map_err(map_err)
        }
        A64Item::Mov { rd, rm, prov, .. } => {
            set_source_location(emitter, prov, source);
            emitter
                .emit_mov_reg(Width::X64, lower_a64_reg(rd), lower_a64_reg(rm))
                .map_err(map_err)
        }
        A64Item::AddImm {
            rd, rn, imm, prov, ..
        } => {
            set_source_location(emitter, prov, source);
            emitter
                .emit_add_imm(
                    Width::X64,
                    lower_a64_reg(rd),
                    lower_a64_reg(rn),
                    parse_imm_u16(imm)?,
                    false,
                )
                .map_err(map_err)
        }
        A64Item::B { target, prov, .. } => {
            set_source_location(emitter, prov, source);
            let label_id = lookup_a64_label(label_map, target)?;
            emitter.emit_b_label(label_id).map_err(map_err)
        }
    }
}

fn assemble_x64(items: &[X64Item], source: &str) -> Result<crate::x64::FinalizedEmission, String> {
    use crate::x64::Emitter;

    let mut emitter = Emitter::new();
    let label_map = allocate_x64_labels(&mut emitter, items);

    for item in items {
        emit_x64_item(&mut emitter, item, &label_map, source)?;
    }

    emitter.finalize().map_err(|e| format!("{e:?}"))
}

fn allocate_x64_labels(
    emitter: &mut crate::x64::Emitter,
    items: &[X64Item],
) -> HashMap<String, crate::x64::LabelId> {
    let mut label_map = HashMap::new();
    for item in items {
        if let X64Item::Label { name, .. } = item {
            label_map
                .entry(name.text.clone())
                .or_insert_with(|| emitter.new_label());
        }
    }
    label_map
}

fn emit_x64_item(
    emitter: &mut crate::x64::Emitter,
    item: &X64Item,
    label_map: &HashMap<String, crate::x64::LabelId>,
    source: &str,
) -> Result<(), String> {
    let map_err = |e| format!("{e:?}");
    match item {
        X64Item::Label { name, .. } => {
            let label_id = lookup_x64_label(label_map, name)?;
            emitter.bind_label(label_id).map_err(map_err)
        }
        X64Item::Ret { prov, .. } => {
            set_source_location(emitter, prov, source);
            emitter.emit_with(crate::x64::encode_ret).map_err(map_err)
        }
        X64Item::Nop { prov, .. } => {
            set_source_location(emitter, prov, source);
            emitter.emit_with(crate::x64::encode_nop).map_err(map_err)
        }
        X64Item::MovImm { rd, imm, prov, .. } => {
            set_source_location(emitter, prov, source);
            let rd = lower_x64_reg(rd);
            let imm = parse_imm_u64(imm)?;
            emitter
                .emit_with(|buf| crate::x64::encode_mov_r64_imm64(rd, imm, buf))
                .map_err(map_err)
        }
        X64Item::MovReg { rd, rm, prov, .. } => {
            set_source_location(emitter, prov, source);
            emitter
                .emit_with(|buf| {
                    crate::x64::encode_mov_r64_r64(lower_x64_reg(rd), lower_x64_reg(rm), buf)
                })
                .map_err(map_err)
        }
        X64Item::AddImm { rd, imm, prov, .. } => {
            set_source_location(emitter, prov, source);
            let imm = parse_imm_u32(imm)?;
            emitter
                .emit_with(|buf| crate::x64::encode_add_r64_imm32(lower_x64_reg(rd), imm, buf))
                .map_err(map_err)
        }
        X64Item::Jmp { target, prov, .. } => {
            set_source_location(emitter, prov, source);
            let label_id = lookup_x64_label(label_map, target)?;
            emitter.emit_jmp_label(label_id).map_err(map_err)
        }
    }
}

fn lower_a64_reg(reg: &A64Reg) -> crate::aarch64::Reg {
    match reg {
        A64Reg::X0(..) => crate::aarch64::Reg::X0,
        A64Reg::X1(..) => crate::aarch64::Reg::X1,
        A64Reg::X2(..) => crate::aarch64::Reg::X2,
        A64Reg::X3(..) => crate::aarch64::Reg::X3,
        A64Reg::Sp(..) => crate::aarch64::Reg::SP,
    }
}

fn lower_x64_reg(reg: &X64Reg) -> u8 {
    match reg {
        X64Reg::Rax(..) => 0,
        X64Reg::Rcx(..) => 1,
        X64Reg::Rdx(..) => 2,
        X64Reg::Rbx(..) => 3,
        X64Reg::Rsp(..) => 4,
        X64Reg::Rbp(..) => 5,
    }
}

fn parse_imm_u16(imm: &Imm) -> Result<u16, String> {
    u16::try_from(imm.value).map_err(|e| format!("invalid u16 immediate `{}`: {e}", imm.value))
}

fn parse_imm_u32(imm: &Imm) -> Result<u32, String> {
    u32::try_from(imm.value).map_err(|e| format!("invalid u32 immediate `{}`: {e}", imm.value))
}

fn parse_imm_u64(imm: &Imm) -> Result<u64, String> {
    Ok(imm.value)
}

fn lookup_a64_label(
    label_map: &HashMap<String, crate::aarch64::LabelId>,
    name: &LabelName,
) -> Result<crate::aarch64::LabelId, String> {
    label_map
        .get(&name.text)
        .copied()
        .ok_or_else(|| format!("unknown AArch64 label `{}`", name.text))
}

fn lookup_x64_label(
    label_map: &HashMap<String, crate::x64::LabelId>,
    name: &LabelName,
) -> Result<crate::x64::LabelId, String> {
    label_map
        .get(&name.text)
        .copied()
        .ok_or_else(|| format!("unknown x86-64 label `{}`", name.text))
}

fn set_source_location(emitter: &mut impl SourceLocEmitter, prov: &Prov, source: &str) {
    emitter.set_source_location(source_location_for_prov(prov, source));
}

fn source_location_for_prov(prov: &Prov, source: &str) -> SourceLocation {
    let file = prov.file_id.unwrap_or(0) as u16;
    let Some(span) = prov.span else {
        return SourceLocation {
            file,
            line: 1,
            column: 1,
        };
    };

    let (line, column) = line_col_for_offset(source, span.start as usize);
    SourceLocation { file, line, column }
}

fn line_col_for_offset(source: &str, offset: usize) -> (u32, u32) {
    let mut line = 1u32;
    let mut column = 1u32;
    for ch in source[..offset.min(source.len())].chars() {
        if ch == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    (line, column)
}

trait SourceLocEmitter {
    fn set_source_location(&mut self, loc: SourceLocation);
}

impl SourceLocEmitter for crate::aarch64::Emitter {
    fn set_source_location(&mut self, loc: SourceLocation) {
        crate::aarch64::Emitter::set_source_location(self, loc);
    }
}

impl SourceLocEmitter for crate::x64::Emitter {
    fn set_source_location(&mut self, loc: SourceLocation) {
        crate::x64::Emitter::set_source_location(self, loc);
    }
}
