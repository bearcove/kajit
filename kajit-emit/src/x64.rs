use crate::{SourceLocation, SourceMap, SourceMapEntry};

use std::ptr;

#[cfg(target_os = "macos")]
const MAP_ANON: i32 = 0x1000;

#[cfg(not(target_os = "macos"))]
const MAP_ANON: i32 = 0x20;

const MAP_PRIVATE: i32 = 0x02;
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const PROT_EXEC: i32 = 4;
const MAP_FAILED: *mut u8 = !0usize as *mut u8;

#[cfg(target_os = "macos")]
const MAP_JIT: i32 = 0x0800;

type MmapProt = i32;

unsafe extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: MmapProt, flags: i32, fd: i32, offset: i64)
    -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn mprotect(addr: *mut u8, len: usize, prot: MmapProt) -> i32;
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn pthread_jit_write_protect_np(enabled: i32);
    fn sys_icache_invalidate(start: *mut u8, len: usize);
}

#[derive(Debug)]
pub struct ExecutableBuffer {
    ptr: *mut u8,
    len: usize,
}

impl ExecutableBuffer {
    fn allocate(bytes: &[u8]) -> Self {
        let len = bytes.len().max(1);
        let mut flags = MAP_PRIVATE | MAP_ANON;
        #[cfg(target_os = "macos")]
        {
            flags |= MAP_JIT;
        }

        let prot = if cfg!(target_os = "macos") {
            PROT_READ | PROT_WRITE | PROT_EXEC
        } else {
            PROT_READ | PROT_WRITE
        };

        let ptr = unsafe { mmap(ptr::null_mut(), len, prot, flags, -1, 0) };
        if ptr == MAP_FAILED {
            panic!("mmap failed to allocate executable memory");
        }

        #[cfg(target_os = "macos")]
        unsafe {
            pthread_jit_write_protect_np(0);
            ptr.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
            pthread_jit_write_protect_np(1);
            sys_icache_invalidate(ptr, bytes.len());
        }

        #[cfg(not(target_os = "macos"))]
        unsafe {
            ptr.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
            if mprotect(ptr, len, PROT_READ | PROT_EXEC) != 0 {
                munmap(ptr, len);
                panic!("mprotect failed to mark executable memory");
            }
        }

        Self { ptr, len }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl AsRef<[u8]> for ExecutableBuffer {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for ExecutableBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len != 0 {
            unsafe {
                munmap(self.ptr, self.len);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mem {
    pub base: u8,
    pub disp: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LabelId(u32);

#[derive(Debug)]
pub struct FinalizedEmission {
    pub exec: ExecutableBuffer,
    pub source_map: SourceMap,
}

impl FinalizedEmission {
    pub fn code_ptr(&self) -> *const u8 {
        self.exec.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.exec.len()
    }

    pub fn trace_entries(&self) -> Result<Vec<crate::TraceEntry>, crate::TraceError> {
        crate::build_trace(self.exec.as_ref(), &self.source_map)
    }

    pub fn trace_text(&self) -> Result<String, crate::TraceError> {
        crate::format_trace(self.exec.as_ref(), &self.source_map)
    }

    pub fn source_map_le(&self) -> Result<Vec<u8>, crate::SourceMapError> {
        crate::encode_source_map_le(&self.source_map)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(u8),
    Mem(Mem),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    InvalidRegister {
        reg: u8,
    },
    LabelOutOfBounds {
        label: LabelId,
    },
    LabelAlreadyBound {
        label: LabelId,
        existing_offset: u32,
    },
    UnboundLabel {
        label: LabelId,
    },
    RelativeOutOfRange {
        at_offset: u32,
        target_offset: u32,
        delta: i64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Lo,
    Hs,
    Hi,
    O,
    No,
    S,
    Ns,
    P,
    Np,
}

impl Condition {
    fn code(self) -> u8 {
        match self {
            Condition::O => 0x0,
            Condition::No => 0x1,
            Condition::Eq => 0x4,
            Condition::Ne => 0x5,
            Condition::Lo => 0x2,
            Condition::Hs => 0x3,
            Condition::Hi => 0x7,
            Condition::S => 0x8,
            Condition::Ns => 0x9,
            Condition::P => 0xa,
            Condition::Np => 0xb,
            Condition::Lt => 0xc,
            Condition::Ge => 0xd,
            Condition::Le => 0xe,
            Condition::Gt => 0xf,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fixup {
    disp_offset: u32,
    next_ip: u32,
    label: LabelId,
}

#[derive(Debug, Clone, Default)]
pub struct Emitter {
    buf: Vec<u8>,
    source_map: SourceMap,
    current_location: SourceLocation,
    last_recorded_location: Option<SourceLocation>,
    labels: Vec<Option<u32>>,
    fixups: Vec<Fixup>,
}

impl Emitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn current_offset(&self) -> u32 {
        self.buf.len() as u32
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn source_map(&self) -> &[SourceMapEntry] {
        &self.source_map
    }

    pub fn set_source_location(&mut self, loc: SourceLocation) {
        self.current_location = loc;
    }

    fn maybe_record_source_map(&mut self) {
        if Some(self.current_location) != self.last_recorded_location {
            self.source_map.push(SourceMapEntry {
                offset: self.current_offset(),
                location: self.current_location,
            });
            self.last_recorded_location = Some(self.current_location);
        }
    }

    pub fn new_label(&mut self) -> LabelId {
        let id = LabelId(self.labels.len() as u32);
        self.labels.push(None);
        id
    }

    pub fn bind_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        let current_offset = self.current_offset();
        let Some(slot) = self.labels.get_mut(label.0 as usize) else {
            return Err(EmitError::LabelOutOfBounds { label });
        };
        if let Some(existing_offset) = *slot {
            return Err(EmitError::LabelAlreadyBound {
                label,
                existing_offset,
            });
        }
        *slot = Some(current_offset);
        Ok(())
    }

    pub fn emit_with<F>(&mut self, f: F) -> Result<(), EmitError>
    where
        F: FnOnce(&mut Vec<u8>) -> Result<(), EmitError>,
    {
        let mut inst = Vec::new();
        f(&mut inst)?;
        self.maybe_record_source_map();
        self.buf.extend_from_slice(&inst);
        Ok(())
    }

    pub fn emit_bytes(&mut self, bytes: &[u8]) {
        self.maybe_record_source_map();
        self.buf.extend_from_slice(bytes);
    }

    fn ensure_label_exists(&self, label: LabelId) -> Result<(), EmitError> {
        if self.labels.get(label.0 as usize).is_some() {
            Ok(())
        } else {
            Err(EmitError::LabelOutOfBounds { label })
        }
    }

    pub fn emit_je_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.emit_jcc_label(label, Condition::Eq)
    }

    pub fn emit_jz_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.emit_je_label(label)
    }

    pub fn emit_jnz_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.emit_jcc_label(label, Condition::Ne)
    }

    pub fn emit_jne_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.emit_jcc_label(label, Condition::Ne)
    }

    pub fn emit_jge_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.emit_jcc_label(label, Condition::Ge)
    }

    pub fn emit_jl_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.emit_jcc_label(label, Condition::Lt)
    }

    pub fn emit_jbe_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.emit_jcc_label(label, Condition::Lo)
    }

    pub fn emit_jae_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.emit_jcc_label(label, Condition::Hs)
    }

    pub fn emit_ja_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.emit_jcc_label(label, Condition::Hi)
    }

    pub fn emit_jcc_label(
        &mut self,
        label: LabelId,
        condition: Condition,
    ) -> Result<(), EmitError> {
        self.ensure_label_exists(label)?;
        let start = self.current_offset();
        self.emit_with(|buf| encode_jcc_rel32(buf, condition, 0))?;
        self.fixups.push(Fixup {
            disp_offset: start + 2,
            next_ip: self.current_offset(),
            label,
        });
        Ok(())
    }

    pub fn emit_jmp_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.ensure_label_exists(label)?;
        let start = self.current_offset();
        self.emit_with(|buf| encode_jmp_rel32(buf, 0))?;
        self.fixups.push(Fixup {
            disp_offset: start + 1,
            next_ip: self.current_offset(),
            label,
        });
        Ok(())
    }

    pub fn emit_call_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.ensure_label_exists(label)?;
        let start = self.current_offset();
        self.emit_with(|buf| encode_call_rel32(buf, 0))?;
        self.fixups.push(Fixup {
            disp_offset: start + 1,
            next_ip: self.current_offset(),
            label,
        });
        Ok(())
    }

    pub fn finalize(mut self) -> Result<FinalizedEmission, EmitError> {
        for fixup in self.fixups.iter().copied() {
            let target = self
                .labels
                .get(fixup.label.0 as usize)
                .ok_or(EmitError::LabelOutOfBounds { label: fixup.label })?
                .ok_or(EmitError::UnboundLabel { label: fixup.label })?;

            let delta = target as i64 - fixup.next_ip as i64;
            if delta < i32::MIN as i64 || delta > i32::MAX as i64 {
                return Err(EmitError::RelativeOutOfRange {
                    at_offset: fixup.disp_offset,
                    target_offset: target,
                    delta,
                });
            }

            let disp = (delta as i32).to_le_bytes();
            self.buf[fixup.disp_offset as usize..fixup.disp_offset as usize + 4]
                .copy_from_slice(&disp);
        }

        Ok(FinalizedEmission {
            exec: ExecutableBuffer::allocate(&self.buf),
            source_map: self.source_map,
        })
    }
}

fn check_reg(reg: u8) -> Result<u8, EmitError> {
    if reg <= 15 {
        Ok(reg)
    } else {
        Err(EmitError::InvalidRegister { reg })
    }
}

fn push_rex(buf: &mut Vec<u8>, w: bool, r: u8, x: u8, b: u8, force: bool) {
    let rex =
        0x40 | ((w as u8) << 3) | (((r >> 3) & 1) << 2) | (((x >> 3) & 1) << 1) | ((b >> 3) & 1);
    if force || rex != 0x40 {
        buf.push(rex);
    }
}

fn emit_modrm(buf: &mut Vec<u8>, reg_field: u8, rm: Operand) {
    match rm {
        Operand::Reg(rm_reg) => {
            let modrm = 0b11_000_000 | ((reg_field & 0x7) << 3) | (rm_reg & 0x7);
            buf.push(modrm);
        }
        Operand::Mem(mem) => {
            let base = mem.base;
            let base_low = base & 0x7;

            let (mod_bits, disp8, disp32) = if mem.disp == 0 && base_low != 0b101 {
                (0b00, None, None)
            } else if (-128..=127).contains(&mem.disp) {
                (0b01, Some(mem.disp as i8), None)
            } else {
                (0b10, None, Some(mem.disp))
            };

            let rm_field = if base_low == 0b100 { 0b100 } else { base_low };
            let modrm = (mod_bits << 6) | ((reg_field & 0x7) << 3) | rm_field;
            buf.push(modrm);

            if rm_field == 0b100 {
                // SIB with no index: scale=0, index=100, base=base_low
                buf.push((0b100 << 3) | base_low);
            }

            if let Some(disp8) = disp8 {
                buf.push(disp8 as u8);
            }
            if let Some(disp32) = disp32 {
                buf.extend_from_slice(&disp32.to_le_bytes());
            }
        }
    }
}

fn emit_op_rm(
    buf: &mut Vec<u8>,
    legacy_prefix: Option<u8>,
    opcodes: &[u8],
    rex_w: bool,
    reg_field: u8,
    rm: Operand,
    force_rex_for_rm8: bool,
) -> Result<(), EmitError> {
    let reg_field = check_reg(reg_field)?;
    let b = match rm {
        Operand::Reg(reg) => check_reg(reg)?,
        Operand::Mem(mem) => check_reg(mem.base)?,
    };

    if let Some(prefix) = legacy_prefix {
        buf.push(prefix);
    }

    let force = force_rex_for_rm8 && matches!(rm, Operand::Reg(r) if (4..=7).contains(&r));
    push_rex(buf, rex_w, reg_field, 0, b, force);
    buf.extend_from_slice(opcodes);
    emit_modrm(buf, reg_field, rm);
    Ok(())
}

pub fn encode_mov_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x89], true, src, Operand::Reg(dst), false)
}

pub fn encode_mov_r64_imm64(dst: u8, imm: u64, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    let dst = check_reg(dst)?;
    push_rex(buf, true, 0, 0, dst, false);
    buf.push(0xB8 + (dst & 0x7));
    buf.extend_from_slice(&imm.to_le_bytes());
    Ok(())
}

pub fn encode_mov_r32_imm32(dst: u8, imm: u32, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    let dst = check_reg(dst)?;
    push_rex(buf, false, 0, 0, dst, false);
    buf.push(0xB8 + (dst & 0x7));
    buf.extend_from_slice(&imm.to_le_bytes());
    Ok(())
}

pub fn encode_mov_m_r64(dst: Mem, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x89], true, src, Operand::Mem(dst), false)
}

pub fn encode_mov_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x8B], true, dst, Operand::Mem(src), false)
}

pub fn encode_mov_r32_r32(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x89], false, src, Operand::Reg(dst), false)
}

pub fn encode_mov_m_r32(dst: Mem, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x89], false, src, Operand::Mem(dst), false)
}

pub fn encode_mov_r32_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x8B], false, dst, Operand::Mem(src), false)
}

pub fn encode_movd_r64_xmm(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0x66),
        &[0x0F, 0x7E],
        false,
        src,
        Operand::Reg(dst),
        false,
    )
}

pub fn encode_movd_xmm_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0x66),
        &[0x0F, 0x6E],
        false,
        dst,
        Operand::Reg(src),
        false,
    )
}

pub fn encode_pshufd_xmm_xmm_imm8(
    dst: u8,
    src: u8,
    imm8: u8,
    buf: &mut Vec<u8>,
) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0x66),
        &[0x0F, 0x70],
        false,
        dst,
        Operand::Reg(src),
        false,
    )?;
    buf.push(imm8);
    Ok(())
}

pub fn encode_movdqa_xmm_xmm(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0x66),
        &[0x0F, 0x6F],
        false,
        dst,
        Operand::Reg(src),
        false,
    )
}

pub fn encode_movdqu_xmm_m(
    dst_xmm: u8,
    base: u8,
    disp: i32,
    buf: &mut Vec<u8>,
) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0x66),
        &[0x0F, 0x6F],
        false,
        dst_xmm,
        Operand::Mem(Mem { base, disp }),
        false,
    )
}

pub fn encode_movdqu_m_xmm(
    base: u8,
    disp: i32,
    src_xmm: u8,
    buf: &mut Vec<u8>,
) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0x66),
        &[0x0F, 0x7F],
        false,
        src_xmm,
        Operand::Mem(Mem { base, disp }),
        false,
    )
}

pub fn encode_pcmpeqb_xmm_xmm(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0x66),
        &[0x0F, 0x74],
        false,
        dst,
        Operand::Reg(src),
        false,
    )
}

pub fn encode_por_xmm_xmm(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0x66),
        &[0x0F, 0xEB],
        false,
        dst,
        Operand::Reg(src),
        false,
    )
}

pub fn encode_pmovmskb_r32_xmm(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0x66),
        &[0x0F, 0xD7],
        false,
        dst,
        Operand::Reg(src),
        false,
    )
}

pub fn encode_mov_m_r8(
    dst_base: u8,
    dst_disp: i32,
    src: u8,
    buf: &mut Vec<u8>,
) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        None,
        &[0x88],
        false,
        src,
        Operand::Mem(Mem {
            base: dst_base,
            disp: dst_disp,
        }),
        true,
    )
}

pub fn encode_mov_r8_m(
    dst: u8,
    src_base: u8,
    src_disp: i32,
    buf: &mut Vec<u8>,
) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        None,
        &[0x8A],
        false,
        dst,
        Operand::Mem(Mem {
            base: src_base,
            disp: src_disp,
        }),
        true,
    )
}

pub fn encode_mov_m_r16(
    dst_base: u8,
    dst_disp: i32,
    src: u8,
    buf: &mut Vec<u8>,
) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0x66),
        &[0x89],
        false,
        src,
        Operand::Mem(Mem {
            base: dst_base,
            disp: dst_disp,
        }),
        false,
    )
}

pub fn encode_mov_r16_m(
    dst: u8,
    src_base: u8,
    src_disp: i32,
    buf: &mut Vec<u8>,
) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0x66),
        &[0x8B],
        false,
        dst,
        Operand::Mem(Mem {
            base: src_base,
            disp: src_disp,
        }),
        false,
    )
}

pub fn encode_movzx_r32_rm8(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xB6], false, dst, src, true)
}

pub fn encode_movzx_r32_rm16(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xB7], false, dst, src, false)
}

pub fn encode_movzx_r64_rm8(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xB6], true, dst, src, true)
}

pub fn encode_movzx_r64_rm16(
    dst: u8,
    base: u8,
    disp: i32,
    buf: &mut Vec<u8>,
) -> Result<(), EmitError> {
    encode_movzx_r64_rm16_op(dst, Operand::Mem(Mem { base, disp }), buf)
}

pub fn encode_movzx_r64_rm16_op(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xB7], true, dst, src, false)
}

pub fn encode_movsx_r64_rm8(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xBE], true, dst, src, true)
}

pub fn encode_movsx_r64_rm16(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xBF], true, dst, src, false)
}

pub fn encode_tzcnt_r32_r32(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        Some(0xF3),
        &[0x0F, 0xBC],
        false,
        dst,
        Operand::Reg(src),
        false,
    )
}

pub fn encode_imul_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        None,
        &[0x0F, 0xAF],
        true,
        src,
        Operand::Reg(dst),
        false,
    )
}

pub fn encode_imul_r64_r64_imm32(
    dst: u8,
    src: u8,
    imm: u32,
    buf: &mut Vec<u8>,
) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x69], true, src, Operand::Reg(dst), false)?;
    buf.extend_from_slice(&imm.to_le_bytes());
    Ok(())
}

pub fn encode_movsxd_r64_rm32(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x63], true, dst, src, false)
}

pub fn encode_lea_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x8D], true, dst, Operand::Mem(src), false)
}

pub fn encode_add_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x01], true, src, Operand::Reg(dst), false)
}

pub fn encode_add_r64_imm32(dst: u8, imm: u32, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x81], true, 0, Operand::Reg(dst), false)?;
    buf.extend_from_slice(&imm.to_le_bytes());
    Ok(())
}

pub fn encode_add_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x03], true, dst, Operand::Mem(src), false)
}

pub fn encode_sub_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x29], true, src, Operand::Reg(dst), false)
}

pub fn encode_sub_r64_imm32(dst: u8, imm: u32, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x81], true, 5, Operand::Reg(dst), false)?;
    buf.extend_from_slice(&imm.to_le_bytes());
    Ok(())
}

pub fn encode_sub_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x2B], true, dst, Operand::Mem(src), false)
}

pub fn encode_and_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x21], true, src, Operand::Reg(dst), false)
}

pub fn encode_and_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x23], true, dst, Operand::Mem(src), false)
}

pub fn encode_or_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x09], true, src, Operand::Reg(dst), false)
}

pub fn encode_or_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0B], true, dst, Operand::Mem(src), false)
}

pub fn encode_xor_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x31], true, src, Operand::Reg(dst), false)
}

pub fn encode_xor_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x33], true, dst, Operand::Mem(src), false)
}

pub fn encode_shl_r64_cl(dst: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xD3], true, 4, Operand::Reg(dst), false)
}

pub fn encode_shr_r64_cl(dst: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xD3], true, 5, Operand::Reg(dst), false)
}

pub fn encode_shl_r64_imm8(dst: u8, imm8: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xC1], true, 4, Operand::Reg(dst), false)?;
    buf.push(imm8);
    Ok(())
}

pub fn encode_shr_r64_imm8(dst: u8, imm8: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xC1], true, 5, Operand::Reg(dst), false)?;
    buf.push(imm8);
    Ok(())
}

pub fn encode_neg_r64(dst: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xF7], true, 3, Operand::Reg(dst), false)
}

pub fn encode_cmp_r64_r64(lhs: u8, rhs: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x39], true, rhs, Operand::Reg(lhs), false)
}

pub fn encode_cmp_r64_imm32(dst: u8, imm: u32, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x81], true, 7, Operand::Reg(dst), false)?;
    buf.extend_from_slice(&imm.to_le_bytes());
    Ok(())
}

pub fn encode_cmp_r64_m(lhs: u8, rhs: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x3B], true, lhs, Operand::Mem(rhs), false)
}

pub fn encode_rep_movsb(buf: &mut Vec<u8>) -> Result<(), EmitError> {
    buf.push(0xF3);
    buf.push(0xA4);
    Ok(())
}

pub fn encode_sar_r64_cl(dst: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xD3], true, 7, Operand::Reg(dst), false)
}

pub fn encode_sar_r64_imm8(dst: u8, imm8: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xC1], true, 7, Operand::Reg(dst), false)?;
    buf.push(imm8);
    Ok(())
}

pub fn encode_mov_r64_imm32_sext(dst: u8, imm: u32, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xC7], true, 0, Operand::Reg(dst), false)?;
    buf.extend_from_slice(&imm.to_le_bytes());
    Ok(())
}

pub fn encode_cld(buf: &mut Vec<u8>) -> Result<(), EmitError> {
    buf.push(0xFC);
    Ok(())
}

pub fn encode_xor_r32_r32(lhs: u8, rhs: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x31], false, rhs, Operand::Reg(lhs), false)
}

pub fn encode_test_r64_r64(lhs: u8, rhs: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x85], true, rhs, Operand::Reg(lhs), false)
}

pub fn encode_test_r32_r32(lhs: u8, rhs: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x85], false, rhs, Operand::Reg(lhs), false)
}

pub fn encode_test_m_r64(lhs: Mem, rhs: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x85], true, rhs, Operand::Mem(lhs), false)
}

pub fn encode_setne_r8(dst: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0x95], false, 0, Operand::Reg(dst), true)
}

pub fn encode_test_r8_r8(dst: u8, rhs: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x84], false, rhs, Operand::Reg(dst), true)
}

pub fn encode_setcc_r8(cond: Condition, dst: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        None,
        &[0x0F, 0x90 + cond.code()],
        false,
        0,
        Operand::Reg(dst),
        true,
    )
}

pub fn encode_cmovcc_r64_r64(
    cond: Condition,
    dst: u8,
    src: u8,
    buf: &mut Vec<u8>,
) -> Result<(), EmitError> {
    emit_op_rm(
        buf,
        None,
        &[0x0F, 0x40 + cond.code()],
        true,
        dst,
        Operand::Reg(src),
        false,
    )
}

pub fn encode_je_rel32(buf: &mut Vec<u8>, disp: i32) -> Result<(), EmitError> {
    buf.extend_from_slice(&[0x0F, 0x84]);
    buf.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

pub fn encode_jcc_rel32(buf: &mut Vec<u8>, cond: Condition, disp: i32) -> Result<(), EmitError> {
    buf.push(0x0F);
    buf.push(0x80 + cond.code());
    buf.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

pub fn encode_jz_rel32(buf: &mut Vec<u8>, disp: i32) -> Result<(), EmitError> {
    encode_je_rel32(buf, disp)
}

pub fn encode_jnz_rel32(buf: &mut Vec<u8>, disp: i32) -> Result<(), EmitError> {
    buf.extend_from_slice(&[0x0F, 0x85]);
    buf.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

pub fn encode_jmp_rel32(buf: &mut Vec<u8>, disp: i32) -> Result<(), EmitError> {
    buf.push(0xE9);
    buf.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

pub fn encode_call_rel32(buf: &mut Vec<u8>, disp: i32) -> Result<(), EmitError> {
    buf.push(0xE8);
    buf.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

pub fn encode_call_r64(reg: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xFF], false, 2, Operand::Reg(reg), false)
}

pub fn encode_jmp_r64(reg: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xFF], false, 4, Operand::Reg(reg), false)
}

pub fn encode_push_r64(reg: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    let reg = check_reg(reg)?;
    push_rex(buf, false, 0, 0, reg, false);
    buf.push(0x50 + (reg & 0x7));
    Ok(())
}

pub fn encode_pop_r64(reg: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    let reg = check_reg(reg)?;
    push_rex(buf, false, 0, 0, reg, false);
    buf.push(0x58 + (reg & 0x7));
    Ok(())
}

pub fn encode_ret(buf: &mut Vec<u8>) -> Result<(), EmitError> {
    buf.push(0xC3);
    Ok(())
}

pub fn encode_nop(buf: &mut Vec<u8>) -> Result<(), EmitError> {
    buf.push(0x90);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_mov_variants() {
        let mut buf = Vec::new();
        encode_mov_r64_r64(11, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x89, 0xd3]);

        buf.clear();
        encode_mov_r64_imm64(3, 0x1122_3344_5566_7788, &mut buf).unwrap();
        assert_eq!(
            buf,
            [0x48, 0xbb, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]
        );

        buf.clear();
        encode_mov_m_r64(Mem { base: 4, disp: 0 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x89, 0x14, 0x24]);

        buf.clear();
        encode_mov_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x8b, 0x14, 0x24]);

        buf.clear();
        encode_mov_r32_r32(11, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x45, 0x89, 0xd3]);

        buf.clear();
        encode_mov_m_r32(Mem { base: 4, disp: 0 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x44, 0x89, 0x14, 0x24]);

        buf.clear();
        encode_mov_r32_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x44, 0x8b, 0x14, 0x24]);

        buf.clear();
        encode_mov_r32_imm32(3, 0x1122_3344, &mut buf).unwrap();
        assert_eq!(buf, [0xbb, 0x44, 0x33, 0x22, 0x11]);

        buf.clear();
        encode_mov_m_r8(9, 32, 7, &mut buf).unwrap();
        assert_eq!(buf, [0x41, 0x88, 0x79, 0x20]);

        buf.clear();
        encode_mov_r8_m(8, 9, 40, &mut buf).unwrap();
        assert_eq!(buf, [0x45, 0x8a, 0x41, 0x28]);

        buf.clear();
        encode_mov_m_r16(10, 16, 1, &mut buf).unwrap();
        assert_eq!(buf, [0x66, 0x41, 0x89, 0x4a, 0x10]);

        buf.clear();
        encode_mov_r16_m(9, 11, 16, &mut buf).unwrap();
        assert_eq!(buf, [0x66, 0x45, 0x8b, 0x4b, 0x10]);
    }

    #[test]
    fn encode_extend_and_lea_variants() {
        let mut buf = Vec::new();
        encode_movzx_r32_rm8(
            10,
            Operand::Mem(Mem {
                base: 14,
                disp: 127,
            }),
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x45, 0x0f, 0xb6, 0x56, 0x7f]);

        buf.clear();
        encode_movzx_r32_rm16(
            10,
            Operand::Mem(Mem {
                base: 14,
                disp: 126,
            }),
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x45, 0x0f, 0xb7, 0x56, 0x7e]);

        buf.clear();
        encode_movsx_r64_rm8(
            10,
            Operand::Mem(Mem {
                base: 14,
                disp: 127,
            }),
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x0f, 0xbe, 0x56, 0x7f]);

        buf.clear();
        encode_movsx_r64_rm16(
            10,
            Operand::Mem(Mem {
                base: 14,
                disp: 126,
            }),
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x0f, 0xbf, 0x56, 0x7e]);

        buf.clear();
        encode_movsxd_r64_rm32(
            10,
            Operand::Mem(Mem {
                base: 14,
                disp: 124,
            }),
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x63, 0x56, 0x7c]);

        buf.clear();
        encode_movzx_r64_rm8(10, Operand::Reg(10), &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x0f, 0xb6, 0xd2]);

        buf.clear();
        encode_movzx_r64_rm16(10, 10, 126, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x0f, 0xb7, 0x52, 0x7e]);

        buf.clear();
        encode_lea_r64_m(10, Mem { base: 4, disp: 16 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x8d, 0x54, 0x24, 0x10]);
    }

    #[test]
    fn encode_integer_and_shift_ops() {
        let mut buf = Vec::new();
        encode_add_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x01, 0xda]);

        buf.clear();
        encode_add_r64_imm32(10, 0x1000, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0x81, 0xc2, 0x00, 0x10, 0x00, 0x00]);

        buf.clear();
        encode_add_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x03, 0x14, 0x24]);

        buf.clear();
        encode_sub_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x29, 0xda]);

        buf.clear();
        encode_sub_r64_imm32(10, 4, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0x81, 0xea, 0x04, 0x00, 0x00, 0x00]);

        buf.clear();
        encode_sub_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x2b, 0x14, 0x24]);

        buf.clear();
        encode_and_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x21, 0xda]);

        buf.clear();
        encode_and_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x23, 0x14, 0x24]);

        buf.clear();
        encode_or_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x09, 0xda]);

        buf.clear();
        encode_or_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x0b, 0x14, 0x24]);

        buf.clear();
        encode_xor_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x31, 0xda]);

        buf.clear();
        encode_xor_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x33, 0x14, 0x24]);

        buf.clear();
        encode_shl_r64_cl(10, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xd3, 0xe2]);

        buf.clear();
        encode_shr_r64_cl(10, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xd3, 0xea]);

        buf.clear();
        encode_shl_r64_imm8(10, 5, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xc1, 0xe2, 0x05]);

        buf.clear();
        encode_shr_r64_imm8(10, 7, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xc1, 0xea, 0x07]);

        buf.clear();
        encode_neg_r64(10, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xf7, 0xda]);
    }

    #[test]
    fn encode_sse2_vector_ops() {
        let mut buf = Vec::new();
        encode_movd_r64_xmm(7, 1, &mut buf).unwrap();
        assert_eq!(buf, [0x66, 0x0f, 0x7e, 0xcf]);

        buf.clear();
        encode_movd_xmm_r64(2, 7, &mut buf).unwrap();
        assert_eq!(buf, [0x66, 0x0f, 0x6e, 0xd7]);

        buf.clear();
        encode_pshufd_xmm_xmm_imm8(3, 1, 0x1b, &mut buf).unwrap();
        assert_eq!(buf, [0x66, 0x0f, 0x70, 0xd9, 0x1b]);

        buf.clear();
        encode_movdqa_xmm_xmm(1, 0, &mut buf).unwrap();
        assert_eq!(buf, [0x66, 0x0f, 0x6f, 0xc8]);

        buf.clear();
        encode_movdqu_xmm_m(2, 1, 0, &mut buf).unwrap();
        assert_eq!(buf, [0x66, 0x0f, 0x6f, 0x11]);

        buf.clear();
        encode_movdqu_m_xmm(3, 0, 2, &mut buf).unwrap();
        assert_eq!(buf, [0x66, 0x0f, 0x7f, 0x13]);

        buf.clear();
        encode_pcmpeqb_xmm_xmm(6, 7, &mut buf).unwrap();
        assert_eq!(buf, [0x66, 0x0f, 0x74, 0xf7]);

        buf.clear();
        encode_por_xmm_xmm(1, 4, &mut buf).unwrap();
        assert_eq!(buf, [0x66, 0x0f, 0xeb, 0xcc]);

        buf.clear();
        encode_pmovmskb_r32_xmm(3, 7, &mut buf).unwrap();
        assert_eq!(buf, [0x66, 0x0f, 0xd7, 0xdf]);
    }

    #[test]
    fn encode_additional_scalar_ops() {
        let mut buf = Vec::new();
        encode_tzcnt_r32_r32(0, 0, &mut buf).unwrap();
        assert_eq!(buf, [0xf3, 0x0f, 0xbc, 0xc0]);

        buf.clear();
        encode_imul_r64_r64(0, 1, &mut buf).unwrap();
        assert_eq!(buf, [0x48, 0x0f, 0xaf, 0xc8]);

        buf.clear();
        encode_imul_r64_r64_imm32(2, 7, 0x1234_5678, &mut buf).unwrap();
        assert_eq!(buf, [0x48, 0x69, 0xfa, 0x78, 0x56, 0x34, 0x12]);

        buf.clear();
        encode_sar_r64_cl(10, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xd3, 0xfa]);

        buf.clear();
        encode_sar_r64_imm8(10, 4, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xc1, 0xfa, 0x04]);

        buf.clear();
        encode_mov_r64_imm32_sext(9, 0x9abcdef0, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xc7, 0xc1, 0xf0, 0xde, 0xbc, 0x9a]);

        buf.clear();
        encode_rep_movsb(&mut buf).unwrap();
        assert_eq!(buf, [0xf3, 0xa4]);

        buf.clear();
        encode_cld(&mut buf).unwrap();
        assert_eq!(buf, [0xfc]);

        buf.clear();
        encode_xor_r32_r32(1, 2, &mut buf).unwrap();
        assert_eq!(buf, [0x31, 0xd1]);
    }

    #[test]
    fn encode_cmp_test_setcc() {
        let mut buf = Vec::new();
        encode_cmp_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x39, 0xda]);

        buf.clear();
        encode_cmp_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x3b, 0x14, 0x24]);

        buf.clear();
        encode_test_r64_r64(10, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x85, 0xd2]);

        buf.clear();
        encode_test_r32_r32(10, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x45, 0x85, 0xd2]);

        buf.clear();
        encode_setne_r8(10, &mut buf).unwrap();
        assert_eq!(buf, [0x41, 0x0f, 0x95, 0xc2]);

        buf.clear();
        encode_test_r8_r8(9, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x45, 0x84, 0xd1]);

        buf.clear();
        encode_setcc_r8(Condition::Ne, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x41, 0x0f, 0x95, 0xc3]);

        buf.clear();
        encode_cmovcc_r64_r64(Condition::Lt, 10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x0f, 0x4c, 0xd3]);

        buf.clear();
        encode_cmp_r64_imm32(11, 3, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0x81, 0xfb, 0x03, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn encode_control_flow_and_stack() {
        let mut buf = Vec::new();
        encode_je_rel32(&mut buf, 0x13d).unwrap();
        assert_eq!(buf, [0x0f, 0x84, 0x3d, 0x01, 0x00, 0x00]);

        buf.clear();
        encode_jz_rel32(&mut buf, 0x137).unwrap();
        assert_eq!(buf, [0x0f, 0x84, 0x37, 0x01, 0x00, 0x00]);

        buf.clear();
        encode_jnz_rel32(&mut buf, 0x131).unwrap();
        assert_eq!(buf, [0x0f, 0x85, 0x31, 0x01, 0x00, 0x00]);

        buf.clear();
        encode_call_rel32(&mut buf, 0x12c).unwrap();
        assert_eq!(buf, [0xe8, 0x2c, 0x01, 0x00, 0x00]);

        buf.clear();
        encode_call_r64(0, &mut buf).unwrap();
        assert_eq!(buf, [0xff, 0xd0]);

        buf.clear();
        encode_call_r64(10, &mut buf).unwrap();
        assert_eq!(buf, [0x41, 0xff, 0xd2]);

        buf.clear();
        encode_push_r64(10, &mut buf).unwrap();
        assert_eq!(buf, [0x41, 0x52]);

        buf.clear();
        encode_pop_r64(10, &mut buf).unwrap();
        assert_eq!(buf, [0x41, 0x5a]);

        buf.clear();
        encode_jmp_r64(10, &mut buf).unwrap();
        assert_eq!(buf, [0x41, 0xff, 0xe2]);

        buf.clear();
        encode_ret(&mut buf).unwrap();
        assert_eq!(buf, [0xc3]);

        buf.clear();
        encode_nop(&mut buf).unwrap();
        assert_eq!(buf, [0x90]);

        buf.clear();
        encode_jcc_rel32(&mut buf, Condition::Hi, 0x14).unwrap();
        assert_eq!(buf, [0x0f, 0x87, 0x14, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn encode_addressing_edge_cases() {
        let mut buf = Vec::new();
        encode_mov_m_r64(Mem { base: 13, disp: 0 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x89, 0x55, 0x00]);

        buf.clear();
        encode_mov_r64_m(10, Mem { base: 13, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x8b, 0x55, 0x00]);

        buf.clear();
        encode_mov_m_r64(
            Mem {
                base: 12,
                disp: 4096,
            },
            10,
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x89, 0x94, 0x24, 0x00, 0x10, 0x00, 0x00]);

        buf.clear();
        encode_lea_r64_m(10, Mem { base: 13, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x8d, 0x55, 0x00]);

        buf.clear();
        encode_lea_r64_m(
            10,
            Mem {
                base: 13,
                disp: 4096,
            },
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x8d, 0x95, 0x00, 0x10, 0x00, 0x00]);

        buf.clear();
        encode_test_m_r64(
            Mem {
                base: 13,
                disp: 4096,
            },
            10,
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x85, 0x95, 0x00, 0x10, 0x00, 0x00]);
    }

    #[test]
    fn emitter_records_source_map_and_resolves_fixups() {
        let mut emitter = Emitter::new();
        let start = emitter.new_label();
        let done = emitter.new_label();

        emitter.bind_label(start).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 3,
            line: 10,
            column: 1,
        });
        emitter.emit_je_label(done).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 3,
            line: 11,
            column: 1,
        });
        emitter.emit_call_label(start).unwrap();
        emitter.bind_label(done).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 3,
            line: 12,
            column: 1,
        });
        emitter.emit_jnz_label(start).unwrap();

        let finalized = emitter.finalize().unwrap();
        assert_eq!(
            finalized.source_map,
            vec![
                crate::SourceMapEntry {
                    offset: 0,
                    location: crate::SourceLocation {
                        file: 3,
                        line: 10,
                        column: 1,
                    },
                },
                crate::SourceMapEntry {
                    offset: 6,
                    location: crate::SourceLocation {
                        file: 3,
                        line: 11,
                        column: 1,
                    },
                },
                crate::SourceMapEntry {
                    offset: 11,
                    location: crate::SourceLocation {
                        file: 3,
                        line: 12,
                        column: 1,
                    },
                },
            ]
        );
        assert_eq!(
            finalized.exec.as_ref(),
            vec![
                0x0f, 0x84, 0x05, 0x00, 0x00, 0x00, // je +5
                0xe8, 0xf5, 0xff, 0xff, 0xff, // call -11
                0x0f, 0x85, 0xef, 0xff, 0xff, 0xff, // jnz -17
            ]
        );

        let trace = finalized.trace_text().unwrap();
        assert_eq!(
            trace,
            "00000000 file=3 line=10 col=1 bytes=0f8405000000\n00000006 file=3 line=11 col=1 bytes=e8f5ffffff\n0000000b file=3 line=12 col=1 bytes=0f85efffffff"
        );

        let source_map_encoded = finalized.source_map_le().unwrap();
        assert_eq!(
            crate::decode_source_map_le(&source_map_encoded).unwrap(),
            finalized.source_map
        );
    }

    #[test]
    fn emitter_reports_unbound_label() {
        let mut emitter = Emitter::new();
        let dangling = emitter.new_label();
        emitter.emit_call_label(dangling).unwrap();
        let err = emitter.finalize().unwrap_err();
        assert!(matches!(err, EmitError::UnboundLabel { .. }));
    }

    #[test]
    fn emitter_reports_out_of_range_fixup() {
        let mut emitter = Emitter::new();
        let far = emitter.new_label();
        emitter.emit_call_label(far).unwrap();
        emitter.labels[far.0 as usize] = Some((i32::MAX as u32).saturating_add(1024));
        let err = emitter.finalize().unwrap_err();
        assert!(matches!(err, EmitError::RelativeOutOfRange { .. }));
    }

    #[test]
    fn invalid_register_is_rejected() {
        let mut buf = Vec::new();
        let err = encode_mov_r64_r64(0, 16, &mut buf).unwrap_err();
        assert!(matches!(err, EmitError::InvalidRegister { reg: 16 }));
    }

    #[test]
    fn encode_disp_size_boundaries() {
        let mut buf = Vec::new();
        encode_mov_m_r64(Mem { base: 5, disp: 0 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x89, 0x55, 0x00]);

        buf.clear();
        encode_mov_m_r64(Mem { base: 5, disp: 127 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x89, 0x55, 0x7f]);

        buf.clear();
        encode_mov_m_r64(Mem { base: 5, disp: 128 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x89, 0x95, 0x80, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn encode_rex_for_low_byte_registers() {
        let mut buf = Vec::new();
        encode_setne_r8(4, &mut buf).unwrap(); // spl
        assert_eq!(buf, [0x40, 0x0f, 0x95, 0xc4]);

        buf.clear();
        encode_movzx_r64_rm8(0, Operand::Reg(4), &mut buf).unwrap(); // movzx rax, spl
        assert_eq!(buf, [0x48, 0x0f, 0xb6, 0xc4]);
    }

    #[test]
    fn emitter_resolves_forward_and_backward_jumps() {
        let mut emitter = Emitter::new();
        let start = emitter.new_label();
        let target = emitter.new_label();
        emitter.bind_label(start).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 4,
            line: 1,
            column: 1,
        });
        emitter.emit_jmp_label(target).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 4,
            line: 2,
            column: 1,
        });
        emitter.emit_bytes(&[0x90, 0x90]); // 2 nops
        emitter.bind_label(target).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 4,
            line: 3,
            column: 1,
        });
        emitter.emit_jz_label(start).unwrap();

        let finalized = emitter.finalize().unwrap();
        assert_eq!(
            finalized.exec.as_ref(),
            vec![
                0xe9, 0x02, 0x00, 0x00, 0x00, // jmp +2
                0x90, 0x90, // nops
                0x0f, 0x84, 0xf3, 0xff, 0xff, 0xff, // jz -13
            ]
        );
    }
}
