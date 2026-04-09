use std::ptr;

use crate::common::EmissionState;
use kajit_types::{
    SourceLocation, SourceMap, SourceMapEntry, SourceMapError, TraceEntry, TraceError,
};

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

// SAFETY: ExecutableBuffer owns an immutable RX mapping after construction.
// Sharing or moving the handle across threads does not permit mutation through
// the raw pointer; destruction is still uniquely owned by the buffer value.
unsafe impl Send for ExecutableBuffer {}
// SAFETY: Shared references only expose read-only byte views and code pointers.
// The mapping itself is immutable after setup, so concurrent access is safe.
unsafe impl Sync for ExecutableBuffer {}

impl ExecutableBuffer {
    fn allocate(bytes: &[u8]) -> Self {
        let len = bytes.len().max(1);
        let prot = if cfg!(target_os = "macos") {
            PROT_READ | PROT_WRITE | PROT_EXEC
        } else {
            PROT_READ | PROT_WRITE
        };
        let mut flags = MAP_PRIVATE | MAP_ANON;
        #[cfg(target_os = "macos")]
        {
            flags |= MAP_JIT;
        }

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
            let ok = mprotect(ptr, len, PROT_READ | PROT_EXEC);
            if ok != 0 {
                munmap(ptr, len);
                panic!("mprotect failed to mark executable memory");
            }
        }

        Self { ptr, len }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Patch a fixed 4-instruction movz/movk/movk/movk sequence at `offset`
    /// to load the given 64-bit `value`. Used for data address relocations
    /// after the buffer is mmap'd and the runtime base address is known.
    ///
    /// # Safety
    /// `offset` must point to a valid 16-byte movz/movk sequence within the buffer.
    pub unsafe fn patch_u64_load(&self, offset: usize, value: u64) {
        assert!(
            offset + 16 <= self.len,
            "patch_u64_load: offset {offset} + 16 > len {}",
            self.len
        );
        let p0 = (value & 0xFFFF) as u16;
        let p1 = ((value >> 16) & 0xFFFF) as u16;
        let p2 = ((value >> 32) & 0xFFFF) as u16;
        let p3 = ((value >> 48) & 0xFFFF) as u16;

        let ptr = unsafe { self.ptr.add(offset) };

        #[cfg(target_os = "macos")]
        unsafe {
            pthread_jit_write_protect_np(0);
        }

        #[cfg(not(target_os = "macos"))]
        unsafe {
            mprotect(self.ptr, self.len, PROT_READ | PROT_WRITE);
        }

        // Patch the imm16 fields of the 4 instructions.
        // movz: bits [20:5] = imm16
        // movk: bits [20:5] = imm16
        for (i, imm) in [p0, p1, p2, p3].iter().enumerate() {
            unsafe {
                let inst_ptr = ptr.add(i * 4) as *mut u32;
                let word = inst_ptr.read_unaligned();
                let patched = (word & !(0xFFFF << 5)) | ((*imm as u32) << 5);
                inst_ptr.write_unaligned(patched);
            }
        }

        #[cfg(target_os = "macos")]
        unsafe {
            sys_icache_invalidate(ptr, 16);
            pthread_jit_write_protect_np(1);
        }

        #[cfg(not(target_os = "macos"))]
        unsafe {
            mprotect(self.ptr, self.len, PROT_READ | PROT_EXEC);
        }
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
pub enum Width {
    W32,
    X64,
}

impl Width {
    fn sf(self) -> u32 {
        match self {
            Self::W32 => 0,
            Self::X64 => 1,
        }
    }

    fn bits(self) -> u8 {
        match self {
            Self::W32 => 32,
            Self::X64 => 64,
        }
    }
}

/// AArch64 general-purpose register (0..=31).
///
/// Named constants are provided for all registers. `XZR` and `SP` are
/// both register 31 — the CPU disambiguates by instruction context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(u8);

impl Reg {
    pub const X0: Self = Self(0);
    pub const X1: Self = Self(1);
    pub const X2: Self = Self(2);
    pub const X3: Self = Self(3);
    pub const X4: Self = Self(4);
    pub const X5: Self = Self(5);
    pub const X6: Self = Self(6);
    pub const X7: Self = Self(7);
    pub const X8: Self = Self(8);
    pub const X9: Self = Self(9);
    pub const X10: Self = Self(10);
    pub const X11: Self = Self(11);
    pub const X12: Self = Self(12);
    pub const X13: Self = Self(13);
    pub const X14: Self = Self(14);
    pub const X15: Self = Self(15);
    pub const X16: Self = Self(16);
    pub const X17: Self = Self(17);
    pub const X18: Self = Self(18);
    pub const X19: Self = Self(19);
    pub const X20: Self = Self(20);
    pub const X21: Self = Self(21);
    pub const X22: Self = Self(22);
    pub const X23: Self = Self(23);
    pub const X24: Self = Self(24);
    pub const X25: Self = Self(25);
    pub const X26: Self = Self(26);
    pub const X27: Self = Self(27);
    pub const X28: Self = Self(28);
    pub const X29: Self = Self(29);
    pub const X30: Self = Self(30);
    pub const XZR: Self = Self(31);
    pub const SP: Self = Self(31);

    /// Create a Reg from a raw register number. Panics if > 31.
    pub const fn from_raw(n: u8) -> Self {
        assert!(n <= 31, "register number must be 0..=31");
        Self(n)
    }

    /// Get the raw register number.
    pub const fn raw(self) -> u8 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shift {
    Lsl = 0,
    Lsr = 1,
    Asr = 2,
    Ror = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    Eq = 0,
    Ne = 1,
    Hs = 2,
    Lo = 3,
    Mi = 4,
    Pl = 5,
    Vs = 6,
    Vc = 7,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
}

impl Condition {
    pub fn invert(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Hs => Self::Lo,
            Self::Lo => Self::Hs,
            Self::Mi => Self::Pl,
            Self::Pl => Self::Mi,
            Self::Vs => Self::Vc,
            Self::Vc => Self::Vs,
            Self::Hi => Self::Ls,
            Self::Ls => Self::Hi,
            Self::Ge => Self::Lt,
            Self::Lt => Self::Ge,
            Self::Gt => Self::Le,
            Self::Le => Self::Gt,
        }
    }

    /// Swap operands: return the condition that tests `b op a` given `a op b`.
    /// E.g., Hi (a > b unsigned) → Lo (b < a unsigned) ≡ Lo.
    pub fn swap_operands(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
            Self::Hi => Self::Lo,
            Self::Lo => Self::Hi,
            Self::Hs => Self::Ls,
            Self::Ls => Self::Hs,
            Self::Gt => Self::Lt,
            Self::Lt => Self::Gt,
            Self::Ge => Self::Le,
            Self::Le => Self::Ge,
            other => other, // Mi/Pl/Vs/Vc don't have swap semantics
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LabelId(u32);

#[derive(Debug)]
pub struct FinalizedEmission {
    pub code: Vec<u8>,
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

    pub fn is_empty(&self) -> bool {
        self.exec.is_empty()
    }

    pub fn trace_entries(&self) -> Result<Vec<TraceEntry>, TraceError> {
        kajit_types::build_trace(&self.code, &self.source_map)
    }

    pub fn trace_text(&self) -> Result<String, TraceError> {
        kajit_types::format_trace(&self.code, &self.source_map)
    }

    pub fn source_map_le(&self) -> Result<Vec<u8>, SourceMapError> {
        kajit_types::encode_source_map_le(&self.source_map)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    InvalidRegister {
        reg: Reg,
    },
    InvalidShiftAmount {
        width: Width,
        amount: u8,
    },
    InvalidMovWideShift {
        width: Width,
        shift: u8,
    },
    InvalidImmediate {
        instruction: &'static str,
        value: i64,
    },
    InvalidOffset {
        instruction: &'static str,
        offset: u32,
        align: u32,
        max: u32,
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
    BranchTargetNotAligned {
        at_offset: u32,
        target_offset: u32,
    },
    BranchOutOfRange {
        bits: u8,
        at_offset: u32,
        target_offset: u32,
        delta_words: i64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FixupKind {
    Imm26,
    Imm19,
    Imm14,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fixup {
    at_offset: u32,
    label: LabelId,
    kind: FixupKind,
}

#[derive(Debug, Clone, Default)]
pub struct Emitter {
    core: EmissionState,
    fixups: Vec<Fixup>,
    /// Label aliases: (from, to) — `from` resolves to whatever `to` resolves to.
    label_aliases: Vec<(LabelId, LabelId)>,
    /// Optional instruction capture for assembly dump/parse
    #[cfg(feature = "old-asm-adapter")]
    captured_instructions: Option<Vec<kajit_reprs::asm::aarch64_asm::Item>>,
}

impl Emitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Current code buffer length (offset of next emitted instruction).
    pub fn code_len(&self) -> usize {
        self.core.code_len()
    }

    /// Append raw bytes to the code buffer (for data sections appended after code).
    pub fn emit_raw_bytes(&mut self, bytes: &[u8]) {
        self.core.emit_raw_bytes(bytes);
    }

    /// Enable instruction capture for assembly dump/parse workflow
    #[cfg(feature = "old-asm-adapter")]
    pub fn enable_capture(&mut self) {
        self.captured_instructions = Some(Vec::new());
    }

    /// Get captured instructions and labels
    #[cfg(feature = "old-asm-adapter")]
    pub fn take_captured_program(&mut self) -> Option<kajit_reprs::asm::aarch64_asm::Program> {
        self.captured_instructions
            .take()
            .map(|items| kajit_reprs::asm::aarch64_asm::Program { items })
    }

    /// Capture an instruction (if capture is enabled)
    #[cfg(feature = "old-asm-adapter")]
    fn capture(&mut self, inst: kajit_reprs::asm::aarch64_asm::Instruction) {
        if let Some(ref mut captured) = self.captured_instructions {
            captured.push(kajit_reprs::asm::aarch64_asm::Item::Instruction(inst));
        }
    }

    /// Capture a label (if capture is enabled)
    #[cfg(feature = "old-asm-adapter")]
    fn capture_label(&mut self, label: LabelId) {
        if let Some(ref mut captured) = self.captured_instructions {
            captured.push(kajit_reprs::asm::aarch64_asm::Item::Label(
                kajit_reprs::asm::aarch64_asm::Label(label.0),
            ));
        }
    }

    #[cfg(not(feature = "old-asm-adapter"))]
    fn capture_label(&mut self, _label: LabelId) {}

    pub fn current_offset(&self) -> u32 {
        self.core.current_offset()
    }

    pub fn bytes(&self) -> &[u8] {
        self.core.bytes()
    }

    pub fn source_map(&self) -> &[SourceMapEntry] {
        self.core.source_map()
    }

    pub fn set_source_location(&mut self, loc: SourceLocation) {
        self.core.set_source_location(loc);
    }

    pub fn current_source_location(&self) -> SourceLocation {
        self.core.current_source_location()
    }

    pub fn new_label(&mut self) -> LabelId {
        LabelId(self.core.new_label())
    }

    pub fn bind_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.core
            .bind_label(label.0)
            .map_err(|existing_offset| match existing_offset {
                Some(existing_offset) => EmitError::LabelAlreadyBound {
                    label,
                    existing_offset,
                },
                None => EmitError::LabelOutOfBounds { label },
            })?;
        self.capture_label(label);
        Ok(())
    }

    /// Make `from` resolve to the same offset as `to` during finalization.
    /// Used to eliminate trampoline blocks: instead of emitting a label + `b target`,
    /// the trampoline's label becomes an alias for the target.
    /// `to` does not need to be bound yet — the alias is resolved during finalization.
    pub fn alias_label(&mut self, from: LabelId, to: LabelId) {
        self.label_aliases.push((from, to));
    }

    pub(crate) fn emit_word(&mut self, word: u32) {
        self.core.maybe_record_source_map();
        self.core.extend_buffer(&word.to_le_bytes());
    }

    pub fn emit_b_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.emit_word(encode_b(0)?);
        self.fixups.push(Fixup {
            at_offset: self.current_offset() - 4,
            label,
            kind: FixupKind::Imm26,
        });
        Ok(())
    }

    pub fn emit_bl_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        self.emit_word(encode_bl(0)?);
        self.fixups.push(Fixup {
            at_offset: self.current_offset() - 4,
            label,
            kind: FixupKind::Imm26,
        });
        Ok(())
    }

    pub fn emit_cbz_label(
        &mut self,
        width: Width,
        rt: Reg,
        label: LabelId,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_cbz(width, rt, 0)?);
        self.fixups.push(Fixup {
            at_offset: self.current_offset() - 4,
            label,
            kind: FixupKind::Imm19,
        });
        Ok(())
    }

    pub fn emit_cbnz_label(
        &mut self,
        width: Width,
        rt: Reg,
        label: LabelId,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_cbnz(width, rt, 0)?);
        self.fixups.push(Fixup {
            at_offset: self.current_offset() - 4,
            label,
            kind: FixupKind::Imm19,
        });
        Ok(())
    }

    pub fn emit_tbz_label(&mut self, rt: Reg, bit: u8, label: LabelId) -> Result<(), EmitError> {
        self.emit_word(encode_tbz(rt, bit, 0)?);
        self.fixups.push(Fixup {
            at_offset: self.current_offset() - 4,
            label,
            kind: FixupKind::Imm14,
        });
        Ok(())
    }

    pub fn emit_tbnz_label(&mut self, rt: Reg, bit: u8, label: LabelId) -> Result<(), EmitError> {
        self.emit_word(encode_tbnz(rt, bit, 0)?);
        self.fixups.push(Fixup {
            at_offset: self.current_offset() - 4,
            label,
            kind: FixupKind::Imm14,
        });
        Ok(())
    }

    pub fn emit_b_cond_label(&mut self, cond: Condition, label: LabelId) -> Result<(), EmitError> {
        self.emit_word(encode_b_cond(cond, 0)?);
        self.fixups.push(Fixup {
            at_offset: self.current_offset() - 4,
            label,
            kind: FixupKind::Imm19,
        });
        Ok(())
    }

    // Non-branch instruction wrappers that capture + emit

    pub fn emit_mov_reg(&mut self, width: Width, rd: Reg, rm: Reg) -> Result<(), EmitError> {
        self.emit_word(encode_mov_reg(width, rd, rm)?);
        Ok(())
    }

    pub fn emit_movz_imm(
        &mut self,
        width: Width,
        rd: Reg,
        imm: u16,
        shift: u8,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_movz(width, rd, imm, shift)?);
        Ok(())
    }

    pub fn emit_movk_imm(
        &mut self,
        width: Width,
        rd: Reg,
        imm: u16,
        shift: u8,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_movk(width, rd, imm, shift)?);
        Ok(())
    }

    pub fn emit_ldr_imm(
        &mut self,
        width: Width,
        rt: Reg,
        rn: Reg,
        offset: u32,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_ldr_imm(width, rt, rn, offset)?);
        Ok(())
    }

    pub fn emit_ldrb_imm(&mut self, rt: Reg, rn: Reg, offset: u32) -> Result<(), EmitError> {
        self.emit_word(encode_ldrb_imm(rt, rn, offset)?);
        Ok(())
    }

    pub fn emit_ldrh_imm(&mut self, rt: Reg, rn: Reg, offset: u32) -> Result<(), EmitError> {
        self.emit_word(encode_ldrh_imm(rt, rn, offset)?);
        Ok(())
    }

    pub fn emit_str_imm(
        &mut self,
        width: Width,
        rt: Reg,
        rn: Reg,
        offset: u32,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_str_imm(width, rt, rn, offset)?);
        Ok(())
    }

    pub fn emit_strb_imm(&mut self, rt: Reg, rn: Reg, offset: u32) -> Result<(), EmitError> {
        self.emit_word(encode_strb_imm(rt, rn, offset)?);
        Ok(())
    }

    pub fn emit_strh_imm(&mut self, rt: Reg, rn: Reg, offset: u32) -> Result<(), EmitError> {
        self.emit_word(encode_strh_imm(rt, rn, offset)?);
        Ok(())
    }

    pub fn emit_add_imm(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        imm: u16,
        shift: bool,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_add_imm(width, rd, rn, imm, shift)?);
        Ok(())
    }

    pub fn emit_sub_imm(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        imm: u16,
        shift: bool,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_sub_imm(width, rd, rn, imm, shift)?);
        Ok(())
    }

    pub fn emit_add_reg(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_add_reg(width, rd, rn, rm)?);
        Ok(())
    }

    pub fn emit_sub_reg(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_sub_reg(width, rd, rn, rm)?);
        Ok(())
    }

    pub fn emit_and_reg(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_and_reg(width, rd, rn, rm, Shift::Lsl, 0)?);
        Ok(())
    }

    pub fn emit_orr_reg(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_orr_reg(width, rd, rn, rm, Shift::Lsl, 0)?);
        Ok(())
    }

    pub fn emit_ret(&mut self) -> Result<(), EmitError> {
        self.emit_word(encode_ret(Reg::X30)?); // X30 is the link register
        Ok(())
    }

    pub fn emit_nop(&mut self) -> Result<(), EmitError> {
        self.emit_word(encode_nop()?);
        Ok(())
    }

    pub fn emit_eor_reg(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_eor_reg(width, rd, rn, rm, Shift::Lsl, 0)?);
        Ok(())
    }

    /// Emit BFI (Bitfield Insert): insert `bit_width` bits from low bits of `rn` into `rd` at position `lsb`.
    pub fn emit_bfi(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        lsb: u8,
        bit_width: u8,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_bfi(width, rd, rn, lsb, bit_width)?);
        Ok(())
    }

    pub fn emit_and_imm(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        imm: u64,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_and_imm(width, rd, rn, imm)?);
        Ok(())
    }

    pub fn emit_orr_imm(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        imm: u64,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_orr_imm(width, rd, rn, imm)?);
        Ok(())
    }

    pub fn emit_eor_imm(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        imm: u64,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_eor_imm(width, rd, rn, imm)?);
        Ok(())
    }

    pub fn emit_lsl_imm(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        shift: u8,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_lsl_imm(width, rd, rn, shift)?);
        Ok(())
    }

    pub fn emit_lsr_imm(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        shift: u8,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_lsr_imm(width, rd, rn, shift)?);
        Ok(())
    }

    pub fn emit_lsl_reg(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_lsl_reg(width, rd, rn, rm)?);
        Ok(())
    }

    pub fn emit_lsr_reg(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_lsr_reg(width, rd, rn, rm)?);
        Ok(())
    }

    pub fn emit_asr_imm(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        shift: u8,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_asr_imm(width, rd, rn, shift)?);
        Ok(())
    }

    pub fn emit_asr_reg(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_asr_reg(width, rd, rn, rm)?);
        Ok(())
    }

    pub fn emit_neg_reg(&mut self, width: Width, rd: Reg, rm: Reg) -> Result<(), EmitError> {
        self.emit_word(encode_neg(width, rd, rm)?);
        Ok(())
    }

    pub fn emit_cmp_reg(&mut self, width: Width, rn: Reg, rm: Reg) -> Result<(), EmitError> {
        self.emit_word(encode_cmp_reg(width, rn, rm)?);
        Ok(())
    }

    pub fn emit_cmp_imm(&mut self, width: Width, rn: Reg, imm: u16) -> Result<(), EmitError> {
        self.emit_word(encode_cmp_imm(width, rn, imm, false)?);
        Ok(())
    }

    pub fn emit_mul_reg(
        &mut self,
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_mul(width, rd, rn, rm)?);
        Ok(())
    }

    pub fn emit_cset(&mut self, width: Width, rd: Reg, cond: Condition) -> Result<(), EmitError> {
        self.emit_word(encode_cset(width, rd, cond)?);
        Ok(())
    }

    pub fn emit_sxtb(&mut self, width: Width, rd: Reg, rn: Reg) -> Result<(), EmitError> {
        self.emit_word(encode_sxtb(rd, rn)?);
        Ok(())
    }

    pub fn emit_sxth(&mut self, width: Width, rd: Reg, rn: Reg) -> Result<(), EmitError> {
        self.emit_word(encode_sxth(rd, rn)?);
        Ok(())
    }

    pub fn emit_sxtw(&mut self, rd: Reg, rn: Reg) -> Result<(), EmitError> {
        self.emit_word(encode_sxtw(rd, rn)?);
        Ok(())
    }

    pub fn emit_stp(
        &mut self,
        width: Width,
        rt1: Reg,
        rt2: Reg,
        rn: Reg,
        offset: i16,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_stp(width, rt1, rt2, rn, offset)?);
        Ok(())
    }

    pub fn emit_ldp(
        &mut self,
        width: Width,
        rt1: Reg,
        rt2: Reg,
        rn: Reg,
        offset: i16,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_ldp(width, rt1, rt2, rn, offset)?);
        Ok(())
    }

    pub fn emit_load_extern(
        &mut self,
        rd: Reg,
        symbol: String,
        addr: u64,
    ) -> Result<(), EmitError> {
        // Capture the symbolic LoadExtern instruction

        // Emit the actual movz/movk sequence
        let p0 = (addr & 0xFFFF) as u16;
        let p1 = ((addr >> 16) & 0xFFFF) as u16;
        let p2 = ((addr >> 32) & 0xFFFF) as u16;
        let p3 = ((addr >> 48) & 0xFFFF) as u16;

        self.emit_word(encode_movz(Width::X64, rd, p0, 0)?);
        if p1 != 0 {
            self.emit_word(encode_movk(Width::X64, rd, p1, 16)?);
        }
        if p2 != 0 {
            self.emit_word(encode_movk(Width::X64, rd, p2, 32)?);
        }
        if p3 != 0 {
            self.emit_word(encode_movk(Width::X64, rd, p3, 48)?);
        }

        Ok(())
    }

    pub fn emit_blr(&mut self, rn: Reg) -> Result<(), EmitError> {
        self.emit_word(encode_blr(rn)?);
        Ok(())
    }

    pub fn finalize(mut self) -> Result<FinalizedEmission, EmitError> {
        // Resolve label aliases: bind aliased labels to their target's offset.
        // Iterate to handle transitive aliases (A → B → C).
        for _ in 0..16 {
            let mut resolved_any = false;
            for &(from, to) in &self.label_aliases {
                if self.core.label_offset(from.0) == Some(None)
                    && let Some(Some(target_offset)) = self.core.label_offset(to.0)
                {
                    *self
                        .core
                        .label_slot_mut(from.0)
                        .expect("alias source label should exist") = Some(target_offset);
                    resolved_any = true;
                }
            }
            if !resolved_any {
                break;
            }
        }

        for fixup in self.fixups.iter().copied() {
            let target = self
                .core
                .label_offset(fixup.label.0)
                .ok_or(EmitError::LabelOutOfBounds { label: fixup.label })?
                .ok_or(EmitError::UnboundLabel { label: fixup.label })?;

            if (target & 0b11) != 0 || (fixup.at_offset & 0b11) != 0 {
                return Err(EmitError::BranchTargetNotAligned {
                    at_offset: fixup.at_offset,
                    target_offset: target,
                });
            }

            let delta_words = (target as i64 - fixup.at_offset as i64) / 4;
            let word = self.core.read_u32_le(fixup.at_offset as usize);

            let patched = match fixup.kind {
                FixupKind::Imm26 => {
                    check_signed_bits("branch26", delta_words, 26)?;
                    (word & !0x03ff_ffff) | ((delta_words as u32) & 0x03ff_ffff)
                }
                FixupKind::Imm19 => {
                    check_signed_bits("branch19", delta_words, 19)?;
                    (word & !(0x7ffff << 5)) | (((delta_words as u32) & 0x7ffff) << 5)
                }
                FixupKind::Imm14 => {
                    check_signed_bits("branch14", delta_words, 14)?;
                    (word & !(0x3fff << 5)) | (((delta_words as u32) & 0x3fff) << 5)
                }
            };

            self.core
                .patch_buffer(fixup.at_offset as usize, &patched.to_le_bytes());
        }

        let (code, source_map) = self.core.into_parts();
        let exec = ExecutableBuffer::allocate(&code);
        Ok(FinalizedEmission {
            code,
            exec,
            source_map,
        })
    }
}

fn check_reg(reg: Reg) -> u32 {
    reg.raw() as u32
}

fn check_shift_amount(width: Width, amount: u8) -> Result<u32, EmitError> {
    if amount < width.bits() {
        Ok(amount as u32)
    } else {
        Err(EmitError::InvalidShiftAmount { width, amount })
    }
}

fn check_signed_bits(instruction: &'static str, value: i64, bits: u8) -> Result<(), EmitError> {
    let min = -(1i64 << (bits - 1));
    let max = (1i64 << (bits - 1)) - 1;
    if value < min || value > max {
        Err(EmitError::InvalidImmediate { instruction, value })
    } else {
        Ok(())
    }
}

fn emit_logical_shifted_reg(
    base: u32,
    width: Width,
    rd: Reg,
    rn: Reg,
    rm: Reg,
    shift: Shift,
    amount: u8,
) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    let amount = check_shift_amount(width, amount)?;
    Ok((width.sf() << 31)
        | base
        | ((shift as u32) << 22)
        | (rm << 16)
        | (amount << 10)
        | (rn << 5)
        | rd)
}

pub fn encode_mov_reg(width: Width, rd: Reg, rm: Reg) -> Result<u32, EmitError> {
    encode_orr_reg(width, rd, Reg::XZR, rm, Shift::Lsl, 0)
}

pub fn encode_movz(width: Width, rd: Reg, imm16: u16, shift: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    if !matches!(shift, 0 | 16 | 32 | 48) {
        return Err(EmitError::InvalidMovWideShift { width, shift });
    }
    if width == Width::W32 && shift > 16 {
        return Err(EmitError::InvalidMovWideShift { width, shift });
    }
    let hw = (shift / 16) as u32;
    Ok((width.sf() << 31) | 0x5280_0000 | (hw << 21) | ((imm16 as u32) << 5) | rd)
}

pub fn encode_movk(width: Width, rd: Reg, imm16: u16, shift: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    if !matches!(shift, 0 | 16 | 32 | 48) {
        return Err(EmitError::InvalidMovWideShift { width, shift });
    }
    if width == Width::W32 && shift > 16 {
        return Err(EmitError::InvalidMovWideShift { width, shift });
    }
    let hw = (shift / 16) as u32;
    Ok((width.sf() << 31) | 0x7280_0000 | (hw << 21) | ((imm16 as u32) << 5) | rd)
}

fn encode_load_store_unsigned(
    base: u32,
    rt: Reg,
    rn: Reg,
    offset: u32,
    align: u32,
) -> Result<u32, EmitError> {
    let rt = check_reg(rt);
    let rn = check_reg(rn);
    let max = 4095 * align;
    if !offset.is_multiple_of(align) || offset > max {
        return Err(EmitError::InvalidOffset {
            instruction: "ld/st unsigned",
            offset,
            align,
            max,
        });
    }
    let imm12 = offset / align;
    Ok(base | (imm12 << 10) | (rn << 5) | rt)
}

fn encode_pair_signed_offset(
    instruction: &'static str,
    base: u32,
    width: Width,
    rt1: Reg,
    rt2: Reg,
    rn: Reg,
    offset: i16,
) -> Result<u32, EmitError> {
    let rt1 = check_reg(rt1);
    let rt2 = check_reg(rt2);
    let rn = check_reg(rn);
    let align = match width {
        Width::W32 => 4,
        Width::X64 => 8,
    } as i16;
    if offset % align != 0 {
        return Err(EmitError::InvalidImmediate {
            instruction,
            value: offset as i64,
        });
    }
    let scaled = offset / align;
    if !(-64..=63).contains(&scaled) {
        return Err(EmitError::InvalidImmediate {
            instruction,
            value: offset as i64,
        });
    }
    let imm7 = (scaled as i32 as u32) & 0x7f;
    Ok(base | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1)
}

pub fn encode_ldr_imm(width: Width, rt: Reg, rn: Reg, offset: u32) -> Result<u32, EmitError> {
    let base = match width {
        Width::W32 => 0xB940_0000,
        Width::X64 => 0xF940_0000,
    };
    let align = match width {
        Width::W32 => 4,
        Width::X64 => 8,
    };
    encode_load_store_unsigned(base, rt, rn, offset, align)
}

pub fn encode_str_imm(width: Width, rt: Reg, rn: Reg, offset: u32) -> Result<u32, EmitError> {
    let base = match width {
        Width::W32 => 0xB900_0000,
        Width::X64 => 0xF900_0000,
    };
    let align = match width {
        Width::W32 => 4,
        Width::X64 => 8,
    };
    encode_load_store_unsigned(base, rt, rn, offset, align)
}

pub fn encode_ldrb_imm(rt: Reg, rn: Reg, offset: u32) -> Result<u32, EmitError> {
    encode_load_store_unsigned(0x3940_0000, rt, rn, offset, 1)
}

pub fn encode_ldrh_imm(rt: Reg, rn: Reg, offset: u32) -> Result<u32, EmitError> {
    encode_load_store_unsigned(0x7940_0000, rt, rn, offset, 2)
}

pub fn encode_strb_imm(rt: Reg, rn: Reg, offset: u32) -> Result<u32, EmitError> {
    encode_load_store_unsigned(0x3900_0000, rt, rn, offset, 1)
}

pub fn encode_strh_imm(rt: Reg, rn: Reg, offset: u32) -> Result<u32, EmitError> {
    encode_load_store_unsigned(0x7900_0000, rt, rn, offset, 2)
}

pub fn encode_stp(
    width: Width,
    rt1: Reg,
    rt2: Reg,
    rn: Reg,
    offset: i16,
) -> Result<u32, EmitError> {
    let base = match width {
        Width::W32 => 0x2900_0000,
        Width::X64 => 0xA900_0000,
    };
    encode_pair_signed_offset("stp", base, width, rt1, rt2, rn, offset)
}

pub fn encode_ldp(
    width: Width,
    rt1: Reg,
    rt2: Reg,
    rn: Reg,
    offset: i16,
) -> Result<u32, EmitError> {
    let base = match width {
        Width::W32 => 0x2940_0000,
        Width::X64 => 0xA940_0000,
    };
    encode_pair_signed_offset("ldp", base, width, rt1, rt2, rn, offset)
}

pub fn encode_ret(rn: Reg) -> Result<u32, EmitError> {
    let rn = check_reg(rn);
    Ok(0xD65F_0000 | (rn << 5))
}

pub fn encode_add_reg(width: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    Ok((width.sf() << 31) | 0x0B00_0000 | (rm << 16) | (rn << 5) | rd)
}

pub fn encode_sub_reg(width: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    Ok((width.sf() << 31) | 0x4B00_0000 | (rm << 16) | (rn << 5) | rd)
}

/// Encode an ARM64 logical immediate value.
/// Returns (N, immr, imms) if the value is encodable, None otherwise.
///
/// ARM64 logical immediates encode bitmasks of consecutive 1s that repeat.
/// The element size can be 2, 4, 8, 16, 32, or 64 bits.
fn encode_logical_imm_params(value: u64, is_64bit: bool) -> Option<(u32, u32, u32)> {
    // All-zeros and all-ones are not encodable
    if value == 0 {
        return None;
    }

    let reg_size = if is_64bit { 64u32 } else { 32u32 };

    // For 32-bit, the value must fit in 32 bits and we replicate it
    let value = if is_64bit {
        value
    } else {
        if (value >> 32) != 0 && (value >> 32) != (value & 0xFFFF_FFFF) {
            return None;
        }
        // Replicate 32-bit pattern to 64-bit for uniform handling
        let lo = value & 0xFFFF_FFFF;
        lo | (lo << 32)
    };

    if value == u64::MAX {
        return None;
    }

    // Find the smallest element size where the pattern repeats
    let mut element_size = reg_size;
    let mut element_mask = u64::MAX;

    // Try progressively smaller element sizes
    for size in [32u32, 16, 8, 4, 2] {
        if size > reg_size {
            continue;
        }
        let mask = (1u64 << size) - 1;
        let element = value & mask;

        // Check if this element repeats across the register
        let mut repeats = true;
        let mut pos = size;
        while pos < 64 {
            if ((value >> pos) & mask) != element {
                repeats = false;
                break;
            }
            pos += size;
        }

        if repeats {
            element_size = size;
            element_mask = mask;
        }
    }

    // Extract the element pattern
    let element = value & element_mask;

    // A valid logical immediate element has a contiguous run of 1s (possibly wrapped)
    // Find the rotation and run length

    // Rotate the element to find a canonical form with 1s at the bottom
    let mut rotated = element;
    let mut rotation = 0u32;

    // Find where the run of 1s starts (rotate until bit 0 is 1 and highest bit is 0)
    // We want the form: 0...0 1...1
    for r in 0..element_size {
        let rot = element.rotate_right(r);
        // Check if this rotation has 1s at bottom and 0s at top
        let ones = rot.trailing_ones();
        if ones > 0 && ones < element_size && (rot >> ones) == 0 {
            rotated = rot;
            rotation = r;
            break;
        }
        // Also check for all-ones element (not valid for logical imm)
        if rot == element_mask {
            return None;
        }
    }

    // Count consecutive 1s from the bottom
    let ones_count = rotated.trailing_ones();
    if ones_count == 0 || ones_count >= element_size {
        return None;
    }

    // Verify the rest is all zeros
    if (rotated >> ones_count) != 0 {
        return None;
    }

    // Encode N, immr, imms
    let n: u32;
    let imms: u32;

    if element_size == 64 {
        n = 1;
        imms = ones_count - 1;
    } else {
        n = 0;
        // imms encodes both element size and run length
        // The top bits of imms (inverted) encode element size:
        // 32-bit: 0xxxxx (bit 5 = 0)
        // 16-bit: 10xxxx (bits 5:4 = 10)
        // 8-bit:  110xxx (bits 5:3 = 110)
        // 4-bit:  1110xx (bits 5:2 = 1110)
        // 2-bit:  11110x (bits 5:1 = 11110)
        let size_encoding = match element_size {
            32 => 0b00_0000,
            16 => 0b10_0000,
            8 => 0b11_0000,
            4 => 0b11_1000,
            2 => 0b11_1100,
            _ => return None,
        };
        imms = size_encoding | (ones_count - 1);
    }

    // immr is the rotation amount within the element
    let immr = if rotation == 0 {
        0
    } else {
        element_size - rotation
    };

    // For 32-bit width, N must be 0
    if !is_64bit && n != 0 {
        return None;
    }

    Some((n, immr, imms))
}

fn encode_add_sub_imm(
    width: Width,
    rd: Reg,
    rn: Reg,
    imm12: u16,
    shift12: bool,
    is_sub: bool,
) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    if imm12 > 0x0fff {
        return Err(EmitError::InvalidImmediate {
            instruction: "add/sub imm12",
            value: imm12 as i64,
        });
    }
    let opcode = match (width, is_sub) {
        (Width::W32, false) => 0x1100_0000,
        (Width::X64, false) => 0x9100_0000,
        (Width::W32, true) => 0x5100_0000,
        (Width::X64, true) => 0xD100_0000,
    };
    Ok(opcode | ((shift12 as u32) << 22) | ((imm12 as u32) << 10) | (rn << 5) | rd)
}

pub fn encode_add_imm(
    width: Width,
    rd: Reg,
    rn: Reg,
    imm12: u16,
    shift12: bool,
) -> Result<u32, EmitError> {
    encode_add_sub_imm(width, rd, rn, imm12, shift12, false)
}

pub fn encode_sub_imm(
    width: Width,
    rd: Reg,
    rn: Reg,
    imm12: u16,
    shift12: bool,
) -> Result<u32, EmitError> {
    encode_add_sub_imm(width, rd, rn, imm12, shift12, true)
}

pub fn encode_and_reg(
    width: Width,
    rd: Reg,
    rn: Reg,
    rm: Reg,
    shift: Shift,
    amount: u8,
) -> Result<u32, EmitError> {
    emit_logical_shifted_reg(0x0A00_0000, width, rd, rn, rm, shift, amount)
}

pub fn encode_orr_reg(
    width: Width,
    rd: Reg,
    rn: Reg,
    rm: Reg,
    shift: Shift,
    amount: u8,
) -> Result<u32, EmitError> {
    emit_logical_shifted_reg(0x2A00_0000, width, rd, rn, rm, shift, amount)
}

pub fn encode_eor_reg(
    width: Width,
    rd: Reg,
    rn: Reg,
    rm: Reg,
    shift: Shift,
    amount: u8,
) -> Result<u32, EmitError> {
    emit_logical_shifted_reg(0x4A00_0000, width, rd, rn, rm, shift, amount)
}

pub fn encode_lsl_reg(width: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    Ok((width.sf() << 31) | 0x1AC0_2000 | (rm << 16) | (rn << 5) | rd)
}

pub fn encode_lsr_reg(width: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    Ok((width.sf() << 31) | 0x1AC0_2400 | (rm << 16) | (rn << 5) | rd)
}

pub fn encode_neg(width: Width, rd: Reg, rm: Reg) -> Result<u32, EmitError> {
    encode_sub_reg(width, rd, Reg::XZR, rm)
}

pub fn encode_cmp_reg(width: Width, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    Ok((width.sf() << 31) | 0x6B00_001F | (rm << 16) | (rn << 5))
}

pub fn encode_cmp_imm(width: Width, rn: Reg, imm12: u16, shift12: bool) -> Result<u32, EmitError> {
    let rn = check_reg(rn);
    if imm12 > 0x0fff {
        return Err(EmitError::InvalidImmediate {
            instruction: "cmp imm12",
            value: imm12 as i64,
        });
    }
    let base = match width {
        Width::W32 => 0x7100_001F,
        Width::X64 => 0xF100_001F,
    };
    Ok(base | ((shift12 as u32) << 22) | ((imm12 as u32) << 10) | (rn << 5))
}

pub fn encode_subs_reg(width: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    Ok((width.sf() << 31) | 0x6B00_0000 | (rm << 16) | (rn << 5) | rd)
}

pub fn encode_cmn_imm(width: Width, rn: Reg, imm12: u16, shift12: bool) -> Result<u32, EmitError> {
    let rn = check_reg(rn);
    if imm12 > 0x0fff {
        return Err(EmitError::InvalidImmediate {
            instruction: "cmn imm12",
            value: imm12 as i64,
        });
    }
    let base = match width {
        Width::W32 => 0x3100_001F,
        Width::X64 => 0xB100_001F,
    };
    Ok(base | ((shift12 as u32) << 22) | ((imm12 as u32) << 10) | (rn << 5))
}

pub fn encode_cbz(width: Width, rt: Reg, imm19: i32) -> Result<u32, EmitError> {
    let rt = check_reg(rt);
    check_signed_bits("cbz", imm19 as i64, 19)?;
    let base = match width {
        Width::W32 => 0x3400_0000,
        Width::X64 => 0xB400_0000,
    };
    Ok(base | (((imm19 as u32) & 0x7ffff) << 5) | rt)
}

pub fn encode_cbnz(width: Width, rt: Reg, imm19: i32) -> Result<u32, EmitError> {
    let rt = check_reg(rt);
    check_signed_bits("cbnz", imm19 as i64, 19)?;
    let base = match width {
        Width::W32 => 0x3500_0000,
        Width::X64 => 0xB500_0000,
    };
    Ok(base | (((imm19 as u32) & 0x7ffff) << 5) | rt)
}

pub fn encode_tbz(rt: Reg, bit: u8, imm14: i32) -> Result<u32, EmitError> {
    let rt = check_reg(rt);
    if bit > 63 {
        return Err(EmitError::InvalidImmediate {
            instruction: "tbz bit",
            value: bit as i64,
        });
    }
    check_signed_bits("tbz", imm14 as i64, 14)?;
    let b5 = (bit >> 5) as u32;
    let b40 = (bit & 0x1f) as u32;
    Ok((b5 << 31) | 0x3600_0000 | (b40 << 19) | (((imm14 as u32) & 0x3fff) << 5) | rt)
}

pub fn encode_tbnz(rt: Reg, bit: u8, imm14: i32) -> Result<u32, EmitError> {
    let rt = check_reg(rt);
    if bit > 63 {
        return Err(EmitError::InvalidImmediate {
            instruction: "tbnz bit",
            value: bit as i64,
        });
    }
    check_signed_bits("tbnz", imm14 as i64, 14)?;
    let b5 = (bit >> 5) as u32;
    let b40 = (bit & 0x1f) as u32;
    Ok((b5 << 31) | 0x3700_0000 | (b40 << 19) | (((imm14 as u32) & 0x3fff) << 5) | rt)
}

pub fn encode_b_cond(cond: Condition, imm19: i32) -> Result<u32, EmitError> {
    check_signed_bits("b.cond", imm19 as i64, 19)?;
    Ok(0x5400_0000 | (((imm19 as u32) & 0x7ffff) << 5) | (cond as u32))
}

pub fn encode_cset(width: Width, rd: Reg, condition: Condition) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let inv = condition.invert() as u32;
    let base = match width {
        Width::W32 => 0x1A9F_07E0,
        Width::X64 => 0x9A9F_07E0,
    };
    Ok(base | (inv << 12) | rd)
}

pub fn encode_tst_reg(width: Width, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    let base = match width {
        Width::W32 => 0x6A00_001F,
        Width::X64 => 0xEA00_001F,
    };
    Ok(base | (rm << 16) | (rn << 5))
}

pub fn encode_tst_imm(width: Width, rn: Reg, imm: u64) -> Result<u32, EmitError> {
    let rn = check_reg(rn);
    if imm == 0 || !imm.is_power_of_two() {
        return Err(EmitError::InvalidImmediate {
            instruction: "tst imm",
            value: imm as i64,
        });
    }
    let width_bits = match width {
        Width::W32 => 32u32,
        Width::X64 => 64u32,
    };
    let bit = imm.trailing_zeros();
    if bit >= width_bits {
        return Err(EmitError::InvalidImmediate {
            instruction: "tst imm",
            value: imm as i64,
        });
    }
    if width == Width::W32 && (imm >> 32) != 0 {
        return Err(EmitError::InvalidImmediate {
            instruction: "tst imm",
            value: imm as i64,
        });
    }
    let n = if width == Width::X64 { 1u32 } else { 0u32 };
    let immr = (width_bits - bit) % width_bits;
    let imms = 0u32;
    let base = match width {
        Width::W32 => 0x7200_001F,
        Width::X64 => 0xF200_001F,
    };
    Ok(base | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5))
}

pub fn encode_sxtb(rd: Reg, rn: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    Ok(0x9340_1C00 | (rn << 5) | rd)
}

pub fn encode_sxth(rd: Reg, rn: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    Ok(0x9340_3C00 | (rn << 5) | rd)
}

pub fn encode_sxtw(rd: Reg, rn: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    Ok(0x9340_7C00 | (rn << 5) | rd)
}

pub fn encode_madd(width: Width, rd: Reg, rn: Reg, rm: Reg, ra: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    let ra = check_reg(ra);
    Ok((width.sf() << 31) | 0x1B00_0000 | (rm << 16) | (ra << 10) | (rn << 5) | rd)
}

pub fn encode_mul(width: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    encode_madd(width, rd, rn, rm, Reg::XZR)
}

pub fn encode_umulh(rd: Reg, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    Ok(0x9BC0_7C00 | (rm << 16) | (rn << 5) | rd)
}

pub fn encode_smull(rd: Reg, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    Ok(0x9B20_7C00 | (rm << 16) | (rn << 5) | rd)
}

pub fn encode_asr_imm(width: Width, rd: Reg, rn: Reg, shift: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let max_shift = match width {
        Width::W32 => 31,
        Width::X64 => 63,
    };
    if shift > max_shift {
        return Err(EmitError::InvalidShiftAmount {
            width,
            amount: shift,
        });
    }
    let base = match width {
        Width::W32 => 0x1300_0000,
        Width::X64 => 0x9340_0000,
    };
    let imms = max_shift as u32;
    Ok(base | ((shift as u32) << 16) | (imms << 10) | (rn << 5) | rd)
}

pub fn encode_asr_reg(width: Width, rd: Reg, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    Ok((width.sf() << 31) | 0x1AC0_2800 | (rm << 16) | (rn << 5) | rd)
}

pub fn encode_lsr_imm(width: Width, rd: Reg, rn: Reg, shift: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let (max_shift, base, imms) = match width {
        Width::W32 => (31u8, 0x5300_0000u32, 31u32),
        Width::X64 => (63u8, 0xD340_0000u32, 63u32),
    };
    if shift > max_shift {
        return Err(EmitError::InvalidShiftAmount {
            width,
            amount: shift,
        });
    }
    Ok(base | ((shift as u32) << 16) | (imms << 10) | (rn << 5) | rd)
}

pub fn encode_lsl_imm(width: Width, rd: Reg, rn: Reg, shift: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let (max_shift, base, width_bits) = match width {
        Width::W32 => (31u8, 0x5300_0000u32, 32u32),
        Width::X64 => (63u8, 0xD340_0000u32, 64u32),
    };
    if shift == 0 || shift > max_shift {
        return Err(EmitError::InvalidShiftAmount {
            width,
            amount: shift,
        });
    }
    let immr = (width_bits - shift as u32) % width_bits;
    let imms = width_bits - 1 - shift as u32;
    Ok(base | (immr << 16) | (imms << 10) | (rn << 5) | rd)
}

pub fn encode_and_imm(width: Width, rd: Reg, rn: Reg, imm: u64) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let is_64bit = width == Width::X64;

    let (n, immr, imms) =
        encode_logical_imm_params(imm, is_64bit).ok_or(EmitError::InvalidImmediate {
            instruction: "and imm",
            value: imm as i64,
        })?;

    let base = match width {
        Width::W32 => 0x1200_0000u32,
        Width::X64 => 0x9200_0000u32,
    };
    Ok(base | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd)
}

pub fn encode_orr_imm(width: Width, rd: Reg, rn: Reg, imm: u64) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let is_64bit = width == Width::X64;

    let (n, immr, imms) =
        encode_logical_imm_params(imm, is_64bit).ok_or(EmitError::InvalidImmediate {
            instruction: "orr imm",
            value: imm as i64,
        })?;

    let base = match width {
        Width::W32 => 0x3200_0000u32,
        Width::X64 => 0xB200_0000u32,
    };
    Ok(base | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd)
}

pub fn encode_eor_imm(width: Width, rd: Reg, rn: Reg, imm: u64) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let is_64bit = width == Width::X64;

    let (n, immr, imms) =
        encode_logical_imm_params(imm, is_64bit).ok_or(EmitError::InvalidImmediate {
            instruction: "eor imm",
            value: imm as i64,
        })?;

    let base = match width {
        Width::W32 => 0x5200_0000u32,
        Width::X64 => 0xD200_0000u32,
    };
    Ok(base | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd)
}

/// Encode BFI (Bitfield Insert): `BFI Rd, Rn, #lsb, #width`
///
/// Inserts `width` bits from the low bits of Rn into Rd starting at bit `lsb`.
/// Other bits of Rd are unchanged. This is an alias of BFM.
pub fn encode_bfi(
    width: Width,
    rd: Reg,
    rn: Reg,
    lsb: u8,
    bit_width: u8,
) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let reg_size: u32 = match width {
        Width::W32 => 32,
        Width::X64 => 64,
    };
    if lsb as u32 >= reg_size || bit_width == 0 || (lsb as u32 + bit_width as u32) > reg_size {
        return Err(EmitError::InvalidImmediate {
            instruction: "bfi",
            value: ((lsb as i64) << 8) | bit_width as i64,
        });
    }
    let immr = (reg_size - lsb as u32) % reg_size;
    let imms = bit_width as u32 - 1;
    let base: u32 = match width {
        Width::W32 => 0x3300_0000,
        Width::X64 => 0xB340_0000,
    };
    Ok(base | (immr << 16) | (imms << 10) | (rn << 5) | rd)
}

pub fn encode_csel(
    width: Width,
    rd: Reg,
    rn: Reg,
    rm: Reg,
    cond: Condition,
) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    Ok((width.sf() << 31) | 0x1A80_0000 | (rm << 16) | ((cond as u32) << 12) | (rn << 5) | rd)
}

pub fn encode_clz(width: Width, rd: Reg, rn: Reg) -> Result<u32, EmitError> {
    let rd = check_reg(rd);
    let rn = check_reg(rn);
    let base = match width {
        Width::W32 => 0x5AC0_1000,
        Width::X64 => 0xDAC0_1000,
    };
    Ok(base | (rn << 5) | rd)
}

pub fn encode_bic(
    width: Width,
    rd: Reg,
    rn: Reg,
    rm: Reg,
    shift: Shift,
    amount: u8,
) -> Result<u32, EmitError> {
    emit_logical_shifted_reg(0x0A20_0000, width, rd, rn, rm, shift, amount)
}

pub fn encode_movi_b16(vd: u8, imm8: u8) -> Result<u32, EmitError> {
    if vd > 31 {
        return Err(EmitError::InvalidRegister { reg: Reg(vd) });
    }
    let a = (imm8 >> 7) & 1;
    let b = (imm8 >> 6) & 1;
    let c = (imm8 >> 5) & 1;
    let defgh = imm8 & 0x1F;
    let vd = vd as u32;
    Ok(0x4F00_E400
        | (a as u32) << 18
        | (b as u32) << 17
        | (c as u32) << 16
        | (defgh as u32) << 5
        | vd)
}

pub fn encode_ld1_b16(vt: u8, rn: Reg) -> Result<u32, EmitError> {
    if vt > 31 {
        return Err(EmitError::InvalidRegister { reg: Reg(vt) });
    }
    let rn = check_reg(rn);
    Ok(0x4C40_7000 | (rn << 5) | vt as u32)
}

pub fn encode_cmeq_b16(vd: u8, vn: u8, vm: u8) -> Result<u32, EmitError> {
    if vd > 31 || vn > 31 || vm > 31 {
        return Err(EmitError::InvalidRegister {
            reg: Reg(if vd > 31 {
                vd
            } else if vn > 31 {
                vn
            } else {
                vm
            }),
        });
    }
    Ok(0x6E20_8C00 | (vm as u32) << 16 | (vn as u32) << 5 | vd as u32)
}

pub fn encode_orr_b16(vd: u8, vn: u8, vm: u8) -> Result<u32, EmitError> {
    if vd > 31 || vn > 31 || vm > 31 {
        return Err(EmitError::InvalidRegister {
            reg: Reg(if vd > 31 {
                vd
            } else if vn > 31 {
                vn
            } else {
                vm
            }),
        });
    }
    Ok(0x4EA0_1C00 | (vm as u32) << 16 | (vn as u32) << 5 | vd as u32)
}

pub fn encode_umaxv_b16(vd: u8, vn: u8) -> Result<u32, EmitError> {
    if vd > 31 || vn > 31 {
        return Err(EmitError::InvalidRegister {
            reg: Reg(if vd > 31 { vd } else { vn }),
        });
    }
    Ok(0x6E30_A800 | (vn as u32) << 5 | vd as u32)
}

pub fn encode_umov_b(rd: Reg, vn: u8, index: u8) -> Result<u32, EmitError> {
    if vn > 31 {
        return Err(EmitError::InvalidRegister { reg: Reg(vn) });
    }
    if index > 15 {
        return Err(EmitError::InvalidImmediate {
            instruction: "umov",
            value: index as i64,
        });
    }
    let rd = check_reg(rd);
    Ok(0x0E00_3C00 | (index as u32) << 17 | 1u32 << 16 | (vn as u32) << 5 | rd)
}

pub fn encode_ldrb_reg(rt: Reg, rn: Reg, rm: Reg) -> Result<u32, EmitError> {
    let rt = check_reg(rt);
    let rn = check_reg(rn);
    let rm = check_reg(rm);
    Ok(0x3860_6800 | (rm << 16) | (rn << 5) | rt)
}

pub fn encode_ucvtf_d_x(vd: u8, rn: Reg) -> Result<u32, EmitError> {
    if vd > 31 {
        return Err(EmitError::InvalidRegister { reg: Reg(vd) });
    }
    let rn = check_reg(rn);
    Ok(0x9E63_0000 | (rn << 5) | vd as u32)
}

pub fn encode_fmov_x_d(rd: Reg, vn: u8) -> Result<u32, EmitError> {
    if vn > 31 {
        return Err(EmitError::InvalidRegister { reg: Reg(vn) });
    }
    let rd = check_reg(rd);
    Ok(0x9E66_0000 | (vn as u32) << 5 | rd)
}

pub fn encode_b(imm26: i32) -> Result<u32, EmitError> {
    check_signed_bits("b", imm26 as i64, 26)?;
    Ok(0x1400_0000 | ((imm26 as u32) & 0x03ff_ffff))
}

pub fn encode_bl(imm26: i32) -> Result<u32, EmitError> {
    check_signed_bits("bl", imm26 as i64, 26)?;
    Ok(0x9400_0000 | ((imm26 as u32) & 0x03ff_ffff))
}

pub fn encode_blr(rn: Reg) -> Result<u32, EmitError> {
    let rn = check_reg(rn);
    Ok(0xD63F_0000 | (rn << 5))
}

pub fn encode_nop() -> Result<u32, EmitError> {
    Ok(0xD503_201F)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(bytes: &[u8]) -> u32 {
        let arr: [u8; 4] = bytes.try_into().expect("need 4 bytes");
        u32::from_le_bytes(arr)
    }

    #[test]
    fn encode_mov_and_mov_wide() {
        assert_eq!(
            encode_mov_reg(Width::X64, Reg::from_raw(1), Reg::from_raw(2)).unwrap(),
            0xAA02_03E1
        );
        assert_eq!(
            encode_mov_reg(Width::W32, Reg::from_raw(1), Reg::from_raw(2)).unwrap(),
            0x2A02_03E1
        );

        assert_eq!(
            encode_movz(Width::X64, Reg::from_raw(3), 0x1234, 0).unwrap(),
            0xD282_4683
        );
        assert_eq!(
            encode_movz(Width::X64, Reg::from_raw(3), 0x1234, 16).unwrap(),
            0xD2A2_4683
        );
        assert_eq!(
            encode_movz(Width::W32, Reg::from_raw(3), 0x1234, 0).unwrap(),
            0x5282_4683
        );
        assert_eq!(
            encode_movz(Width::W32, Reg::from_raw(3), 0x1234, 16).unwrap(),
            0x52A2_4683
        );

        assert_eq!(
            encode_movk(Width::X64, Reg::from_raw(3), 0x5678, 0).unwrap(),
            0xF28A_CF03
        );
        assert_eq!(
            encode_movk(Width::X64, Reg::from_raw(3), 0x5678, 32).unwrap(),
            0xF2CA_CF03
        );
        assert_eq!(
            encode_movk(Width::W32, Reg::from_raw(3), 0x5678, 0).unwrap(),
            0x728A_CF03
        );
        assert_eq!(
            encode_movk(Width::W32, Reg::from_raw(3), 0x5678, 16).unwrap(),
            0x72AA_CF03
        );
    }

    #[test]
    fn encode_load_store_unsigned_offset() {
        assert_eq!(
            encode_ldr_imm(Width::X64, Reg::from_raw(4), Reg::from_raw(5), 16).unwrap(),
            0xF940_08A4
        );
        assert_eq!(
            encode_ldr_imm(Width::W32, Reg::from_raw(4), Reg::from_raw(5), 16).unwrap(),
            0xB940_10A4
        );
        assert_eq!(
            encode_ldrb_imm(Reg::from_raw(4), Reg::from_raw(5), 15).unwrap(),
            0x3940_3CA4
        );
        assert_eq!(
            encode_ldrh_imm(Reg::from_raw(4), Reg::from_raw(5), 14).unwrap(),
            0x7940_1CA4
        );

        assert_eq!(
            encode_str_imm(Width::X64, Reg::from_raw(4), Reg::from_raw(5), 16).unwrap(),
            0xF900_08A4
        );
        assert_eq!(
            encode_str_imm(Width::W32, Reg::from_raw(4), Reg::from_raw(5), 16).unwrap(),
            0xB900_10A4
        );
        assert_eq!(
            encode_strb_imm(Reg::from_raw(4), Reg::from_raw(5), 15).unwrap(),
            0x3900_3CA4
        );
        assert_eq!(
            encode_strh_imm(Reg::from_raw(4), Reg::from_raw(5), 14).unwrap(),
            0x7900_1CA4
        );
    }

    #[test]
    fn encode_integer_arithmetic_and_bitwise() {
        assert_eq!(
            encode_add_reg(
                Width::X64,
                Reg::from_raw(6),
                Reg::from_raw(7),
                Reg::from_raw(8)
            )
            .unwrap(),
            0x8B08_00E6
        );
        assert_eq!(
            encode_add_reg(
                Width::W32,
                Reg::from_raw(6),
                Reg::from_raw(7),
                Reg::from_raw(8)
            )
            .unwrap(),
            0x0B08_00E6
        );
        assert_eq!(
            encode_add_imm(Width::X64, Reg::from_raw(6), Reg::from_raw(7), 123, false).unwrap(),
            0x9101_ECE6
        );
        assert_eq!(
            encode_add_imm(Width::W32, Reg::from_raw(6), Reg::from_raw(7), 123, false).unwrap(),
            0x1101_ECE6
        );
        assert_eq!(
            encode_add_imm(Width::X64, Reg::from_raw(6), Reg::from_raw(7), 0x123, true).unwrap(),
            0x9144_8CE6
        );

        assert_eq!(
            encode_sub_reg(
                Width::X64,
                Reg::from_raw(6),
                Reg::from_raw(7),
                Reg::from_raw(8)
            )
            .unwrap(),
            0xCB08_00E6
        );
        assert_eq!(
            encode_sub_reg(
                Width::W32,
                Reg::from_raw(6),
                Reg::from_raw(7),
                Reg::from_raw(8)
            )
            .unwrap(),
            0x4B08_00E6
        );
        assert_eq!(
            encode_sub_imm(Width::X64, Reg::from_raw(6), Reg::from_raw(7), 123, false).unwrap(),
            0xD101_ECE6
        );
        assert_eq!(
            encode_sub_imm(Width::W32, Reg::from_raw(6), Reg::from_raw(7), 123, false).unwrap(),
            0x5101_ECE6
        );
        assert_eq!(
            encode_sub_imm(Width::X64, Reg::from_raw(6), Reg::from_raw(7), 0x123, true).unwrap(),
            0xD144_8CE6
        );

        assert_eq!(
            encode_and_reg(
                Width::X64,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11),
                Shift::Lsl,
                0
            )
            .unwrap(),
            0x8A0B_0149
        );
        assert_eq!(
            encode_orr_reg(
                Width::X64,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11),
                Shift::Lsl,
                0
            )
            .unwrap(),
            0xAA0B_0149
        );
        assert_eq!(
            encode_eor_reg(
                Width::X64,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11),
                Shift::Lsl,
                0
            )
            .unwrap(),
            0xCA0B_0149
        );

        assert_eq!(
            encode_and_reg(
                Width::X64,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11),
                Shift::Lsr,
                5
            )
            .unwrap(),
            0x8A4B_1549
        );
        assert_eq!(
            encode_orr_reg(
                Width::X64,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11),
                Shift::Lsr,
                5
            )
            .unwrap(),
            0xAA4B_1549
        );
        assert_eq!(
            encode_eor_reg(
                Width::X64,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11),
                Shift::Lsr,
                5
            )
            .unwrap(),
            0xCA4B_1549
        );
    }

    #[test]
    fn encode_shifts_cmp_and_misc() {
        assert_eq!(
            encode_lsl_reg(
                Width::X64,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11)
            )
            .unwrap(),
            0x9ACB_2149
        );
        assert_eq!(
            encode_lsl_reg(
                Width::W32,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11)
            )
            .unwrap(),
            0x1ACB_2149
        );
        assert_eq!(
            encode_lsr_reg(
                Width::X64,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11)
            )
            .unwrap(),
            0x9ACB_2549
        );
        assert_eq!(
            encode_lsr_reg(
                Width::W32,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11)
            )
            .unwrap(),
            0x1ACB_2549
        );

        assert_eq!(
            encode_neg(Width::X64, Reg::from_raw(12), Reg::from_raw(13)).unwrap(),
            0xCB0D_03EC
        );
        assert_eq!(
            encode_neg(Width::W32, Reg::from_raw(12), Reg::from_raw(13)).unwrap(),
            0x4B0D_03EC
        );

        assert_eq!(
            encode_cmp_reg(Width::X64, Reg::from_raw(14), Reg::from_raw(15)).unwrap(),
            0xEB0F_01DF
        );
        assert_eq!(
            encode_cmp_reg(Width::W32, Reg::from_raw(14), Reg::from_raw(15)).unwrap(),
            0x6B0F_01DF
        );
        assert_eq!(
            encode_cmp_imm(Width::X64, Reg::from_raw(14), 0xff, false).unwrap(),
            0xF103_FDDF
        );
        assert_eq!(
            encode_cmp_imm(Width::W32, Reg::from_raw(14), 0xff, false).unwrap(),
            0x7103_FDDF
        );
        assert_eq!(
            encode_cmp_imm(Width::X64, Reg::from_raw(14), 0x123, true).unwrap(),
            0xF144_8DDF
        );

        assert_eq!(
            encode_cset(Width::X64, Reg::from_raw(12), Condition::Ne).unwrap(),
            0x9A9F_07EC
        );
        assert_eq!(
            encode_cset(Width::W32, Reg::from_raw(12), Condition::Ne).unwrap(),
            0x1A9F_07EC
        );
        assert_eq!(
            encode_cset(Width::X64, Reg::from_raw(12), Condition::Eq).unwrap(),
            0x9A9F_17EC
        );
        assert_eq!(
            encode_cset(Width::X64, Reg::from_raw(12), Condition::Gt).unwrap(),
            0x9A9F_D7EC
        );

        assert_eq!(
            encode_sxtb(Reg::from_raw(18), Reg::from_raw(19)).unwrap(),
            0x9340_1E72
        );
        assert_eq!(
            encode_sxth(Reg::from_raw(18), Reg::from_raw(19)).unwrap(),
            0x9340_3E72
        );
        assert_eq!(
            encode_sxtw(Reg::from_raw(18), Reg::from_raw(19)).unwrap(),
            0x9340_7E72
        );

        assert_eq!(encode_blr(Reg::from_raw(20)).unwrap(), 0xD63F_0280);
    }

    #[test]
    fn lsr_imm_w32() {
        let w = encode_lsr_imm(Width::W32, Reg::X10, Reg::X9, 1).unwrap();
        assert_eq!(w & 0x1f, 10);
        assert_eq!((w >> 5) & 0x1f, 9);
        assert_eq!((w >> 16) & 0x3f, 1);
        assert_eq!((w >> 10) & 0x3f, 31);
        assert_eq!(w >> 24, 0x53);
    }

    #[test]
    fn lsl_imm_w32() {
        let w = encode_lsl_imm(Width::W32, Reg::X10, Reg::X9, 1).unwrap();
        assert_eq!(w & 0x1f, 10);
        assert_eq!((w >> 5) & 0x1f, 9);
        assert_eq!((w >> 16) & 0x3f, 31);
        assert_eq!((w >> 10) & 0x3f, 30);
    }

    #[test]
    fn and_imm_w32_bit0() {
        let w = encode_and_imm(Width::W32, Reg::X11, Reg::X9, 1).unwrap();
        assert_eq!(w & 0x1f, 11);
        assert_eq!((w >> 5) & 0x1f, 9);
    }

    #[test]
    fn orr_imm_w32_0x80() {
        let w = encode_orr_imm(Width::W32, Reg::X9, Reg::X11, 0x80).unwrap();
        assert_eq!(w & 0x1f, 9);
        assert_eq!((w >> 5) & 0x1f, 11);
    }

    #[test]
    fn orr_imm_x64_bit1() {
        let w = encode_orr_imm(Width::X64, Reg::X13, Reg::X13, 2).unwrap();
        assert_eq!(w & 0x1f, 13);
        assert_eq!((w >> 5) & 0x1f, 13);
    }

    #[test]
    fn logical_imm_bitmasks() {
        // 0x7f = 7 consecutive 1s (0111_1111)
        let w = encode_and_imm(Width::X64, Reg::X9, Reg::X10, 0x7f).unwrap();
        assert_eq!(w & 0x1f, 9); // rd
        assert_eq!((w >> 5) & 0x1f, 10); // rn

        // 0xff = 8 consecutive 1s
        assert!(encode_and_imm(Width::X64, Reg::X9, Reg::X10, 0xff).is_ok());

        // 0x1f = 5 consecutive 1s
        assert!(encode_and_imm(Width::X64, Reg::X9, Reg::X10, 0x1f).is_ok());

        // 0xffff = 16 consecutive 1s
        assert!(encode_and_imm(Width::X64, Reg::X9, Reg::X10, 0xffff).is_ok());

        // 0xff00 = 8 consecutive 1s shifted
        assert!(encode_and_imm(Width::X64, Reg::X9, Reg::X10, 0xff00).is_ok());

        // ORR with bitmasks
        assert!(encode_orr_imm(Width::X64, Reg::X9, Reg::X10, 0x7f).is_ok());
        assert!(encode_orr_imm(Width::X64, Reg::X9, Reg::X10, 0xff).is_ok());

        // EOR with bitmasks
        assert!(encode_eor_imm(Width::X64, Reg::X9, Reg::X10, 0x7f).is_ok());

        // 32-bit width
        assert!(encode_and_imm(Width::W32, Reg::X9, Reg::X10, 0x7f).is_ok());
        assert!(encode_and_imm(Width::W32, Reg::X9, Reg::X10, 0xff).is_ok());

        // Invalid: all zeros
        assert!(encode_and_imm(Width::X64, Reg::X9, Reg::X10, 0).is_err());

        // Invalid: all ones
        assert!(encode_and_imm(Width::X64, Reg::X9, Reg::X10, u64::MAX).is_err());
    }

    #[test]
    fn movi_b16_basic() {
        let w = encode_movi_b16(1, 0x22).unwrap();
        assert_eq!(w & 0x1f, 1);
    }

    #[test]
    fn ld1_b16_basic() {
        let w = encode_ld1_b16(0, Reg::X19).unwrap();
        assert_eq!(w & 0x1f, 0);
        assert_eq!((w >> 5) & 0x1f, 19);
    }

    #[test]
    fn cmeq_b16_basic() {
        let w = encode_cmeq_b16(3, 0, 1).unwrap();
        assert_eq!(w & 0x1f, 3);
        assert_eq!((w >> 5) & 0x1f, 0);
        assert_eq!((w >> 16) & 0x1f, 1);
    }

    #[test]
    fn orr_b16_basic() {
        let w = encode_orr_b16(1, 2, 3).unwrap();
        assert_eq!(w & 0x1f, 1);
        assert_eq!((w >> 5) & 0x1f, 2);
        assert_eq!((w >> 16) & 0x1f, 3);
    }

    #[test]
    fn umaxv_b16_basic() {
        let w = encode_umaxv_b16(5, 6).unwrap();
        assert_eq!(w & 0x1f, 5);
        assert_eq!((w >> 5) & 0x1f, 6);
    }

    #[test]
    fn umov_b16_basic() {
        let w = encode_umov_b(Reg::X9, 5, 7).unwrap();
        assert_eq!(w & 0x1f, 9);
        assert_eq!((w >> 5) & 0x1f, 5);
        assert_eq!(((w >> 16) & 0x1f), 15);
        assert_eq!((w >> 17) & 0xf, 7);
    }

    #[test]
    fn ldrb_reg_basic() {
        let w = encode_ldrb_reg(Reg::X9, Reg::X19, Reg::X10).unwrap();
        assert_eq!(w & 0x1f, 9);
        assert_eq!((w >> 5) & 0x1f, 19);
        assert_eq!((w >> 16) & 0x1f, 10);
    }

    #[test]
    fn ucvtf_d_x_basic() {
        let w = encode_ucvtf_d_x(0, Reg::X9).unwrap();
        assert_eq!(w & 0x1f, 0);
        assert_eq!((w >> 5) & 0x1f, 9);
        assert_eq!(w >> 24, 0x9E);
    }

    #[test]
    fn fmov_x_d_basic() {
        let w = encode_fmov_x_d(Reg::X7, 4).unwrap();
        assert_eq!(w & 0x1f, 7);
        assert_eq!((w >> 5) & 0x1f, 4);
        assert_eq!(w >> 24, 0x9E);
    }

    #[test]
    fn encode_pair_load_store_ret_and_branches() {
        assert_eq!(
            encode_stp(
                Width::X64,
                Reg::from_raw(29),
                Reg::from_raw(30),
                Reg::from_raw(31),
                0
            )
            .unwrap(),
            0xA900_7BFD
        );
        assert_eq!(
            encode_ldp(
                Width::X64,
                Reg::from_raw(29),
                Reg::from_raw(30),
                Reg::from_raw(31),
                0
            )
            .unwrap(),
            0xA940_7BFD
        );
        assert_eq!(
            encode_stp(
                Width::W32,
                Reg::from_raw(1),
                Reg::from_raw(2),
                Reg::from_raw(3),
                -4
            )
            .unwrap(),
            0x293F_8861
        );
        assert_eq!(
            encode_ldp(
                Width::W32,
                Reg::from_raw(1),
                Reg::from_raw(2),
                Reg::from_raw(3),
                -4
            )
            .unwrap(),
            0x297F_8861
        );
        assert_eq!(encode_ret(Reg::from_raw(30)).unwrap(), 0xD65F_03C0);

        assert_eq!(encode_tbz(Reg::from_raw(1), 7, 3).unwrap(), 0x3638_0061);
        assert_eq!(encode_tbnz(Reg::from_raw(1), 7, 3).unwrap(), 0x3738_0061);
        assert_eq!(encode_b_cond(Condition::Eq, 3).unwrap(), 0x5400_0060);
    }

    #[test]
    fn encode_test_and_misc_integer_ops() {
        assert_eq!(
            encode_tst_reg(Width::X64, Reg::from_raw(14), Reg::from_raw(15)).unwrap(),
            0xEA0F_01DF
        );
        assert_eq!(
            encode_tst_reg(Width::W32, Reg::from_raw(14), Reg::from_raw(15)).unwrap(),
            0x6A0F_01DF
        );
        assert_eq!(
            encode_tst_imm(Width::X64, Reg::from_raw(9), 1).unwrap(),
            0xF240_013F
        );
        assert_eq!(
            encode_tst_imm(Width::W32, Reg::from_raw(9), 1).unwrap(),
            0x7200_013F
        );

        assert_eq!(
            encode_madd(
                Width::X64,
                Reg::from_raw(4),
                Reg::from_raw(1),
                Reg::from_raw(2),
                Reg::from_raw(3)
            )
            .unwrap(),
            0x9B02_0C24
        );
        assert_eq!(
            encode_madd(
                Width::W32,
                Reg::from_raw(4),
                Reg::from_raw(1),
                Reg::from_raw(2),
                Reg::from_raw(3)
            )
            .unwrap(),
            0x1B02_0C24
        );
        assert_eq!(
            encode_asr_imm(Width::X64, Reg::from_raw(1), Reg::from_raw(2), 5).unwrap(),
            0x9345_FC41
        );
        assert_eq!(
            encode_asr_imm(Width::W32, Reg::from_raw(1), Reg::from_raw(2), 5).unwrap(),
            0x1305_7C41
        );
        assert_eq!(
            encode_csel(
                Width::X64,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11),
                Condition::Ne
            )
            .unwrap(),
            0x9A8B_1149
        );
        assert_eq!(
            encode_subs_reg(
                Width::X64,
                Reg::from_raw(6),
                Reg::from_raw(7),
                Reg::from_raw(8)
            )
            .unwrap(),
            0xEB08_00E6
        );
        assert_eq!(
            encode_subs_reg(
                Width::W32,
                Reg::from_raw(6),
                Reg::from_raw(7),
                Reg::from_raw(8)
            )
            .unwrap(),
            0x6B08_00E6
        );
        assert_eq!(
            encode_cmn_imm(Width::X64, Reg::from_raw(14), 0xff, false).unwrap(),
            0xB103_FDDF
        );
        assert_eq!(
            encode_mul(
                Width::X64,
                Reg::from_raw(6),
                Reg::from_raw(7),
                Reg::from_raw(8)
            )
            .unwrap(),
            0x9B08_7CE6
        );
        assert_eq!(
            encode_umulh(Reg::from_raw(6), Reg::from_raw(7), Reg::from_raw(8)).unwrap(),
            0x9BC8_7CE6
        );
        assert_eq!(
            encode_clz(Width::X64, Reg::from_raw(9), Reg::from_raw(10)).unwrap(),
            0xDAC0_1149
        );
        assert_eq!(
            encode_clz(Width::W32, Reg::from_raw(9), Reg::from_raw(10)).unwrap(),
            0x5AC0_1149
        );
        assert_eq!(
            encode_bic(
                Width::X64,
                Reg::from_raw(9),
                Reg::from_raw(10),
                Reg::from_raw(11),
                Shift::Lsl,
                0
            )
            .unwrap(),
            0x8A2B_0149
        );
        assert_eq!(
            encode_smull(Reg::from_raw(9), Reg::from_raw(10), Reg::from_raw(11)).unwrap(),
            0x9B2B_7D49
        );
    }

    #[test]
    fn encode_branches_and_cb_branches() {
        assert_eq!(encode_b(1).unwrap(), 0x1400_0001);
        assert_eq!(encode_b(-1).unwrap(), 0x17FF_FFFF);
        assert_eq!(encode_bl(3).unwrap(), 0x9400_0003);
        assert_eq!(encode_bl(-4).unwrap(), 0x97FF_FFFC);

        assert_eq!(
            encode_cbz(Width::X64, Reg::from_raw(16), 10).unwrap(),
            0xB400_0150
        );
        assert_eq!(
            encode_cbnz(Width::W32, Reg::from_raw(17), 9).unwrap(),
            0x3500_0131
        );
        assert_eq!(
            encode_cbz(Width::X64, Reg::from_raw(16), -2).unwrap(),
            0xB4FF_FFD0
        );
        assert_eq!(
            encode_cbnz(Width::W32, Reg::from_raw(17), -3).unwrap(),
            0x35FF_FFB1
        );
    }

    #[test]
    fn emitter_records_source_map_and_resolves_fixups() {
        let mut emitter = Emitter::new();
        let start = emitter.new_label();
        let done = emitter.new_label();

        emitter.bind_label(start).unwrap();
        emitter.set_source_location(kajit_reprs::asm::SourceLocation {
            file: 1,
            line: 10,
            column: 1,
        });
        emitter.emit_b_label(done).unwrap();
        emitter.set_source_location(kajit_reprs::asm::SourceLocation {
            file: 1,
            line: 11,
            column: 1,
        });
        emitter
            .emit_cbz_label(Width::X64, Reg::from_raw(16), start)
            .unwrap();
        emitter.bind_label(done).unwrap();
        emitter.set_source_location(kajit_reprs::asm::SourceLocation {
            file: 1,
            line: 12,
            column: 1,
        });
        emitter.emit_bl_label(start).unwrap();

        let finalized = emitter.finalize().unwrap();
        let code = finalized.exec.as_ref();
        assert_eq!(
            finalized.source_map,
            vec![
                kajit_reprs::asm::SourceMapEntry {
                    offset: 0,
                    location: kajit_reprs::asm::SourceLocation {
                        file: 1,
                        line: 10,
                        column: 1,
                    },
                },
                kajit_reprs::asm::SourceMapEntry {
                    offset: 4,
                    location: kajit_reprs::asm::SourceLocation {
                        file: 1,
                        line: 11,
                        column: 1,
                    },
                },
                kajit_reprs::asm::SourceMapEntry {
                    offset: 8,
                    location: kajit_reprs::asm::SourceLocation {
                        file: 1,
                        line: 12,
                        column: 1,
                    },
                },
            ]
        );

        assert_eq!(word(&code[0..4]), 0x1400_0002);
        assert_eq!(word(&code[4..8]), 0xB4FF_FFF0);
        assert_eq!(word(&code[8..12]), 0x97FF_FFFE);

        let trace = finalized.trace_text().unwrap();
        assert_eq!(
            trace,
            "00000000 file=1 line=10 col=1 bytes=02000014\n00000004 file=1 line=11 col=1 bytes=f0ffffb4\n00000008 file=1 line=12 col=1 bytes=feffff97"
        );

        let source_map_encoded = finalized.source_map_le().unwrap();
        assert_eq!(
            kajit_reprs::asm::decode_source_map_le(&source_map_encoded).unwrap(),
            finalized.source_map
        );
    }

    #[test]
    fn emitter_reports_unbound_label() {
        let mut emitter = Emitter::new();
        let dangling = emitter.new_label();
        emitter.emit_b_label(dangling).unwrap();
        let err = emitter.finalize().unwrap_err();
        assert!(matches!(err, EmitError::UnboundLabel { .. }));
    }

    #[test]
    fn emitter_reports_out_of_range_fixup() {
        let mut emitter = Emitter::new();
        let far = emitter.new_label();
        emitter.emit_cbz_label(Width::X64, Reg::X0, far).unwrap();
        for _ in 0..=262_143 {
            emitter.emit_word(0xD503_201F); // nop
        }
        emitter.bind_label(far).unwrap();

        let err = emitter.finalize().unwrap_err();
        assert!(matches!(
            err,
            EmitError::InvalidImmediate {
                instruction: "branch19",
                ..
            }
        ));
    }

    #[test]
    fn emitter_resolves_tbz_tbnz_and_b_cond_fixups() {
        let mut emitter = Emitter::new();
        let start = emitter.new_label();
        let done = emitter.new_label();

        emitter.bind_label(start).unwrap();
        emitter.set_source_location(kajit_reprs::asm::SourceLocation {
            file: 2,
            line: 1,
            column: 1,
        });
        emitter.emit_tbz_label(Reg::X0, 7, done).unwrap();
        emitter.set_source_location(kajit_reprs::asm::SourceLocation {
            file: 2,
            line: 2,
            column: 1,
        });
        emitter.emit_tbnz_label(Reg::from_raw(1), 5, start).unwrap();
        emitter.set_source_location(kajit_reprs::asm::SourceLocation {
            file: 2,
            line: 3,
            column: 1,
        });
        emitter.emit_b_cond_label(Condition::Eq, start).unwrap();
        emitter.bind_label(done).unwrap();

        let finalized = emitter.finalize().unwrap();
        let code = finalized.exec.as_ref();
        assert_eq!(
            finalized.source_map,
            vec![
                kajit_reprs::asm::SourceMapEntry {
                    offset: 0,
                    location: kajit_reprs::asm::SourceLocation {
                        file: 2,
                        line: 1,
                        column: 1,
                    },
                },
                kajit_reprs::asm::SourceMapEntry {
                    offset: 4,
                    location: kajit_reprs::asm::SourceLocation {
                        file: 2,
                        line: 2,
                        column: 1,
                    },
                },
                kajit_reprs::asm::SourceMapEntry {
                    offset: 8,
                    location: kajit_reprs::asm::SourceLocation {
                        file: 2,
                        line: 3,
                        column: 1,
                    },
                },
            ]
        );

        assert_eq!(word(&code[0..4]), 0x3638_0060);
        assert_eq!(word(&code[4..8]), 0x372F_FFE1);
        assert_eq!(word(&code[8..12]), 0x54FF_FFC0);
    }

    #[test]
    fn encode_boundaries_and_rejections() {
        assert_eq!(encode_b(-(1 << 25)).unwrap(), 0x1600_0000);
        assert_eq!(encode_b((1 << 25) - 1).unwrap(), 0x15FF_FFFF);
        assert!(matches!(
            encode_b(-(1 << 25) - 1),
            Err(EmitError::InvalidImmediate {
                instruction: "b",
                ..
            })
        ));
        assert!(matches!(
            encode_b(1 << 25),
            Err(EmitError::InvalidImmediate {
                instruction: "b",
                ..
            })
        ));

        assert_eq!(
            encode_cbz(Width::X64, Reg::from_raw(0), -(1 << 18)).unwrap(),
            0xB480_0000
        );
        assert_eq!(
            encode_cbz(Width::X64, Reg::from_raw(0), (1 << 18) - 1).unwrap(),
            0xB47F_FFE0
        );
        assert!(matches!(
            encode_cbz(Width::X64, Reg::from_raw(0), -(1 << 18) - 1),
            Err(EmitError::InvalidImmediate {
                instruction: "cbz",
                ..
            })
        ));
        assert!(matches!(
            encode_cbz(Width::X64, Reg::from_raw(0), 1 << 18),
            Err(EmitError::InvalidImmediate {
                instruction: "cbz",
                ..
            })
        ));

        assert_eq!(
            encode_ldr_imm(Width::X64, Reg::from_raw(1), Reg::from_raw(2), 4095 * 8).unwrap(),
            0xF97F_FC41
        );
        assert!(matches!(
            encode_ldr_imm(Width::X64, Reg::from_raw(1), Reg::from_raw(2), 4095 * 8 + 8),
            Err(EmitError::InvalidOffset { .. })
        ));
        assert!(matches!(
            encode_ldr_imm(Width::X64, Reg::from_raw(1), Reg::from_raw(2), 10),
            Err(EmitError::InvalidOffset { .. })
        ));

        assert!(matches!(
            encode_movz(Width::W32, Reg::from_raw(1), 0x1234, 32),
            Err(EmitError::InvalidMovWideShift { .. })
        ));
        assert!(matches!(
            encode_add_imm(
                Width::X64,
                Reg::from_raw(0),
                Reg::from_raw(0),
                0x1000,
                false
            ),
            Err(EmitError::InvalidImmediate {
                instruction: "add/sub imm12",
                ..
            })
        ));
    }

    #[test]
    fn emitter_rejects_double_label_bind() {
        let mut emitter = Emitter::new();
        let label = emitter.new_label();
        emitter.bind_label(label).unwrap();
        let err = emitter.bind_label(label).unwrap_err();
        assert!(matches!(err, EmitError::LabelAlreadyBound { .. }));
    }

    #[test]
    fn emitter_captures_instructions() {
        let mut emitter = Emitter::new();
        emitter.enable_capture();

        let label0 = emitter.new_label();
        let label1 = emitter.new_label();

        emitter.bind_label(label0).unwrap();
        emitter.emit_movz_imm(Width::X64, Reg::X0, 0x42, 0).unwrap();
        emitter
            .emit_add_imm(Width::X64, Reg::X1, Reg::X0, 7, false)
            .unwrap();
        emitter
            .emit_ldr_imm(Width::W32, Reg::X2, Reg::X1, 0)
            .unwrap();
        emitter.emit_cbz_label(Width::X64, Reg::X2, label1).unwrap();
        emitter.emit_ret().unwrap();
        emitter.bind_label(label1).unwrap();
        emitter.emit_mov_reg(Width::X64, Reg::X0, Reg::X2).unwrap();
        emitter.emit_b_label(label0).unwrap();

        let program = emitter.take_captured_program().unwrap();
        assert_eq!(program.items.len(), 9);

        // Verify we can format it
        let formatted = format!("{}", program);
        println!("{}", formatted);
        assert!(formatted.contains(".L0:"));
        assert!(formatted.contains(".L1:"));
        assert!(formatted.contains("movz x0, #0x42"));
        assert!(formatted.contains("add x1, x0, #7"));
        assert!(formatted.contains("ldr w2, [x1]"));
        assert!(formatted.contains("cbz x2, .L1"));
        assert!(formatted.contains("ret"));
        assert!(formatted.contains("mov x0, x2"));
        assert!(formatted.contains("b .L0"));
    }
}
