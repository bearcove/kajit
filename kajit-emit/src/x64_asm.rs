//! x86_64 assembly instruction representation and text format.
//!
//! This module provides a high-level representation of x86_64 instructions
//! that can be dumped to text, hand-edited, and parsed back for emission.

use crate::x64::{Condition, LabelId, Mem, Operand};

/// A label identifier for branch targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(pub u32);

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ".L{}", self.0)
    }
}

/// x86_64 assembly instruction (Intel syntax)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    // Moves - 64-bit
    MovR64R64 { dst: u8, src: u8 },
    MovR64Imm64 { dst: u8, imm: u64 },
    MovR64Imm32Sext { dst: u8, imm: u32 },
    MovR32Imm32 { dst: u8, imm: u32 },
    MovR32R32 { dst: u8, src: u8 },

    // Moves - memory
    MovR64M { dst: u8, src: Mem },
    MovMR64 { dst: Mem, src: u8 },
    MovR32M { dst: u8, src: Mem },
    MovMR32 { dst: Mem, src: u8 },
    MovMR16 { dst: Mem, src: u8 },
    MovMR8 { dst: Mem, src: u8 },

    // Moves - zero/sign extend
    MovzxR32Rm8 { dst: u8, src: Operand },
    MovzxR32Rm16 { dst: u8, src: Operand },
    MovzxR64Rm8 { dst: u8, src: Operand },
    MovsxR64Rm8 { dst: u8, src: Operand },
    MovsxR64Rm16 { dst: u8, src: Operand },
    MovsxdR64Rm32 { dst: u8, src: Operand },

    // Arithmetic
    AddR64R64 { dst: u8, src: u8 },
    AddR64Imm32 { dst: u8, imm: u32 },
    AddR64M { dst: u8, src: Mem },
    SubR64R64 { dst: u8, src: u8 },
    SubR64Imm32 { dst: u8, imm: u32 },
    SubR64M { dst: u8, src: Mem },
    ImulR64R64 { dst: u8, src: u8 },
    NegR64 { dst: u8 },
    LeaR64M { dst: u8, src: Mem },

    // Logical
    AndR64R64 { dst: u8, src: u8 },
    AndR64M { dst: u8, src: Mem },
    OrR64R64 { dst: u8, src: u8 },
    OrR64M { dst: u8, src: Mem },
    XorR64R64 { dst: u8, src: u8 },
    XorR64M { dst: u8, src: Mem },
    XorR32R32 { dst: u8, src: u8 },

    // Shifts
    ShlR64Imm8 { dst: u8, imm: u8 },
    ShlR64Cl { dst: u8 },
    ShrR64Imm8 { dst: u8, imm: u8 },
    ShrR64Cl { dst: u8 },
    SarR64Imm8 { dst: u8, imm: u8 },
    SarR64Cl { dst: u8 },

    // Compare/Test
    CmpR64R64 { lhs: u8, rhs: u8 },
    CmpR64Imm32 { dst: u8, imm: u32 },
    CmpR64M { lhs: u8, rhs: Mem },
    TestR64R64 { lhs: u8, rhs: u8 },
    TestR32R32 { lhs: u8, rhs: u8 },
    SetccR8 { cond: Condition, dst: u8 },

    // Control flow - local labels
    Jmp { target: Label },
    Je { target: Label },
    Jne { target: Label },
    Jz { target: Label },
    Jnz { target: Label },
    Jcc { cond: Condition, target: Label },

    // Control flow - emitter labels
    JmpLabel { label: LabelId },
    JeLabel { label: LabelId },
    JzLabel { label: LabelId },
    JnzLabel { label: LabelId },
    JccLabel { cond: Condition, label: LabelId },
    CallR64 { reg: u8 },
    CallLabel { label: LabelId },

    // Stack
    PushR64 { reg: u8 },
    PopR64 { reg: u8 },

    // Other
    Ret,
    Nop,
}

fn reg64_name(r: u8) -> &'static str {
    match r {
        0 => "rax",
        1 => "rcx",
        2 => "rdx",
        3 => "rbx",
        4 => "rsp",
        5 => "rbp",
        6 => "rsi",
        7 => "rdi",
        8 => "r8",
        9 => "r9",
        10 => "r10",
        11 => "r11",
        12 => "r12",
        13 => "r13",
        14 => "r14",
        15 => "r15",
        _ => unreachable!("invalid register {}", r),
    }
}

fn reg32_name(r: u8) -> &'static str {
    match r {
        0 => "eax",
        1 => "ecx",
        2 => "edx",
        3 => "ebx",
        4 => "esp",
        5 => "ebp",
        6 => "esi",
        7 => "edi",
        8 => "r8d",
        9 => "r9d",
        10 => "r10d",
        11 => "r11d",
        12 => "r12d",
        13 => "r13d",
        14 => "r14d",
        15 => "r15d",
        _ => unreachable!("invalid register {}", r),
    }
}

fn reg8_name(r: u8) -> &'static str {
    match r {
        0 => "al",
        1 => "cl",
        2 => "dl",
        3 => "bl",
        4 => "spl",
        5 => "bpl",
        6 => "sil",
        7 => "dil",
        8 => "r8b",
        9 => "r9b",
        10 => "r10b",
        11 => "r11b",
        12 => "r12b",
        13 => "r13b",
        14 => "r14b",
        15 => "r15b",
        _ => unreachable!("invalid register {}", r),
    }
}

fn reg16_name(r: u8) -> &'static str {
    match r {
        0 => "ax",
        1 => "cx",
        2 => "dx",
        3 => "bx",
        4 => "sp",
        5 => "bp",
        6 => "si",
        7 => "di",
        8 => "r8w",
        9 => "r9w",
        10 => "r10w",
        11 => "r11w",
        12 => "r12w",
        13 => "r13w",
        14 => "r14w",
        15 => "r15w",
        _ => unreachable!("invalid register {}", r),
    }
}

fn fmt_mem(f: &mut std::fmt::Formatter<'_>, mem: &Mem) -> std::fmt::Result {
    let base = reg64_name(mem.base);
    match mem.disp.cmp(&0) {
        std::cmp::Ordering::Equal => write!(f, "[{}]", base),
        std::cmp::Ordering::Greater => write!(f, "[{} + {}]", base, mem.disp),
        std::cmp::Ordering::Less => write!(f, "[{} - {}]", base, -mem.disp),
    }
}

fn fmt_operand32(f: &mut std::fmt::Formatter<'_>, op: &Operand) -> std::fmt::Result {
    match op {
        Operand::Reg(r) => write!(f, "{}", reg32_name(*r)),
        Operand::Mem(m) => fmt_mem(f, m),
    }
}

fn fmt_operand8(f: &mut std::fmt::Formatter<'_>, op: &Operand) -> std::fmt::Result {
    match op {
        Operand::Reg(r) => write!(f, "{}", reg8_name(*r)),
        Operand::Mem(m) => {
            write!(f, "byte ptr ")?;
            fmt_mem(f, m)
        }
    }
}

fn fmt_operand16(f: &mut std::fmt::Formatter<'_>, op: &Operand) -> std::fmt::Result {
    match op {
        Operand::Reg(r) => write!(f, "{}", reg16_name(*r)),
        Operand::Mem(m) => {
            write!(f, "word ptr ")?;
            fmt_mem(f, m)
        }
    }
}

fn cond_name(cond: Condition) -> &'static str {
    match cond {
        Condition::Eq => "eq",
        Condition::Ne => "ne",
        Condition::Lt => "l",
        Condition::Le => "le",
        Condition::Gt => "g",
        Condition::Ge => "ge",
        Condition::Lo => "b",
        Condition::Hs => "ae",
        Condition::Hi => "a",
        Condition::Ls => "be",
        Condition::O => "o",
        Condition::No => "no",
        Condition::S => "s",
        Condition::Ns => "ns",
        Condition::P => "p",
        Condition::Np => "np",
    }
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Moves - 64-bit register
            Instruction::MovR64R64 { dst, src } => {
                write!(f, "mov {}, {}", reg64_name(*dst), reg64_name(*src))
            }
            Instruction::MovR64Imm64 { dst, imm } => {
                write!(f, "movabs {}, {:#x}", reg64_name(*dst), imm)
            }
            Instruction::MovR64Imm32Sext { dst, imm } => {
                write!(f, "mov {}, {:#x}", reg64_name(*dst), imm)
            }
            Instruction::MovR32Imm32 { dst, imm } => {
                write!(f, "mov {}, {:#x}", reg32_name(*dst), imm)
            }
            Instruction::MovR32R32 { dst, src } => {
                write!(f, "mov {}, {}", reg32_name(*dst), reg32_name(*src))
            }

            // Moves - memory
            Instruction::MovR64M { dst, src } => {
                write!(f, "mov {}, ", reg64_name(*dst))?;
                fmt_mem(f, src)
            }
            Instruction::MovMR64 { dst, src } => {
                write!(f, "mov ")?;
                fmt_mem(f, dst)?;
                write!(f, ", {}", reg64_name(*src))
            }
            Instruction::MovR32M { dst, src } => {
                write!(f, "mov {}, ", reg32_name(*dst))?;
                fmt_mem(f, src)
            }
            Instruction::MovMR32 { dst, src } => {
                write!(f, "mov dword ptr ")?;
                fmt_mem(f, dst)?;
                write!(f, ", {}", reg32_name(*src))
            }
            Instruction::MovMR16 { dst, src } => {
                write!(f, "mov word ptr ")?;
                fmt_mem(f, dst)?;
                write!(f, ", {}", reg16_name(*src))
            }
            Instruction::MovMR8 { dst, src } => {
                write!(f, "mov byte ptr ")?;
                fmt_mem(f, dst)?;
                write!(f, ", {}", reg8_name(*src))
            }

            // Moves - zero/sign extend
            Instruction::MovzxR32Rm8 { dst, src } => {
                write!(f, "movzx {}, ", reg32_name(*dst))?;
                fmt_operand8(f, src)
            }
            Instruction::MovzxR32Rm16 { dst, src } => {
                write!(f, "movzx {}, ", reg32_name(*dst))?;
                fmt_operand16(f, src)
            }
            Instruction::MovzxR64Rm8 { dst, src } => {
                write!(f, "movzx {}, ", reg64_name(*dst))?;
                fmt_operand8(f, src)
            }
            Instruction::MovsxR64Rm8 { dst, src } => {
                write!(f, "movsx {}, ", reg64_name(*dst))?;
                fmt_operand8(f, src)
            }
            Instruction::MovsxR64Rm16 { dst, src } => {
                write!(f, "movsx {}, ", reg64_name(*dst))?;
                fmt_operand16(f, src)
            }
            Instruction::MovsxdR64Rm32 { dst, src } => {
                write!(f, "movsxd {}, ", reg64_name(*dst))?;
                fmt_operand32(f, src)
            }

            // Arithmetic
            Instruction::AddR64R64 { dst, src } => {
                write!(f, "add {}, {}", reg64_name(*dst), reg64_name(*src))
            }
            Instruction::AddR64Imm32 { dst, imm } => {
                write!(f, "add {}, {:#x}", reg64_name(*dst), imm)
            }
            Instruction::AddR64M { dst, src } => {
                write!(f, "add {}, ", reg64_name(*dst))?;
                fmt_mem(f, src)
            }
            Instruction::SubR64R64 { dst, src } => {
                write!(f, "sub {}, {}", reg64_name(*dst), reg64_name(*src))
            }
            Instruction::SubR64Imm32 { dst, imm } => {
                write!(f, "sub {}, {:#x}", reg64_name(*dst), imm)
            }
            Instruction::SubR64M { dst, src } => {
                write!(f, "sub {}, ", reg64_name(*dst))?;
                fmt_mem(f, src)
            }
            Instruction::ImulR64R64 { dst, src } => {
                write!(f, "imul {}, {}", reg64_name(*dst), reg64_name(*src))
            }
            Instruction::NegR64 { dst } => {
                write!(f, "neg {}", reg64_name(*dst))
            }
            Instruction::LeaR64M { dst, src } => {
                write!(f, "lea {}, ", reg64_name(*dst))?;
                fmt_mem(f, src)
            }

            // Logical
            Instruction::AndR64R64 { dst, src } => {
                write!(f, "and {}, {}", reg64_name(*dst), reg64_name(*src))
            }
            Instruction::AndR64M { dst, src } => {
                write!(f, "and {}, ", reg64_name(*dst))?;
                fmt_mem(f, src)
            }
            Instruction::OrR64R64 { dst, src } => {
                write!(f, "or {}, {}", reg64_name(*dst), reg64_name(*src))
            }
            Instruction::OrR64M { dst, src } => {
                write!(f, "or {}, ", reg64_name(*dst))?;
                fmt_mem(f, src)
            }
            Instruction::XorR64R64 { dst, src } => {
                write!(f, "xor {}, {}", reg64_name(*dst), reg64_name(*src))
            }
            Instruction::XorR64M { dst, src } => {
                write!(f, "xor {}, ", reg64_name(*dst))?;
                fmt_mem(f, src)
            }
            Instruction::XorR32R32 { dst, src } => {
                write!(f, "xor {}, {}", reg32_name(*dst), reg32_name(*src))
            }

            // Shifts
            Instruction::ShlR64Imm8 { dst, imm } => {
                write!(f, "shl {}, {}", reg64_name(*dst), imm)
            }
            Instruction::ShlR64Cl { dst } => {
                write!(f, "shl {}, cl", reg64_name(*dst))
            }
            Instruction::ShrR64Imm8 { dst, imm } => {
                write!(f, "shr {}, {}", reg64_name(*dst), imm)
            }
            Instruction::ShrR64Cl { dst } => {
                write!(f, "shr {}, cl", reg64_name(*dst))
            }
            Instruction::SarR64Imm8 { dst, imm } => {
                write!(f, "sar {}, {}", reg64_name(*dst), imm)
            }
            Instruction::SarR64Cl { dst } => {
                write!(f, "sar {}, cl", reg64_name(*dst))
            }

            // Compare/Test
            Instruction::CmpR64R64 { lhs, rhs } => {
                write!(f, "cmp {}, {}", reg64_name(*lhs), reg64_name(*rhs))
            }
            Instruction::CmpR64Imm32 { dst, imm } => {
                write!(f, "cmp {}, {:#x}", reg64_name(*dst), imm)
            }
            Instruction::CmpR64M { lhs, rhs } => {
                write!(f, "cmp {}, ", reg64_name(*lhs))?;
                fmt_mem(f, rhs)
            }
            Instruction::TestR64R64 { lhs, rhs } => {
                write!(f, "test {}, {}", reg64_name(*lhs), reg64_name(*rhs))
            }
            Instruction::TestR32R32 { lhs, rhs } => {
                write!(f, "test {}, {}", reg32_name(*lhs), reg32_name(*rhs))
            }
            Instruction::SetccR8 { cond, dst } => {
                write!(f, "set{} {}", cond_name(*cond), reg8_name(*dst))
            }

            // Control flow - local labels
            Instruction::Jmp { target } => write!(f, "jmp {}", target),
            Instruction::Je { target } => write!(f, "je {}", target),
            Instruction::Jne { target } => write!(f, "jne {}", target),
            Instruction::Jz { target } => write!(f, "jz {}", target),
            Instruction::Jnz { target } => write!(f, "jnz {}", target),
            Instruction::Jcc { cond, target } => {
                write!(f, "j{} {}", cond_name(*cond), target)
            }

            // Control flow - emitter labels
            Instruction::JmpLabel { label } => write!(f, "jmp {:?}", label),
            Instruction::JeLabel { label } => write!(f, "je {:?}", label),
            Instruction::JzLabel { label } => write!(f, "jz {:?}", label),
            Instruction::JnzLabel { label } => write!(f, "jnz {:?}", label),
            Instruction::JccLabel { cond, label } => {
                write!(f, "j{} {:?}", cond_name(*cond), label)
            }
            Instruction::CallR64 { reg } => write!(f, "call {}", reg64_name(*reg)),
            Instruction::CallLabel { label } => write!(f, "call {:?}", label),

            // Stack
            Instruction::PushR64 { reg } => write!(f, "push {}", reg64_name(*reg)),
            Instruction::PopR64 { reg } => write!(f, "pop {}", reg64_name(*reg)),

            // Other
            Instruction::Ret => write!(f, "ret"),
            Instruction::Nop => write!(f, "nop"),
        }
    }
}

/// A sequence of instructions with labels
#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Label(Label),
    Instruction(Instruction),
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for item in &self.items {
            match item {
                Item::Label(label) => writeln!(f, "{}:", label)?,
                Item::Instruction(inst) => writeln!(f, "  {}", inst)?,
            }
        }
        Ok(())
    }
}
