//! ARM64 assembly instruction representation and text format.
//!
//! This module provides a high-level representation of ARM64 instructions
//! that can be dumped to text, hand-edited, and parsed back for emission.

use crate::aarch64::{Condition, Reg, Width};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format() {
        let program = Program {
            items: vec![
                Item::Label(Label(0)),
                Item::Instruction(Instruction::MovzImm {
                    width: Width::X64,
                    rd: Reg::X0,
                    imm: 0x42,
                    shift: 0,
                }),
                Item::Instruction(Instruction::AddImm {
                    width: Width::X64,
                    rd: Reg::X1,
                    rn: Reg::X0,
                    imm: 7,
                }),
                Item::Instruction(Instruction::LdrImm {
                    width: Width::W32,
                    rt: Reg::X2,
                    rn: Reg::X1,
                    offset: 0,
                }),
                Item::Instruction(Instruction::AndImm {
                    width: Width::X64,
                    rd: Reg::X3,
                    rn: Reg::X2,
                    imm: 0x7f,
                }),
                Item::Instruction(Instruction::CmpImm {
                    width: Width::X64,
                    rn: Reg::X3,
                    imm: 0,
                }),
                Item::Instruction(Instruction::BCond {
                    cond: Condition::Eq,
                    target: Label(1),
                }),
                Item::Instruction(Instruction::StrImm {
                    width: Width::W32,
                    rt: Reg::X3,
                    rn: Reg::X20,
                    offset: 0,
                }),
                Item::Instruction(Instruction::Ret),
                Item::Label(Label(1)),
                Item::Instruction(Instruction::MovzImm {
                    width: Width::X64,
                    rd: Reg::X0,
                    imm: 1,
                    shift: 0,
                }),
                Item::Instruction(Instruction::Ret),
            ],
        };

        let output = format!("{}", program);
        eprintln!("\n{}", output);

        // Check some basic formatting
        assert!(output.contains(".L0:"));
        assert!(output.contains("movz x0, #0x42"));
        assert!(output.contains("add x1, x0, #7"));
        assert!(output.contains("ldr w2, [x1]"));
        assert!(output.contains("b.eq .L1"));
    }
}

/// A label identifier for branch targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(pub u32);

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ".L{}", self.0)
    }
}

/// ARM64 assembly instruction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    // Data processing - immediate
    AddImm {
        width: Width,
        rd: Reg,
        rn: Reg,
        imm: u32,
    },
    SubImm {
        width: Width,
        rd: Reg,
        rn: Reg,
        imm: u32,
    },
    CmpImm {
        width: Width,
        rn: Reg,
        imm: u32,
    },

    // Data processing - register
    AddReg {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    SubReg {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    CmpReg {
        width: Width,
        rn: Reg,
        rm: Reg,
    },
    MulReg {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    NegReg {
        width: Width,
        rd: Reg,
        rm: Reg,
    },

    // Logical - immediate
    AndImm {
        width: Width,
        rd: Reg,
        rn: Reg,
        imm: u64,
    },
    OrrImm {
        width: Width,
        rd: Reg,
        rn: Reg,
        imm: u64,
    },
    EorImm {
        width: Width,
        rd: Reg,
        rn: Reg,
        imm: u64,
    },

    // Logical - register
    AndReg {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    OrrReg {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    EorReg {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },

    // Shifts
    LslImm {
        width: Width,
        rd: Reg,
        rn: Reg,
        shift: u8,
    },
    LsrImm {
        width: Width,
        rd: Reg,
        rn: Reg,
        shift: u8,
    },
    LslReg {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    LsrReg {
        width: Width,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },

    // Move
    MovReg {
        width: Width,
        rd: Reg,
        rm: Reg,
    },
    MovzImm {
        width: Width,
        rd: Reg,
        imm: u16,
        shift: u8,
    },
    MovkImm {
        width: Width,
        rd: Reg,
        imm: u16,
        shift: u8,
    },

    // Conditional
    Cset {
        width: Width,
        rd: Reg,
        cond: Condition,
    },

    // Load/Store
    LdrImm {
        width: Width,
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    LdrbImm {
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    LdrhImm {
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    StrImm {
        width: Width,
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    StrbImm {
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    StrhImm {
        rt: Reg,
        rn: Reg,
        offset: u32,
    },
    Stp {
        width: Width,
        rt1: Reg,
        rt2: Reg,
        rn: Reg,
        offset: i16,
    },
    Ldp {
        width: Width,
        rt1: Reg,
        rt2: Reg,
        rn: Reg,
        offset: i16,
    },

    // Sign extend
    Sxtb {
        width: Width,
        rd: Reg,
        rn: Reg,
    },
    Sxth {
        width: Width,
        rd: Reg,
        rn: Reg,
    },
    Sxtw {
        rd: Reg,
        rn: Reg,
    },

    // Branches
    B {
        target: Label,
    },
    Bl {
        target: Label,
    },
    Blr {
        rn: Reg,
    },
    Ret,
    Cbz {
        width: Width,
        rt: Reg,
        target: Label,
    },
    Cbnz {
        width: Width,
        rt: Reg,
        target: Label,
    },
    Tbz {
        rt: Reg,
        bit: u8,
        target: Label,
    },
    Tbnz {
        rt: Reg,
        bit: u8,
        target: Label,
    },
    BCond {
        cond: Condition,
        target: Label,
    },

    // Pseudo-instructions (expanded during parsing/emission)
    LoadExtern {
        rd: Reg,
        symbol: String,
    },

    // Special
    Nop,
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Data processing - immediate
            Instruction::AddImm { width, rd, rn, imm } => {
                write!(
                    f,
                    "add {}, {}, #{}",
                    reg_name_or_sp(*rd, *width),
                    reg_name_or_sp(*rn, *width),
                    imm
                )
            }
            Instruction::SubImm { width, rd, rn, imm } => {
                write!(
                    f,
                    "sub {}, {}, #{}",
                    reg_name_or_sp(*rd, *width),
                    reg_name_or_sp(*rn, *width),
                    imm
                )
            }
            Instruction::CmpImm { width, rn, imm } => {
                write!(f, "cmp {}, #{}", reg_name(*rn, *width), imm)
            }

            // Data processing - register
            Instruction::AddReg { width, rd, rn, rm } => {
                write!(
                    f,
                    "add {}, {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    reg_name(*rm, *width)
                )
            }
            Instruction::SubReg { width, rd, rn, rm } => {
                write!(
                    f,
                    "sub {}, {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    reg_name(*rm, *width)
                )
            }
            Instruction::CmpReg { width, rn, rm } => {
                write!(
                    f,
                    "cmp {}, {}",
                    reg_name(*rn, *width),
                    reg_name(*rm, *width)
                )
            }
            Instruction::MulReg { width, rd, rn, rm } => {
                write!(
                    f,
                    "mul {}, {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    reg_name(*rm, *width)
                )
            }
            Instruction::NegReg { width, rd, rm } => {
                write!(
                    f,
                    "neg {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rm, *width)
                )
            }

            // Logical - immediate
            Instruction::AndImm { width, rd, rn, imm } => {
                write!(
                    f,
                    "and {}, {}, #{:#x}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    imm
                )
            }
            Instruction::OrrImm { width, rd, rn, imm } => {
                write!(
                    f,
                    "orr {}, {}, #{:#x}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    imm
                )
            }
            Instruction::EorImm { width, rd, rn, imm } => {
                write!(
                    f,
                    "eor {}, {}, #{:#x}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    imm
                )
            }

            // Logical - register
            Instruction::AndReg { width, rd, rn, rm } => {
                write!(
                    f,
                    "and {}, {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    reg_name(*rm, *width)
                )
            }
            Instruction::OrrReg { width, rd, rn, rm } => {
                write!(
                    f,
                    "orr {}, {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    reg_name(*rm, *width)
                )
            }
            Instruction::EorReg { width, rd, rn, rm } => {
                write!(
                    f,
                    "eor {}, {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    reg_name(*rm, *width)
                )
            }

            // Shifts
            Instruction::LslImm {
                width,
                rd,
                rn,
                shift,
            } => {
                write!(
                    f,
                    "lsl {}, {}, #{}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    shift
                )
            }
            Instruction::LsrImm {
                width,
                rd,
                rn,
                shift,
            } => {
                write!(
                    f,
                    "lsr {}, {}, #{}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    shift
                )
            }
            Instruction::LslReg { width, rd, rn, rm } => {
                write!(
                    f,
                    "lsl {}, {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    reg_name(*rm, *width)
                )
            }
            Instruction::LsrReg { width, rd, rn, rm } => {
                write!(
                    f,
                    "lsr {}, {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rn, *width),
                    reg_name(*rm, *width)
                )
            }

            // Move
            Instruction::MovReg { width, rd, rm } => {
                write!(
                    f,
                    "mov {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rm, *width)
                )
            }
            Instruction::MovzImm {
                width,
                rd,
                imm,
                shift,
            } => {
                if *shift == 0 {
                    write!(f, "movz {}, #{:#x}", reg_name(*rd, *width), imm)
                } else {
                    write!(
                        f,
                        "movz {}, #{:#x}, lsl #{}",
                        reg_name(*rd, *width),
                        imm,
                        shift
                    )
                }
            }
            Instruction::MovkImm {
                width,
                rd,
                imm,
                shift,
            } => {
                if *shift == 0 {
                    write!(f, "movk {}, #{:#x}", reg_name(*rd, *width), imm)
                } else {
                    write!(
                        f,
                        "movk {}, #{:#x}, lsl #{}",
                        reg_name(*rd, *width),
                        imm,
                        shift
                    )
                }
            }

            // Conditional
            Instruction::Cset { width, rd, cond } => {
                write!(f, "cset {}, {}", reg_name(*rd, *width), cond_name(*cond))
            }

            // Load/Store
            Instruction::LdrImm {
                width,
                rt,
                rn,
                offset,
            } => {
                if *offset == 0 {
                    write!(
                        f,
                        "ldr {}, [{}]",
                        reg_name(*rt, *width),
                        reg_name_or_sp(*rn, Width::X64)
                    )
                } else {
                    write!(
                        f,
                        "ldr {}, [{}, #{}]",
                        reg_name(*rt, *width),
                        reg_name_or_sp(*rn, Width::X64),
                        offset
                    )
                }
            }
            Instruction::LdrbImm { rt, rn, offset } => {
                if *offset == 0 {
                    write!(
                        f,
                        "ldrb {}, [{}]",
                        reg_name(*rt, Width::W32),
                        reg_name_or_sp(*rn, Width::X64)
                    )
                } else {
                    write!(
                        f,
                        "ldrb {}, [{}, #{}]",
                        reg_name(*rt, Width::W32),
                        reg_name_or_sp(*rn, Width::X64),
                        offset
                    )
                }
            }
            Instruction::LdrhImm { rt, rn, offset } => {
                if *offset == 0 {
                    write!(
                        f,
                        "ldrh {}, [{}]",
                        reg_name(*rt, Width::W32),
                        reg_name_or_sp(*rn, Width::X64)
                    )
                } else {
                    write!(
                        f,
                        "ldrh {}, [{}, #{}]",
                        reg_name(*rt, Width::W32),
                        reg_name_or_sp(*rn, Width::X64),
                        offset
                    )
                }
            }
            Instruction::StrImm {
                width,
                rt,
                rn,
                offset,
            } => {
                if *offset == 0 {
                    write!(
                        f,
                        "str {}, [{}]",
                        reg_name(*rt, *width),
                        reg_name_or_sp(*rn, Width::X64)
                    )
                } else {
                    write!(
                        f,
                        "str {}, [{}, #{}]",
                        reg_name(*rt, *width),
                        reg_name_or_sp(*rn, Width::X64),
                        offset
                    )
                }
            }
            Instruction::StrbImm { rt, rn, offset } => {
                if *offset == 0 {
                    write!(
                        f,
                        "strb {}, [{}]",
                        reg_name(*rt, Width::W32),
                        reg_name_or_sp(*rn, Width::X64)
                    )
                } else {
                    write!(
                        f,
                        "strb {}, [{}, #{}]",
                        reg_name(*rt, Width::W32),
                        reg_name_or_sp(*rn, Width::X64),
                        offset
                    )
                }
            }
            Instruction::StrhImm { rt, rn, offset } => {
                if *offset == 0 {
                    write!(
                        f,
                        "strh {}, [{}]",
                        reg_name(*rt, Width::W32),
                        reg_name_or_sp(*rn, Width::X64)
                    )
                } else {
                    write!(
                        f,
                        "strh {}, [{}, #{}]",
                        reg_name(*rt, Width::W32),
                        reg_name_or_sp(*rn, Width::X64),
                        offset
                    )
                }
            }
            Instruction::Stp {
                width,
                rt1,
                rt2,
                rn,
                offset,
            } => {
                if *offset == 0 {
                    write!(
                        f,
                        "stp {}, {}, [{}]",
                        reg_name(*rt1, *width),
                        reg_name(*rt2, *width),
                        reg_name_or_sp(*rn, Width::X64)
                    )
                } else {
                    write!(
                        f,
                        "stp {}, {}, [{}, #{}]",
                        reg_name(*rt1, *width),
                        reg_name(*rt2, *width),
                        reg_name_or_sp(*rn, Width::X64),
                        offset
                    )
                }
            }
            Instruction::Ldp {
                width,
                rt1,
                rt2,
                rn,
                offset,
            } => {
                if *offset == 0 {
                    write!(
                        f,
                        "ldp {}, {}, [{}]",
                        reg_name(*rt1, *width),
                        reg_name(*rt2, *width),
                        reg_name_or_sp(*rn, Width::X64)
                    )
                } else {
                    write!(
                        f,
                        "ldp {}, {}, [{}, #{}]",
                        reg_name(*rt1, *width),
                        reg_name(*rt2, *width),
                        reg_name_or_sp(*rn, Width::X64),
                        offset
                    )
                }
            }

            // Sign extend
            Instruction::Sxtb { width, rd, rn } => {
                write!(
                    f,
                    "sxtb {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rn, Width::W32)
                )
            }
            Instruction::Sxth { width, rd, rn } => {
                write!(
                    f,
                    "sxth {}, {}",
                    reg_name(*rd, *width),
                    reg_name(*rn, Width::W32)
                )
            }
            Instruction::Sxtw { rd, rn } => {
                write!(
                    f,
                    "sxtw {}, {}",
                    reg_name(*rd, Width::X64),
                    reg_name(*rn, Width::W32)
                )
            }

            // Branches
            Instruction::B { target } => write!(f, "b {}", target),
            Instruction::Bl { target } => write!(f, "bl {}", target),
            Instruction::Blr { rn } => write!(f, "blr {}", reg_name_or_sp(*rn, Width::X64)),
            Instruction::Ret => write!(f, "ret"),
            Instruction::Cbz { width, rt, target } => {
                write!(f, "cbz {}, {}", reg_name(*rt, *width), target)
            }
            Instruction::Cbnz { width, rt, target } => {
                write!(f, "cbnz {}, {}", reg_name(*rt, *width), target)
            }
            Instruction::Tbz { rt, bit, target } => {
                write!(f, "tbz {}, #{}, {}", reg_name(*rt, Width::X64), bit, target)
            }
            Instruction::Tbnz { rt, bit, target } => {
                write!(
                    f,
                    "tbnz {}, #{}, {}",
                    reg_name(*rt, Width::X64),
                    bit,
                    target
                )
            }
            Instruction::BCond { cond, target } => {
                write!(f, "b.{} {}", cond_name(*cond), target)
            }

            // Pseudo-instructions
            Instruction::LoadExtern { rd, symbol } => {
                write!(f, "load_extern {}, @{}", reg_name(*rd, Width::X64), symbol)
            }

            // Special
            Instruction::Nop => write!(f, "nop"),
        }
    }
}

fn reg_name(reg: Reg, width: Width) -> String {
    let num = reg.raw();
    match width {
        Width::W32 => {
            if num == 31 {
                "wzr".to_string()
            } else {
                format!("w{}", num)
            }
        }
        Width::X64 => {
            if num == 31 {
                "xzr".to_string()
            } else {
                format!("x{}", num)
            }
        }
    }
}

/// Format register name, treating register 31 as SP instead of XZR.
/// Used for instructions where register 31 refers to the stack pointer.
fn reg_name_or_sp(reg: Reg, width: Width) -> String {
    let num = reg.raw();
    match width {
        Width::W32 => {
            if num == 31 {
                "wsp".to_string()
            } else {
                format!("w{}", num)
            }
        }
        Width::X64 => {
            if num == 31 {
                "sp".to_string()
            } else {
                format!("x{}", num)
            }
        }
    }
}

fn cond_name(cond: Condition) -> &'static str {
    match cond {
        Condition::Eq => "eq",
        Condition::Ne => "ne",
        Condition::Hs => "hs",
        Condition::Lo => "lo",
        Condition::Mi => "mi",
        Condition::Pl => "pl",
        Condition::Vs => "vs",
        Condition::Vc => "vc",
        Condition::Hi => "hi",
        Condition::Ls => "ls",
        Condition::Ge => "ge",
        Condition::Lt => "lt",
        Condition::Gt => "gt",
        Condition::Le => "le",
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
