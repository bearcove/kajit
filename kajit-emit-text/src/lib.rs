use chumsky::prelude::*;

use kajit_emit::aarch64::{Condition, Reg, Width};
use kajit_emit::aarch64_asm::{Instruction, Item, Label, Program};

type Extra<'src> = extra::Err<Rich<'src, char>>;

fn ws<'src>() -> impl Parser<'src, &'src str, (), Extra<'src>> + Clone {
    any()
        .filter(|c: &char| c.is_whitespace())
        .repeated()
        .ignored()
}

fn token<'src>(text: &'static str) -> impl Parser<'src, &'src str, (), Extra<'src>> + Clone {
    just(text).padded_by(ws()).ignored()
}

fn uint32<'src>() -> impl Parser<'src, &'src str, u32, Extra<'src>> + Clone {
    let hex = just("0x")
        .ignore_then(text::int::<_, Extra<'_>>(16))
        .map(|s: &str| u32::from_str_radix(s, 16).unwrap());
    let dec = text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u32>().unwrap());
    hex.or(dec).padded_by(ws())
}

fn uint64<'src>() -> impl Parser<'src, &'src str, u64, Extra<'src>> + Clone {
    let hex = just("0x")
        .ignore_then(text::int::<_, Extra<'_>>(16))
        .map(|s: &str| u64::from_str_radix(s, 16).unwrap());
    let dec = text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u64>().unwrap());
    hex.or(dec).padded_by(ws())
}

fn uint16<'src>() -> impl Parser<'src, &'src str, u16, Extra<'src>> + Clone {
    let hex = just("0x")
        .ignore_then(text::int::<_, Extra<'_>>(16))
        .map(|s: &str| u16::from_str_radix(s, 16).unwrap());
    let dec = text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u16>().unwrap());
    hex.or(dec).padded_by(ws())
}

fn uint8<'src>() -> impl Parser<'src, &'src str, u8, Extra<'src>> + Clone {
    text::int::<_, Extra<'_>>(10)
        .map(|s: &str| s.parse::<u8>().unwrap())
        .padded_by(ws())
}

// Parse a label like .L0:
fn label<'src>() -> impl Parser<'src, &'src str, Label, Extra<'src>> + Clone {
    just(".L")
        .ignore_then(uint32())
        .then_ignore(just(':'))
        .map(Label)
        .padded_by(ws())
}

// Parse a register like x0, w0, xzr, wzr
fn register<'src>() -> impl Parser<'src, &'src str, (Reg, Width), Extra<'src>> + Clone {
    let xzr = just("xzr").to((Reg::XZR, Width::X64));
    let wzr = just("wzr").to((Reg::XZR, Width::W32)); // wzr is the 32-bit version of xzr (reg 31)
    let sp = just("sp").to((Reg::SP, Width::X64));

    let x_reg = just('x')
        .ignore_then(text::int::<_, Extra<'_>>(10))
        .map(|s: &str| {
            let num = s.parse::<u8>().unwrap();
            (Reg::from_raw(num), Width::X64)
        });

    let w_reg = just('w')
        .ignore_then(text::int::<_, Extra<'_>>(10))
        .map(|s: &str| {
            let num = s.parse::<u8>().unwrap();
            (Reg::from_raw(num), Width::W32)
        });

    choice((xzr, wzr, sp, x_reg, w_reg)).padded_by(ws())
}

// Parse an immediate like #42 or #0x1234
fn immediate<'src>() -> impl Parser<'src, &'src str, u64, Extra<'src>> + Clone {
    just('#').ignore_then(uint64())
}

// Parse a condition code like eq, ne, gt, etc.
fn condition<'src>() -> impl Parser<'src, &'src str, Condition, Extra<'src>> + Clone {
    choice((
        just("eq").to(Condition::Eq),
        just("ne").to(Condition::Ne),
        just("hs").to(Condition::Hs),
        just("lo").to(Condition::Lo),
        just("mi").to(Condition::Mi),
        just("pl").to(Condition::Pl),
        just("vs").to(Condition::Vs),
        just("vc").to(Condition::Vc),
        just("hi").to(Condition::Hi),
        just("ls").to(Condition::Ls),
        just("ge").to(Condition::Ge),
        just("lt").to(Condition::Lt),
        just("gt").to(Condition::Gt),
        just("le").to(Condition::Le),
    ))
    .padded_by(ws())
}

// Parse a label reference like .L0
fn label_ref<'src>() -> impl Parser<'src, &'src str, Label, Extra<'src>> + Clone {
    just(".L").ignore_then(uint32()).map(Label).padded_by(ws())
}

// Parse a symbol name like @kajit_encode_postcard_string
fn symbol<'src>() -> impl Parser<'src, &'src str, String, Extra<'src>> + Clone {
    just('@')
        .ignore_then(
            any()
                .filter(|c: &char| c.is_alphanumeric() || *c == '_')
                .repeated()
                .at_least(1)
                .collect::<String>(),
        )
        .padded_by(ws())
}

// Parse memory operand like [x0] or [x0, #16]
fn memory_operand<'src>() -> impl Parser<'src, &'src str, (Reg, u32), Extra<'src>> + Clone {
    just('[')
        .ignore_then(register())
        .then(
            just(',')
                .ignore_then(ws())
                .ignore_then(immediate())
                .map(|imm| imm as u32)
                .or_not(),
        )
        .then_ignore(just(']'))
        .map(|((reg, _width), offset)| (reg, offset.unwrap_or(0)))
        .padded_by(ws())
}

// Parse instruction - comprehensive implementation
fn instruction<'src>() -> impl Parser<'src, &'src str, Instruction, Extra<'src>> + Clone {
    // Simple zero-operand instructions
    let ret = just("ret").to(Instruction::Ret);
    let nop = just("nop").to(Instruction::Nop);

    // Branches
    let b = just("b ")
        .ignore_then(label_ref())
        .map(|target| Instruction::B { target });

    let bl = just("bl ")
        .ignore_then(label_ref())
        .map(|target| Instruction::Bl { target });

    let blr = just("blr ")
        .ignore_then(register())
        .map(|(rn, _)| Instruction::Blr { rn });

    // Conditional branches
    let b_cond = just("b.")
        .ignore_then(condition())
        .then_ignore(ws())
        .then(label_ref())
        .map(|(cond, target)| Instruction::BCond { cond, target });

    let cbz = just("cbz ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(label_ref())
        .map(|((rt, width), target)| Instruction::Cbz { width, rt, target });

    let cbnz = just("cbnz ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(label_ref())
        .map(|((rt, width), target)| Instruction::Cbnz { width, rt, target });

    let tbz = just("tbz ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(immediate().map(|i| i as u8))
        .then_ignore(token(","))
        .then(label_ref())
        .map(|(((rt, _), bit), target)| Instruction::Tbz { rt, bit, target });

    let tbnz = just("tbnz ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(immediate().map(|i| i as u8))
        .then_ignore(token(","))
        .then(label_ref())
        .map(|(((rt, _), bit), target)| Instruction::Tbnz { rt, bit, target });

    // Move instructions
    let mov = just("mov ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|((rd, width), (rm, _))| Instruction::MovReg { width, rd, rm });

    let movz = just("movz ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(immediate().map(|i| i as u16))
        .then(
            token(",")
                .ignore_then(just("lsl "))
                .ignore_then(immediate().map(|i| i as u8))
                .or_not()
                .map(|s| s.unwrap_or(0)),
        )
        .map(|(((rd, width), imm), shift)| Instruction::MovzImm {
            width,
            rd,
            imm,
            shift,
        });

    let movk = just("movk ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(immediate().map(|i| i as u16))
        .then(
            token(",")
                .ignore_then(just("lsl "))
                .ignore_then(immediate().map(|i| i as u8))
                .or_not()
                .map(|s| s.unwrap_or(0)),
        )
        .map(|(((rd, width), imm), shift)| Instruction::MovkImm {
            width,
            rd,
            imm,
            shift,
        });

    let load_extern = just("load_extern ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(symbol())
        .map(|((rd, _width), symbol)| Instruction::LoadExtern { rd, symbol });

    // Arithmetic immediate
    let add_imm = just("add ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(immediate())
        .map(|(((rd, width), (rn, _)), imm)| Instruction::AddImm {
            width,
            rd,
            rn,
            imm: imm as u32,
        });

    let sub_imm = just("sub ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(immediate())
        .map(|(((rd, width), (rn, _)), imm)| Instruction::SubImm {
            width,
            rd,
            rn,
            imm: imm as u32,
        });

    let cmp_imm = just("cmp ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(immediate())
        .map(|((rn, width), imm)| Instruction::CmpImm {
            width,
            rn,
            imm: imm as u32,
        });

    // Arithmetic register
    let add_reg = just("add ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|(((rd, width), (rn, _)), (rm, _))| Instruction::AddReg { width, rd, rn, rm });

    let sub_reg = just("sub ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|(((rd, width), (rn, _)), (rm, _))| Instruction::SubReg { width, rd, rn, rm });

    let cmp_reg = just("cmp ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|((rn, width), (rm, _))| Instruction::CmpReg { width, rn, rm });

    let mul = just("mul ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|(((rd, width), (rn, _)), (rm, _))| Instruction::MulReg { width, rd, rn, rm });

    let neg = just("neg ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|((rd, width), (rm, _))| Instruction::NegReg { width, rd, rm });

    // Logical immediate
    let and_imm = just("and ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(immediate())
        .map(|(((rd, width), (rn, _)), imm)| Instruction::AndImm { width, rd, rn, imm });

    let orr_imm = just("orr ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(immediate())
        .map(|(((rd, width), (rn, _)), imm)| Instruction::OrrImm { width, rd, rn, imm });

    let eor_imm = just("eor ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(immediate())
        .map(|(((rd, width), (rn, _)), imm)| Instruction::EorImm { width, rd, rn, imm });

    // Logical register
    let and_reg = just("and ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|(((rd, width), (rn, _)), (rm, _))| Instruction::AndReg { width, rd, rn, rm });

    let orr_reg = just("orr ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|(((rd, width), (rn, _)), (rm, _))| Instruction::OrrReg { width, rd, rn, rm });

    let eor_reg = just("eor ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|(((rd, width), (rn, _)), (rm, _))| Instruction::EorReg { width, rd, rn, rm });

    // Shifts
    let lsl_imm = just("lsl ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(immediate().map(|i| i as u8))
        .map(|(((rd, width), (rn, _)), shift)| Instruction::LslImm {
            width,
            rd,
            rn,
            shift,
        });

    let lsr_imm = just("lsr ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(immediate().map(|i| i as u8))
        .map(|(((rd, width), (rn, _)), shift)| Instruction::LsrImm {
            width,
            rd,
            rn,
            shift,
        });

    let lsl_reg = just("lsl ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|(((rd, width), (rn, _)), (rm, _))| Instruction::LslReg { width, rd, rn, rm });

    let lsr_reg = just("lsr ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|(((rd, width), (rn, _)), (rm, _))| Instruction::LsrReg { width, rd, rn, rm });

    let asr_imm = just("asr ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(immediate().map(|i| i as u8))
        .map(|(((rd, width), (rn, _)), shift)| Instruction::AsrImm {
            width,
            rd,
            rn,
            shift,
        });

    let asr_reg = just("asr ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|(((rd, width), (rn, _)), (rm, _))| Instruction::AsrReg { width, rd, rn, rm });

    // Load/Store
    let ldr = just("ldr ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(memory_operand())
        .map(|((rt, width), (rn, offset))| Instruction::LdrImm {
            width,
            rt,
            rn,
            offset,
        });

    let ldrb = just("ldrb ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(memory_operand())
        .map(|((rt, _), (rn, offset))| Instruction::LdrbImm { rt, rn, offset });

    let ldrh = just("ldrh ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(memory_operand())
        .map(|((rt, _), (rn, offset))| Instruction::LdrhImm { rt, rn, offset });

    let str = just("str ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(memory_operand())
        .map(|((rt, width), (rn, offset))| Instruction::StrImm {
            width,
            rt,
            rn,
            offset,
        });

    let strb = just("strb ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(memory_operand())
        .map(|((rt, _), (rn, offset))| Instruction::StrbImm { rt, rn, offset });

    let strh = just("strh ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(memory_operand())
        .map(|((rt, _), (rn, offset))| Instruction::StrhImm { rt, rn, offset });

    // Sign extend
    let sxtb = just("sxtb ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|((rd, width), (rn, _))| Instruction::Sxtb { width, rd, rn });

    let sxth = just("sxth ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|((rd, width), (rn, _))| Instruction::Sxth { width, rd, rn });

    let sxtw = just("sxtw ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .map(|((rd, _), (rn, _))| Instruction::Sxtw { rd, rn });

    // Conditional
    let cset = just("cset ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(condition())
        .map(|((rd, width), cond)| Instruction::Cset { width, rd, cond });

    // Group parsers to avoid hitting choice tuple size limits
    let simple = choice((ret, nop));

    let branches = choice((b_cond, bl, blr, b, cbz, cbnz, tbz, tbnz));

    let pseudo = load_extern;

    let moves = choice((movz, movk, mov));

    let arith_imm = choice((add_imm, sub_imm, cmp_imm));

    let arith_reg = choice((add_reg, sub_reg, cmp_reg, mul, neg));

    let logical = choice((
        and_imm.or(and_reg),
        orr_imm.or(orr_reg),
        eor_imm.or(eor_reg),
    ));

    let shifts = choice((
        lsl_imm.or(lsl_reg),
        lsr_imm.or(lsr_reg),
        asr_imm.or(asr_reg),
    ));

    let stp = just("stp ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(memory_operand())
        .map(
            |(((rt1, width), (rt2, _)), (rn, offset))| Instruction::Stp {
                width,
                rt1,
                rt2,
                rn,
                offset: offset as i16,
            },
        );

    let ldp = just("ldp ")
        .ignore_then(register())
        .then_ignore(token(","))
        .then(register())
        .then_ignore(token(","))
        .then(memory_operand())
        .map(
            |(((rt1, width), (rt2, _)), (rn, offset))| Instruction::Ldp {
                width,
                rt1,
                rt2,
                rn,
                offset: offset as i16,
            },
        );

    let load_store = choice((stp, ldp, ldr, ldrb, ldrh, str, strb, strh));

    let sign_extend = choice((sxtb, sxth, sxtw));

    let conditional = cset;

    // Combine all groups
    choice((
        simple,
        branches,
        pseudo,
        moves,
        arith_imm,
        arith_reg,
        logical,
        shifts,
        load_store,
        sign_extend,
        conditional,
    ))
}

// Parse a program item (label or instruction)
fn item<'src>() -> impl Parser<'src, &'src str, Item, Extra<'src>> + Clone {
    let label_item = label().map(Item::Label);
    let inst_item = instruction().map(Item::Instruction);

    choice((label_item, inst_item))
}

// Parse a complete program
fn program<'src>() -> impl Parser<'src, &'src str, Program, Extra<'src>> + Clone {
    ws().ignore_then(item().repeated().collect::<Vec<_>>())
        .then_ignore(ws())
        .then_ignore(end())
        .map(|items| Program { items })
}

/// Parse ARM64 assembly text into a Program
pub fn parse_asm(text: &str) -> Result<Program, Vec<Rich<'_, char>>> {
    program().parse(text).into_result()
}

/// Expand pseudo-instructions (like LoadExtern) into real instructions.
///
/// The symbol_resolver function maps symbol names to 64-bit addresses.
pub fn expand_pseudo_instructions<F>(program: Program, symbol_resolver: F) -> Program
where
    F: Fn(&str) -> Option<u64>,
{
    let mut expanded_items = Vec::new();

    for item in program.items {
        match item {
            Item::Instruction(Instruction::LoadExtern { rd, symbol }) => {
                let addr = symbol_resolver(&symbol)
                    .unwrap_or_else(|| panic!("Unknown symbol: {}", symbol));

                // Expand to movz/movk sequence
                let p0 = (addr & 0xFFFF) as u16;
                let p1 = ((addr >> 16) & 0xFFFF) as u16;
                let p2 = ((addr >> 32) & 0xFFFF) as u16;
                let p3 = ((addr >> 48) & 0xFFFF) as u16;

                expanded_items.push(Item::Instruction(Instruction::MovzImm {
                    width: Width::X64,
                    rd,
                    imm: p0,
                    shift: 0,
                }));

                if p1 != 0 {
                    expanded_items.push(Item::Instruction(Instruction::MovkImm {
                        width: Width::X64,
                        rd,
                        imm: p1,
                        shift: 16,
                    }));
                }

                if p2 != 0 {
                    expanded_items.push(Item::Instruction(Instruction::MovkImm {
                        width: Width::X64,
                        rd,
                        imm: p2,
                        shift: 32,
                    }));
                }

                if p3 != 0 {
                    expanded_items.push(Item::Instruction(Instruction::MovkImm {
                        width: Width::X64,
                        rd,
                        imm: p3,
                        shift: 48,
                    }));
                }
            }
            other => expanded_items.push(other),
        }
    }

    Program {
        items: expanded_items,
    }
}

/// Assemble a Program into executable machine code.
///
/// This function replays the program through an Emitter, handling label
/// binding and resolution automatically.
#[cfg(target_arch = "aarch64")]
pub fn assemble(program: &Program) -> Result<kajit_emit::aarch64::FinalizedEmission, String> {
    use kajit_emit::aarch64::Emitter;
    use std::collections::HashMap;

    let mut emitter = Emitter::new();
    let mut label_map: HashMap<Label, kajit_emit::aarch64::LabelId> = HashMap::new();

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
    emitter: &mut kajit_emit::aarch64::Emitter,
    inst: &Instruction,
    label_map: &std::collections::HashMap<Label, kajit_emit::aarch64::LabelId>,
) -> Result<(), String> {
    use kajit_emit::aarch64::{Condition, Reg, Width};

    let map_err = |e: kajit_emit::aarch64::EmitError| format!("{:?}", e);

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
            .emit_and_imm(*width, *rd, *rn, *imm as u64)
            .map_err(map_err),
        Instruction::AndReg { width, rd, rn, rm } => {
            emitter.emit_and_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
        Instruction::OrrImm { width, rd, rn, imm } => emitter
            .emit_orr_imm(*width, *rd, *rn, *imm as u64)
            .map_err(map_err),
        Instruction::OrrReg { width, rd, rn, rm } => {
            emitter.emit_orr_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
        Instruction::EorImm { width, rd, rn, imm } => emitter
            .emit_eor_imm(*width, *rd, *rn, *imm as u64)
            .map_err(map_err),
        Instruction::EorReg { width, rd, rn, rm } => {
            emitter.emit_eor_reg(*width, *rd, *rn, *rm).map_err(map_err)
        }
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
            .emit_ldr_imm(*width, *rt, *rn, *offset as u32)
            .map_err(map_err),
        Instruction::LdrbImm { rt, rn, offset } => emitter
            .emit_ldrb_imm(*rt, *rn, *offset as u32)
            .map_err(map_err),
        Instruction::LdrhImm { rt, rn, offset } => emitter
            .emit_ldrh_imm(*rt, *rn, *offset as u32)
            .map_err(map_err),
        Instruction::StrImm {
            width,
            rt,
            rn,
            offset,
        } => emitter
            .emit_str_imm(*width, *rt, *rn, *offset as u32)
            .map_err(map_err),
        Instruction::StrbImm { rt, rn, offset } => emitter
            .emit_strb_imm(*rt, *rn, *offset as u32)
            .map_err(map_err),
        Instruction::StrhImm { rt, rn, offset } => emitter
            .emit_strh_imm(*rt, *rn, *offset as u32)
            .map_err(map_err),
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
            return Err(
                "LoadExtern pseudo-instruction must be expanded before assembly".to_string(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_extern() {
        let asm = "load_extern x16, @test_function\n";
        let parsed = parse_asm(asm).expect("Failed to parse");

        assert_eq!(parsed.items.len(), 1);
        match &parsed.items[0] {
            Item::Instruction(Instruction::LoadExtern { rd, symbol }) => {
                assert_eq!(*rd, Reg::X16);
                assert_eq!(symbol, "test_function");
            }
            _ => panic!("Expected LoadExtern instruction"),
        }

        // Test expansion
        let symbol_resolver = |name: &str| {
            if name == "test_function" {
                Some(0x123456789abcdef0)
            } else {
                None
            }
        };

        let expanded = expand_pseudo_instructions(parsed, symbol_resolver);

        // Should expand to movz + 3 movk instructions (all parts non-zero)
        assert_eq!(expanded.items.len(), 4);

        // Check movz
        match &expanded.items[0] {
            Item::Instruction(Instruction::MovzImm { rd, imm, shift, .. }) => {
                assert_eq!(*rd, Reg::X16);
                assert_eq!(*imm, 0xdef0);
                assert_eq!(*shift, 0);
            }
            _ => panic!("Expected MovzImm"),
        }

        // Check first movk
        match &expanded.items[1] {
            Item::Instruction(Instruction::MovkImm { rd, imm, shift, .. }) => {
                assert_eq!(*rd, Reg::X16);
                assert_eq!(*imm, 0x9abc);
                assert_eq!(*shift, 16);
            }
            _ => panic!("Expected MovkImm"),
        }
    }

    #[test]
    fn test_load_extern_display() {
        let prog = Program {
            items: vec![Item::Instruction(Instruction::LoadExtern {
                rd: Reg::X16,
                symbol: "my_function".to_string(),
            })],
        };

        let formatted = format!("{}", prog);
        assert_eq!(formatted.trim(), "load_extern x16, @my_function");

        // Round-trip test
        let parsed = parse_asm(&formatted).expect("Failed to parse formatted output");
        assert_eq!(parsed.items.len(), 1);
        match &parsed.items[0] {
            Item::Instruction(Instruction::LoadExtern { rd, symbol }) => {
                assert_eq!(*rd, Reg::X16);
                assert_eq!(symbol, "my_function");
            }
            _ => panic!("Expected LoadExtern instruction"),
        }
    }

    #[test]
    fn parse_label() {
        let result = label().parse(".L0:").into_result();
        assert_eq!(result, Ok(Label(0)));

        let result = label().parse(".L42:").into_result();
        assert_eq!(result, Ok(Label(42)));
    }

    #[test]
    fn parse_registers() {
        let result = register().parse("x0").into_result();
        assert_eq!(result, Ok((Reg::X0, Width::X64)));

        let result = register().parse("w15").into_result();
        assert_eq!(result, Ok((Reg::from_raw(15), Width::W32)));

        let result = register().parse("xzr").into_result();
        assert_eq!(result, Ok((Reg::XZR, Width::X64)));
    }

    #[test]
    fn parse_immediates() {
        let result = immediate().parse("#42").into_result();
        assert_eq!(result, Ok(42));

        let result = immediate().parse("#0x1234").into_result();
        assert_eq!(result, Ok(0x1234));
    }

    #[test]
    fn parse_simple_program() {
        let asm = r#"
.L0:
  movz x0, #0x42
  add x1, x0, #7
  ret
"#;

        let result = parse_asm(asm);
        assert!(result.is_ok(), "Parse failed: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.items.len(), 4); // label + 3 instructions
    }

    #[test]
    fn round_trip_program() {
        use kajit_emit::aarch64::{Reg, Width};

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
                Item::Instruction(Instruction::Ret),
            ],
        };

        let formatted = format!("{}", program);
        let parsed = parse_asm(&formatted).expect("Failed to parse formatted output");

        assert_eq!(parsed.items.len(), program.items.len());
    }
}
