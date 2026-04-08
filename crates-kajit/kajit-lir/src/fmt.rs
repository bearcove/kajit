// ─── Display ─────────────────────────────────────────────────────────────────

use core::fmt;
use kajit_ir::IntrinsicRegistry;

use crate::{LinearIr, LinearOp};

impl fmt::Display for LinearIr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let display = LinearIrDisplay {
            linear: self,
            registry: None,
        };
        fmt::Display::fmt(&display, f)
    }
}

pub struct LinearIrDisplay<'a> {
    linear: &'a LinearIr,
    registry: Option<&'a IntrinsicRegistry>,
}

impl<'a> fmt::Display for LinearIrDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for op in &self.linear.ops {
            // Labels get no indentation, everything else gets 2 spaces.
            match op {
                LinearOp::Label(label) => {
                    writeln!(f, "L{}:", label.index())?;
                }
                LinearOp::FuncStart {
                    lambda_id, label, ..
                } => {
                    writeln!(f, "func λ{} ({label}):", lambda_id.index())?;
                }
                LinearOp::FuncEnd => {
                    writeln!(f, "end")?;
                }
                _ => {
                    write!(f, "  ")?;
                    fmt_op(f, op, self.registry)?;
                    writeln!(f)?;
                }
            }
        }
        Ok(())
    }
}

impl LinearIr {
    pub fn display_with_registry<'a>(
        &'a self,
        registry: &'a IntrinsicRegistry,
    ) -> LinearIrDisplay<'a> {
        LinearIrDisplay {
            linear: self,
            registry: Some(registry),
        }
    }
}

fn fmt_vreg(f: &mut fmt::Formatter<'_>, v: VReg) -> fmt::Result {
    write!(f, "v{}", v.index())
}

fn fmt_op(
    f: &mut fmt::Formatter<'_>,
    op: &LinearOp,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    match op {
        LinearOp::Const { dst, value } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = const ")?;
            fmt_const(f, *value, registry)
        }
        LinearOp::DataAddr { dst, blob_id } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = data_addr({blob_id})")
        }
        LinearOp::ExternAddr { dst, symbol, .. } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = extern_addr(@{symbol})")
        }
        LinearOp::BinOp { op, dst, lhs, rhs } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = ")?;
            fmt_vreg(f, *lhs)?;
            write!(f, " {op:?} ")?;
            fmt_vreg(f, *rhs)
        }
        LinearOp::UnaryOp { op, dst, src } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = {op:?} ")?;
            fmt_vreg(f, *src)
        }
        LinearOp::Copy { dst, src } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = copy ")?;
            fmt_vreg(f, *src)
        }
        LinearOp::StackAlloc { dst, id } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = stack_alloc({})", id.index())
        }
        LinearOp::StoreToAddr { addr, src, width } => {
            write!(f, "store_addr [{width}] ")?;
            fmt_vreg(f, *addr)?;
            write!(f, ", ")?;
            fmt_vreg(f, *src)
        }
        LinearOp::LoadFromAddr { dst, addr, width } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = load_addr [{width}] ")?;
            fmt_vreg(f, *addr)
        }
        LinearOp::Call { func, args, dst } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = call ")?;
            fmt_intrinsic(f, *func, registry)?;
            write!(f, "(")?;
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_vreg(f, *a)?;
            }
            write!(f, ")")
        }
        LinearOp::Branch { target, phi_args } => {
            write!(f, "br L{}", target.index())?;
            if !phi_args.is_empty() {
                write!(f, " phi[")?;
                for (i, (src, dst)) in phi_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *src)?;
                    write!(f, "→")?;
                    fmt_vreg(f, *dst)?;
                }
                write!(f, "]")?;
            }
            Ok(())
        }
        LinearOp::BranchIf {
            cond,
            target,
            phi_args,
            fallthrough_phi_args,
        } => {
            write!(f, "br_if ")?;
            fmt_vreg(f, *cond)?;
            write!(f, " L{}", target.index())?;
            if !phi_args.is_empty() {
                write!(f, " phi[")?;
                for (i, (src, dst)) in phi_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *src)?;
                    write!(f, "→")?;
                    fmt_vreg(f, *dst)?;
                }
                write!(f, "]")?;
            }
            if !fallthrough_phi_args.is_empty() {
                write!(f, " fall[")?;
                for (i, (src, dst)) in fallthrough_phi_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *src)?;
                    write!(f, "→")?;
                    fmt_vreg(f, *dst)?;
                }
                write!(f, "]")?;
            }
            Ok(())
        }
        LinearOp::BranchIfZero {
            cond,
            target,
            phi_args,
            fallthrough_phi_args,
        } => {
            write!(f, "br_zero ")?;
            fmt_vreg(f, *cond)?;
            write!(f, " L{}", target.index())?;
            if !phi_args.is_empty() {
                write!(f, " phi[")?;
                for (i, (src, dst)) in phi_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *src)?;
                    write!(f, "→")?;
                    fmt_vreg(f, *dst)?;
                }
                write!(f, "]")?;
            }
            if !fallthrough_phi_args.is_empty() {
                write!(f, " fall[")?;
                for (i, (src, dst)) in fallthrough_phi_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *src)?;
                    write!(f, "→")?;
                    fmt_vreg(f, *dst)?;
                }
                write!(f, "]")?;
            }
            Ok(())
        }
        LinearOp::JumpTable {
            predicate,
            labels,
            default,
        } => {
            write!(f, "jump_table ")?;
            fmt_vreg(f, *predicate)?;
            write!(f, " [")?;
            for (i, l) in labels.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "L{}", l.index())?;
            }
            write!(f, "] default L{}", default.index())
        }
        LinearOp::CallLambda {
            target,
            args,
            results,
        } => {
            if !results.is_empty() {
                for (i, r) in results.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *r)?;
                }
                write!(f, " = ")?;
            }
            write!(f, "call λ{}(", target.index())?;
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_vreg(f, *a)?;
            }
            write!(f, ")")
        }
        // FuncStart/FuncEnd/Label handled in Display for LinearIr
        LinearOp::Label(_) | LinearOp::FuncStart { .. } | LinearOp::FuncEnd => {
            unreachable!("handled in Display for LinearIr")
        }
    }
}

fn fmt_intrinsic(
    f: &mut fmt::Formatter<'_>,
    func: FnPtr,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    if let Some(registry) = registry
        && let Some(name) = registry.name_of(func)
    {
        return write!(f, "@{name}");
    }
    write!(f, "{func}")
}

fn fmt_const(
    f: &mut fmt::Formatter<'_>,
    value: u64,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    if let Some(registry) = registry
        && let Some(name) = registry.const_name_of(value)
    {
        return write!(f, "@{name}");
    }
    write!(f, "{value}")
}

impl fmt::Debug for LinearIr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LinearIr {{")?;
        writeln!(
            f,
            "  labels: {}, vregs: {}, slots: {}",
            self.label_count, self.vreg_count, self.slot_count
        )?;
        for op in &self.ops {
            writeln!(f, "  {op:?}")?;
        }
        writeln!(f, "}}")
    }
}
