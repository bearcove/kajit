pub use super::{format_trace, format_trace_entries};

pub fn to_text(program: &super::aarch64_asm::Program) -> String {
    program.to_string()
}
