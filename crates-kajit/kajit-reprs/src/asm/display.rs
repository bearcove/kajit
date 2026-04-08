pub use kajit_asm::{format_trace, format_trace_entries};

pub fn to_text(program: &kajit_asm::aarch64_asm::Program) -> String {
    program.to_string()
}
