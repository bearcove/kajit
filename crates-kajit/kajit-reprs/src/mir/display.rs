pub(crate) use crate::ir as kajit_ir;

pub fn to_text(program: &super::Program) -> String {
    program.to_string()
}

pub fn to_text_with_registry(
    program: &super::Program,
    registry: &kajit_ir::IntrinsicRegistry,
) -> String {
    program.display_with_registry(registry).to_string()
}
