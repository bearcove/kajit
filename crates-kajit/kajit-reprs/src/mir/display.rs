pub fn to_text(program: &kajit_mir::Program) -> String {
    program.to_string()
}

pub fn to_text_with_registry(
    program: &kajit_mir::Program,
    registry: &kajit_ir::IntrinsicRegistry,
) -> String {
    program.display_with_registry(registry).to_string()
}
