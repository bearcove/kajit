pub fn to_text(func: &kajit_ir::IrFunc) -> String {
    func.to_string()
}

pub fn to_text_with_registry(
    func: &kajit_ir::IrFunc,
    registry: &kajit_ir::IntrinsicRegistry,
) -> String {
    func.display_with_registry(registry).to_string()
}
