pub fn to_text(func: &super::IrFunc) -> String {
    func.to_string()
}

pub fn to_text_with_registry(func: &super::IrFunc, registry: &super::IntrinsicRegistry) -> String {
    func.display_with_registry(registry).to_string()
}
