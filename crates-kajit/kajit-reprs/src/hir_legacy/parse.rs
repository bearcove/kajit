use crate::hir_legacy::Module;

pub fn parse_hir(text: &str) -> Result<Module, String> {
    crate::hir_legacy::token_parser::parse_hir_v2(text)
}
