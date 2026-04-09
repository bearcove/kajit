use crate::hir::Module;

pub fn parse_hir(text: &str) -> Result<Module, String> {
    crate::hir::token_parser::parse_hir_v2(text)
}
