mod defs;

pub use defs::{LangDef, read_from_file};

/// A single generated Rust source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedFile {
    /// The relative Rust source file path to emit, such as `parser.rs`.
    pub path: String,
    /// The full stubbed contents to write into the generated file.
    pub contents: String,
}

/// The set of Rust source files produced for a language definition.
///
/// These are stubbed for now and can be written out by a caller however it wants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedFiles {
    /// The generated AST module source file.
    pub ast: GeneratedFile,
    /// The generated parser module source file.
    pub parser: GeneratedFile,
    /// The generated formatter module source file.
    pub formatter: GeneratedFile,
    /// The generated crate root source file.
    pub lib: GeneratedFile,
}

impl GeneratedFiles {
    pub fn all(&self) -> [&GeneratedFile; 4] {
        [&self.ast, &self.parser, &self.formatter, &self.lib]
    }
}

pub fn codegen(def: &LangDef) -> Result<GeneratedFiles, String> {
    Ok(GeneratedFiles {
        ast: GeneratedFile {
            path: "ast.rs".to_string(),
            contents: stub_ast(def),
        },
        parser: GeneratedFile {
            path: "parser.rs".to_string(),
            contents: stub_parser(def),
        },
        formatter: GeneratedFile {
            path: "formatter.rs".to_string(),
            contents: stub_formatter(def),
        },
        lib: GeneratedFile {
            path: "lib.rs".to_string(),
            contents: stub_lib(def),
        },
    })
}

fn stub_ast(def: &LangDef) -> String {
    format!(
        "// Generated AST for {name}\n\
         // TODO: flesh out generated AST types.\n\n\
         #[derive(Debug, Clone, PartialEq, Eq)]\n\
         pub struct Ast;\n",
        name = def.name
    )
}

fn stub_parser(def: &LangDef) -> String {
    format!(
        "// Generated parser for {name}\n\
         // TODO: flesh out generated parser.\n\n\
         use crate::ast::Ast;\n\n\
         pub fn parse(_input: &str) -> Result<Ast, String> {{\n\
         \tErr(\"parser not implemented yet\".to_string())\n\
         }}\n",
        name = def.name
    )
}

fn stub_formatter(def: &LangDef) -> String {
    format!(
        "// Generated formatter for {name}\n\
         // TODO: flesh out generated formatter.\n\n\
         use crate::ast::Ast;\n\n\
         pub fn format(_ast: &Ast) -> String {{\n\
         \tString::new()\n\
         }}\n",
        name = def.name
    )
}

fn stub_lib(def: &LangDef) -> String {
    format!(
        "// Generated library root for {name}\n\
         // TODO: flesh out generated module exports.\n\n\
         pub mod ast;\n\
         pub mod formatter;\n\
         pub mod parser;\n\n\
         pub use ast::Ast;\n\
         pub use parser::parse;\n",
        name = def.name
    )
}
