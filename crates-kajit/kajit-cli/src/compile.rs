use std::fs;
use std::path::Path;

use kajit_asm::schema_poc::FinalizedSchemaPocEmission;
use kajit_reprs::schema_poc;

use crate::validate::path_matches_ext;

pub(crate) fn cmd_compile(path: &Path) -> Result<(), String> {
    let source =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    let output = if path_matches_ext(path, schema_poc::asm::REPR_FILE_EXT) {
        let program = schema_poc::asm::parse_root_text(&source)
            .map_err(|err| format!("{}:\n{err}", path.display()))?;
        kajit_asm::schema_poc::assemble_schema_poc_program(&program, &source)?
    } else {
        return match path.extension().and_then(|ext| ext.to_str()) {
            Some(other) => Err(format!(
                "compile does not support .{other} yet for {}",
                path.display()
            )),
            None => Err(format!("cannot determine file type for {}", path.display())),
        };
    };

    print_compilation(path, &output)?;
    Ok(())
}

fn print_compilation(path: &Path, output: &FinalizedSchemaPocEmission) -> Result<(), String> {
    println!("input: {}", path.display());
    println!("repr: {}", schema_poc::asm::REPR_NAME);
    println!("bytes: {}", output.len());
    println!();
    println!("hex:");

    for (offset, chunk) in output.bytes().chunks(16).enumerate() {
        print!("{:04x}: ", offset * 16);
        for (idx, byte) in chunk.iter().enumerate() {
            if idx > 0 {
                print!(" ");
            }
            print!("{byte:02x}");
        }
        println!();
    }

    println!();
    println!("trace:");
    print!("{}", output.trace_text()?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_aarch64_schema_poc_asm() {
        let path = Path::new("/tmp/example.k-asm");
        let source = "asm aarch64 { start: movz x0, 42 ret }";
        let program = schema_poc::asm::parse_root_text(source).expect("expected valid .k-asm");
        let output =
            kajit_asm::schema_poc::assemble_schema_poc_program(&program, source).expect("emit");
        assert!(matches!(output, FinalizedSchemaPocEmission::AArch64(_)));
        assert!(output.len() > 0);
        assert!(output.trace_text().is_ok());
        assert!(path_matches_ext(path, schema_poc::asm::REPR_FILE_EXT));
    }

    #[test]
    fn compiles_x64_schema_poc_asm() {
        let path = Path::new("/tmp/example.k-asm");
        let source = "asm x86_64 { entry: mov rax, 42 ret }";
        let program = schema_poc::asm::parse_root_text(source).expect("expected valid .k-asm");
        let output =
            kajit_asm::schema_poc::assemble_schema_poc_program(&program, source).expect("emit");
        assert!(matches!(output, FinalizedSchemaPocEmission::X64(_)));
        assert!(output.len() > 0);
        assert!(output.trace_text().is_ok());
        assert!(path_matches_ext(path, schema_poc::asm::REPR_FILE_EXT));
    }
}
