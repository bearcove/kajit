use std::fs;
use std::path::Path;

use kajit_asm::schema_poc::FinalizedSchemaPocEmission;
use kajit_reprs::schema_poc;
use kajit_wares::{ObjectInput, PrintMainExecutableInput, TargetArch};

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

    let exe_path = write_schema_poc_asm_executable(path, &output)?;
    println!("{}", exe_path.display());
    Ok(())
}

fn write_schema_poc_asm_executable(
    path: &Path,
    output: &FinalizedSchemaPocEmission,
) -> Result<std::path::PathBuf, String> {
    let output_dir = std::path::PathBuf::from("target").join("kajit-compile");
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| format!("cannot determine output name for {}", path.display()))?;

    let (arch_suffix, object) = match output {
        FinalizedSchemaPocEmission::AArch64(_) => (
            "aarch64",
            ObjectInput {
                target_arch: TargetArch::Aarch64,
                code: output.bytes(),
                entry_offset: 0,
                function_name: "kajit_main",
                dwarf: None,
                intrinsic_calls: &[],
                extern_addr_relocs: &[],
            },
        ),
        FinalizedSchemaPocEmission::X64(_) => (
            "x86_64",
            ObjectInput {
                target_arch: TargetArch::X86_64,
                code: output.bytes(),
                entry_offset: 0,
                function_name: "kajit_main",
                dwarf: None,
                intrinsic_calls: &[],
                extern_addr_relocs: &[],
            },
        ),
    };
    let base_name = format!("{stem}-{arch_suffix}");
    let input = PrintMainExecutableInput { object };
    kajit_wares::write_print_main_executable(&input, &output_dir, &base_name)
        .map_err(|err| format!("failed to build executable for {}: {err}", path.display()))
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
