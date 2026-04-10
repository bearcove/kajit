use std::fs;
use std::path::Path;

use kajit_asm::repr::FinalizedReprEmission;
use kajit_mir::lower;
use kajit_reprs::{self as reprs, ResolutionSet};
use kajit_wares::{ObjectInput, PrintMainExecutableInput, TargetArch};

use crate::validate::path_matches_ext;

pub(crate) fn cmd_compile(path: &Path) -> Result<(), String> {
    let source =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    let output = if path_matches_ext(path, reprs::asm::REPR_FILE_EXT) {
        let resolutions =
            reprs::asm::resolve(&source).map_err(|err| format!("{}:\n{err}", path.display()))?;
        ensure_all_references_resolved(path, &source, &resolutions)?;
        let program = reprs::asm::parse_root_text(&source)
            .map_err(|err| format!("{}:\n{err}", path.display()))?;
        kajit_asm::repr::assemble_repr_program(&program, &source)?
    } else if path_matches_ext(path, reprs::mir::REPR_FILE_EXT) {
        let mut graph = reprs::mir::Graph::new();
        let handle = reprs::mir::parse_root_into_graph(&mut graph, &source)
            .map_err(|err| format!("{}:\n{err}", path.display()))?;
        let program = graph
            .root(handle)
            .ok_or_else(|| format!("{}:\nfailed to load MIR root", path.display()))?;

        let lowered = lower::lower_program_to_asm(&graph, program)
            .map_err(|err| format!("{}:\n{err}", path.display()))?;
        let resolutions = reprs::asm::resolve(&lowered.source)
            .map_err(|err| format!("{}:\n{err}", path.display()))?;
        ensure_all_references_resolved(path, &lowered.source, &resolutions)?;
        kajit_asm::repr::assemble_repr_program(&lowered.program, &lowered.source)?
    } else {
        return match path.extension().and_then(|ext| ext.to_str()) {
            Some(other) => Err(format!(
                "compile does not support .{other} yet for {}",
                path.display()
            )),
            None => Err(format!("cannot determine file type for {}", path.display())),
        };
    };

    let exe_path = write_repr_asm_executable(path, &output)?;
    println!("{}", exe_path.display());
    Ok(())
}

fn ensure_all_references_resolved(
    path: &Path,
    source: &str,
    resolutions: &ResolutionSet,
) -> Result<(), String> {
    let unresolved = resolutions
        .references
        .iter()
        .filter(|reference| reference.target.is_none())
        .collect::<Vec<_>>();
    if unresolved.is_empty() {
        return Ok(());
    }

    let messages = unresolved
        .into_iter()
        .map(|reference| {
            let start = offset_to_line_col(source, reference.reference.start as usize);
            format!(
                "{}:{}:{}: unresolved {:?} reference `{}`",
                path.display(),
                start.0,
                start.1,
                reference.reference.kind,
                reference.reference.name
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    Err(messages)
}

fn offset_to_line_col(content: &str, offset: usize) -> (u32, u32) {
    let mut line = 1_u32;
    let mut col = 1_u32;
    for ch in content[..offset.min(content.len())].chars() {
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

fn write_repr_asm_executable(
    path: &Path,
    output: &FinalizedReprEmission,
) -> Result<std::path::PathBuf, String> {
    let output_dir = std::path::PathBuf::from("target").join("kajit-compile");
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| format!("cannot determine output name for {}", path.display()))?;

    let (arch_suffix, object) = match output {
        FinalizedReprEmission::AArch64(_) => (
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
        FinalizedReprEmission::X64(_) => (
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
    fn compiles_aarch64_repr_asm() {
        let path = Path::new("/tmp/example.k-asm");
        let source = "asm aarch64 { start: movz x0, 42 ret }";
        let program = reprs::asm::parse_root_text(source).expect("expected valid .k-asm");
        let output = kajit_asm::repr::assemble_repr_program(&program, source).expect("emit");
        assert!(matches!(output, FinalizedReprEmission::AArch64(_)));
        assert!(output.len() > 0);
        assert!(output.trace_text().is_ok());
        assert!(path_matches_ext(path, reprs::asm::REPR_FILE_EXT));
    }

    #[test]
    fn compiles_x64_repr_asm() {
        let path = Path::new("/tmp/example.k-asm");
        let source = "asm x86_64 { entry: mov rax, 42 ret }";
        let program = reprs::asm::parse_root_text(source).expect("expected valid .k-asm");
        let output = kajit_asm::repr::assemble_repr_program(&program, source).expect("emit");
        assert!(matches!(output, FinalizedReprEmission::X64(_)));
        assert!(output.len() > 0);
        assert!(output.trace_text().is_ok());
        assert!(path_matches_ext(path, reprs::asm::REPR_FILE_EXT));
    }

    #[test]
    fn compile_reuses_shared_resolver_for_unresolved_asm_labels() {
        let path = Path::new("/tmp/broken.k-asm");
        let source = "asm x86_64 { jmp missing }";
        let resolutions = reprs::asm::resolve(source).expect("resolver should run");
        let err = ensure_all_references_resolved(path, source, &resolutions)
            .expect_err("expected unresolved label error");
        assert!(err.contains("unresolved Label reference `missing`"));
        assert!(err.contains("/tmp/broken.k-asm:1:"));
    }

    #[test]
    fn compiles_pilot_mir_const_ret_to_host_asm() {
        let source = r#"
cfg_program vregs=1 slots=0 {
    cfg_func @0 f0 entry=b0 {
        data_args: []
        data_results: [0]
        block b0 params=[] insts=[0] term=t0 preds=[] succs=[]
        inst i0: v0:gpr/ret0 = const(42)
        term t0: return
    }
}
"#;

        let mut graph = reprs::mir::Graph::new();
        let handle = reprs::mir::parse_root_into_graph(&mut graph, source).expect("parse MIR");
        let program = graph.root(handle).expect("root");
        let lowered = lower::lower_program_to_asm(&graph, program).expect("lower");
        let output = kajit_asm::repr::assemble_repr_program(&lowered.program, &lowered.source)
            .expect("assemble");

        if cfg!(target_arch = "aarch64") {
            assert!(matches!(output, FinalizedReprEmission::AArch64(_)));
        } else if cfg!(target_arch = "x86_64") {
            assert!(matches!(output, FinalizedReprEmission::X64(_)));
        } else {
            // If the host arch isn't supported, the lowerer should've errored.
            unreachable!("unexpected host arch in tests");
        }
    }
}
