use kajit_ir::IntrinsicRegistry;
use std::path::Path;

fn cfg_mir_transform_test(path: &Path) -> datatest_stable::Result<()> {
    let input = std::fs::read_to_string(path)?;
    let registry = IntrinsicRegistry::empty();
    let mut program = kajit_mir_text::parse_cfg_mir_with_registry(&input, &registry)
        .map_err(|e| format!("parse failed for {}: {e}", path.display()))?;

    // The pass name is the parent directory name.
    let pass_name = path
        .parent()
        .unwrap()
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();

    kajit_mir::opt::reduce::run_named_pass(&mut program, pass_name);

    // Validate SSA after the pass.
    for func in &program.funcs {
        if let Err(errors) = kajit_mir::opt::validate_ssa::validate_ssa(func) {
            return Err(format!(
                "SSA validation failed after {pass_name} for {}:\n{}",
                path.display(),
                errors
                    .iter()
                    .map(|e| format!("  {e:?}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
            .into());
        }
    }

    let actual = format!("{}", program.display_with_registry(&registry));

    // Compare against .expected file next to the input.
    let expected_path = path.with_extension("expected");
    if expected_path.exists() {
        let expected = std::fs::read_to_string(&expected_path)?;
        if actual.trim() != expected.trim() {
            return Err(format!(
                "output mismatch for {}\n\n--- expected ---\n{}\n--- actual ---\n{}\n",
                path.display(),
                expected.trim(),
                actual.trim()
            )
            .into());
        }
    } else {
        // No .expected file yet — write it and fail so the user reviews it.
        std::fs::write(&expected_path, &actual)?;
        return Err(format!(
            "no .expected file for {} — created one. Review and re-run.",
            path.display(),
        )
        .into());
    }

    Ok(())
}

datatest_stable::harness! {
    { test = cfg_mir_transform_test, root = "tests/passes", pattern = r"\.cfg-mir$" },
}
