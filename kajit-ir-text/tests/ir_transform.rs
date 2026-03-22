use kajit_ir::{IntrinsicRegistry, verify};
use std::path::Path;

/// Registry of IR passes by directory name.
fn run_pass(name: &str, func: &mut kajit_ir::IrFunc) {
    match name {
        "slot2reg" => kajit_ir::slot2reg::slot_to_reg(func),
        other => panic!("unknown pass directory: {other}"),
    }
}

fn ir_transform_test(path: &Path) -> datatest_stable::Result<()> {
    let input = std::fs::read_to_string(path)?;
    let registry = IntrinsicRegistry::empty();
    let mut func = kajit_ir_text::parse_ir(&input, &registry)
        .map_err(|e| format!("parse failed for {}: {e}", path.display()))?;

    // The pass name is the parent directory name.
    let pass_name = path
        .parent()
        .unwrap()
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();

    run_pass(pass_name, &mut func);

    if let Err(e) = verify(&func) {
        return Err(format!(
            "verification failed after {pass_name} for {}: {e}",
            path.display()
        )
        .into());
    }

    let actual = format!("{}", func.display_with_registry(&registry));

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
    { test = ir_transform_test, root = "tests/slot2reg", pattern = r"\.vixen-ir$" },
}
