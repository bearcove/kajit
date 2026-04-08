use std::fs;
use std::path::{Path, PathBuf};

mod normalize;
mod parser_codegen;
mod render_helpers;
mod render_module;
mod schema;

pub fn generate_repr_poc(workspace_root: &Path) -> Result<Vec<PathBuf>, String> {
    let schema_path = workspace_root.join("notes/unified-ast/pilot/hir.repr.styx");
    let repr = schema::load_hir_pilot_schema(&schema_path)?;

    let out_dir = workspace_root.join("crates-kajit/kajit-reprs/src/schema_poc");
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

    let mod_path = out_dir.join("mod.rs");
    fs::write(&mod_path, "pub mod hir;\n")
        .map_err(|e| format!("failed to write {}: {e}", mod_path.display()))?;

    let hir_path = out_dir.join("hir.rs");
    fs::write(&hir_path, render_module::render_hir_poc_module(&repr))
        .map_err(|e| format!("failed to write {}: {e}", hir_path.display()))?;

    Ok(vec![mod_path, hir_path])
}
