use std::fs;
use std::path::{Path, PathBuf};

mod formatter_codegen;
mod normalize;
mod parser_codegen;
mod render_helpers;
mod render_module;
mod schema;

pub fn generate_repr_poc(workspace_root: &Path) -> Result<Vec<PathBuf>, String> {
    let schema_path = workspace_root.join("notes/unified-ast/pilot/hir.repr.styx");
    let loaded = schema::load_hir_pilot_schema(&schema_path)?;
    let repr = normalize::normalize_repr(&loaded.body)?;
    let repr = normalize::with_module_doc(repr, loaded.doc);

    let out_dir = workspace_root.join("crates-kajit/kajit-reprs/src/schema_poc");
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

    let old_hir_path = out_dir.join("hir.rs");
    if old_hir_path.exists() {
        fs::remove_file(&old_hir_path)
            .map_err(|e| format!("failed to remove {}: {e}", old_hir_path.display()))?;
    }

    let mut written = Vec::new();
    for generated in render_module::render_hir_poc_files(&repr) {
        let path = out_dir.join(&generated.relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
        fs::write(&path, generated.contents)
            .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
        written.push(path);
    }

    Ok(written)
}
