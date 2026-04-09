use std::fs;
use std::path::{Path, PathBuf};

mod formatter_codegen;
mod hover_codegen;
mod normalize;
mod parser_codegen;
mod render_helpers;
mod render_module;
mod schema;
mod semantic_codegen;

#[cfg(test)]
mod tests;

pub fn generate_repr_poc(workspace_root: &Path) -> Result<Vec<PathBuf>, String> {
    let schema_paths = [
        workspace_root.join("notes/unified-ast/pilot/hir.repr.styx"),
        workspace_root.join("notes/unified-ast/pilot/asm.repr.styx"),
    ];
    let mut reprs = Vec::new();
    for schema_path in schema_paths {
        if !schema_path.exists() {
            continue;
        }
        let loaded = schema::load_pilot_schema(&schema_path)?;
        let repr = normalize::normalize_repr(&loaded.body)?;
        reprs.push(normalize::with_module_doc(repr, loaded.doc));
    }

    let out_dir = workspace_root.join("crates-kajit/kajit-reprs/src/schema_poc");
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

    let old_hir_path = out_dir.join("hir.rs");
    if old_hir_path.exists() {
        fs::remove_file(&old_hir_path)
            .map_err(|e| format!("failed to remove {}: {e}", old_hir_path.display()))?;
    }

    let mut written = Vec::new();
    for generated in render_module::render_repr_poc_files(&reprs) {
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
