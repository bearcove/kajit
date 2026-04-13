use std::fs;
use std::path::{Path, PathBuf};

mod schema;

#[cfg(test)]
mod tests;

pub fn generate_repr_poc(workspace_root: &Path) -> Result<Vec<PathBuf>, String> {
    let schema_paths = [
        workspace_root.join("notes/unified-ast/pilot/hir.repr.styx"),
        workspace_root.join("notes/unified-ast/pilot/asm.repr.styx"),
        workspace_root.join("notes/unified-ast/pilot/mir.repr.styx"),
    ];
    let mut schemas = Vec::new();
    for schema_path in schema_paths {
        if !schema_path.exists() {
            continue;
        }
        let schema = schema::read_from_file(&schema_path)?;
        schemas.push(schema);
    }

    let out_dir = workspace_root.join("crates-kajit/kajit-reprs/src");
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

    todo!("woops I deleted 80% of the crate because it was bad");
}
