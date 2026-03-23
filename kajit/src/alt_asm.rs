//! Support for loading and benchmarking hand-optimized assembly variants.

use std::path::PathBuf;

/// Check if a `.alt.vixen-asm` file exists for the given group/format/case.
pub fn has_alt_asm(group: &str, format: &str) -> bool {
    alt_asm_path(group, format).exists()
}

/// Get the path to the `.alt.vixen-asm` file for the given group/format.
fn alt_asm_path(group: &str, format: &str) -> PathBuf {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().expect("workspace root");
    let dump_dir = workspace_root.join("target/kajit-stage-dumps");

    let arch = std::env::consts::ARCH;
    let filename = format!("{format}__{group}__{}__asm.alt.vixen-asm", arch);
    dump_dir.join(filename)
}

/// Load and parse a `.alt.vixen-asm` file.
///
/// Returns None if the file doesn't exist or fails to parse.
#[cfg(target_arch = "aarch64")]
pub fn load_alt_asm(group: &str, format: &str) -> Option<kajit_emit::aarch64_asm::Program> {
    let path = alt_asm_path(group, format);
    let text = std::fs::read_to_string(&path).ok()?;
    kajit_emit_text::parse_asm(&text).ok()
}
