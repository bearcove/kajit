//! Support for loading and benchmarking hand-optimized assembly variants.

use std::path::PathBuf;

/// A decoder compiled from hand-optimized assembly.
#[cfg(target_arch = "aarch64")]
pub struct AltDecoder {
    buf: kajit_asm::aarch64::FinalizedEmission,
    func: *const u8,
}

// Safety: the assembled code buffer is immutable and pinned for the lifetime of AltDecoder.
#[cfg(target_arch = "aarch64")]
unsafe impl Send for AltDecoder {}
#[cfg(target_arch = "aarch64")]
unsafe impl Sync for AltDecoder {}

#[cfg(target_arch = "aarch64")]
impl AltDecoder {
    /// Raw entry point pointer. Caller casts to the appropriate signature.
    pub fn func_ptr(&self) -> *const u8 {
        self.func
    }

    /// The raw executable code buffer.
    pub fn code(&self) -> &[u8] {
        &self.buf.code
    }

    /// Entry point offset into the code.
    pub fn entry_offset(&self) -> usize {
        0 // Alt decoders start at offset 0
    }
}

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

/// Load, assemble, and wrap a `.alt.vixen-asm` file as an AltDecoder.
///
/// Returns None if the file doesn't exist, fails to parse, or fails to assemble.
#[cfg(target_arch = "aarch64")]
pub fn load_alt_decoder(
    group: &str,
    format: &str,
    intrinsic_registry: &crate::ir::IntrinsicRegistry,
) -> Option<AltDecoder> {
    let path = alt_asm_path(group, format);
    let text = std::fs::read_to_string(&path).ok()?;
    let program = kajit_emit_text::parse_asm(&text).ok()?;

    // Expand pseudo-instructions (like load_extern)
    let expanded = kajit_emit_text::expand_pseudo_instructions(program, |symbol| {
        intrinsic_registry.address_of_name(symbol)
    });

    // Assemble to executable code
    let buf = kajit_emit_text::assemble(&expanded).ok()?;

    let func = buf.code_ptr();

    Some(AltDecoder { buf, func })
}
