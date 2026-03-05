use crate::linearize::LinearIr;
use crate::regalloc_engine::AllocatedProgram;
use crate::regalloc_mir::RaProgram;

#[cfg(target_arch = "x86_64")]
pub struct LinearBackendResult {
    pub buf: kajit_emit::x64::FinalizedEmission,
    pub entry: u32,
    pub source_map: Option<kajit_emit::SourceMap>,
}

#[cfg(target_arch = "aarch64")]
pub struct LinearBackendResult {
    pub buf: kajit_emit::aarch64::FinalizedEmission,
    pub entry: u32,
    pub source_map: Option<kajit_emit::SourceMap>,
}

pub fn compile_linear_ir(ir: &LinearIr) -> LinearBackendResult {
    let ra_mir = crate::regalloc_mir::lower_linear_ir(ir);
    let alloc = crate::regalloc_engine::allocate_program(&ra_mir)
        .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));
    compile_linear_ir_with_alloc(ir, &ra_mir, &alloc)
}

pub fn compile_linear_ir_with_alloc(
    ir: &LinearIr,
    ra_mir: &RaProgram,
    alloc: &AllocatedProgram,
) -> LinearBackendResult {
    let max_spillslots = alloc
        .functions
        .iter()
        .map(|f| f.num_spillslots)
        .max()
        .unwrap_or(0);

    #[cfg(target_arch = "x86_64")]
    {
        let _ = (ir, max_spillslots); // x64 backend reads from ra_mir directly
        crate::backends::x86_64::compile(ra_mir, alloc)
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = (ir, max_spillslots); // aarch64 backend reads from ra_mir directly
        crate::backends::aarch64::compile(ra_mir, alloc)
    }
}

/// Compile directly from an RaProgram (no LinearIr needed).
///
/// This is the entry point for the RA-MIR text test workflow: parse RA-MIR
/// text into an RaProgram, then run regalloc2 + codegen without needing a
/// LinearIr.
pub fn compile_ra_program(ra_mir: &RaProgram, alloc: &AllocatedProgram) -> LinearBackendResult {
    #[cfg(target_arch = "x86_64")]
    {
        crate::backends::x86_64::compile(ra_mir, alloc)
    }

    #[cfg(target_arch = "aarch64")]
    {
        crate::backends::aarch64::compile(ra_mir, alloc)
    }
}
