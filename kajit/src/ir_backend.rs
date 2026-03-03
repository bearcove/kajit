use dynasmrt::AssemblyOffset;

use crate::linearize::LinearIr;
use crate::regalloc_engine::AllocatedProgram;
use crate::regalloc_mir::RaProgram;

pub struct LinearBackendResult {
    pub buf: dynasmrt::ExecutableBuffer,
    pub entry: AssemblyOffset,
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
        crate::ir_backend_x64::compile(ra_mir, alloc)
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = ra_mir; // aarch64 backend still uses flat LinearIr
        crate::ir_backend_aarch64::compile(ir, max_spillslots, alloc)
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
        crate::ir_backend_x64::compile(ra_mir, alloc)
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = (ra_mir, alloc);
        panic!("compile_ra_program is not yet supported on aarch64 (backend needs refactoring to consume RaProgram directly)")
    }
}
