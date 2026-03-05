use crate::linearize::LinearIr;
use crate::regalloc_engine::{AllocatedCfgProgram, cfg_mir};

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
    let cfg_program = cfg_mir::lower_linear_ir(ir);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg_program)
        .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));
    compile_linear_ir_with_alloc(ir, &cfg_program, &alloc)
}

pub fn compile_linear_ir_with_alloc(
    ir: &LinearIr,
    cfg_program: &cfg_mir::Program,
    alloc: &AllocatedCfgProgram,
) -> LinearBackendResult {
    compile_linear_ir_with_alloc_and_mode(ir, cfg_program, alloc, true)
}

pub fn compile_linear_ir_with_alloc_and_mode(
    ir: &LinearIr,
    cfg_program: &cfg_mir::Program,
    alloc: &AllocatedCfgProgram,
    apply_regalloc_edits: bool,
) -> LinearBackendResult {
    let max_spillslots = alloc
        .functions
        .iter()
        .map(|f| f.num_spillslots)
        .max()
        .unwrap_or(0);

    #[cfg(target_arch = "x86_64")]
    {
        let _ = (ir, max_spillslots, apply_regalloc_edits);
        crate::backends::x86_64::compile(cfg_program, alloc)
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = (ir, max_spillslots); // aarch64 backend reads from ra_mir directly
        crate::backends::aarch64::compile(cfg_program, alloc, apply_regalloc_edits)
    }
}

/// Compile directly from an RaProgram (no LinearIr needed).
///
/// This is the entry point for the RA-MIR text test workflow: parse RA-MIR
/// text into an RaProgram, then run regalloc2 + codegen without needing a
/// LinearIr.
pub fn compile_ra_program(
    _ra_mir: &crate::regalloc_mir::RaProgram,
    _alloc: &crate::regalloc_engine::AllocatedProgram,
) -> LinearBackendResult {
    compile_ra_program_with_mode(_ra_mir, _alloc, true)
}

pub fn compile_ra_program_with_mode(
    _ra_mir: &crate::regalloc_mir::RaProgram,
    _alloc: &crate::regalloc_engine::AllocatedProgram,
    apply_regalloc_edits: bool,
) -> LinearBackendResult {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = apply_regalloc_edits;
        panic!("x86_64 strict backend path not wired yet")
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = apply_regalloc_edits;
        panic!("compile_ra_program is disabled on strict CFG backend path")
    }
}
