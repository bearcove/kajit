use crate::linearize::LinearIr;
use crate::regalloc_engine::{AllocatedCfgProgram, cfg_mir};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCodeRange {
    pub start_offset: u32,
    pub end_offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendOpDebugInfo {
    pub lambda_id: u32,
    pub op_id: cfg_mir::OpId,
    pub line: u32,
    pub code_ranges: Vec<BackendCodeRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BackendDebugInfo {
    pub op_infos: Vec<BackendOpDebugInfo>,
}

#[cfg(target_arch = "x86_64")]
pub struct LinearBackendResult {
    pub buf: kajit_emit::x64::FinalizedEmission,
    pub entry: u32,
    pub source_map: Option<kajit_emit::SourceMap>,
    pub backend_debug_info: Option<BackendDebugInfo>,
}

#[cfg(target_arch = "aarch64")]
pub struct LinearBackendResult {
    pub buf: kajit_emit::aarch64::FinalizedEmission,
    pub entry: u32,
    pub source_map: Option<kajit_emit::SourceMap>,
    pub backend_debug_info: Option<BackendDebugInfo>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IrBuilder, Width};
    use facet::Facet;

    #[test]
    fn compile_linear_ir_records_backend_op_ranges() {
        let mut builder = IrBuilder::new(<u32 as Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let value = rb.const_val(42);
            rb.write_to_field(value, 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        crate::ir_passes::run_default_passes(&mut func);
        let linear = crate::linearize::linearize(&mut func);
        let cfg_program = cfg_mir::lower_linear_ir(&linear);
        let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg_program)
            .unwrap_or_else(|err| panic!("regalloc2 allocation failed in test: {err}"));

        let result = compile_linear_ir_with_alloc_and_mode(&linear, &cfg_program, &alloc, true);
        let backend_debug_info = result
            .backend_debug_info
            .expect("backend debug info should be present");
        assert!(!backend_debug_info.op_infos.is_empty());
        assert!(backend_debug_info.op_infos.iter().all(|op| {
            op.code_ranges
                .iter()
                .all(|range| range.start_offset < range.end_offset)
        }));
    }
}
