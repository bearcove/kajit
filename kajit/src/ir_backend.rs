use crate::regalloc_engine::cfg_mir;

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
    pub intrinsic_call_sites:
        Vec<crate::backends::x86_64::regalloc3_backend::IntrinsicCallSiteInfo>,
    pub data_relocs: Vec<crate::backends::x86_64::regalloc3_backend::DataRelocInfo>,
}

#[cfg(target_arch = "aarch64")]
pub struct LinearBackendResult {
    pub buf: kajit_emit::aarch64::FinalizedEmission,
    pub entry: u32,
    pub source_map: Option<kajit_emit::SourceMap>,
    pub backend_debug_info: Option<BackendDebugInfo>,
    pub asm_program: Option<kajit_emit::aarch64_asm::Program>,
    pub intrinsic_call_sites:
        Vec<crate::backends::aarch64::regalloc3_backend::IntrinsicCallSiteInfo>,
    pub data_relocs: Vec<crate::backends::aarch64::regalloc3_backend::DataRelocInfo>,
}
