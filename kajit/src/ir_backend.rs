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

/// Recorded intrinsic call site for harness relocation.
#[derive(Debug, Clone)]
pub struct IntrinsicCallSiteInfo {
    pub code_offset: usize,
    pub func: kajit_ir::IntrinsicFn,
}

/// Recorded data blob address site for relocation.
#[derive(Debug, Clone)]
pub struct DataRelocInfo {
    pub code_offset: usize,
    pub blob_id: u32,
}

/// Recorded external symbol address site for relocation (vtable function pointers etc.).
#[derive(Debug, Clone)]
pub struct ExternAddrRelocInfo {
    /// Offset in the code buffer of the first instruction in the fixed-length load sequence.
    pub code_offset: usize,
    /// Symbol name for linker relocation in standalone harness mode.
    pub symbol: kajit_types::SymbolName,
}

pub struct LinearBackendResult {
    pub buf: BackendBuf,
    pub entry: u32,
    pub source_map: Option<kajit_emit::SourceMap>,
    pub backend_debug_info: Option<BackendDebugInfo>,
    pub asm_program: Option<kajit_emit::aarch64_asm::Program>,
    pub intrinsic_call_sites: Vec<IntrinsicCallSiteInfo>,
    pub data_relocs: Vec<DataRelocInfo>,
    pub extern_addr_relocs: Vec<ExternAddrRelocInfo>,
}

/// Architecture-specific finalized code buffer.
pub enum BackendBuf {
    X86_64(kajit_emit::x64::FinalizedEmission),
    Aarch64(kajit_emit::aarch64::FinalizedEmission),
}

impl BackendBuf {
    pub fn code(&self) -> &[u8] {
        match self {
            BackendBuf::X86_64(buf) => buf.exec.as_ref(),
            BackendBuf::Aarch64(buf) => &buf.code,
        }
    }

    pub fn code_ptr(&self) -> *const u8 {
        match self {
            BackendBuf::X86_64(buf) => buf.exec.as_ptr(),
            BackendBuf::Aarch64(buf) => buf.code_ptr(),
        }
    }

    pub fn len(&self) -> usize {
        self.code().len()
    }

    pub fn source_map(&self) -> &kajit_emit::SourceMap {
        match self {
            BackendBuf::X86_64(buf) => &buf.source_map,
            BackendBuf::Aarch64(buf) => &buf.source_map,
        }
    }
}
