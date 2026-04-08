use std::collections::HashMap;
use std::fmt;

/// A single function argument value with its type.
#[derive(Clone, Copy, Debug)]
pub enum ArgValue {
    /// 64-bit integer/pointer — passed in GPR (x0-x7 on aarch64, rdi/rsi/... on x86_64).
    U64(u64),
    /// 64-bit float — passed in FPR (d0-d7 on aarch64, xmm0-xmm7 on x86_64).
    F64(f64),
}

impl ArgValue {
    pub fn as_u64(&self) -> u64 {
        match self {
            ArgValue::U64(v) => *v,
            ArgValue::F64(_) => panic!("expected U64, got F64"),
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            ArgValue::F64(v) => *v,
            ArgValue::U64(_) => panic!("expected F64, got U64"),
        }
    }
}

/// Typed function arguments, laid out according to calling convention.
#[derive(Clone, Debug, Default)]
pub struct Arguments {
    items: Vec<ArgValue>,
}

impl Arguments {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn push(&mut self, arg: ArgValue) {
        self.items.push(arg);
    }

    pub fn push_ptr<T>(&mut self, ptr: *mut T) {
        self.items.push(ArgValue::U64(ptr as u64));
    }

    pub fn items(&self) -> &[ArgValue] {
        &self.items
    }

    /// Assign arguments to GPR and FPR slots according to SysV ABI.
    /// Returns (gpr_args, fpr_args) — each ordered by register index.
    pub fn to_register_slots(&self) -> (Vec<u64>, Vec<f64>) {
        let mut gprs = Vec::new();
        let mut fprs = Vec::new();
        for arg in &self.items {
            match arg {
                ArgValue::U64(v) => gprs.push(*v),
                ArgValue::F64(v) => fprs.push(*v),
            }
        }
        (gprs, fprs)
    }

    /// Lay out arguments according to a calling convention.
    /// Returns one `ArgLocation` per argument in the same order as `items()`.
    pub fn layout(&self, cc: &CallingConvention) -> Vec<ArgLocation> {
        let kinds: Vec<ArgKind> = self.items.iter().map(ArgKind::of).collect();
        layout_args(&kinds, cc)
    }
}

/// The type of an argument, without a value. Used at compile time
/// (e.g. on CFG-MIR function params) when we need to know the register
/// class but don't have a runtime value yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgKind {
    /// Integer or pointer — GPR.
    U64,
    /// Float — FPR.
    F64,
}

impl ArgKind {
    /// Get the kind of an ArgValue.
    pub fn of(v: &ArgValue) -> Self {
        match v {
            ArgValue::U64(_) => ArgKind::U64,
            ArgValue::F64(_) => ArgKind::F64,
        }
    }
}

/// Lay out a sequence of argument kinds according to a calling convention.
/// Same logic as `Arguments::layout` but without needing runtime values.
pub fn layout_args(kinds: &[ArgKind], cc: &CallingConvention) -> Vec<ArgLocation> {
    let mut gpr_idx: u8 = 0;
    let mut fpr_idx: u8 = 0;
    let mut stack_offset: u32 = 0;
    let mut result = Vec::with_capacity(kinds.len());
    for kind in kinds {
        match kind {
            ArgKind::U64 => {
                if (gpr_idx as usize) < cc.gpr_args.len() {
                    result.push(ArgLocation::Gpr(cc.gpr_args[gpr_idx as usize]));
                    gpr_idx += 1;
                } else {
                    result.push(ArgLocation::Stack {
                        offset: stack_offset,
                    });
                    stack_offset += 8;
                }
            }
            ArgKind::F64 => {
                if (fpr_idx as usize) < cc.fpr_args.len() {
                    result.push(ArgLocation::Fpr(cc.fpr_args[fpr_idx as usize]));
                    fpr_idx += 1;
                } else {
                    result.push(ArgLocation::Stack {
                        offset: stack_offset,
                    });
                    stack_offset += 8;
                }
            }
        }
    }
    result
}

/// Where an argument ends up after calling convention layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgLocation {
    /// General-purpose register (index into the platform's GPR arg list).
    Gpr(u8),
    /// Floating-point register (index into the platform's FPR arg list).
    Fpr(u8),
    /// Stack slot at the given byte offset from the stack args base.
    Stack { offset: u32 },
}

/// Calling convention descriptor. Lists the registers available for
/// passing arguments, in order. Once registers are exhausted, arguments
/// spill to the stack.
#[derive(Clone, Debug)]
pub struct CallingConvention {
    /// GPR argument registers, in assignment order.
    /// On aarch64 SysV: [0, 1, 2, 3, 4, 5, 6, 7] (x0-x7)
    /// On x86_64 SysV: [7, 6, 2, 1, 8, 9] (rdi, rsi, rdx, rcx, r8, r9)
    pub gpr_args: &'static [u8],
    /// FPR argument registers, in assignment order.
    /// On aarch64 SysV: [0, 1, 2, 3, 4, 5, 6, 7] (d0-d7)
    /// On x86_64 SysV: [0, 1, 2, 3, 4, 5, 6, 7] (xmm0-xmm7)
    pub fpr_args: &'static [u8],
}

/// SysV ABI calling convention for aarch64.
pub const SYSV_AARCH64: CallingConvention = CallingConvention {
    gpr_args: &[0, 1, 2, 3, 4, 5, 6, 7],
    fpr_args: &[0, 1, 2, 3, 4, 5, 6, 7],
};

/// SysV ABI calling convention for x86_64.
/// GPR order: rdi(7), rsi(6), rdx(2), rcx(1), r8(8), r9(9)
/// using hardware register encoding.
pub const SYSV_X86_64: CallingConvention = CallingConvention {
    gpr_args: &[7, 6, 2, 1, 8, 9],
    fpr_args: &[0, 1, 2, 3, 4, 5, 6, 7],
};

/// Returns the calling convention for the current compilation target.
pub fn target_calling_convention() -> &'static CallingConvention {
    #[cfg(target_arch = "aarch64")]
    {
        &SYSV_AARCH64
    }
    #[cfg(target_arch = "x86_64")]
    {
        &SYSV_X86_64
    }
}

/// A named external symbol (e.g. a vtable function pointer).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymbolName(String);

impl SymbolName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SymbolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A runtime function/data address resolved from a symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuntimeAddr(usize);

impl RuntimeAddr {
    pub fn from_ptr<T>(ptr: *const T) -> Self {
        Self(ptr as usize)
    }

    pub fn as_usize(self) -> usize {
        self.0
    }

    pub fn as_u64(self) -> u64 {
        self.0 as u64
    }
}

/// Maps external symbol names to their runtime addresses.
///
/// Populated by format frontends (e.g. postcard) with vtable function pointers.
/// Consumed by backends (to emit load-immediate) and interpreters (to resolve
/// ExternAddr ops).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolTable {
    symbols: HashMap<SymbolName, RuntimeAddr>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: SymbolName, addr: RuntimeAddr) {
        self.symbols.insert(name, addr);
    }

    pub fn resolve(&self, name: &SymbolName) -> RuntimeAddr {
        *self
            .symbols
            .get(name)
            .unwrap_or_else(|| panic!("unresolved extern symbol: {name}"))
    }
}

/// Describes a pointer-valued field within a data_arg struct.
/// Used by the debugger to seed shadow memory so loads through
/// data_arg pointers recover provenance correctly.
#[derive(Clone, Debug)]
pub struct PointerField {
    /// Byte offset of this pointer within the struct.
    pub offset: u64,
    /// Human-readable label (e.g. "bytes.ptr", "input_ptr").
    pub label: String,
}

/// Layout metadata for a single data_arg, describing which fields
/// within the pointed-to struct contain pointers.
///
/// This is format-specific knowledge (e.g. postcard knows its cursor
/// struct has a `bytes.ptr` pointer at offset 0) that the generic
/// interpreter needs for shadow memory tracking.
#[derive(Clone, Debug, Default)]
pub struct DataArgLayout {
    /// Human-readable name for this arg (e.g. "cursor", "out", "ctx").
    pub name: String,
    /// Pointer-valued fields within this struct.
    pub pointer_fields: Vec<PointerField>,
}
