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
