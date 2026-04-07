use std::collections::HashMap;
use std::fmt;

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
