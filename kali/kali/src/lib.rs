/// Uniquely identifies a file that was parsed
pub struct FileId(u32);

/// Identifies the start and length of a parsed bit of syntax
pub struct Span {
    /// Start in bytes
    pub start: u32,

    /// Length in bytes
    pub len: u32,
}
