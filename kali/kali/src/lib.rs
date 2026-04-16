use std::collections::HashMap;

use camino::Utf8PathBuf;

/// A set of files that were parsed
pub struct ParseSet {
    pub files: HashMap<FileId, FileInfo>,
}

/// File information
pub struct FileInfo {
    /// Path on disk
    pub absolute_path: Utf8PathBuf,
}

/// Uniquely identifies a file that was parsed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileId(pub u32);

/// Identifies the start and length of a parsed bit of syntax
pub struct Span {
    /// Start in bytes
    pub start: u32,

    /// Length in bytes
    pub len: u32,
}

/// Provenance for a parsed bit of syntax, combining the file and span
pub struct Prov {
    /// The file this syntax came from
    pub file: FileId,

    /// The span within that file
    pub span: Span,
}
