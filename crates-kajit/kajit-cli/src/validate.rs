use std::fs;
use std::path::Path;

use kajit_reprs::schema_poc;

pub(crate) fn cmd_validate(path: &Path) -> Result<(), String> {
    let Some(handler) = schema_poc::REPRS
        .iter()
        .find(|repr| path_matches_ext(path, repr.file_ext))
    else {
        return match path.extension().and_then(|ext| ext.to_str()) {
            Some(other) => Err(format!(
                "unsupported file extension .{other} for {}",
                path.display()
            )),
            None => Err(format!("cannot determine file type for {}", path.display())),
        };
    };

    let source =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    (handler.validate)(&source)
        .map(|_| ())
        .map_err(|err| format!("{}:\n{err}", path.display()))
}

pub(crate) fn path_matches_ext(path: &Path, file_ext: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(file_ext))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_pilot_hir_text() {
        let path = Path::new("/tmp/example.k-hir");
        let source = "module { fn main() -> Value { return 42 } }";
        let result = schema_poc::REPRS
            .iter()
            .find(|repr| path_matches_ext(path, repr.file_ext))
            .map(|repr| (repr.validate)(source))
            .expect("expected .k-hir validation handler");
        assert!(result.is_ok());
    }

    #[test]
    fn rejects_unknown_extensions() {
        let err = cmd_validate(Path::new("/tmp/example.txt")).unwrap_err();
        assert!(err.contains("unsupported file extension"));
    }
}
