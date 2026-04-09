use std::fs;
use std::path::Path;

use kajit_reprs::schema_poc;

use crate::validate::path_matches_ext;

pub(crate) fn cmd_format(path: &Path) -> Result<(), String> {
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
    let formatted =
        (handler.format)(&source).map_err(|err| format!("{}:\n{err}", path.display()))?;
    print!("{formatted}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_pilot_hir_text() {
        let path = Path::new("/tmp/example.k-hir");
        let source = "module { fn main() -> Value { return 42 } }";
        let result = schema_poc::REPRS
            .iter()
            .find(|repr| path_matches_ext(path, repr.file_ext))
            .map(|repr| (repr.format)(source))
            .expect("expected .k-hir formatter handler");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            "module {\nfn main() -> Value {\nreturn 42\n}\n}"
        );
    }
}
