use std::{
    fs,
    path::{Path, PathBuf},
};

use camino::Utf8Path;

mod defs;

pub fn generate_repr_poc(lang_def_path: &Utf8Path) -> Result<Vec<PathBuf>, String> {
    let def = defs::read_from_file(&lang_def_path)?;
    todo!("and now to generate something")
}
