use std::path::{Path, PathBuf};

fn main() {
    let mut args = std::env::args();
    let _bin = args.next();
    let command = args.next();

    match command.as_deref() {
        Some("repr-poc") => {
            let workspace_root = workspace_root();
            let files = kajit_foundation::generate_repr_poc(&workspace_root).unwrap_or_else(|err| {
                eprintln!("{err}");
                std::process::exit(1);
            });
            for file in files {
                println!("{}", file.display());
            }
        }
        _ => {
            eprintln!(
                "usage: cargo run -p kajit-foundation-cli -- <repr-poc>"
            );
            std::process::exit(2);
        }
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("kajit-foundation-cli should live in crates-kajit/")
        .to_path_buf()
}
