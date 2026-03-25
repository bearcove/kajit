use cc::Build;

use std::{fs, process::Command};

fn get_llvm_output(arg: &str) -> String {
    let llvm_config = std::env::var("LLVM_CONFIG").unwrap_or_else(|_| "llvm-config".into());
    let res = Command::new(llvm_config).arg(arg).output().unwrap();
    if !res.status.success() {
        panic!(
            "Could not run \"llvm-config {}\": {}",
            arg,
            res.status.code().unwrap()
        );
    }
    String::from_utf8(res.stdout).unwrap().trim().to_string()
}

fn match_libname(name: &str) -> Option<String> {
    if name.starts_with("liblldb.so") || name.starts_with("liblldb-") {
        if let Some(pos) = name.rfind(".so") {
            return Some(name["lib".len()..pos].into());
        }
    }
    if name.starts_with("liblldb") && name.ends_with(".dylib") {
        // Trim the leading "lib" and trailing ".dylib"
        return Some(name[3..name.len() - 6].into());
    }
    if name.starts_with("liblldb") && name.ends_with(".lib") {
        // windows will have liblldb.lib
        // Trim the trailing ".lib"
        return Some(name[0..name.len() - 4].into());
    }
    None
}

#[cfg(test)]
#[test]
fn test_match_libname() {
    assert_eq!(match_libname("liblldb.so"), Some("lldb"));
    assert_eq!(match_libname("liblldb-3.8.so"), Some("lldb-3.8"));
    assert_eq!(match_libname("liblldbIntelFeatures.so"), None);
    assert_eq!(match_libname("liblldb.lib"), Some("liblldb"));
}

fn get_compiler_config() -> Build {
    // We use the `llvm-config` utility to get the include and library paths
    // as well as the name of the shared library.
    println!("cargo:rerun-if-env-changed=LLVM_CONFIG");
    println!("cargo:rerun-if-env-changed=LLDB_LIB_PATH");
    println!("cargo:rerun-if-env-changed=LLDB_ADDITIONAL_INCLUDE_DIRS");
    let llvm_headers_path = get_llvm_output("--includedir");
    let llvm_lib_path = get_llvm_output("--libdir");

    // Some systems (ex. NixOS) do not have liblldb.so in llvm-config's libdir.
    // As a workaround, allow users to specifiy path to search
    // for the aforementioned library.
    let lldb_lib_path = match std::env::var_os("LLDB_LIB_PATH") {
        Some(path) => &path
            .into_string()
            .expect("LLDB_LIB_PATH contains invalid Unicode data"),
        None => &llvm_lib_path,
    };

    let lib_name = fs::read_dir(lldb_lib_path)
        .expect("failed to stat libdir from llvm-config")
        .filter_map(|entry| match_libname(entry.unwrap().file_name().to_str().unwrap()))
        .next()
        .expect("unable to locate shared library of liblldb");
    println!("cargo:rustc-link-search={llvm_lib_path}");
    println!("cargo:rustc-link-lib={lib_name}");
    let mut res = cc::Build::new();
    res.include(llvm_headers_path);
    // if llvm is in the development tree, (in other words, just after build)
    // we may need to add several other directories to include lldb
    // those directories are not constant (might depend on build system) so
    // we allow user to specify with PATH
    if let Some(dirs) = std::env::var_os("LLDB_ADDITIONAL_INCLUDE_DIRS") {
        for path in std::env::split_paths(&dirs) {
            res.include(path);
        }
    }
    res
}

fn main() {
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }
    get_compiler_config()
        .cpp(true)
        .std("c++14")
        .warnings(false)
        .include("src")
        .file("src/lldb/UnityBuild.cpp")
        .compile("liblldb-c.a");
}
