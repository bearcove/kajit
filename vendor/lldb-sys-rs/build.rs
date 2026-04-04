use cc::Build;

use std::{collections::BTreeSet, fs, path::{Path, PathBuf}, process::Command};

fn llvm_config_candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    if let Ok(explicit) = std::env::var("LLVM_CONFIG") {
        candidates.push(explicit);
    }
    candidates.extend(
        [
            "llvm-config",
            "llvm-config-21",
            "llvm-config-20",
            "/usr/bin/llvm-config",
            "/usr/bin/llvm-config-21",
            "/usr/bin/llvm-config-20",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    candidates
}

fn run_llvm_config(llvm_config: &str, arg: &str) -> Result<String, String> {
    match Command::new(llvm_config).arg(arg).output() {
        Ok(res) if res.status.success() => Ok(String::from_utf8(res.stdout)
            .unwrap()
            .trim()
            .to_string()),
        Ok(res) => Err(format!(
            "Could not run \"{} {}\": {}",
            llvm_config,
            arg,
            res.status.code().unwrap_or(-1)
        )),
        Err(err) => Err(format!(
            "Could not spawn \"{} {}\": {}",
            llvm_config, arg, err
        )),
    }
}

fn resolve_llvm_config() -> Option<String> {
    for llvm_config in llvm_config_candidates() {
        if run_llvm_config(&llvm_config, "--includedir").is_ok()
            && run_llvm_config(&llvm_config, "--libdir").is_ok()
        {
            return Some(llvm_config);
        }
    }
    None
}

fn get_llvm_output(llvm_config: &str, arg: &str) -> Option<String> {
    run_llvm_config(llvm_config, arg).ok()
}

fn lldb_header_exists(dir: &Path) -> bool {
    dir.join("lldb/API/LLDB.h").exists()
}

fn lldb_lib_exists(dir: &Path) -> bool {
    fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .any(|entry| match_libname(entry.file_name().to_string_lossy().as_ref()).is_some())
}

fn push_dir(candidates: &mut Vec<PathBuf>, seen: &mut BTreeSet<PathBuf>, path: PathBuf) {
    if seen.insert(path.clone()) {
        candidates.push(path);
    }
}

fn common_llvm_prefixes() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();

    for version in [21, 20, 19, 18, 17, 16] {
        push_dir(
            &mut candidates,
            &mut seen,
            PathBuf::from(format!("/usr/lib/llvm-{version}")),
        );
    }
    if let Ok(entries) = fs::read_dir("/usr/lib") {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("llvm-"))
                .unwrap_or(false)
            {
                push_dir(&mut candidates, &mut seen, path);
            }
        }
    }
    for prefix in [
        "/usr/local/opt/llvm",
        "/opt/homebrew/opt/llvm",
        "/opt/local/libexec/llvm",
    ] {
        push_dir(&mut candidates, &mut seen, PathBuf::from(prefix));
    }
    candidates
}

fn candidate_include_dirs(llvm_config: Option<&str>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();

    if let Some(dirs) = std::env::var_os("LLDB_ADDITIONAL_INCLUDE_DIRS") {
        for path in std::env::split_paths(&dirs) {
            push_dir(&mut candidates, &mut seen, path);
        }
    }
    if let Some(llvm_config) = llvm_config
        && let Some(include_dir) = get_llvm_output(llvm_config, "--includedir")
    {
        push_dir(&mut candidates, &mut seen, PathBuf::from(include_dir));
    }
    for prefix in common_llvm_prefixes() {
        push_dir(&mut candidates, &mut seen, prefix.join("include"));
    }
    candidates
}

fn candidate_lib_dirs(llvm_config: Option<&str>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();

    if let Some(path) = std::env::var_os("LLDB_LIB_PATH") {
        push_dir(
            &mut candidates,
            &mut seen,
            PathBuf::from(path.into_string().expect("LLDB_LIB_PATH contains invalid Unicode data")),
        );
    }
    if let Some(llvm_config) = llvm_config
        && let Some(lib_dir) = get_llvm_output(llvm_config, "--libdir")
    {
        push_dir(&mut candidates, &mut seen, PathBuf::from(lib_dir));
    }
    for prefix in common_llvm_prefixes() {
        push_dir(&mut candidates, &mut seen, prefix.join("lib"));
    }
    for path in ["/usr/lib", "/usr/local/lib"] {
        push_dir(&mut candidates, &mut seen, PathBuf::from(path));
    }
    candidates
}

fn resolve_lldb_include_dir(llvm_config: Option<&str>) -> PathBuf {
    candidate_include_dirs(llvm_config)
        .into_iter()
        .find(|dir| lldb_header_exists(dir))
        .unwrap_or_else(|| {
            panic!(
                "unable to locate LLDB headers (looking for lldb/API/LLDB.h); tried LLVM_CONFIG={:?}, LLDB_ADDITIONAL_INCLUDE_DIRS={:?}",
                llvm_config,
                std::env::var_os("LLDB_ADDITIONAL_INCLUDE_DIRS")
            )
        })
}

fn resolve_lldb_lib_dir(llvm_config: Option<&str>) -> PathBuf {
    candidate_lib_dirs(llvm_config)
        .into_iter()
        .find(|dir| lldb_lib_exists(dir))
        .unwrap_or_else(|| {
            panic!(
                "unable to locate liblldb shared library; tried LLVM_CONFIG={:?}, LLDB_LIB_PATH={:?}",
                llvm_config,
                std::env::var_os("LLDB_LIB_PATH")
            )
        })
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
    let llvm_config = resolve_llvm_config();
    let lldb_include_dir = resolve_lldb_include_dir(llvm_config.as_deref());
    let lldb_lib_dir = resolve_lldb_lib_dir(llvm_config.as_deref());

    let lib_name = fs::read_dir(&lldb_lib_dir)
        .expect("failed to stat libdir from llvm-config")
        .filter_map(|entry| match_libname(entry.unwrap().file_name().to_str().unwrap()))
        .next()
        .expect("unable to locate shared library of liblldb");
    println!("cargo:rustc-link-search={}", lldb_lib_dir.display());
    println!("cargo:rustc-link-lib={lib_name}");
    let mut res = cc::Build::new();
    res.include(lldb_include_dir);
    if let Some(llvm_config) = llvm_config.as_deref()
        && let Some(llvm_headers_path) = get_llvm_output(llvm_config, "--includedir")
    {
        res.include(llvm_headers_path);
    }
    if let Some(dirs) = std::env::var_os("LLDB_ADDITIONAL_INCLUDE_DIRS") {
        for path in std::env::split_paths(&dirs) {
            res.include(path);
        }
    }
    res
}

fn main() {
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-changed=src/lldb/UnityBuild.cpp");
    println!("cargo:rerun-if-changed=src/lldb/Bindings");
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
