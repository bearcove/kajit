use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn main() {
    let mut args = std::env::args();
    let _bin = args.next();
    let command = args.next();
    let command_args: Vec<String> = args.collect();

    match command.as_deref() {
        Some("install") => {
            let dev = command_args.iter().any(|a| a == "--dev");
            let sanitize = !command_args.iter().any(|a| a == "--no-sanitize");
            install(dev, sanitize);
        }
        _ => {
            eprintln!("usage: cargo xtask install");
            std::process::exit(2);
        }
    }
}

fn install(dev: bool, sanitize: bool) {
    let root = workspace_root();
    let package = "kajit-cli";
    let binary = platform_binary_name("kajit");

    let (mode, profile_dir) = if dev {
        ("debug", "debug")
    } else {
        ("release", "release")
    };

    if sanitize {
        println!("building {package} in {mode} mode with AddressSanitizer...");
    } else {
        println!("building {package} in {mode} mode...");
    }

    let mut cmd = Command::new("cargo");
    if sanitize {
        cmd.arg("+nightly");
    }
    cmd.args(["build", "-p", package]);
    if !dev {
        cmd.arg("--release");
    }
    if sanitize {
        let target = if cfg!(target_arch = "aarch64") {
            "aarch64-apple-darwin"
        } else {
            "x86_64-apple-darwin"
        };
        cmd.arg("--target").arg(target);
        cmd.env("RUSTFLAGS", "-Zsanitizer=address");
    }

    let status = cmd
        .current_dir(&root)
        .status()
        .expect("failed to run cargo build");
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }

    let src = if sanitize {
        let target = if cfg!(target_arch = "aarch64") {
            "aarch64-apple-darwin"
        } else {
            "x86_64-apple-darwin"
        };
        root.join("target")
            .join(target)
            .join(profile_dir)
            .join(&binary)
    } else {
        root.join("target").join(profile_dir).join(&binary)
    };

    if !src.exists() {
        eprintln!("build finished but binary not found at {}", src.display());
        std::process::exit(1);
    }

    let dst_dir = cargo_bin_dir();
    if let Err(err) = fs::create_dir_all(&dst_dir) {
        eprintln!(
            "failed to create cargo bin directory {}: {err}",
            dst_dir.display()
        );
        std::process::exit(1);
    }

    let dst = dst_dir.join(&binary);
    let tmp = dst_dir.join(format!("{binary}.tmp-{}", std::process::id()));

    if let Err(err) = fs::copy(&src, &tmp) {
        eprintln!(
            "failed to copy {} to {}: {err}",
            src.display(),
            tmp.display()
        );
        std::process::exit(1);
    }

    if let Err(err) = fs::rename(&tmp, &dst) {
        #[cfg(windows)]
        {
            let _ = fs::remove_file(&dst);
            if let Err(err) = fs::rename(&tmp, &dst) {
                eprintln!(
                    "failed to install {} to {}: {err}",
                    src.display(),
                    dst.display()
                );
                std::process::exit(1);
            }
        }
        #[cfg(not(windows))]
        {
            let _ = fs::remove_file(&tmp);
            eprintln!(
                "failed to install {} to {}: {err}",
                src.display(),
                dst.display()
            );
            std::process::exit(1);
        }
    }

    println!("copied {package} to {}", dst.display());

    #[cfg(target_os = "macos")]
    {
        println!("codesigning installed binary...");
        let status = Command::new("codesign")
            .args(["--sign", "-", "--force"])
            .arg(&dst)
            .status()
            .expect("failed to run codesign");
        if !status.success() {
            eprintln!("warning: codesign failed, continuing anyway");
        } else {
            let verify = Command::new("codesign")
                .args(["--verify", "--verbose=2"])
                .arg(&dst)
                .status()
                .expect("failed to run codesign --verify");
            if !verify.success() {
                eprintln!("warning: codesign verification failed, continuing anyway");
            }
        }
    }

    println!("validating installed binary...");
    let output = Command::new(&dst)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|err| {
            eprintln!("failed to execute {}: {err}", dst.display());
            std::process::exit(1);
        });

    if !output.status.success() {
        eprintln!(
            "installed binary exited with {} while validating",
            output.status
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            eprintln!("stderr:\n{stderr}");
        }
        std::process::exit(output.status.code().unwrap_or(1));
    }

    println!("installed and validated: {}", dst.display());

    print_mcp_setup_instructions(&dst);
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask should live in workspace/xtask")
        .to_path_buf()
}

fn platform_binary_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    }
}

fn cargo_bin_dir() -> PathBuf {
    if let Some(cargo_home) = std::env::var_os("CARGO_HOME") {
        let cargo_home = PathBuf::from(cargo_home);
        if cargo_home.is_absolute() {
            return cargo_home.join("bin");
        }
        return home_dir().join(cargo_home).join("bin");
    }
    home_dir().join(".cargo").join("bin")
}

fn home_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home);
    }
    if let Some(home) = std::env::var_os("USERPROFILE") {
        return PathBuf::from(home);
    }
    panic!("unable to determine home directory (HOME/USERPROFILE are unset)");
}

fn print_mcp_setup_instructions(installed_binary: &Path) {
    let binary = installed_binary.display();
    println!();
    println!("MCP setup (copy/paste):");
    println!("  codex  => codex mcp add kajit -- {binary} mcp");
    println!("  claude => claude mcp add --transport stdio kajit -- {binary} mcp");
    println!();
    println!("After adding, restart the client so it picks up the new MCP server.");
}
