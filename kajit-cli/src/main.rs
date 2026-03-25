mod mcp;

use facet::Facet;
use figue as args;

/// kajit — JIT deserializer toolkit
#[derive(Facet, Debug)]
struct Args {
    #[facet(args::subcommand)]
    command: Command,
}

#[derive(Facet, Debug)]
#[repr(u8)]
enum Command {
    /// Run the MCP debugger server
    Mcp {
        /// Run in real mode (direct MCP protocol, not proxy)
        #[facet(args::named, default)]
        real: bool,
    },

    /// Evaluate a CFG-MIR program and print the result
    Eval {
        /// Path to CFG-MIR text file
        #[facet(args::positional)]
        cfg_mir: String,

        /// Input bytes as hex string
        #[facet(args::positional, default)]
        input_hex: Option<String>,
    },
}

fn main() {
    let args: Args = figue::from_std_args().unwrap();
    let cmd = args.command;

    match cmd {
        Command::Mcp { real } => {
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            let result = if real {
                rt.block_on(mcp::run_real())
            } else {
                rt.block_on(mcp::run_mcp_proxy())
            };
            if let Err(error) = result {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        Command::Eval { cfg_mir, input_hex } => {
            cmd_eval(&cfg_mir, input_hex.as_deref().unwrap_or(""));
        }
    }
}

fn cmd_eval(cfg_mir_path: &str, input_hex: &str) {
    let mir_text = std::fs::read_to_string(cfg_mir_path).unwrap_or_else(|e| {
        eprintln!("error: failed to read {cfg_mir_path}: {e}");
        std::process::exit(1);
    });

    let program = kajit_mir_text::parse_cfg_mir(&mir_text).unwrap_or_else(|e| {
        eprintln!("error: failed to parse CFG-MIR: {e}");
        std::process::exit(1);
    });

    let input = parse_hex(input_hex);
    let session = kajit_mir::DebuggerSession::new(&program, &input).unwrap_or_else(|e| {
        eprintln!("error: failed to create session: {e}");
        std::process::exit(1);
    });

    // Run to completion
    let mut session = session;
    let events = session
        .run_until(kajit_mir::RunUntilTarget::Return, 100_000)
        .unwrap_or_else(|e| {
            eprintln!("error: execution failed: {e}");
            std::process::exit(1);
        });

    let state = session.state();

    // Print result
    println!("steps: {}", state.step_count);
    println!("cursor: {}", state.cursor);
    if let Some(trap) = &state.trap {
        println!("TRAP: {:?} at offset {}", trap.code, trap.offset);
    }
    if state.returned {
        println!("returned: yes");
    }
    println!("output: {}", encode_hex(&state.output));

    // Non-zero vregs
    let nonzero: Vec<_> = state
        .vregs
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0)
        .collect();
    if !nonzero.is_empty() {
        print!("vregs:");
        for (idx, val) in &nonzero {
            print!(" v{}={}", idx, val);
        }
        println!();
    }
}

fn parse_hex(s: &str) -> Vec<u8> {
    let cleaned: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    if cleaned.is_empty() {
        return Vec::new();
    }
    cleaned
        .as_bytes()
        .chunks_exact(2)
        .map(|chunk| {
            let s = std::str::from_utf8(chunk).unwrap();
            u8::from_str_radix(s, 16).unwrap()
        })
        .collect()
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
