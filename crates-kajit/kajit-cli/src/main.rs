use facet::Facet;
use figue as args;
use std::path::PathBuf;

mod lsp;
mod validate;

/// kajit — JIT deserializer toolkit
#[derive(Facet, Debug)]
struct Args {
    /// Standard CLI options
    #[facet(flatten)]
    builtins: args::FigueBuiltins,

    #[facet(args::subcommand)]
    command: Command,
}

#[derive(Facet, Debug)]
#[repr(u8)]
enum Command {
    /// Run the Kajit LSP server
    Lsp {
        /// Run over stdio
        #[facet(args::named, default = true)]
        stdio: bool,
    },

    /// Validate a Kajit text file based on its extension
    Validate {
        /// Path to the file to validate
        #[facet(args::positional)]
        path: PathBuf,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Args = figue::from_std_args().unwrap();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    match args.command {
        Command::Lsp { stdio } => lsp::cmd_lsp(stdio).await,
        Command::Validate { path } => {
            if let Err(err) = validate::cmd_validate(&path) {
                eprintln!("{err}");
                std::process::exit(1);
            }
        }
    }
}
