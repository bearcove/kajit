use facet::Facet;
use figue as args;

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
}

fn main() {
    let args: Args = figue::from_std_args().unwrap();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    match args.command {
        Command::Lsp { stdio } => cmd_lsp(stdio),
    }
}

fn cmd_lsp(stdio: bool) {
    if !stdio {
        eprintln!("error: kajit lsp currently only supports --stdio");
        std::process::exit(2);
    }

    eprintln!("error: kajit lsp is not implemented yet");
    std::process::exit(1);
}
