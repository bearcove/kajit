use std::collections::BTreeMap;

const KAJIT_OPTS_ENV: &str = "KAJIT_OPTS";
const KAJIT_OPTS_HELP: &str = "help";
const PASS_OPTION_PREFIX: &str = "pass.";

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PipelineOptions {
    pub all_opts: Option<bool>,
    pub regalloc: Option<bool>,
    pass_overrides: BTreeMap<String, bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    message: String,
}

impl ParseError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParseError {}

impl PipelineOptions {
    // r[impl compiler.opts]
    // r[impl compiler.opts.defaults]
    // r[impl compiler.opts.help]
    // r[impl compiler.opts.invalid]
    pub fn from_env() -> Self {
        let Some(raw) = std::env::var_os(KAJIT_OPTS_ENV) else {
            return Self::default();
        };
        let raw = raw.to_string_lossy();
        let raw = raw.trim();
        if raw.is_empty() {
            return Self::default();
        }
        if raw.eq_ignore_ascii_case(KAJIT_OPTS_HELP) {
            panic!("{}", Self::help_text());
        }
        Self::parse(raw).unwrap_or_else(|error| {
            panic!(
                "invalid {KAJIT_OPTS_ENV}={raw:?}: {error}\n\n{}",
                Self::help_text()
            )
        })
    }

    // r[impl compiler.opts.syntax]
    // r[impl compiler.opts.composition]
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let mut opts = Self::default();
        for token in input.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }

            let (enabled, name) = match token.as_bytes()[0] {
                b'+' => (true, &token[1..]),
                b'-' => (false, &token[1..]),
                _ => (true, token),
            };

            if name.is_empty() {
                return Err(ParseError::new(format!("empty option token in {input:?}")));
            }
            opts.set_option(name, enabled)?;
        }

        Ok(opts)
    }

    // r[impl compiler.opts.pass-registry]
    pub fn set_option(&mut self, name: &str, enabled: bool) -> Result<(), ParseError> {
        match name {
            "all_opts" => {
                self.all_opts = Some(enabled);
                return Ok(());
            }
            "regalloc" => {
                self.regalloc = Some(enabled);
                return Ok(());
            }
            _ => {}
        }

        let pass_name = name.strip_prefix(PASS_OPTION_PREFIX).unwrap_or(name);
        if !is_known_pass(pass_name) {
            return Err(ParseError::new(format!("unknown option {name:?}")));
        }

        self.pass_overrides.insert(pass_name.to_owned(), enabled);
        Ok(())
    }

    // r[impl compiler.opts.api]
    pub fn enable_option(&mut self, name: &str) -> Result<(), ParseError> {
        self.set_option(name, true)
    }

    // r[impl compiler.opts.api]
    pub fn disable_option(&mut self, name: &str) -> Result<(), ParseError> {
        self.set_option(name, false)
    }

    pub fn resolve_all_opts(&self, default_enabled: bool) -> bool {
        self.all_opts.unwrap_or(default_enabled)
    }

    pub fn resolve_regalloc(&self, default_enabled: bool) -> bool {
        self.regalloc.unwrap_or(default_enabled)
    }

    pub fn resolve_pass(&self, pass_name: &str, default_enabled: bool) -> bool {
        self.pass_overrides
            .get(pass_name)
            .copied()
            .unwrap_or(default_enabled)
    }

    pub fn help_text() -> String {
        let mut out = String::new();
        out.push_str("KAJIT_OPTS usage:\n");
        out.push_str("  comma-separated tokens, each optionally prefixed with '+' or '-'\n");
        out.push_str("  examples: -all_opts, +regalloc, -pass.inline_apply\n\n");
        out.push_str("Top-level options:\n");
        out.push_str("  all_opts  : enable/disable pre-linearization default passes\n");
        out.push_str("  regalloc  : enable/disable regalloc allocation+edits (disabled = canonical vreg->stack lowering)\n\n");
        out.push_str("Per-pass options:\n");
        for pass in crate::ir_passes::default_pass_registry() {
            out.push_str(&format!("  pass.{}  : {}\n", pass.name, pass.description));
        }
        out.push_str("\nSet KAJIT_OPTS=help to print this message.");
        out
    }
}

fn is_known_pass(name: &str) -> bool {
    crate::ir_passes::default_pass_registry()
        .iter()
        .any(|pass| pass.name == name)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{KAJIT_OPTS_ENV, PipelineOptions};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_kajit_opts_env(value: Option<&str>, f: impl FnOnce()) {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        match value {
            Some(value) => unsafe {
                std::env::set_var(KAJIT_OPTS_ENV, value);
            },
            None => unsafe {
                std::env::remove_var(KAJIT_OPTS_ENV);
            },
        }
        f();
        unsafe {
            std::env::remove_var(KAJIT_OPTS_ENV);
        }
    }

    // r[verify compiler.opts.defaults]
    #[test]
    fn parse_empty_string_keeps_defaults() {
        assert_eq!(
            PipelineOptions::parse("").unwrap(),
            PipelineOptions::default()
        );
    }

    // r[verify compiler.opts.syntax]
    // r[verify compiler.opts.composition]
    #[test]
    fn parse_supports_enabling_disabling_and_last_token_wins() {
        let got = PipelineOptions::parse("-all_opts,+all_opts,-regalloc").unwrap();
        assert_eq!(got.all_opts, Some(true));
        assert_eq!(got.regalloc, Some(false));
    }

    // r[verify compiler.opts.syntax]
    #[test]
    fn parse_bare_tokens_as_enable() {
        let got = PipelineOptions::parse("all_opts,regalloc").unwrap();
        assert_eq!(got.all_opts, Some(true));
        assert_eq!(got.regalloc, Some(true));
    }

    #[test]
    fn parse_supports_per_pass_toggles() {
        let got = PipelineOptions::parse("-pass.inline_apply,+theta_loop_invariant_hoist").unwrap();
        assert_eq!(got.resolve_pass("inline_apply", true), false);
        assert_eq!(got.resolve_pass("theta_loop_invariant_hoist", false), true);
    }

    // r[verify compiler.opts.invalid]
    #[test]
    fn parse_rejects_unknown_tokens() {
        let error = PipelineOptions::parse("-does_not_exist").unwrap_err();
        assert!(
            error.to_string().contains("unknown option"),
            "unexpected parse error: {error}"
        );
    }

    #[test]
    // r[verify compiler.opts.api]
    fn resolve_falls_back_to_callsite_defaults() {
        let got = PipelineOptions::default();
        assert_eq!(got.resolve_all_opts(false), false);
        assert_eq!(got.resolve_regalloc(true), true);
        assert_eq!(got.resolve_pass("inline_apply", true), true);
    }

    #[test]
    // r[verify compiler.opts.api]
    fn resolve_prefers_explicit_overrides() {
        let got = PipelineOptions::parse("-all_opts,+regalloc").unwrap();
        assert_eq!(got.resolve_all_opts(true), false);
        assert_eq!(got.resolve_regalloc(false), true);
    }

    #[test]
    // r[verify compiler.opts]
    fn from_env_reads_kajit_opts() {
        with_kajit_opts_env(Some("-regalloc"), || {
            let got = PipelineOptions::from_env();
            assert_eq!(got.resolve_regalloc(true), false);
        });
    }

    #[test]
    // r[verify compiler.opts.pass-registry]
    fn from_env_reads_per_pass_toggles() {
        with_kajit_opts_env(Some("-pass.inline_apply"), || {
            let got = PipelineOptions::from_env();
            assert_eq!(got.resolve_pass("inline_apply", true), false);
        });
    }

    #[test]
    // r[verify compiler.opts.help]
    fn from_env_help_panics_with_help_text() {
        with_kajit_opts_env(Some("help"), || {
            let panic = std::panic::catch_unwind(PipelineOptions::from_env)
                .expect_err("KAJIT_OPTS=help should panic with usage text");
            let msg = match panic.downcast::<String>() {
                Ok(msg) => *msg,
                Err(panic) => match panic.downcast::<&str>() {
                    Ok(msg) => (*msg).to_owned(),
                    Err(_) => "<non-string panic>".to_owned(),
                },
            };
            assert!(
                msg.contains("KAJIT_OPTS usage"),
                "unexpected help panic: {msg}"
            );
        });
    }
}
