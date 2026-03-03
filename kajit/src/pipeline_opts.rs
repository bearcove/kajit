const KAJIT_OPTS_ENV: &str = "KAJIT_OPTS";
const KNOWN_OPTIONS: &[&str] = &["all_opts", "regalloc"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineOptions {
    pub all_opts: Option<bool>,
    pub regalloc: Option<bool>,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            all_opts: None,
            regalloc: None,
        }
    }
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
    // r[impl compiler.opts.invalid]
    pub fn from_env() -> Self {
        let Some(raw) = std::env::var_os(KAJIT_OPTS_ENV) else {
            return Self::default();
        };
        let raw = raw.to_string_lossy();
        if raw.trim().is_empty() {
            return Self::default();
        }
        Self::parse(&raw).unwrap_or_else(|error| {
            panic!(
                "invalid {KAJIT_OPTS_ENV}={raw:?}: {error}. Supported options: {}",
                KNOWN_OPTIONS.join(", ")
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

            match name {
                "all_opts" => opts.all_opts = Some(enabled),
                "regalloc" => opts.regalloc = Some(enabled),
                other => {
                    return Err(ParseError::new(format!(
                        "unknown option {other:?} in {input:?}"
                    )));
                }
            }
        }

        Ok(opts)
    }

    pub fn resolve_all_opts(self, default_enabled: bool) -> bool {
        self.all_opts.unwrap_or(default_enabled)
    }

    pub fn resolve_regalloc(self, default_enabled: bool) -> bool {
        self.regalloc.unwrap_or(default_enabled)
    }
}

#[cfg(test)]
mod tests {
    use super::PipelineOptions;

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
    fn resolve_falls_back_to_callsite_defaults() {
        let got = PipelineOptions::default();
        assert_eq!(got.resolve_all_opts(false), false);
        assert_eq!(got.resolve_regalloc(true), true);
    }

    #[test]
    fn resolve_prefers_explicit_overrides() {
        let got = PipelineOptions::parse("-all_opts,+regalloc").unwrap();
        assert_eq!(got.resolve_all_opts(true), false);
        assert_eq!(got.resolve_regalloc(false), true);
    }
}
