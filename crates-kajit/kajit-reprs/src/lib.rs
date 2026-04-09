use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::prelude::Rich;

pub mod hir;
pub mod schema_poc;

/// Format chumsky `Rich` parse errors with ariadne for readable diagnostics.
///
/// This is the first shared helper moved up from `kajit-parse-util` so the
/// representation crate can own parse/display support over time.
pub fn format_rich_errors(source: &str, errs: Vec<Rich<char>>) -> String {
    let mut buf = Vec::new();
    for e in errs {
        Report::build(ReportKind::Error, ((), e.span().into_range()))
            .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Byte))
            .with_message(e.to_string())
            .with_label(
                Label::new(((), e.span().into_range()))
                    .with_message(e.reason().to_string())
                    .with_color(Color::Red),
            )
            .finish()
            .write(Source::from(source), &mut buf)
            .unwrap();
    }
    String::from_utf8(buf).unwrap_or_else(|e| format!("(error formatting parse errors: {e})"))
}
