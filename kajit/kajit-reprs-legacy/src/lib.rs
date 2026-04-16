use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::prelude::Rich;

pub mod asm;
pub mod hir;
pub mod ir;
pub mod lir;
pub mod mir;

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
