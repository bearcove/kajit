use chumsky::prelude::Rich;

/// Format chumsky `Rich` parse errors with line/column numbers and source context.
pub fn format_rich_errors(source: &str, errs: Vec<Rich<char>>) -> String {
    errs.into_iter()
        .map(|e| {
            let offset = e.span().start;
            let prefix = &source[..offset.min(source.len())];
            let line = prefix.chars().filter(|&c| c == '\n').count() + 1;
            let col = prefix.rfind('\n').map_or(offset, |nl| offset - nl - 1) + 1;
            let line_start = prefix.rfind('\n').map_or(0, |nl| nl + 1);
            let line_end = source[offset..]
                .find('\n')
                .map_or(source.len(), |nl| offset + nl);
            let context_line = &source[line_start..line_end];
            format!(
                "line {line}, col {col}: {e}\n  | {context_line}\n  | {}^",
                " ".repeat(col.saturating_sub(1))
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
