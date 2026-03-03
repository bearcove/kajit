//! Parse NDJSON bench output from stdin and generate bench_report/{index.html,results.json,results.md}.
//!
//! Reads line-by-line, showing live progress on stderr.
//!
//!   cargo bench 2>/dev/null | cargo run --example bench_report

use std::fmt::Write as _;
use std::io::BufRead as _;

use facet::Facet;
#[cfg(target_arch = "x86_64")]
use yaxpeax_arch::LengthedInstruction;
use yaxpeax_arch::{Decoder, U8Reader};

fn main() {
    let check_vec_signals = std::env::args().any(|arg| arg == "--check-vec-signals");

    let stdin = std::io::stdin();
    let mut lines: Vec<String> = Vec::new();

    // Progress state
    let mut total_expected: usize = 0;
    let mut completed: usize = 0;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() || !trimmed.starts_with('{') {
            continue;
        }

        // Parse for progress display
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            match v["type"].as_str() {
                Some("suite") => {
                    let n = v["total"].as_u64().unwrap_or(0) as usize;
                    total_expected += n;
                    eprint!("\r\x1b[2K  [{completed}/{total_expected}] waiting...");
                }
                Some("start") => {
                    if let Some(name) = v["name"].as_str() {
                        eprint!("\r\x1b[2K  [{completed}/{total_expected}] {name}");
                    }
                }
                Some("result") => {
                    completed += 1;
                    if let (Some(name), Some(median)) =
                        (v["name"].as_str(), v["median_ns"].as_f64())
                    {
                        eprint!(
                            "\r\x1b[2K  [{completed}/{total_expected}] {name} ... {}",
                            fmt_time(median)
                        );
                    }
                }
                _ => {}
            }
        }

        lines.push(line);
    }

    // Clear progress line
    eprint!("\r\x1b[2K");

    let input = lines.join("\n");
    let sections = parse_ndjson(&input);

    if sections.is_empty() || sections.iter().all(|s| s.groups.is_empty()) {
        eprintln!("no groups parsed — pipe in `cargo bench` NDJSON output");
        std::process::exit(1);
    }

    let meta = Meta::collect();
    let vec_signals = collect_vec_scalar_signals(&sections);
    let html = render(&sections, &meta, &vec_signals);
    let json = render_json(&sections, &meta, &vec_signals);
    let md = render_markdown(&sections, &meta, &vec_signals);
    std::fs::create_dir_all("bench_report").unwrap();
    std::fs::write("bench_report/index.html", &html).unwrap();
    std::fs::write("bench_report/results.json", &json).unwrap();
    std::fs::write("bench_report/results.md", &md).unwrap();
    eprintln!("→ bench_report/index.html  ({completed} benchmarks)");
    eprintln!("→ bench_report/results.json");
    eprintln!("→ bench_report/results.md");

    if check_vec_signals {
        match check_vec_scalar_signals(&vec_signals) {
            Ok(summary) => eprintln!("PASS vec-signal check: {summary}"),
            Err(details) => {
                eprintln!("FAIL vec-signal check:\n{details}");
                std::process::exit(2);
            }
        }
    }
}

// ── Types ──────────────────────────────────────────────────────────────────

struct Meta {
    datetime: String,
    commit_short: String,
    commit_full: String,
    os_name: String,
    platform: String,
}

impl Meta {
    fn collect() -> Self {
        let datetime = sh("date", &["+%Y-%m-%d %H:%M %Z"]);
        let commit_short = sh("git", &["rev-parse", "--short", "HEAD"]);
        let commit_full = sh("git", &["rev-parse", "HEAD"]);
        let uname = sh("uname", &["-srm"]);
        let os_name = sh("uname", &["-s"]);
        let cpu = cpu_name();
        let platform = if cpu.is_empty() {
            uname
        } else {
            format!("{uname} · {cpu}")
        };
        Meta {
            datetime,
            commit_short,
            commit_full,
            os_name,
            platform,
        }
    }
}

fn sh(prog: &str, args: &[&str]) -> String {
    std::process::Command::new(prog)
        .args(args)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn cpu_name() -> String {
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in info.lines() {
            if line.starts_with("model name") {
                if let Some(val) = line.split(':').nth(1) {
                    return val.trim().to_string();
                }
            }
        }
    }
    sh("sysctl", &["-n", "machdep.cpu.brand_string"])
}

struct Section {
    label: String,
    groups: Vec<Group>,
}

struct Group {
    name: String,
    rows: Vec<Row>,
}

struct Row {
    name: String,
    median_ns: f64,
    p5_ns: f64,
    p95_ns: f64,
}

#[derive(Clone, Copy)]
struct AsmSignals {
    frame_size_bytes: Option<u32>,
    stack_mem_ops: usize,
    regalloc_edits: Option<usize>,
}

struct VecScalarSignalsRow {
    bench_name: &'static str,
    legacy: AsmSignals,
    ir: AsmSignals,
}

const VEC_SCALAR_BENCHES: [&str; 3] = ["vec_scalar_small", "vec_scalar_medium", "vec_scalar_large"];

#[derive(Facet)]
struct VecScalarSignalShape {
    values: Vec<u32>,
}

fn collect_vec_scalar_signals(sections: &[Section]) -> Vec<VecScalarSignalsRow> {
    let present: Vec<&'static str> = VEC_SCALAR_BENCHES
        .iter()
        .copied()
        .filter(|name| has_postcard_deser_group(sections, name))
        .collect();
    if present.is_empty() {
        return Vec::new();
    }

    // r[impl ir.regalloc.regressions]
    let legacy =
        kajit::compile_decoder(VecScalarSignalShape::SHAPE, &kajit::postcard::KajitPostcard);
    let ir = kajit::compile_decoder(VecScalarSignalShape::SHAPE, &kajit::postcard::KajitPostcard);

    let legacy_signals = analyze_codegen_signals(legacy.code(), None);
    let ir_edits =
        kajit::regalloc_edit_count(VecScalarSignalShape::SHAPE, &kajit::postcard::KajitPostcard);
    let ir_signals = analyze_codegen_signals(ir.code(), Some(ir_edits));

    present
        .into_iter()
        .map(|bench_name| VecScalarSignalsRow {
            bench_name,
            legacy: legacy_signals,
            ir: ir_signals,
        })
        .collect()
}

fn check_vec_scalar_signals(rows: &[VecScalarSignalsRow]) -> Result<String, String> {
    if rows.is_empty() {
        return Err("no vec scalar signals were collected".to_string());
    }

    let mut failures = Vec::<String>::new();
    for row in rows {
        let bench = row.bench_name;
        if row.ir.regalloc_edits.is_none() {
            failures.push(format!("{bench}: missing ir edit count"));
        }

        if let (Some(legacy_frame), Some(ir_frame)) =
            (row.legacy.frame_size_bytes, row.ir.frame_size_bytes)
            && ir_frame > legacy_frame
        {
            failures.push(format!(
                "{bench}: ir frame {ir_frame}B is larger than legacy {legacy_frame}B"
            ));
        }

        if row.legacy.stack_mem_ops == 0 {
            failures.push(format!(
                "{bench}: legacy stack_mem_ops is zero, ratio undefined"
            ));
            continue;
        }
        let ratio = row.ir.stack_mem_ops as f64 / row.legacy.stack_mem_ops as f64;
        if ratio > 2.0 {
            failures.push(format!(
                "{bench}: ir/legacy stack-op ratio {ratio:.2}x exceeds 2.00x"
            ));
        }
    }

    if failures.is_empty() {
        let ratios = rows
            .iter()
            .map(|row| {
                if row.legacy.stack_mem_ops == 0 {
                    0.0
                } else {
                    row.ir.stack_mem_ops as f64 / row.legacy.stack_mem_ops as f64
                }
            })
            .collect::<Vec<_>>();
        let worst_ratio = ratios.into_iter().fold(0.0_f64, f64::max);
        Ok(format!(
            "{} benches checked, worst stack-op ratio {:.2}x (limit 2.00x)",
            rows.len(),
            worst_ratio
        ))
    } else {
        Err(failures.join("\n"))
    }
}

fn has_postcard_deser_group(sections: &[Section], group_name: &str) -> bool {
    sections.iter().any(|section| {
        section.label == "postcard deser" && section.groups.iter().any(|g| g.name == group_name)
    })
}

fn analyze_codegen_signals(code: &[u8], regalloc_edits: Option<usize>) -> AsmSignals {
    let insts = decode_instructions(code);
    AsmSignals {
        frame_size_bytes: detect_frame_size_bytes(&insts),
        stack_mem_ops: insts.iter().filter(|inst| is_stack_mem_op(inst)).count(),
        regalloc_edits,
    }
}

#[cfg(target_arch = "aarch64")]
fn decode_instructions(code: &[u8]) -> Vec<String> {
    use yaxpeax_arm::armv8::a64::InstDecoder;

    let decoder = InstDecoder::default();
    let mut reader = U8Reader::new(code);
    let mut insts = Vec::new();
    let mut offset = 0usize;
    while offset + 4 <= code.len() {
        match decoder.decode(&mut reader) {
            Ok(inst) => insts.push(format!("{inst}")),
            Err(_) => break,
        }
        offset += 4;
    }
    insts
}

#[cfg(target_arch = "x86_64")]
fn decode_instructions(code: &[u8]) -> Vec<String> {
    use yaxpeax_x86::amd64::InstDecoder;

    let decoder = InstDecoder::default();
    let mut reader = U8Reader::new(code);
    let mut insts = Vec::new();
    let mut offset = 0usize;
    while offset < code.len() {
        match decoder.decode(&mut reader) {
            Ok(inst) => {
                let len = inst.len().to_const() as usize;
                if len == 0 {
                    break;
                }
                insts.push(format!("{inst}"));
                offset += len;
            }
            Err(_) => break,
        }
    }
    insts
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn decode_instructions(_code: &[u8]) -> Vec<String> {
    Vec::new()
}

fn detect_frame_size_bytes(insts: &[String]) -> Option<u32> {
    #[cfg(target_arch = "aarch64")]
    {
        for inst in insts {
            if let Some(rest) = inst.strip_prefix("sub sp, sp, #") {
                if let Some(bytes) = parse_imm_u32(rest) {
                    return Some(bytes);
                }
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        for inst in insts {
            if let Some(rest) = inst.strip_prefix("sub rsp, ") {
                if let Some(bytes) = parse_imm_u32(rest) {
                    return Some(bytes);
                }
            }
        }
    }
    None
}

fn parse_imm_u32(text: &str) -> Option<u32> {
    let token = text
        .trim()
        .trim_start_matches('#')
        .split([',', ' '])
        .next()
        .unwrap_or_default();
    if token.is_empty() {
        return None;
    }
    if let Some(hex) = token.strip_prefix("0x") {
        return u32::from_str_radix(hex, 16).ok();
    }
    token.parse::<u32>().ok()
}

fn is_stack_mem_op(inst: &str) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        return inst.contains("[sp");
    }
    #[cfg(target_arch = "x86_64")]
    {
        let mnemonic = inst.split_whitespace().next().unwrap_or_default();
        if matches!(mnemonic, "push" | "pop") {
            return true;
        }
        if mnemonic == "lea" {
            return false;
        }
        return inst.contains("[rsp") || inst.contains("[rbp");
    }
    #[allow(unreachable_code)]
    false
}

// ── Parser (NDJSON) ───────────────────────────────────────────────────────

fn parse_ndjson(input: &str) -> Vec<Section> {
    let mut entries: Vec<(Vec<String>, f64, f64, f64)> = Vec::new();

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || !line.starts_with('{') {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v["type"].as_str() != Some("result") {
            continue;
        }
        let Some(name) = v["name"].as_str() else {
            continue;
        };
        let Some(median_ns) = v["median_ns"].as_f64() else {
            continue;
        };
        let p5_ns = v["p5_ns"].as_f64().unwrap_or(median_ns);
        let p95_ns = v["p95_ns"].as_f64().unwrap_or(median_ns);

        let parts: Vec<String> = name.split('/').map(String::from).collect();
        entries.push((parts, median_ns, p5_ns, p95_ns));
    }

    organize_entries(&entries)
}

/// Reorganize flat entries into Section/Group/Row hierarchy.
///
/// - 3-component paths `[type, format, fn]` from the `bench!` macro:
///   section = "{format} {direction}", group = type, row = fn
/// - 2-component paths `[group, fn]` from manual bench groups:
///   infer format from group name prefix, same routing
/// - 1-component paths `[fn]` from flat benchmarks:
///   section = "json deser", group = first component, row = fn
fn organize_entries(entries: &[(Vec<String>, f64, f64, f64)]) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();

    for (path, median_ns, p5_ns, p95_ns) in entries {
        let (section_label, group_name, row_name) = match path.len() {
            3 => {
                let direction = if path[2].ends_with("_ser") {
                    "ser"
                } else {
                    "deser"
                };
                (
                    format!("{} {direction}", path[1]),
                    path[0].clone(),
                    path[2].clone(),
                )
            }
            2 => {
                // Manual groups: infer format from group name prefix
                let (format, clean_name) = if path[0].starts_with("postcard_") {
                    ("postcard", path[0]["postcard_".len()..].to_string())
                } else {
                    ("json", path[0].clone())
                };
                let direction = if path[1].ends_with("_ser") {
                    "ser"
                } else {
                    "deser"
                };
                (format!("{format} {direction}"), clean_name, path[1].clone())
            }
            1 => ("json deser".to_string(), path[0].clone(), path[0].clone()),
            _ => continue,
        };

        // Find or create section
        let section = match sections.iter_mut().find(|s| s.label == section_label) {
            Some(s) => s,
            None => {
                sections.push(Section {
                    label: section_label,
                    groups: Vec::new(),
                });
                sections.last_mut().unwrap()
            }
        };

        // Find or create group within section
        let group = match section.groups.iter_mut().find(|g| g.name == group_name) {
            Some(g) => g,
            None => {
                section.groups.push(Group {
                    name: group_name,
                    rows: Vec::new(),
                });
                section.groups.last_mut().unwrap()
            }
        };

        group.rows.push(Row {
            name: row_name,
            median_ns: *median_ns,
            p5_ns: *p5_ns,
            p95_ns: *p95_ns,
        });
    }

    sections
}

fn fmt_time_html(ns: f64) -> String {
    let (num, unit) = if ns >= 1_000_000_000.0 {
        (format!("{:.2}", ns / 1_000_000_000.0), "s")
    } else if ns >= 1_000_000.0 {
        (format!("{:.2}", ns / 1_000_000.0), "ms")
    } else if ns >= 1_000.0 {
        (format!("{:.2}", ns / 1_000.0), "µs")
    } else {
        (format!("{:.1}", ns), "ns")
    };
    format!(r#"{}<span class="unit"> {}</span>"#, num, unit)
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Strip _deser/_ser suffix for display, then apply legacy renames.
fn display_name(name: &str) -> String {
    let base = name
        .strip_suffix("_deser")
        .or_else(|| name.strip_suffix("_ser"))
        .or_else(|| name.strip_suffix("_compile"))
        .unwrap_or(name);
    match normalize_impl_name(base) {
        "serde" => "serde".to_string(),
        "kajit" => "kajit".to_string(),
        other => other.to_string(),
    }
}

#[derive(Clone, Copy)]
struct PairDef {
    id: &'static str,
    left: &'static str,
    right: &'static str,
    label: &'static str,
}

const PAIRS: [PairDef; 1] = [PairDef {
    id: "serde-vs-kajit",
    left: "serde",
    right: "kajit",
    label: "serde vs kajit",
}];

fn normalize_impl_name(name: &str) -> &str {
    match name {
        "postcard_serde" | "serde_json" => "serde",
        "kajit" | "legacy" | "kajit_legacy" | "kajit_dynasm" | "ir" | "kajit_ir" => "kajit",
        other => other,
    }
}

fn split_row_name(name: &str) -> (&str, &str) {
    if let Some(base) = name.strip_suffix("_deser") {
        return (base, "deser");
    }
    if let Some(base) = name.strip_suffix("_ser") {
        return (base, "ser");
    }
    (name, "other")
}

fn section_direction(section: &Section) -> &str {
    if section.label.ends_with(" ser") {
        "ser"
    } else {
        "deser"
    }
}

fn find_row_for_impl<'a>(group: &'a Group, impl_name: &str, direction: &str) -> Option<&'a Row> {
    group.rows.iter().find(|r| {
        let (base, dir) = split_row_name(&r.name);
        dir == direction && normalize_impl_name(base) == impl_name
    })
}

// ── JSON ───────────────────────────────────────────────────────────────────

fn render_json(sections: &[Section], meta: &Meta, vec_signals: &[VecScalarSignalsRow]) -> String {
    let mut j = String::new();
    j.push_str("{\n");
    write!(j, r#"  "datetime": "{}","#, meta.datetime).unwrap();
    j.push('\n');
    write!(j, r#"  "commit": "{}","#, meta.commit_short).unwrap();
    j.push('\n');
    write!(
        j,
        r#"  "platform": "{}","#,
        meta.platform.replace('"', r#"\""#)
    )
    .unwrap();
    j.push('\n');
    j.push_str(r#"  "sections": ["#);
    j.push('\n');
    for (si, section) in sections.iter().enumerate() {
        j.push_str("    {\n");
        write!(j, r#"      "label": "{}","#, section.label).unwrap();
        j.push('\n');
        j.push_str(r#"      "groups": ["#);
        j.push('\n');
        for (gi, group) in section.groups.iter().enumerate() {
            j.push_str("        {\n");
            write!(j, r#"          "name": "{}","#, group.name).unwrap();
            j.push('\n');
            j.push_str(r#"          "rows": ["#);
            j.push('\n');
            for (ri, row) in group.rows.iter().enumerate() {
                let comma = if ri + 1 < group.rows.len() { "," } else { "" };
                write!(j, r#"            {{ "name": "{}", "median_ns": {:.1}, "p5_ns": {:.1}, "p95_ns": {:.1} }}{comma}"#, row.name, row.median_ns, row.p5_ns, row.p95_ns).unwrap();
                j.push('\n');
            }
            j.push_str("          ]\n");
            let comma = if gi + 1 < section.groups.len() {
                ","
            } else {
                ""
            };
            write!(j, "        }}{comma}").unwrap();
            j.push('\n');
        }
        j.push_str("      ]\n");
        let comma = if si + 1 < sections.len() { "," } else { "" };
        write!(j, "    }}{comma}").unwrap();
        j.push('\n');
    }

    j.push_str("  ],\n");
    j.push_str(r#"  "vec_scalar_signals": ["#);
    j.push('\n');
    for (i, row) in vec_signals.iter().enumerate() {
        let comma = if i + 1 < vec_signals.len() { "," } else { "" };
        let legacy_frame = row
            .legacy
            .frame_size_bytes
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string());
        let ir_frame = row
            .ir
            .frame_size_bytes
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string());
        let ir_edits = row
            .ir
            .regalloc_edits
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string());
        write!(
            j,
            r#"    {{ "name": "{}", "legacy": {{ "frame_size_bytes": {}, "stack_mem_ops": {} }}, "ir": {{ "frame_size_bytes": {}, "stack_mem_ops": {}, "regalloc_edits": {} }} }}{comma}"#,
            row.bench_name,
            legacy_frame,
            row.legacy.stack_mem_ops,
            ir_frame,
            row.ir.stack_mem_ops,
            ir_edits
        )
        .unwrap();
        j.push('\n');
    }
    j.push_str("  ]\n");
    j.push_str("}\n");
    j
}

// ── Markdown ───────────────────────────────────────────────────────────────

fn fmt_time(ns: f64) -> String {
    if ns >= 1_000_000_000.0 {
        format!("{:.2}s", ns / 1_000_000_000.0)
    } else if ns >= 1_000_000.0 {
        format!("{:.2}ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.2}µs", ns / 1_000.0)
    } else {
        format!("{:.1}ns", ns)
    }
}

fn render_markdown(
    sections: &[Section],
    meta: &Meta,
    vec_signals: &[VecScalarSignalsRow],
) -> String {
    let mut m = String::new();
    writeln!(m, "# Bench Report").unwrap();
    writeln!(m).unwrap();
    writeln!(
        m,
        "> {} · {} · {}",
        meta.datetime, meta.commit_short, meta.platform
    )
    .unwrap();
    writeln!(m).unwrap();

    for section in sections {
        let direction = section_direction(section);
        let mut wrote_section_header = false;

        for pair in PAIRS {
            let mut comparisons: Vec<(&str, f64, f64)> = Vec::new();
            for group in &section.groups {
                let left = find_row_for_impl(group, pair.left, direction);
                let right = find_row_for_impl(group, pair.right, direction);
                if let (Some(l), Some(r)) = (left, right) {
                    comparisons.push((&group.name, l.median_ns, r.median_ns));
                }
            }

            if comparisons.is_empty() {
                continue;
            }
            if !wrote_section_header {
                writeln!(m, "## {}", section.label).unwrap();
                writeln!(m).unwrap();
                wrote_section_header = true;
            }

            comparisons.sort_by(|a, b| {
                let ra = a.2 / a.1;
                let rb = b.2 / b.1;
                rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
            });

            let left_label = display_name(pair.left);
            let right_label = display_name(pair.right);

            writeln!(m, "### {left_label} vs {right_label}").unwrap();
            writeln!(m).unwrap();
            writeln!(
                m,
                "| Benchmark | {left_label} | {right_label} | ratio ({right_label}/{left_label}) |"
            )
            .unwrap();
            writeln!(m, "|-----------|-------|-------|-------|").unwrap();

            for (name, left_ns, right_ns) in &comparisons {
                let ratio = right_ns / left_ns;
                let indicator = if ratio > 1.05 {
                    format!("**{ratio:.2}x** {left_label} faster")
                } else if ratio < 0.95 {
                    format!("**{:.2}x** {right_label} faster", 1.0 / ratio)
                } else {
                    "~tie".to_string()
                };
                writeln!(
                    m,
                    "| {name} | {} | {} | {indicator} |",
                    fmt_time(*left_ns),
                    fmt_time(*right_ns)
                )
                .unwrap();
            }

            writeln!(m).unwrap();

            let left_wins = comparisons.iter().filter(|(_, l, r)| l < r).count();
            let right_wins = comparisons.iter().filter(|(_, l, r)| l > r).count();
            let ties = comparisons.len() - left_wins - right_wins;
            write!(m, "**{left_label} wins {left_wins}").unwrap();
            if ties > 0 {
                write!(m, ", ties {ties}").unwrap();
            }
            writeln!(m, ", {right_label} wins {right_wins}**").unwrap();
            writeln!(m).unwrap();
        }
    }

    if !vec_signals.is_empty() {
        writeln!(m, "## Postcard Vec Scalar Signals").unwrap();
        writeln!(m).unwrap();
        writeln!(
            m,
            "| Benchmark | legacy frame | ir frame | legacy stack ops | ir stack ops | stack ratio (ir/legacy) | ir edits |"
        )
        .unwrap();
        writeln!(
            m,
            "|-----------|--------------|----------|------------------|--------------|-------------------------|----------|"
        )
        .unwrap();
        for row in vec_signals {
            let legacy_frame = row
                .legacy
                .frame_size_bytes
                .map(|v| format!("{v} B"))
                .unwrap_or_else(|| "n/a".to_string());
            let ir_frame = row
                .ir
                .frame_size_bytes
                .map(|v| format!("{v} B"))
                .unwrap_or_else(|| "n/a".to_string());
            let stack_ratio = if row.legacy.stack_mem_ops > 0 {
                format!(
                    "{:.2}x",
                    row.ir.stack_mem_ops as f64 / row.legacy.stack_mem_ops as f64
                )
            } else {
                "n/a".to_string()
            };
            let ir_edits = row
                .ir
                .regalloc_edits
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            writeln!(
                m,
                "| {} | {} | {} | {} | {} | {} | {} |",
                row.bench_name,
                legacy_frame,
                ir_frame,
                row.legacy.stack_mem_ops,
                row.ir.stack_mem_ops,
                stack_ratio,
                ir_edits
            )
            .unwrap();
        }
        writeln!(m).unwrap();
    }

    m
}

// ── HTML ───────────────────────────────────────────────────────────────────

fn group_sort_key_for_pair(group: &Group, pair: PairDef, direction: &str) -> f64 {
    match (
        find_row_for_impl(group, pair.left, direction),
        find_row_for_impl(group, pair.right, direction),
    ) {
        (Some(left), Some(right)) if left.median_ns > 0.0 => right.median_ns / left.median_ns,
        _ => -1.0,
    }
}

/// Map a ratio to a fill percentage of one half of the bar (0–50%).
fn ratio_to_fill(ratio: f64) -> f64 {
    let r = if ratio >= 1.0 { ratio } else { 1.0 / ratio };
    ((r - 1.0) / 3.0 * 50.0_f64).min(50.0)
}

fn render(sections: &[Section], meta: &Meta, vec_signals: &[VecScalarSignalsRow]) -> String {
    let mut h = String::new();

    let active_sections: Vec<&Section> = sections.iter().collect();
    let default_pair_id = PAIRS[0].id;
    let default_tab_id = active_sections
        .first()
        .map(|s| tab_id(&s.label))
        .unwrap_or_else(|| "tab".to_string());

    h.push_str(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bench Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#070A0F;--surface:#0C1118;--border:rgba(255,255,255,0.07);
  --track:#1A2535;
  --text:#AAB8C8;--bright:#E4EEF8;--dim:#6A7A8A;
  --lhs:#34829c;--rhs:#74d44a;
  --mono:"IBM Plex Mono",monospace;--sans:"Plus Jakarta Sans",sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  background:var(--bg);color:var(--text);
  font-family:var(--mono);font-size:13px;
  padding:24px 20px;max-width:960px;margin:0 auto;
}
.controls{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.controls label{font-size:12px;color:var(--dim);letter-spacing:.03em}
.controls select{
  background:var(--surface);color:var(--bright);border:1px solid var(--border);
  border-radius:0;padding:6px 8px;font-family:var(--mono);font-size:12px;
}
.tabs{display:flex;gap:0;margin-bottom:16px;border-bottom:1px solid rgba(255,255,255,0.07);flex-wrap:wrap}
.tab{
  font-family:var(--sans);
  font-size:13px;font-weight:700;letter-spacing:-.01em;
  color:var(--dim);background:none;border:none;cursor:pointer;
  padding:8px 16px 9px;border-bottom:2px solid transparent;margin-bottom:-1px;
}
.tab:hover{color:var(--text)}
.tab.active{color:var(--bright);border-bottom-color:var(--lhs)}
.panel{display:none}
.panel.active{display:block}
.bench-row{
  display:grid;
  grid-template-columns:180px 44px 64px 1fr 64px;
  gap:0 10px;align-items:center;margin-bottom:10px;
}
.bench-row:last-child{margin-bottom:0}
.bname{
  font-size:12px;font-weight:600;color:#A0B8CC;
  letter-spacing:.04em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.ratio-col{
  font-size:11px;font-weight:600;
  font-variant-numeric:tabular-nums;
  text-align:right;white-space:nowrap;
}
.ratio-col.win{color:var(--lhs)}
.ratio-col.lose{color:var(--rhs)}
.ratio-col.na{color:var(--dim)}
.t-left{
  font-size:12px;font-weight:500;color:var(--bright);
  text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;
}
.t-right{
  font-size:12px;font-weight:500;color:var(--bright);
  font-variant-numeric:tabular-nums;white-space:nowrap;
}
.unit{font-size:11px;color:var(--text);font-weight:400}
.pair-row,.pair-summary{display:none}
.delta-track{
  position:relative;height:8px;
  background:var(--track);
}
.delta-track::after{
  content:'';position:absolute;
  left:calc(50% - 0.5px);top:0;bottom:0;width:1px;
  background:rgba(255,255,255,0.2);z-index:1;pointer-events:none;
}
.delta-fill{position:absolute;top:0;bottom:0;opacity:0.55}
.delta-fill.left-side{right:50%;background:var(--lhs)}
.delta-fill.right-side{left:50%;background:var(--rhs)}
.whisker{position:absolute;top:0;bottom:0;width:1px;z-index:2;background:rgba(255,255,255,0.8)}
.whisker-line{position:absolute;top:50%;height:1px;z-index:2;background:rgba(255,255,255,0.5)}
.summary-cell{
  font-family:var(--sans);font-size:13px;font-weight:700;
  text-align:center;padding-bottom:6px;color:var(--text);
}
.summary-cell .lhs-n{color:var(--lhs)}
.summary-cell .rhs-n{color:var(--rhs)}
.summary-cell .sep{color:var(--dim);margin:0 8px}
.signals{
  margin-top:16px;padding:10px 12px;
  border:1px solid var(--border);background:var(--surface);
}
.signals h2{
  font-family:var(--sans);font-size:13px;font-weight:700;color:var(--bright);
  margin-bottom:8px;
}
.signals table{width:100%;border-collapse:collapse}
.signals th,.signals td{
  font-size:12px;padding:4px 6px;text-align:left;
  border-bottom:1px solid rgba(255,255,255,0.06);
}
.signals tr:last-child td{border-bottom:none}
.signals th{color:var(--dim);font-weight:600}
footer{
  margin-top:24px;padding-top:12px;
  border-top:1px solid var(--border);
  font-size:12px;color:var(--dim);
}
footer a:hover{text-decoration:underline}
</style>
</head>
<body>
"#);

    h.push_str(
        r#"<div class="controls"><label for="pair-select">Delta Pair</label><select id="pair-select" onchange="onPairChange(this.value)">"#,
    );
    for pair in PAIRS {
        write!(
            h,
            r#"<option value="{}">{}</option>"#,
            esc(pair.id),
            esc(pair.label)
        )
        .unwrap();
    }
    h.push_str("</select></div>\n");

    h.push_str(r#"<div class="tabs">"#);
    for (i, section) in active_sections.iter().enumerate() {
        let active = if i == 0 { " active" } else { "" };
        let tab_id = tab_id(&section.label);
        write!(
            h,
            r#"<button class="tab{}" data-tab="{}" onclick="switchTab('{}')">{}</button>"#,
            active,
            tab_id,
            tab_id,
            esc(&section.label)
        )
        .unwrap();
    }
    h.push_str("</div>\n");

    for (i, section) in active_sections.iter().enumerate() {
        let active = if i == 0 { " active" } else { "" };
        let tab_id = tab_id(&section.label);
        write!(h, r#"<div class="panel{}" id="panel-{}">"#, active, tab_id).unwrap();

        let direction = section_direction(section);
        let mut groups: Vec<&Group> = section.groups.iter().collect();
        groups.sort_by(|a, b| {
            group_sort_key_for_pair(b, PAIRS[0], direction)
                .partial_cmp(&group_sort_key_for_pair(a, PAIRS[0], direction))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for pair in PAIRS {
            let any_total = groups
                .iter()
                .filter(|g| {
                    find_row_for_impl(g, pair.left, direction).is_some()
                        || find_row_for_impl(g, pair.right, direction).is_some()
                })
                .count();
            if any_total == 0 {
                continue;
            }

            let comparable_total = groups
                .iter()
                .filter(|g| {
                    find_row_for_impl(g, pair.left, direction).is_some()
                        && find_row_for_impl(g, pair.right, direction).is_some()
                })
                .count();
            if comparable_total > 0 {
                let left_wins = groups
                    .iter()
                    .filter(|g| {
                        let Some(left) = find_row_for_impl(g, pair.left, direction) else {
                            return false;
                        };
                        let Some(right) = find_row_for_impl(g, pair.right, direction) else {
                            return false;
                        };
                        left.median_ns <= right.median_ns
                    })
                    .count();
                let right_wins = comparable_total - left_wins;
                write!(
                    h,
                    r#"<div class="bench-row pair-summary pair-{}"><span></span><span></span><span></span><div class="summary-cell">{} wins <span class="lhs-n">{}</span><span class="sep">&middot;</span>{} wins <span class="rhs-n">{}</span></div><span></span></div>"#,
                    esc(pair.id),
                    esc(&display_name(pair.left)),
                    left_wins,
                    esc(&display_name(pair.right)),
                    right_wins
                )
                .unwrap();
            }
        }

        for group in &groups {
            for pair in PAIRS {
                let left_row = find_row_for_impl(group, pair.left, direction);
                let right_row = find_row_for_impl(group, pair.right, direction);
                if left_row.is_none() && right_row.is_none() {
                    continue;
                }

                let default_na = r#"<span class="unit">n/a</span>"#.to_string();
                let left_cell = left_row
                    .map(|r| fmt_time_html(r.median_ns))
                    .unwrap_or_else(|| default_na.clone());
                let right_cell = right_row
                    .map(|r| fmt_time_html(r.median_ns))
                    .unwrap_or_else(|| default_na.clone());

                let mut ratio_text = "--".to_string();
                let mut ratio_cls = "na";
                let mut bar = String::new();

                if let (Some(left_row), Some(right_row)) = (left_row, right_row) {
                    let ratio = right_row.median_ns / left_row.median_ns;
                    let left_wins = ratio >= 1.0;
                    let fill = ratio_to_fill(ratio);
                    let fill_class = if left_wins {
                        "delta-fill left-side"
                    } else {
                        "delta-fill right-side"
                    };
                    ratio_text = format!("{ratio:.2}×");
                    ratio_cls = if left_wins { "win" } else { "lose" };

                    let ratio_best = right_row.p95_ns / left_row.p5_ns;
                    let ratio_worst = right_row.p5_ns / left_row.p95_ns;
                    let best_wins = ratio_best >= 1.0;
                    let worst_wins = ratio_worst >= 1.0;
                    let best_fill = ratio_to_fill(ratio_best);
                    let worst_fill = ratio_to_fill(ratio_worst);

                    bar = format!(r#"<div class="{fill_class}" style="width:{fill:.1}%"></div>"#);
                    if best_wins == worst_wins {
                        let side = if best_wins { "left-side" } else { "right-side" };
                        let (lo, hi) = if worst_fill < best_fill {
                            (worst_fill, best_fill)
                        } else {
                            (best_fill, worst_fill)
                        };
                        if best_wins {
                            write!(bar, r#"<div class="whisker-line {side}" style="right:calc(50% + {lo:.1}%);width:{:.1}%"></div>"#, hi - lo).unwrap();
                            write!(bar, r#"<div class="whisker {side}" style="right:calc(50% + {hi:.1}%)"></div>"#).unwrap();
                            write!(bar, r#"<div class="whisker {side}" style="right:calc(50% + {lo:.1}%)"></div>"#).unwrap();
                        } else {
                            write!(bar, r#"<div class="whisker-line {side}" style="left:calc(50% + {lo:.1}%);width:{:.1}%"></div>"#, hi - lo).unwrap();
                            write!(bar, r#"<div class="whisker {side}" style="left:calc(50% + {hi:.1}%)"></div>"#).unwrap();
                            write!(bar, r#"<div class="whisker {side}" style="left:calc(50% + {lo:.1}%)"></div>"#).unwrap();
                        }
                    } else if best_wins {
                        write!(bar, r#"<div class="whisker-line left-side" style="right:50%;width:{best_fill:.1}%"></div>"#).unwrap();
                        write!(bar, r#"<div class="whisker left-side" style="right:calc(50% + {best_fill:.1}%)"></div>"#).unwrap();
                        write!(bar, r#"<div class="whisker-line right-side" style="left:50%;width:{worst_fill:.1}%"></div>"#).unwrap();
                        write!(bar, r#"<div class="whisker right-side" style="left:calc(50% + {worst_fill:.1}%)"></div>"#).unwrap();
                    } else {
                        write!(bar, r#"<div class="whisker-line right-side" style="left:50%;width:{best_fill:.1}%"></div>"#).unwrap();
                        write!(bar, r#"<div class="whisker right-side" style="left:calc(50% + {best_fill:.1}%)"></div>"#).unwrap();
                        write!(bar, r#"<div class="whisker-line left-side" style="right:50%;width:{worst_fill:.1}%"></div>"#).unwrap();
                        write!(bar, r#"<div class="whisker left-side" style="right:calc(50% + {worst_fill:.1}%)"></div>"#).unwrap();
                    }
                }

                if bar.is_empty() {
                    bar.push_str(r#"<div class="delta-fill left-side" style="width:0%"></div>"#);
                }

                write!(
                    h,
                    r#"<div class="bench-row pair-row pair-{}"><span class="bname">{}</span><span class="ratio-col {}">{}</span><span class="t-left">{}</span><div class="delta-track">{}</div><span class="t-right">{}</span></div>"#,
                    esc(pair.id),
                    esc(&group.name),
                    ratio_cls,
                    ratio_text,
                    left_cell,
                    bar,
                    right_cell,
                )
                .unwrap();
            }
        }

        h.push_str("</div>\n");
    }

    if !vec_signals.is_empty() {
        h.push_str(r#"<section class="signals"><h2>Postcard Vec Scalar Signals</h2><table><thead><tr><th>benchmark</th><th>legacy frame</th><th>ir frame</th><th>legacy stack ops</th><th>ir stack ops</th><th>stack ratio</th><th>ir edits</th></tr></thead><tbody>"#);
        for row in vec_signals {
            let legacy_frame = row
                .legacy
                .frame_size_bytes
                .map(|v| format!("{v} B"))
                .unwrap_or_else(|| "n/a".to_string());
            let ir_frame = row
                .ir
                .frame_size_bytes
                .map(|v| format!("{v} B"))
                .unwrap_or_else(|| "n/a".to_string());
            let stack_ratio = if row.legacy.stack_mem_ops > 0 {
                format!(
                    "{:.2}x",
                    row.ir.stack_mem_ops as f64 / row.legacy.stack_mem_ops as f64
                )
            } else {
                "n/a".to_string()
            };
            let ir_edits = row
                .ir
                .regalloc_edits
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            write!(
                h,
                r#"<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>"#,
                esc(row.bench_name),
                esc(&legacy_frame),
                esc(&ir_frame),
                row.legacy.stack_mem_ops,
                row.ir.stack_mem_ops,
                esc(&stack_ratio),
                esc(&ir_edits),
            )
            .unwrap();
        }
        h.push_str("</tbody></table></section>\n");
    }

    h.push_str("<footer>");
    write!(h, "{} · ", esc(&meta.datetime)).unwrap();
    write!(
        h,
        r#"<a href="https://github.com/bearcove/kajit/commit/{}" style="color:var(--dim);text-decoration:none">{}</a>"#,
        esc(&meta.commit_full),
        esc(&meta.commit_short)
    )
    .unwrap();
    write!(h, " · ").unwrap();
    h.push_str(os_icon(&meta.os_name));
    write!(h, " {}", esc(&meta.platform)).unwrap();
    h.push_str("</footer>\n");

    write!(
        h,
        r#"<script>
const DEFAULT_TAB = "{}";
const DEFAULT_PAIR = "{}";

function parseHash() {{
  const out = {{}};
  const raw = window.location.hash.replace(/^#/, '');
  if (!raw) return out;
  for (const part of raw.split('&')) {{
    if (!part) continue;
    const [k, v] = part.split('=');
    out[decodeURIComponent(k)] = decodeURIComponent(v || '');
  }}
  return out;
}}

function setHashState(tab, pair) {{
  const hash = `#tab=${{encodeURIComponent(tab)}}&pair=${{encodeURIComponent(pair)}}`;
  if (window.location.hash !== hash) {{
    window.location.hash = hash;
  }} else {{
    applyState();
  }}
}}

function applyState() {{
  const state = parseHash();
  const pairSelect = document.getElementById('pair-select');
  const validPairs = Array.from(pairSelect.options).map(o => o.value);
  let pair = state.pair || DEFAULT_PAIR;
  if (!validPairs.includes(pair)) pair = DEFAULT_PAIR;
  pairSelect.value = pair;

  const tabs = Array.from(document.querySelectorAll('.tab'));
  const validTabs = tabs.map(t => t.dataset.tab);
  let tab = state.tab || DEFAULT_TAB;
  if (!validTabs.includes(tab)) tab = DEFAULT_TAB;

  tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + tab));
  document.querySelectorAll('.pair-row, .pair-summary').forEach(el => {{
    el.style.display = el.classList.contains(`pair-${{pair}}`) ? 'grid' : 'none';
  }});
}}

function switchTab(id) {{
  const pair = document.getElementById('pair-select').value || DEFAULT_PAIR;
  setHashState(id, pair);
}}

function onPairChange(pair) {{
  const activeTab = document.querySelector('.tab.active');
  const tab = activeTab ? activeTab.dataset.tab : DEFAULT_TAB;
  setHashState(tab, pair || DEFAULT_PAIR);
}}

window.addEventListener('hashchange', applyState);
window.addEventListener('DOMContentLoaded', applyState);
</script>
</body>
</html>
"#,
        esc(&default_tab_id),
        esc(default_pair_id),
    )
    .unwrap();

    h
}

/// Inline SVG icon for the OS.
fn os_icon(os_name: &str) -> &'static str {
    match os_name {
        "Darwin" => {
            r#"<svg style="vertical-align:-2px" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.8-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M13 3.5c.73-.83 1.94-1.46 2.94-1.5.13 1.17-.34 2.35-1.04 3.19-.69.85-1.83 1.51-2.95 1.42-.15-1.15.41-2.35 1.05-3.11z"/></svg>"#
        }
        "Linux" => {
            r#"<svg style="vertical-align:-2px" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12.504 0c-.155 0-.315.008-.48.021-4.226.333-3.105 4.807-3.17 6.298-.076 1.092-.3 1.953-1.05 3.02-.885 1.051-2.127 2.75-2.716 4.521-.278.832-.41 1.684-.287 2.489a.424.424 0 00-.11.135c-.26.268-.45.6-.663.839-.199.199-.485.267-.797.4-.313.136-.658.269-.864.68-.09.189-.136.394-.132.602 0 .199.027.4.055.536.058.399.116.728.04.97-.249.68-.28 1.145-.106 1.484.174.334.535.47.94.601.81.2 1.91.135 2.774.6.926.466 1.866.67 2.616.47.526-.116.97-.464 1.208-.946.587-.003 1.23-.269 2.26-.334.699-.058 1.574.267 2.577.2.025.134.063.198.114.333l.003.003c.391.778 1.113 1.345 1.884 1.345.358 0 .705-.094 1.053-.283.591-.32.974-.77 1.143-1.272.17-.505.138-1.12-.164-1.768-.096-.2-.238-.381-.394-.556-.104-.131-.196-.237-.238-.356a.723.723 0 01-.024-.344c.018-.174.107-.377.224-.578.202-.362.483-.775.67-1.264l.003-.007c.183-.471.298-1.03.248-1.694-.045-.6-.228-1.278-.583-2.029-.177-.375-.385-.739-.63-1.1-.24-.35-.512-.669-.792-.982-.095-.104-.185-.21-.262-.337-.087-.145-.16-.293-.21-.439.005-.009.009-.019.009-.03.24-.682.359-1.515.359-2.485 0-1.252-.37-2.293-.978-3.036-.598-.733-1.378-1.096-2.078-1.096-.4 0-.766.108-1.055.287-.29.178-.556.45-.768.816a12.262 12.262 0 00-.48.891 10.078 10.078 0 00-.162.381.25.25 0 00-.026.07c-.15.3-.315.505-.464.633-.165.138-.31.192-.449.208-.096.01-.248-.04-.425-.192a2.474 2.474 0 01-.398-.441c-.3-.398-.657-.963-1.047-1.467-.39-.502-.844-.973-1.413-1.28-.57-.307-1.165-.438-1.843-.434z"/></svg>"#
        }
        _ if os_name.contains("Windows")
            || os_name.contains("MINGW")
            || os_name.contains("MSYS") =>
        {
            r#"<svg style="vertical-align:-2px" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M0 3.449L9.75 2.1v9.451H0m10.949-9.602L24 0v11.4H10.949M0 12.6h9.75v9.451L0 20.699M10.949 12.6H24V24l-12.9-1.801"/></svg>"#
        }
        _ => "",
    }
}

/// Produce a CSS-safe ID from a section label (e.g., "json deser" → "json-deser").
fn tab_id(label: &str) -> String {
    label.replace(' ', "-")
}
