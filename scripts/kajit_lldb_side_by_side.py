import os
import re

SECTION_RE = re.compile(r"^=== line (\d+) step (\d+) ===$")


def _load_sections():
    path = os.environ.get("KAJIT_LLDB_REF_FILE")
    if not path:
        return None, {}, []

    sections = {}
    ordered = []
    current_line = None
    current_lines = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            match = SECTION_RE.match(line)
            if match:
                current_line = int(match.group(1))
                current_lines = [line]
                continue
            if line == "=== end ===" and current_line is not None:
                current_lines.append(line)
                text = "\n".join(current_lines)
                sections[current_line] = text
                ordered.append(current_line)
                current_line = None
                current_lines = []
                continue
            if current_line is not None:
                current_lines.append(line)

    return path, sections, ordered


REF_PATH, REF_SECTIONS, REF_ORDER = _load_sections()


def _current_line(debugger):
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    if not process.IsValid():
        return None
    thread = process.GetSelectedThread()
    if not thread.IsValid():
        return None
    frame = thread.GetSelectedFrame()
    if not frame.IsValid():
        return None
    line_entry = frame.GetLineEntry()
    if not line_entry.IsValid():
        return None
    line = line_entry.GetLine()
    return line if line > 0 else None


def kajit_here(debugger, command, result, _internal_dict):
    if not REF_SECTIONS:
        result.PutCString(
            "kajit side-by-side reference is not loaded; set KAJIT_LLDB_REF_FILE before launching LLDB"
        )
        return

    raw = command.strip()
    line = int(raw) if raw else _current_line(debugger)
    if line is None:
        result.PutCString("no source line is selected")
        return

    section = REF_SECTIONS.get(line)
    if section is None:
        result.PutCString(f"no kajit interpreter reference for line {line}")
        return

    result.PutCString(section)


def kajit_list(_debugger, command, result, _internal_dict):
    if not REF_SECTIONS:
        result.PutCString(
            "kajit side-by-side reference is not loaded; set KAJIT_LLDB_REF_FILE before launching LLDB"
        )
        return

    raw = command.strip()
    if raw:
        line = int(raw)
        section = REF_SECTIONS.get(line)
        if section is None:
            result.PutCString(f"no kajit interpreter reference for line {line}")
        else:
            result.PutCString(section)
        return

    result.PutCString(
        "available kajit reference lines: " + " ".join(str(line) for line in REF_ORDER)
    )


def kajit_step(debugger, _command, result, _internal_dict):
    debugger.HandleCommand("thread step-over")
    line = _current_line(debugger)
    if line is None:
        result.PutCString("stepped, but no source line is selected")
        return
    section = REF_SECTIONS.get(line)
    if section is None:
        result.PutCString(f"stepped to line {line}; no kajit interpreter reference")
        return
    result.PutCString(section)


def kajit_help(_debugger, _command, result, _internal_dict):
    parts = [
        "kajit-break [regex] - set a JIT-code breakpoint (default regex: kajit::decode::)",
        "kajit-here [line]  - show interpreter reference for current or explicit CFG-MIR line",
        "kajit-list [line]  - list available reference lines or dump one explicit line",
        "kajit-step         - thread step-over, then print interpreter reference for the new line",
    ]
    if REF_PATH:
        parts.insert(0, f"reference file: {REF_PATH}")
    result.PutCString("\n".join(parts))


def kajit_break(debugger, command, result, _internal_dict):
    raw = command.strip()
    pattern = raw if raw else "kajit::decode::"
    escaped = pattern.replace("\\", "\\\\").replace('"', '\\"')
    debugger.HandleCommand(f'breakpoint set -r "{escaped}"')
    result.PutCString(f"set JIT breakpoint regex: {pattern}")


def __lldb_init_module(debugger, _internal_dict):
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_break kajit-break"
    )
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_here kajit-here"
    )
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_list kajit-list"
    )
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_step kajit-step"
    )
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_help kajit-help"
    )
