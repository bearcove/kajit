import lldb
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


def _current_frame(debugger):
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
    return frame


def _variable_state(debugger, name):
    frame = _current_frame(debugger)
    if frame is None:
        return "missing-frame", None
    value = frame.FindVariable(name)
    if value is None or not value.IsValid():
        return "missing", None
    if hasattr(value, "IsInScope") and not value.IsInScope():
        return "unavailable", None
    rendered = value.GetValue()
    if rendered is None:
        rendered = value.GetSummary()
    if rendered is None and value.GetNumChildren() == 0:
        return "unavailable", None
    return "available", rendered


def _listed_variable_names(debugger):
    frame = _current_frame(debugger)
    if frame is None:
        return None, "missing-frame"
    values = frame.GetVariables(False, True, False, True)
    names = []
    for index in range(values.GetSize()):
        value = values.GetValueAtIndex(index)
        if value is None or not value.IsValid():
            continue
        name = value.GetName()
        if name:
            names.append(name)
    return names, None


def _step_to_line(debugger, target_line):
    current = _current_line(debugger)
    if current is None:
        return None, "no source line is selected"
    if current > target_line:
        return current, f"current line {current} is already past requested line {target_line}"
    for _ in range(256):
        current = _current_line(debugger)
        if current == target_line:
            return current, None
        if current is not None and current > target_line:
            return current, f"stepped past requested line {target_line}, now at {current}"
        debugger.HandleCommand("thread step-over")
        current = _current_line(debugger)
        if current == target_line:
            return current, None
    return _current_line(debugger), f"did not reach line {target_line} within step budget"


def _evaluate_u64(debugger, expr_text):
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    if not process.IsValid():
        return None, "no process"
    thread = process.GetSelectedThread()
    if not thread.IsValid():
        return None, "no thread"
    frame = thread.GetSelectedFrame()
    if not frame.IsValid():
        return None, "no frame"
    value = frame.EvaluateExpression(expr_text)
    if value is None or not value.IsValid() or value.GetError().Fail():
        err = value.GetError().GetCString() if value is not None and value.IsValid() else None
        return None, err or f"failed to evaluate expression: {expr_text}"
    unsigned = value.GetValueAsUnsigned()
    return unsigned, None


def _read_bytes(debugger, address, count):
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    error = lldb.SBError()
    data = process.ReadMemory(address, count, error)
    if error.Fail():
        return None, error.GetCString() or f"failed to read memory at 0x{address:x}"
    return data, None


def _format_ascii(data):
    chars = []
    for byte in data:
        if 32 <= byte <= 126:
            chars.append(chr(byte))
        else:
            chars.append(".")
    return "".join(chars)


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
        "kajit-bytes [count] - read bytes at input_ptr as hex + ASCII (default: 16)",
        "kajit-text [count]  - read bytes at input_ptr as printable text (default: 32)",
        "kajit-break [regex] - set a JIT-code breakpoint (default regex: kajit::decode::)",
        "kajit-step-to <line> - step over until the current CFG-MIR line matches",
        "kajit-var-state <var> - print whether one variable is available in the current frame",
        "kajit-listed-vars - print local variable names visible in the current lexical scope",
        "kajit-expect <line> <var> <available|unavailable|listed|unlisted> - step, check, and print OK/FAIL",
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


def kajit_step_to(debugger, command, result, _internal_dict):
    raw = command.strip()
    if not raw:
        result.PutCString("usage: kajit-step-to <line>")
        return
    target_line = int(raw)
    line, err = _step_to_line(debugger, target_line)
    if err is not None:
        result.PutCString(f"FAIL line={target_line} error={err}")
        return
    result.PutCString(f"OK line={line}")


def kajit_var_state(debugger, command, result, _internal_dict):
    name = command.strip()
    if not name:
        result.PutCString("usage: kajit-var-state <var>")
        return
    state, rendered = _variable_state(debugger, name)
    line = _current_line(debugger)
    if rendered is None:
        result.PutCString(f"VAR line={line} name={name} state={state}")
    else:
        result.PutCString(f"VAR line={line} name={name} state={state} value={rendered}")


def kajit_listed_vars(debugger, _command, result, _internal_dict):
    names, err = _listed_variable_names(debugger)
    if err is not None:
        result.PutCString(f"FAIL error={err}")
        return
    line = _current_line(debugger)
    result.PutCString(f"LISTED line={line} names={' '.join(names)}")


def kajit_expect(debugger, command, result, _internal_dict):
    parts = command.strip().split()
    if len(parts) != 3:
        result.PutCString("usage: kajit-expect <line> <var> <available|unavailable|listed|unlisted>")
        return
    target_line = int(parts[0])
    name = parts[1]
    expected = parts[2]
    line, err = _step_to_line(debugger, target_line)
    if err is not None:
        result.PutCString(f"FAIL line={target_line} name={name} error={err}")
        return
    if expected in ("listed", "unlisted"):
        names, list_err = _listed_variable_names(debugger)
        if list_err is not None:
            result.PutCString(f"FAIL line={line} name={name} error={list_err}")
            return
        actual = "listed" if name in names else "unlisted"
        rendered = None
    else:
        actual, rendered = _variable_state(debugger, name)
    if actual == expected:
        if rendered is None:
            result.PutCString(f"OK line={line} name={name} state={actual}")
        else:
            result.PutCString(f"OK line={line} name={name} state={actual} value={rendered}")
        return
    if rendered is None:
        result.PutCString(
            f"FAIL line={line} name={name} expected={expected} actual={actual}"
        )
    else:
        result.PutCString(
            f"FAIL line={line} name={name} expected={expected} actual={actual} value={rendered}"
        )


def kajit_bytes(debugger, command, result, _internal_dict):
    raw = command.strip()
    count = int(raw) if raw else 16
    if count <= 0:
        result.PutCString("count must be positive")
        return

    input_ptr, err = _evaluate_u64(debugger, "input_ptr")
    if err is not None:
        result.PutCString(f"could not read input_ptr: {err}")
        return
    input_end, err = _evaluate_u64(debugger, "input_end")
    if err is not None:
        result.PutCString(f"could not read input_end: {err}")
        return

    available = max(0, input_end - input_ptr)
    count = min(count, available)
    data, err = _read_bytes(debugger, input_ptr, count)
    if err is not None:
        result.PutCString(err)
        return

    hex_bytes = " ".join(f"{byte:02x}" for byte in data)
    ascii_bytes = _format_ascii(data)
    result.PutCString(
        "\n".join(
            [
                f"input_ptr=0x{input_ptr:x} input_end=0x{input_end:x} remaining={available}",
                f"hex:   {hex_bytes}",
                f"ascii: {ascii_bytes}",
            ]
        )
    )


def kajit_text(debugger, command, result, _internal_dict):
    raw = command.strip()
    count = int(raw) if raw else 32
    if count <= 0:
        result.PutCString("count must be positive")
        return

    input_ptr, err = _evaluate_u64(debugger, "input_ptr")
    if err is not None:
        result.PutCString(f"could not read input_ptr: {err}")
        return
    input_end, err = _evaluate_u64(debugger, "input_end")
    if err is not None:
        result.PutCString(f"could not read input_end: {err}")
        return

    available = max(0, input_end - input_ptr)
    count = min(count, available)
    data, err = _read_bytes(debugger, input_ptr, count)
    if err is not None:
        result.PutCString(err)
        return

    escaped = data.decode("utf-8", errors="backslashreplace")
    result.PutCString(
        "\n".join(
            [
                f"input_ptr=0x{input_ptr:x} input_end=0x{input_end:x} remaining={available}",
                f"text: {escaped}",
            ]
        )
    )


def __lldb_init_module(debugger, _internal_dict):
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_bytes kajit-bytes"
    )
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_text kajit-text"
    )
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_break kajit-break"
    )
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_step_to kajit-step-to"
    )
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_var_state kajit-var-state"
    )
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_listed_vars kajit-listed-vars"
    )
    debugger.HandleCommand(
        "command script add -f kajit_lldb_side_by_side.kajit_expect kajit-expect"
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
