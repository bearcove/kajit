# Captaining

Captaining is a debugging and investigation workflow where one AI session (the captain) coordinates multiple independent AI agents (investigators) without doing the investigation work itself.

## The Problem It Solves

When a codebase has multiple independent failures, a single AI session will:
- Lose context as it context-switches between problems
- Make premature fixes based on incomplete understanding
- Burn context window on investigation that produces no code
- Serialize work that could be parallelized

Captaining addresses this by strict role separation.

## Roles

**Captain** (this session):
- Runs tests, collects failure output
- Clusters failures into groups by likely root cause
- Writes investigation prompts — precise, bounded, no-fix mandates
- Receives diagnoses, updates the tracking document
- Applies fixes once a diagnosis is confirmed
- Commits work in clean logical units

**Investigator** (spawned agents):
- Given one prompt, one question, one mandate: diagnose only, do not fix
- Has full codebase access but a scoped task
- Returns a diagnosis with file:line citations
- Does not need to understand the other groups

## What Makes a Good Investigation Prompt

1. **State the symptom exactly** — error message, test name, observed vs expected
2. **Give the hypothesis** — what you think might be wrong and why
3. **Name the diagnostic pattern** — e.g. "small values pass, large fail → multi-byte varint"
4. **Point at the likely area** — specific files, modules, function names
5. **List specific questions** — numbered, answerable, bounded
6. **Explicitly forbid fixing** — "do not fix anything, produce a diagnosis with file:line evidence"

The prompt should be self-contained: the investigator gets no other context.

## The Tracking Document

The captain maintains a document (e.g. `internal/x86-64-failures.md`) that:
- Groups failures by likely root cause (not by test name)
- Records the full error messages
- Records the diagnosis when it comes back
- Gets updated as groups are resolved or found moot
- Serves as the brief for each new investigator prompt

This document is the captain's working memory. It persists across sessions.

## Workflow

```
run tests --no-fail-fast
    │
    ▼
cluster failures by root cause
    │
    ▼
write tracking doc with groups + error messages
    │
    ▼
for each group:
    write investigation prompt
    spawn investigator agent (can parallelize)
    │
    ▼
receive diagnosis
    │
    ├─ diagnosis clear → apply fix → run tests → commit
    │
    ├─ diagnosis unclear → write follow-up prompt → respawn
    │
    └─ group moot (e.g. test deleted) → mark moot in doc
```

## Key Disciplines

**No premature fixing.** The captain does not touch code until a diagnosis comes back with file:line evidence. Guessing at fixes without understanding the root cause produces noise commits and masks real problems.

**Diagnose before clustering.** Two tests with identical error messages may have different root causes. Look for structural patterns (which versions fail, what values, what types) before grouping.

**Commit in logical units.** Cleanup, fixes, and doc updates are separate commits. A fix commit message explains the mechanism, not just the symptom.

**Investigators are disposable.** If a diagnosis comes back incomplete or wrong, write a sharper prompt and respawn. Don't try to correct an investigator mid-flight.

**The tracking doc is always current.** Every resolved or moot group gets updated immediately. The doc reflects reality, not aspirations.

## What Captaining Is Not

- It is not project management theater. The tracking doc exists to brief investigators and record findings, not to satisfy a process.
- It is not a way to avoid reading code. The captain reads enough to cluster failures and write good prompts. Investigators read deeply.
- It is not sequential. Multiple investigators can run in parallel on independent groups.
