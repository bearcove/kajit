# Make state domains frontend-declared

Remove the hardcoded `CURSOR_STATE_DOMAIN = StateDomainId(0)` and let frontends declare their own domains.

## Current state

- `kajit-ir/src/ir.rs:179` — `pub const CURSOR_STATE_DOMAIN: StateDomainId = StateDomainId::new(0)`
- `CURSOR_STATE_DOMAIN_NAME`, `CURSOR_STATE_PORT`, `CURSOR_EFFECT` constants
- `IrFunc::builtin_state_domains()` always creates cursor as domain 0
- `verify.rs` requires CURSOR_STATE_DOMAIN to exist
- All cursor op builders (`read_bytes`, `peek_byte`, etc.) use `CURSOR_STATE_DOMAIN`

## Target state

- `IrFunc` has no built-in state domains — the frontend adds them via `func.add_state_domain("cursor")`
- The postcard frontend declares cursor + output domains
- Other frontends can declare whatever domains they need (or none)
- The compiler treats all state domains uniformly

## What changes

- Delete `CURSOR_STATE_DOMAIN`, `CURSOR_STATE_PORT`, `CURSOR_EFFECT` constants
- Make `builtin_state_domains()` return empty or remove it
- Cursor op builders take a `StateDomainId` parameter instead of hardcoding domain 0
- Update `verify.rs` to not require cursor domain
- Update all call sites that reference these constants

## Risk

Low — this is a naming/API change, not a semantic change. The cursor domain still exists, it's just declared by the frontend instead of the IR layer.
