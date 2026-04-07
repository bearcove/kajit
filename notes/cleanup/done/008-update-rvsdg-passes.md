# Delete CURSOR_STATE_DOMAIN

After 007, no ops use CURSOR_STATE_DOMAIN. Remove it.

## What to delete

- `CURSOR_STATE_DOMAIN` constant (`kajit-ir/src/ir.rs:179`)
- `CURSOR_STATE_DOMAIN_NAME` constant
- `CURSOR_STATE_PORT` constant  
- `CURSOR_EFFECT` constant
- Remove cursor domain from `builtin_state_domains()` — it no longer creates domain 0
- Renumber: OUTPUT becomes domain 0, MEMORY becomes domain 1 (or just let the frontend add them)
- Update `verify.rs` to not require cursor domain
- Update all IR text tests that reference `%cs` (cursor state) args

## Risk

Low — mechanical deletion. The domain has no users after 007.

## Depends on

007 (cursor ops replaced with memory ops).

## Enables

009 (merge output into memory).
