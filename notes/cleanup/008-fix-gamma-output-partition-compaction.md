# Fix gamma_output_partition compaction

**Phase:** 2

`gamma_output_partition` doesn't compact outputs. There's a regalloc assumption preventing it.

1. Write a text test where compaction is needed
2. Fix the regalloc assumption
3. Enable compaction
