# Fix post_unroll_canonicalize 800-node cap

**Phase:** 2

Currently capped at 800 nodes to avoid breakage.

1. Write a text test with 900 nodes
2. Understand what breaks
3. Fix it, remove the cap
