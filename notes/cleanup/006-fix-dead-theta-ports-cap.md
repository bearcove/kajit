# Fix dead_theta_ports 3-port cap

**Phase:** 2

The bug is in port index shifting during multi-port removal. Currently capped at 3 ports to avoid it.

1. Write a text test with 5 dead ports
2. Fix the indexing
3. Remove the cap
