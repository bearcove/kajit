# Clean up const_fold

**Phase:** 1

The existing `const_fold` is A- quality — keep it. But clean up the duplicate constant resolver. There are currently three constant resolvers; this pass should use one shared one.
