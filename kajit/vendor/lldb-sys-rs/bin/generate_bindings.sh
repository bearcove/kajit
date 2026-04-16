#! /bin/sh

bindgen --match SB --output src/lldb_sys.rs.in src/lldb/Bindings/LLDBBinding.h -- -Isrc -DBINDGEN
sed -f bin/rename_enums.sed < src/lldb_sys.rs.in > src/lldb_sys.rs
rm src/lldb_sys.rs.in
