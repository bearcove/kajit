// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # lldb-sys
//!
//! [LLDB](https://lldb.llvm.org/) is a debugger from the
//! [LLVM](https://llvm.org/) project. It is usable as both a
//! standalone program as well as via an API.
//!
//! This crate provides a low-level, fairly direct mapping of the
//! [public C++ API](https://lldb.llvm.org/cpp_reference/namespacelldb.html)
//! for LLDB into unsafe Rust.
//!
//! It is primarily for use by a higher level crate, such as
//! [lldb.rs](https://crates.io/crates/lldb/).

mod lldb_sys;

pub use crate::lldb_sys::*;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
