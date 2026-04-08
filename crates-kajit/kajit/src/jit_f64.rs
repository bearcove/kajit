// r[impl deser.json.scalar.float.uscale.table]
//! Constants and table pointer for the JIT f64 parser.
//!
//! The uscale algorithm converts `d * 10^p` to IEEE 754 f64 via precomputed
//! 128-bit powers of five. This module re-exports the table and provides
//! compile-time constants that the aarch64/x86_64 emitters bake into code.

use crate::pow10tab::{POW10_MAX, POW10_MIN, POW10_TAB};

/// Base address of the power-of-five table. Loaded as a 64-bit immediate
/// at JIT-compile time.
pub fn pow10_tab_ptr() -> u64 {
    POW10_TAB.as_ptr() as u64
}

/// Minimum power-of-10 exponent in the table (-348).
pub const P_MIN: i32 = POW10_MIN;

/// Maximum power-of-10 exponent in the table (347).
pub const P_MAX: i32 = POW10_MAX;

/// Each table entry is `(u64, u64)` = 16 bytes.
pub const ENTRY_SIZE: u32 = 16;

/// Offset to subtract from `p` to get the table index: `idx = p - P_MIN`.
/// Since P_MIN is -348, this is equivalent to `idx = p + 348`.
pub const P_OFFSET: i32 = -P_MIN; // 348

/// log₂(10) ≈ 108853/2^15. Used in `(p * 108853) >> 15`.
pub const LOG2_10_NUM: i32 = 108853;
pub const LOG2_10_SHIFT: u32 = 15;

/// Maximum significant digits that fit in a u64.
pub const MAX_SIG_DIGITS: u32 = 19;

/// Threshold for the overflow check in uscale: `unmin(2^53) = (2^53 << 2) - 2`.
pub const UNMIN_2_53: u64 = (1u64 << 55) - 2;

/// IEEE 754 constants.
pub const F64_SIGN_BIT: u64 = 1u64 << 63;
pub const F64_EXP_BIAS: i32 = 1075;
pub const F64_INF_BITS: u64 = 0x7FF0_0000_0000_0000;
pub const F64_MANTISSA_BIT: u64 = 1u64 << 52;
