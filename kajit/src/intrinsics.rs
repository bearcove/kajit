use std::borrow::Cow;

use facet::{MapFromPairSliceFn, PtrUninit};

use crate::context::{DeserContext, ErrorCode};

// r[impl callconv.intrinsics]

// --- Varint helpers ---

/// Read a LEB128-encoded unsigned varint, returning up to 64 bits.
/// Sets error on malformed varint or EOF.
unsafe fn read_varint_u64(ctx: &mut DeserContext) -> u64 {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;

    loop {
        if ctx.input_ptr >= ctx.input_end {
            ctx.error.code = ErrorCode::UnexpectedEof as u32;
            return 0;
        }

        let byte = unsafe { *ctx.input_ptr };
        ctx.input_ptr = unsafe { ctx.input_ptr.add(1) };

        let value_bits = (byte & 0x7F) as u64;

        if shift >= 63 && (byte & 0x7E) != 0 {
            ctx.error.code = ErrorCode::InvalidVarint as u32;
            return 0;
        }

        result |= value_bits << shift;
        shift += 7;

        if byte & 0x80 == 0 {
            return result;
        }

        if shift >= 70 {
            ctx.error.code = ErrorCode::InvalidVarint as u32;
            return 0;
        }
    }
}

/// Read a LEB128-encoded unsigned varint, returning up to 128 bits.
/// Sets error on malformed varint or EOF.
unsafe fn read_varint_u128(ctx: &mut DeserContext) -> u128 {
    let mut result: u128 = 0;
    let mut shift: u32 = 0;

    loop {
        if ctx.input_ptr >= ctx.input_end {
            ctx.error.code = ErrorCode::UnexpectedEof as u32;
            return 0;
        }

        let byte = unsafe { *ctx.input_ptr };
        ctx.input_ptr = unsafe { ctx.input_ptr.add(1) };

        let value_bits = (byte & 0x7F) as u128;

        if shift >= 127 && (byte & 0x7E) != 0 {
            ctx.error.code = ErrorCode::InvalidVarint as u32;
            return 0;
        }

        result |= value_bits << shift;
        shift += 7;

        if byte & 0x80 == 0 {
            return result;
        }

        if shift >= 140 {
            ctx.error.code = ErrorCode::InvalidVarint as u32;
            return 0;
        }
    }
}

/// Decode a ZigZag-encoded i64 from a u64.
fn zigzag_decode(encoded: u64) -> i64 {
    ((encoded >> 1) as i64) ^ -((encoded & 1) as i64)
}

/// Decode a ZigZag-encoded i128 from a u128.
fn zigzag_decode_i128(encoded: u128) -> i128 {
    ((encoded >> 1) as i128) ^ -((encoded & 1) as i128)
}

// --- Unsigned integer intrinsics ---

// r[impl deser.postcard.scalar.bool]

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to a `bool`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_bool(ctx: *mut DeserContext, out: *mut bool) {
    let ctx = unsafe { &mut *ctx };
    if ctx.input_ptr >= ctx.input_end {
        ctx.error.code = ErrorCode::UnexpectedEof as u32;
        return;
    }
    let byte = unsafe { *ctx.input_ptr };
    ctx.input_ptr = unsafe { ctx.input_ptr.add(1) };
    match byte {
        0 => unsafe { *out = false },
        1 => unsafe { *out = true },
        _ => ctx.error.code = ErrorCode::InvalidBool as u32,
    }
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to a `u8`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_u8(ctx: *mut DeserContext, out: *mut u8) {
    let ctx = unsafe { &mut *ctx };
    if ctx.input_ptr >= ctx.input_end {
        ctx.error.code = ErrorCode::UnexpectedEof as u32;
        return;
    }
    let byte = unsafe { *ctx.input_ptr };
    ctx.input_ptr = unsafe { ctx.input_ptr.add(1) };
    unsafe { *out = byte };
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to a `u16`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_u16(ctx: *mut DeserContext, out: *mut u16) {
    let ctx = unsafe { &mut *ctx };
    let val = unsafe { read_varint_u64(ctx) };
    if ctx.error.code != 0 {
        return;
    }
    if val > u16::MAX as u64 {
        ctx.error.code = ErrorCode::NumberOutOfRange as u32;
        return;
    }
    unsafe { *out = val as u16 };
}

// r[impl deser.postcard.scalar.varint]

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to a `u32`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_varint_u32(ctx: *mut DeserContext, out: *mut u32) {
    let ctx = unsafe { &mut *ctx };
    let val = unsafe { read_varint_u64(ctx) };
    if ctx.error.code != 0 {
        return;
    }
    if val > u32::MAX as u64 {
        ctx.error.code = ErrorCode::NumberOutOfRange as u32;
        return;
    }
    unsafe { *out = val as u32 };
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to a `u64`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_u64(ctx: *mut DeserContext, out: *mut u64) {
    let ctx = unsafe { &mut *ctx };
    let val = unsafe { read_varint_u64(ctx) };
    if ctx.error.code != 0 {
        return;
    }
    unsafe { *out = val };
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to a `u128`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_u128(ctx: *mut DeserContext, out: *mut u128) {
    let ctx = unsafe { &mut *ctx };
    let val = unsafe { read_varint_u128(ctx) };
    if ctx.error.code != 0 {
        return;
    }
    unsafe { *out = val };
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to a `usize`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_usize(ctx: *mut DeserContext, out: *mut usize) {
    let ctx = unsafe { &mut *ctx };
    let val = unsafe { read_varint_u64(ctx) };
    if ctx.error.code != 0 {
        return;
    }
    let v = match usize::try_from(val) {
        Ok(v) => v,
        Err(_) => {
            ctx.error.code = ErrorCode::NumberOutOfRange as u32;
            return;
        }
    };
    unsafe { *out = v };
}

// --- Signed integer intrinsics ---
// i8: raw byte in two's complement (per postcard spec)
// i16/i32/i64: ZigZag + varint

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to an `i8`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_i8(ctx: *mut DeserContext, out: *mut i8) {
    let ctx = unsafe { &mut *ctx };
    if ctx.input_ptr >= ctx.input_end {
        ctx.error.code = ErrorCode::UnexpectedEof as u32;
        return;
    }
    let byte = unsafe { *ctx.input_ptr };
    ctx.input_ptr = unsafe { ctx.input_ptr.add(1) };
    unsafe { *out = byte as i8 };
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to an `i16`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_i16(ctx: *mut DeserContext, out: *mut i16) {
    let ctx = unsafe { &mut *ctx };
    let val = unsafe { read_varint_u64(ctx) };
    if ctx.error.code != 0 {
        return;
    }
    let decoded = zigzag_decode(val);
    if decoded < i16::MIN as i64 || decoded > i16::MAX as i64 {
        ctx.error.code = ErrorCode::NumberOutOfRange as u32;
        return;
    }
    unsafe { *out = decoded as i16 };
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to an `i32`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_i32(ctx: *mut DeserContext, out: *mut i32) {
    let ctx = unsafe { &mut *ctx };
    let val = unsafe { read_varint_u64(ctx) };
    if ctx.error.code != 0 {
        return;
    }
    let decoded = zigzag_decode(val);
    if decoded < i32::MIN as i64 || decoded > i32::MAX as i64 {
        ctx.error.code = ErrorCode::NumberOutOfRange as u32;
        return;
    }
    unsafe { *out = decoded as i32 };
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to an `i64`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_i64(ctx: *mut DeserContext, out: *mut i64) {
    let ctx = unsafe { &mut *ctx };
    let val = unsafe { read_varint_u64(ctx) };
    if ctx.error.code != 0 {
        return;
    }
    unsafe { *out = zigzag_decode(val) };
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to an `i128`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_i128(ctx: *mut DeserContext, out: *mut i128) {
    let ctx = unsafe { &mut *ctx };
    let val = unsafe { read_varint_u128(ctx) };
    if ctx.error.code != 0 {
        return;
    }
    unsafe { *out = zigzag_decode_i128(val) };
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to an `isize`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_isize(ctx: *mut DeserContext, out: *mut isize) {
    let ctx = unsafe { &mut *ctx };
    let val = unsafe { read_varint_u64(ctx) };
    if ctx.error.code != 0 {
        return;
    }
    let decoded = zigzag_decode(val);
    let v = match isize::try_from(decoded) {
        Ok(v) => v,
        Err(_) => {
            ctx.error.code = ErrorCode::NumberOutOfRange as u32;
            return;
        }
    };
    unsafe { *out = v };
}

// --- Float intrinsics (little-endian IEEE 754) ---

// r[impl deser.postcard.scalar.float]

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to an `f32`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_f32(ctx: *mut DeserContext, out: *mut f32) {
    let ctx = unsafe { &mut *ctx };
    let remaining = unsafe { ctx.input_end.offset_from(ctx.input_ptr) as usize };
    if remaining < 4 {
        ctx.error.code = ErrorCode::UnexpectedEof as u32;
        return;
    }
    let bytes: [u8; 4] = unsafe { core::ptr::read_unaligned(ctx.input_ptr as *const [u8; 4]) };
    ctx.input_ptr = unsafe { ctx.input_ptr.add(4) };
    unsafe { *out = f32::from_le_bytes(bytes) };
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to an `f64`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_f64(ctx: *mut DeserContext, out: *mut f64) {
    let ctx = unsafe { &mut *ctx };
    let remaining = unsafe { ctx.input_end.offset_from(ctx.input_ptr) as usize };
    if remaining < 8 {
        ctx.error.code = ErrorCode::UnexpectedEof as u32;
        return;
    }
    let bytes: [u8; 8] = unsafe { core::ptr::read_unaligned(ctx.input_ptr as *const [u8; 8]) };
    ctx.input_ptr = unsafe { ctx.input_ptr.add(8) };
    unsafe { *out = f64::from_le_bytes(bytes) };
}

// --- String intrinsic ---

/// Read a postcard length-prefixed string from the input, write to `*out`.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to uninitialized `String` memory
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_postcard_string(ctx: *mut DeserContext, out: *mut String) {
    let ctx = unsafe { &mut *ctx };
    let s = match unsafe { read_postcard_string_borrowed(ctx) } {
        Some(s) => s,
        None => return,
    };
    unsafe { out.write(s.to_owned()) };
}

unsafe fn read_postcard_string_borrowed(ctx: &mut DeserContext) -> Option<&str> {
    let mut len: u32 = 0;
    unsafe { kajit_read_varint_u32(ctx as *mut _, &mut len) };
    if ctx.error.code != 0 {
        return None;
    }
    unsafe { read_postcard_string_borrowed_with_len(ctx, len) }
}

unsafe fn read_postcard_string_borrowed_with_len(
    ctx: &mut DeserContext,
    len_u32: u32,
) -> Option<&str> {
    let len = len_u32 as usize;

    let remaining = unsafe { ctx.input_end.offset_from(ctx.input_ptr) as usize };
    if remaining < len {
        ctx.error.code = ErrorCode::UnexpectedEof as u32;
        return None;
    }

    let bytes = unsafe { core::slice::from_raw_parts(ctx.input_ptr, len) };
    let s = if ctx.trusted_utf8 {
        // SAFETY: trusted mode is enabled only when the caller opted into
        // pre-validated UTF-8 input for a compatible format.
        unsafe { core::str::from_utf8_unchecked(bytes) }
    } else {
        match core::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => {
                ctx.error.code = ErrorCode::InvalidUtf8 as u32;
                return None;
            }
        }
    };

    ctx.input_ptr = unsafe { ctx.input_ptr.add(len) };
    Some(s)
}

/// Read a postcard string with pre-decoded length and write to `*out`.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to uninitialized `String` memory
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_postcard_string_with_len(
    ctx: *mut DeserContext,
    len: u32,
    out: *mut String,
) {
    let ctx = unsafe { &mut *ctx };
    let s = match unsafe { read_postcard_string_borrowed_with_len(ctx, len) } {
        Some(s) => s,
        None => return,
    };
    unsafe { out.write(s.to_owned()) };
}

/// Read a postcard string with pre-decoded length and write as borrowed `&str`.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to uninitialized `&str` memory
/// - The borrowed slice is only valid for the lifetime of the input buffer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_postcard_str_with_len(
    ctx: *mut DeserContext,
    len: u32,
    out: *mut &str,
) {
    let ctx = unsafe { &mut *ctx };
    let s = match unsafe { read_postcard_string_borrowed_with_len(ctx, len) } {
        Some(s) => s,
        None => return,
    };
    let s_static: &'static str = unsafe { core::mem::transmute::<&str, &'static str>(s) };
    unsafe { out.write(s_static) };
}

/// Read a postcard string with pre-decoded length and write as `Cow<str>`.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to uninitialized `Cow<'static, str>` memory
/// - The borrowed slice is only valid for the lifetime of the input buffer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_postcard_cow_str_with_len(
    ctx: *mut DeserContext,
    len: u32,
    out: *mut Cow<'static, str>,
) {
    let ctx = unsafe { &mut *ctx };
    let s = match unsafe { read_postcard_string_borrowed_with_len(ctx, len) } {
        Some(s) => s,
        None => return,
    };
    let s_static: &'static str = unsafe { core::mem::transmute::<&str, &'static str>(s) };
    unsafe { out.write(Cow::Borrowed(s_static)) };
}

/// Read a postcard string and write it as a borrowed `&str`.
///
/// This is always zero-copy: postcard strings are length-prefixed UTF-8 and
/// require no unescaping.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to uninitialized `&str` memory
/// - The borrowed slice is only valid for the lifetime of the input buffer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_postcard_str(ctx: *mut DeserContext, out: *mut &str) {
    let ctx = unsafe { &mut *ctx };
    let s = match unsafe { read_postcard_string_borrowed(ctx) } {
        Some(s) => s,
        None => return,
    };
    let s_static: &'static str = unsafe { core::mem::transmute::<&str, &'static str>(s) };
    unsafe { out.write(s_static) };
}

/// Read a postcard string and write it as `Cow<str>`.
///
/// Postcard strings are always borrowable (no escape sequences), so this always
/// produces `Cow::Borrowed`.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to uninitialized `Cow<'static, str>` memory
/// - The borrowed slice is only valid for the lifetime of the input buffer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_postcard_cow_str(
    ctx: *mut DeserContext,
    out: *mut Cow<'static, str>,
) {
    let ctx = unsafe { &mut *ctx };
    let s = match unsafe { read_postcard_string_borrowed(ctx) } {
        Some(s) => s,
        None => return,
    };
    let s_static: &'static str = unsafe { core::mem::transmute::<&str, &'static str>(s) };
    unsafe { out.write(Cow::Borrowed(s_static)) };
}

/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to a `char`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_read_char(ctx: *mut DeserContext, out: *mut char) {
    let mut len: u32 = 0;
    unsafe { kajit_read_varint_u32(ctx, &mut len) };
    let ctx = unsafe { &mut *ctx };
    if ctx.error.code != 0 {
        return;
    }
    let len = len as usize;
    if len == 0 || len > 4 {
        ctx.error.code = ErrorCode::InvalidUtf8 as u32;
        return;
    }
    let remaining = unsafe { ctx.input_end.offset_from(ctx.input_ptr) as usize };
    if remaining < len {
        ctx.error.code = ErrorCode::UnexpectedEof as u32;
        return;
    }
    let bytes = unsafe { core::slice::from_raw_parts(ctx.input_ptr, len) };
    let s = match core::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => {
            ctx.error.code = ErrorCode::InvalidUtf8 as u32;
            return;
        }
    };
    let mut chars = s.chars();
    let ch = match chars.next() {
        Some(ch) => ch,
        None => {
            ctx.error.code = ErrorCode::InvalidUtf8 as u32;
            return;
        }
    };
    if chars.next().is_some() {
        ctx.error.code = ErrorCode::InvalidUtf8 as u32;
        return;
    }
    unsafe { *out = ch };
    ctx.input_ptr = unsafe { ctx.input_ptr.add(len) };
}

// --- Option intrinsics ---

/// Initialize an Option with None using the vtable's init_none function.
///
/// Wraps the facet OptionVTable's init_none, which takes wide pointer types
/// (PtrUninit), into a thin `extern "C"` interface callable from JIT code.
///
/// # Safety
///
/// - `init_none_fn` must be a valid `OptionInitNoneFn` for the target Option type
/// - `out` must be a valid, aligned, non-null pointer to uninitialized memory
///   sized and aligned for the target `Option<T>`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_option_init_none(
    init_none_fn: facet::OptionInitNoneFn,
    out: *mut u8,
) {
    let ptr_uninit = facet::PtrUninit::new_sized(out);
    unsafe { (init_none_fn)(ptr_uninit) };
}

/// Initialize an Option with Some(value) using the vtable's init_some function.
///
/// `value_ptr` points to an already-deserialized T. init_some will _move_ it
/// (read + write into the Option), so the caller must not use value_ptr afterwards.
///
/// # Safety
///
/// - `init_some_fn` must be a valid `OptionInitSomeFn` for the target Option type
/// - `out` must be a valid, aligned, non-null pointer to uninitialized memory
///   sized and aligned for the target `Option<T>`
/// - `value_ptr` must be a valid, aligned, non-null pointer to an initialized `T`;
///   it is consumed (moved) and must not be used after this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_option_init_some(
    init_some_fn: facet::OptionInitSomeFn,
    out: *mut u8,
    value_ptr: *mut u8,
) {
    let ptr_uninit = facet::PtrUninit::new_sized(out);
    let ptr_mut = facet::PtrMut::new_sized(value_ptr);
    unsafe { (init_some_fn)(ptr_uninit, ptr_mut) };
}

/// IR-callable wrapper for Option::None init.
///
/// ABI shape matches linear IR side-effect intrinsic calls:
/// `fn(ctx, arg0, out)`.
///
/// # Safety
///
/// Same requirements as [`kajit_option_init_none`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_option_init_none_ctx(
    _ctx: *mut DeserContext,
    init_none_fn: facet::OptionInitNoneFn,
    out: *mut u8,
) {
    unsafe { kajit_option_init_none(init_none_fn, out) };
}

/// IR-callable wrapper for Option::Some init.
///
/// ABI shape matches linear IR side-effect intrinsic calls:
/// `fn(ctx, arg0, arg1, out)`.
///
/// # Safety
///
/// Same requirements as [`kajit_option_init_some`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_option_init_some_ctx(
    _ctx: *mut DeserContext,
    init_some_fn: facet::OptionInitSomeFn,
    value_ptr: *mut u8,
    out: *mut u8,
) {
    unsafe { kajit_option_init_some(init_some_fn, out, value_ptr) };
}

// r[impl deser.pointer.new-into]

/// Wrap an already-deserialized T into a smart pointer (Box, Arc, Rc) using the
/// vtable's `new_into_fn`.
///
/// Same ABI shape as `kajit_option_init_some`: bridges thin raw pointers from JIT
/// code to facet's wide pointer types (PtrUninit, PtrMut).
///
/// `value_ptr` points to an already-deserialized T. new_into_fn will _move_ it
/// (read + write into the pointer), so the caller must not use value_ptr afterwards.
///
/// # Safety
///
/// - `new_into_fn` must be a valid `NewIntoFn` for the target smart pointer type
/// - `out` must be a valid, aligned, non-null pointer to uninitialized memory
///   sized and aligned for the target smart pointer (Box/Arc/Rc)
/// - `value_ptr` must be a valid, aligned, non-null pointer to an initialized `T`;
///   it is consumed (moved) and must not be used after this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_pointer_new_into(
    new_into_fn: facet::NewIntoFn,
    out: *mut u8,
    value_ptr: *mut u8,
) {
    let ptr_uninit = facet::PtrUninit::new_sized(out);
    let ptr_mut = facet::PtrMut::new_sized(value_ptr);
    unsafe { (new_into_fn)(ptr_uninit, ptr_mut) };
}

/// Validate UTF-8 and allocate a String from a raw byte slice, write to `*out`.
///
/// This is the "lean" string intrinsic — it does NOT read the length varint,
/// bounds check the input, or advance the cursor. The JIT inlines those parts.
/// This intrinsic only handles the work that can't be inlined: UTF-8 validation
/// and heap allocation.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `out` must be a valid, aligned, non-null pointer to uninitialized `String` memory
/// - `data_ptr` must point to at least `data_len` readable bytes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_postcard_validate_and_alloc_string(
    ctx: *mut DeserContext,
    out: *mut String,
    data_ptr: *const u8,
    data_len: u32,
) {
    let len = data_len as usize;
    let bytes = unsafe { core::slice::from_raw_parts(data_ptr, len) };
    let ctx = unsafe { &mut *ctx };
    let s = if ctx.trusted_utf8 {
        // SAFETY: trusted mode is enabled only when the caller opted into
        // pre-validated UTF-8 input for a compatible format.
        unsafe { core::str::from_utf8_unchecked(bytes) }
    } else {
        match core::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => {
                ctx.error.code = ErrorCode::InvalidUtf8 as u32;
                return;
            }
        }
    };
    unsafe { out.write(s.to_owned()) };
}

/// Validate UTF-8, allocate raw buffer, and copy bytes. Returns buffer pointer.
///
/// Malum string intrinsic — the JIT writes the returned pointer + len directly
/// into the String's `(ptr, len, cap)` fields at discovered offsets, bypassing
/// the intermediate `String` object.
///
/// Returns:
/// - On success: pointer to allocated buffer containing the string bytes
/// - On empty (data_len == 0): `1 as *mut u8` (dangling aligned pointer)
/// - On error: null pointer (error code set on ctx)
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `data_ptr` must point to at least `data_len` readable bytes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_string_validate_alloc_copy(
    ctx: *mut DeserContext,
    data_ptr: *const u8,
    data_len: u32,
) -> *mut u8 {
    let len = data_len as usize;

    // Empty string: return dangling pointer, JIT writes ptr/0/0.
    if len == 0 {
        return std::ptr::dangling_mut::<u8>();
    }

    // Validate UTF-8 unless input is already trusted.
    if !unsafe { (*ctx).trusted_utf8 } {
        let bytes = unsafe { core::slice::from_raw_parts(data_ptr, len) };
        if core::str::from_utf8(bytes).is_err() {
            let ctx = unsafe { &mut *ctx };
            ctx.error.code = ErrorCode::InvalidUtf8 as u32;
            return core::ptr::null_mut();
        }
    }

    // Allocate raw buffer (same allocator String uses).
    let layout = unsafe { std::alloc::Layout::from_size_align_unchecked(len, 1) };
    let buf = unsafe { std::alloc::alloc(layout) };
    if buf.is_null() {
        let ctx = unsafe { &mut *ctx };
        ctx.error.code = ErrorCode::AllocError as u32;
        return core::ptr::null_mut();
    }

    // Copy bytes.
    unsafe { core::ptr::copy_nonoverlapping(data_ptr, buf, len) };
    buf
}

// --- Vec intrinsics ---

// r[impl seq.malum.alloc-compat]

/// Allocate a buffer for `count` elements of `elem_size` bytes, `elem_align` alignment.
///
/// Uses `std::alloc::alloc` with `Layout::from_size_align(count * elem_size, elem_align)` —
/// the same allocator and layout that `Vec<T>` would use, so the resulting buffer can be
/// owned by a Vec and deallocated normally.
///
/// Returns a pointer to the allocated buffer. On allocation failure, sets an error on ctx
/// and returns a null pointer.
///
/// # Safety
/// - `count` must be > 0 (caller handles the empty case)
/// - `elem_size` and `elem_align` must be valid for `Layout::from_size_align`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_vec_alloc(
    ctx: *mut DeserContext,
    count: usize,
    elem_size: usize,
    elem_align: usize,
) -> *mut u8 {
    let size = count.checked_mul(elem_size).unwrap_or(0);
    if size == 0 {
        return core::ptr::null_mut();
    }
    let layout = match std::alloc::Layout::from_size_align(size, elem_align) {
        Ok(layout) => layout,
        Err(_) => {
            let ctx = unsafe { &mut *ctx };
            ctx.error.code = ErrorCode::AllocError as u32;
            return core::ptr::null_mut();
        }
    };
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        let ctx = unsafe { &mut *ctx };
        ctx.error.code = ErrorCode::AllocError as u32;
    }
    ptr
}

/// Grow a Vec buffer: allocate a new buffer of `new_cap * elem_size`, copy
/// `len * elem_size` bytes from `old_buf`, and deallocate `old_buf`.
///
/// Returns the new buffer pointer. On allocation failure, the old buffer is NOT freed
/// and an error is set on ctx.
///
/// # Safety
/// - `old_buf` must have been allocated with `Layout::from_size_align(old_cap * elem_size, elem_align)`
/// - `len <= old_cap`
/// - `new_cap > old_cap`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_vec_grow(
    ctx: *mut DeserContext,
    old_buf: *mut u8,
    len: usize,
    old_cap: usize,
    new_cap: usize,
    elem_size: usize,
    elem_align: usize,
) -> *mut u8 {
    let new_size = new_cap * elem_size;
    let new_layout = match std::alloc::Layout::from_size_align(new_size, elem_align) {
        Ok(layout) => layout,
        Err(_) => {
            let ctx = unsafe { &mut *ctx };
            ctx.error.code = ErrorCode::AllocError as u32;
            return old_buf;
        }
    };
    let new_buf = unsafe { std::alloc::alloc(new_layout) };
    if new_buf.is_null() {
        let ctx = unsafe { &mut *ctx };
        ctx.error.code = ErrorCode::AllocError as u32;
        return old_buf;
    }

    // Copy existing elements.
    let copy_size = len * elem_size;
    if copy_size > 0 {
        unsafe { core::ptr::copy_nonoverlapping(old_buf, new_buf, copy_size) };
    }

    // Free old buffer.
    let old_size = old_cap * elem_size;
    if old_size > 0 {
        let old_layout =
            unsafe { std::alloc::Layout::from_size_align_unchecked(old_size, elem_align) };
        unsafe { std::alloc::dealloc(old_buf, old_layout) };
    }

    new_buf
}

/// Free a Vec buffer. Called on error paths to clean up partially-built Vecs.
///
/// # Safety
/// - `buf` must have been allocated with `Layout::from_size_align(cap * elem_size, elem_align)`
/// - `buf` must not be null (caller checks)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_vec_free(
    buf: *mut u8,
    cap: usize,
    elem_size: usize,
    elem_align: usize,
) {
    let size = cap * elem_size;
    if size > 0 && !buf.is_null() {
        let layout = unsafe { std::alloc::Layout::from_size_align_unchecked(size, elem_align) };
        unsafe { std::alloc::dealloc(buf, layout) };
    }
}

// --- Default field intrinsics ---

// r[impl deser.default]
// r[impl deser.default.fn-ptr]

/// Initialize a field to its default value using the type's `Default` impl.
///
/// Wraps the facet `TypeOpsDirect.default_in_place` function, which has the ABI
/// `unsafe fn(*mut ())`, into a thin `extern "C"` trampoline callable from JIT code.
///
/// # Safety
/// - `default_fn` must be a valid `unsafe fn(*mut ())` from `TypeOpsDirect.default_in_place`
/// - `out` must point to uninitialized memory of the correct size/alignment for the type
#[unsafe(no_mangle)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "C" fn kajit_field_default_trait(default_fn: unsafe fn(*mut ()), out: *mut u8) {
    unsafe { default_fn(out as *mut ()) };
}

/// Initialize a field to its default value using a custom default expression.
///
/// Wraps a `DefaultInPlaceFn` (`unsafe fn(PtrUninit) -> PtrMut`) into a thin
/// `extern "C"` trampoline. Constructs the `PtrUninit` from the raw output pointer.
///
/// # Safety
/// - `default_fn` must be a valid `DefaultInPlaceFn`
/// - `out` must point to uninitialized memory of the correct size/alignment for the type
#[unsafe(no_mangle)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "C" fn kajit_field_default_custom(
    default_fn: facet::DefaultInPlaceFn,
    out: *mut u8,
) {
    let ptr_uninit = PtrUninit::new_sized(out);
    unsafe { default_fn(ptr_uninit) };
}

/// Initialize a field to its default value using an indirect `TypeOpsIndirect.default_in_place`.
///
/// Indirect types (generic containers like `Option<T>`, `Vec<T>`) use wide pointers
/// (`OxPtrUninit` = pointer + shape) instead of thin pointers. This trampoline constructs
/// the `OxPtrUninit` from the raw output pointer and shape.
///
/// # Safety
/// - `default_fn` must be a valid `unsafe fn(OxPtrUninit) -> bool`
/// - `out` must point to uninitialized memory of the correct size/alignment
/// - `shape` must be the correct `&'static Shape` for the type
#[unsafe(no_mangle)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "C" fn kajit_field_default_indirect(
    default_fn: unsafe fn(facet::OxPtrUninit) -> bool,
    out: *mut u8,
    shape: &'static facet::Shape,
) {
    let ptr_uninit = PtrUninit::new_sized(out);
    let ox = facet::OxPtrUninit::new(ptr_uninit, shape);
    unsafe { default_fn(ox) };
}

/// Trampoline: call `from_pair_slice` with a plain *mut u8 map pointer.
///
/// JIT code cannot directly call `from_pair_slice` because its first argument,
/// `PtrUninit`, is a 16-byte `#[repr(C)]` struct — passed in two registers on
/// aarch64 / Linux x64 but by pointer on Windows x64.  This trampoline takes
/// only pointer-/usize-sized arguments (all single-register) and constructs the
/// `PtrUninit` value internally using Rust's correct ABI handling.
///
/// # Safety
/// - `from_pair_slice_fn` must be a valid `MapFromPairSliceFn` function pointer.
/// - `map_ptr` must point to uninitialised memory sized and aligned for the map type.
/// - `pairs_ptr` must point to a contiguous `count` pairs (or may be null when `count == 0`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_map_build(
    from_pair_slice_fn: *const u8,
    map_ptr: *mut u8,
    pairs_ptr: *mut u8,
    count: usize,
) {
    let f: MapFromPairSliceFn = unsafe { core::mem::transmute(from_pair_slice_fn) };
    let uninit = PtrUninit::new(map_ptr);
    unsafe { f(uninit, pairs_ptr, count) };
}

// --- Encode intrinsics ---

use crate::context::EncodeContext;

/// Grow the output buffer to accommodate at least `needed` additional bytes.
/// Called by JIT code when a bounds check against output_end fails.
///
/// The JIT must flush output_ptr to ctx before calling this, and reload
/// output_ptr and output_end after (same pattern as deser cursor flush/reload).
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to an `EncodeContext`
/// - `ctx.output_ptr` must have been flushed (written back) before this call
pub unsafe extern "C" fn kajit_output_grow(ctx: *mut EncodeContext, needed: usize) {
    let ctx = unsafe { &mut *ctx };
    unsafe { ctx.grow(needed) };
}

/// Helper: write a u32 as a varint into the output buffer.
/// Assumes sufficient capacity has already been ensured.
/// Returns the number of bytes written.
unsafe fn write_varint32(ptr: *mut u8, mut value: u32) -> usize {
    let start = ptr;
    let mut p = ptr;
    while value >= 0x80 {
        unsafe {
            *p = (value as u8) | 0x80;
            p = p.add(1);
        }
        value >>= 7;
    }
    unsafe {
        *p = value as u8;
        p = p.add(1);
        p.offset_from(start) as usize
    }
}

/// Helper: write a u64 as a varint into the output buffer.
/// Assumes sufficient capacity has already been ensured.
/// Returns the number of bytes written.
#[allow(dead_code)]
unsafe fn write_varint64(ptr: *mut u8, mut value: u64) -> usize {
    let start = ptr;
    let mut p = ptr;
    while value >= 0x80 {
        unsafe {
            *p = (value as u8) | 0x80;
            p = p.add(1);
        }
        value >>= 7;
    }
    unsafe {
        *p = value as u8;
        p = p.add(1);
        p.offset_from(start) as usize
    }
}

/// Helper: write a u128 as a varint into the output buffer.
/// Assumes sufficient capacity has already been ensured.
/// Returns the number of bytes written.
unsafe fn write_varint128(ptr: *mut u8, mut value: u128) -> usize {
    let start = ptr;
    let mut p = ptr;
    while value >= 0x80 {
        unsafe {
            *p = (value as u8) | 0x80;
            p = p.add(1);
        }
        value >>= 7;
    }
    unsafe {
        *p = value as u8;
        p = p.add(1);
        p.offset_from(start) as usize
    }
}

/// Ensure the output buffer has at least `needed` bytes of capacity.
/// If not, grows the buffer.
unsafe fn ensure_capacity(ctx: &mut EncodeContext, needed: usize) {
    if ctx.remaining() < needed {
        unsafe { ctx.grow(needed) };
    }
}

/// Encode a String/&str/Cow<str> as postcard: varint(len) + raw UTF-8 bytes.
///
/// Called as: fn(ctx, field_ptr) where field_ptr = input + field_offset.
/// Reads ptr and len from the string's memory layout using discovered offsets.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to an `EncodeContext`
/// - `field_ptr` must point to a valid, initialized String/&str/Cow<str> value
pub unsafe extern "C" fn kajit_encode_postcard_string(
    ctx: *mut EncodeContext,
    field_ptr: *const u8,
) {
    let ctx = unsafe { &mut *ctx };
    let offsets = crate::malum::discover_string_offsets();

    // Read the data pointer and length from the string's layout.
    let data_ptr = unsafe { *(field_ptr.add(offsets.ptr_offset as usize) as *const *const u8) };
    let data_len = unsafe { *(field_ptr.add(offsets.len_offset as usize) as *const usize) };

    // Max varint32 is 5 bytes for the length prefix.
    let needed = 5 + data_len;
    unsafe { ensure_capacity(ctx, needed) };

    // Write varint-encoded length.
    let written = unsafe { write_varint32(ctx.output_ptr, data_len as u32) };
    ctx.output_ptr = unsafe { ctx.output_ptr.add(written) };

    // Copy string bytes.
    if data_len > 0 {
        unsafe {
            core::ptr::copy_nonoverlapping(data_ptr, ctx.output_ptr, data_len);
            ctx.output_ptr = ctx.output_ptr.add(data_len);
        }
    }
}

/// Encode a u128 as a varint.
///
/// Called as: fn(ctx, field_ptr) where field_ptr = input + field_offset.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to an `EncodeContext`
/// - `field_ptr` must be a valid, aligned pointer to an initialized `u128`
pub unsafe extern "C" fn kajit_encode_u128(ctx: *mut EncodeContext, field_ptr: *const u8) {
    let ctx = unsafe { &mut *ctx };
    let value = unsafe { *(field_ptr as *const u128) };
    // u128 varint is at most 19 bytes.
    unsafe { ensure_capacity(ctx, 19) };
    let written = unsafe { write_varint128(ctx.output_ptr, value) };
    ctx.output_ptr = unsafe { ctx.output_ptr.add(written) };
}

/// Encode an i128 as zigzag + varint.
///
/// Called as: fn(ctx, field_ptr) where field_ptr = input + field_offset.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to an `EncodeContext`
/// - `field_ptr` must be a valid, aligned pointer to an initialized `i128`
pub unsafe extern "C" fn kajit_encode_i128(ctx: *mut EncodeContext, field_ptr: *const u8) {
    let ctx = unsafe { &mut *ctx };
    let value = unsafe { *(field_ptr as *const i128) };
    // Zigzag encode: (value << 1) ^ (value >> 127)
    let encoded = ((value << 1) ^ (value >> 127)) as u128;
    unsafe { ensure_capacity(ctx, 19) };
    let written = unsafe { write_varint128(ctx.output_ptr, encoded) };
    ctx.output_ptr = unsafe { ctx.output_ptr.add(written) };
}

/// Encode a char as length-prefixed UTF-8 bytes (postcard wire format).
///
/// Called as: fn(ctx, field_ptr) where field_ptr = input + field_offset.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to an `EncodeContext`
/// - `field_ptr` must be a valid, aligned pointer to an initialized `char`
pub unsafe extern "C" fn kajit_encode_char(ctx: *mut EncodeContext, field_ptr: *const u8) {
    let ctx = unsafe { &mut *ctx };
    let value = unsafe { *(field_ptr as *const char) };
    let mut buf = [0u8; 4];
    let utf8 = value.encode_utf8(&mut buf);
    let len = utf8.len();
    // varint(len) takes at most 1 byte (len <= 4), plus up to 4 UTF-8 bytes
    unsafe { ensure_capacity(ctx, 5) };
    let varint_written = unsafe { write_varint32(ctx.output_ptr, len as u32) };
    ctx.output_ptr = unsafe { ctx.output_ptr.add(varint_written) };
    unsafe { core::ptr::copy_nonoverlapping(buf.as_ptr(), ctx.output_ptr, len) };
    ctx.output_ptr = unsafe { ctx.output_ptr.add(len) };
}

/// Returns all known postcard intrinsics as `(name, IntrinsicFn)` pairs.
pub fn known_intrinsics() -> Vec<(&'static str, crate::ir::IntrinsicFn)> {
    use crate::ir::IntrinsicFn;
    vec![
        (
            "kajit_read_bool",
            IntrinsicFn(kajit_read_bool as *const () as usize),
        ),
        (
            "kajit_read_u8",
            IntrinsicFn(kajit_read_u8 as *const () as usize),
        ),
        (
            "kajit_read_u16",
            IntrinsicFn(kajit_read_u16 as *const () as usize),
        ),
        (
            "kajit_read_varint_u32",
            IntrinsicFn(kajit_read_varint_u32 as *const () as usize),
        ),
        (
            "kajit_read_u64",
            IntrinsicFn(kajit_read_u64 as *const () as usize),
        ),
        (
            "kajit_read_u128",
            IntrinsicFn(kajit_read_u128 as *const () as usize),
        ),
        (
            "kajit_read_usize",
            IntrinsicFn(kajit_read_usize as *const () as usize),
        ),
        (
            "kajit_read_i8",
            IntrinsicFn(kajit_read_i8 as *const () as usize),
        ),
        (
            "kajit_read_i16",
            IntrinsicFn(kajit_read_i16 as *const () as usize),
        ),
        (
            "kajit_read_i32",
            IntrinsicFn(kajit_read_i32 as *const () as usize),
        ),
        (
            "kajit_read_i64",
            IntrinsicFn(kajit_read_i64 as *const () as usize),
        ),
        (
            "kajit_read_i128",
            IntrinsicFn(kajit_read_i128 as *const () as usize),
        ),
        (
            "kajit_read_isize",
            IntrinsicFn(kajit_read_isize as *const () as usize),
        ),
        (
            "kajit_read_f32",
            IntrinsicFn(kajit_read_f32 as *const () as usize),
        ),
        (
            "kajit_read_f64",
            IntrinsicFn(kajit_read_f64 as *const () as usize),
        ),
        (
            "kajit_read_char",
            IntrinsicFn(kajit_read_char as *const () as usize),
        ),
        (
            "kajit_read_postcard_string",
            IntrinsicFn(kajit_read_postcard_string as *const () as usize),
        ),
        (
            "kajit_read_postcard_string_with_len",
            IntrinsicFn(kajit_read_postcard_string_with_len as *const () as usize),
        ),
        (
            "kajit_read_postcard_str",
            IntrinsicFn(kajit_read_postcard_str as *const () as usize),
        ),
        (
            "kajit_read_postcard_str_with_len",
            IntrinsicFn(kajit_read_postcard_str_with_len as *const () as usize),
        ),
        (
            "kajit_read_postcard_cow_str",
            IntrinsicFn(kajit_read_postcard_cow_str as *const () as usize),
        ),
        (
            "kajit_read_postcard_cow_str_with_len",
            IntrinsicFn(kajit_read_postcard_cow_str_with_len as *const () as usize),
        ),
        (
            "kajit_option_init_none",
            IntrinsicFn(kajit_option_init_none as *const () as usize),
        ),
        (
            "kajit_option_init_some",
            IntrinsicFn(kajit_option_init_some as *const () as usize),
        ),
        (
            "kajit_option_init_none_ctx",
            IntrinsicFn(kajit_option_init_none_ctx as *const () as usize),
        ),
        (
            "kajit_option_init_some_ctx",
            IntrinsicFn(kajit_option_init_some_ctx as *const () as usize),
        ),
        (
            "kajit_pointer_new_into",
            IntrinsicFn(kajit_pointer_new_into as *const () as usize),
        ),
        (
            "kajit_postcard_validate_and_alloc_string",
            IntrinsicFn(kajit_postcard_validate_and_alloc_string as *const () as usize),
        ),
        (
            "kajit_string_validate_alloc_copy",
            IntrinsicFn(kajit_string_validate_alloc_copy as *const () as usize),
        ),
        (
            "kajit_vec_alloc",
            IntrinsicFn(kajit_vec_alloc as *const () as usize),
        ),
        (
            "kajit_vec_grow",
            IntrinsicFn(kajit_vec_grow as *const () as usize),
        ),
        (
            "kajit_vec_free",
            IntrinsicFn(kajit_vec_free as *const () as usize),
        ),
        (
            "kajit_field_default_trait",
            IntrinsicFn(kajit_field_default_trait as *const () as usize),
        ),
        (
            "kajit_field_default_custom",
            IntrinsicFn(kajit_field_default_custom as *const () as usize),
        ),
        (
            "kajit_field_default_indirect",
            IntrinsicFn(kajit_field_default_indirect as *const () as usize),
        ),
        (
            "kajit_map_build",
            IntrinsicFn(kajit_map_build as *const () as usize),
        ),
        (
            "kajit_output_grow",
            IntrinsicFn(kajit_output_grow as *const () as usize),
        ),
        (
            "kajit_encode_postcard_string",
            IntrinsicFn(kajit_encode_postcard_string as *const () as usize),
        ),
        (
            "kajit_encode_u128",
            IntrinsicFn(kajit_encode_u128 as *const () as usize),
        ),
        (
            "kajit_encode_i128",
            IntrinsicFn(kajit_encode_i128 as *const () as usize),
        ),
        (
            "kajit_encode_char",
            IntrinsicFn(kajit_encode_char as *const () as usize),
        ),
    ]
}
