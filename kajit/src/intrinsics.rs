use std::borrow::Cow;

use facet::{MapFromPairSliceFn, PtrUninit};

use crate::context::{DeserContext, ErrorCode};

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
pub unsafe extern "C" fn kajit_validate_and_alloc_string(
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

/// Validate that a raw byte range is UTF-8.
///
/// This is the lean borrowed-string helper for generated HIR: the decoder
/// computes the byte range and cursor movement itself, and this helper only
/// checks UTF-8 validity in the current trust mode.
///
/// # Safety
///
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `data_ptr` must point to at least `data_len` readable bytes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_validate_utf8_range(
    ctx: *mut DeserContext,
    data_ptr: *const u8,
    data_len: u32,
) {
    let len = data_len as usize;
    let ctx = unsafe { &mut *ctx };
    if ctx.trusted_utf8 {
        return;
    }
    let bytes = unsafe { core::slice::from_raw_parts(data_ptr, len) };
    if core::str::from_utf8(bytes).is_err() {
        if std::env::var_os("KAJIT_TRACE_UTF8").is_some() {
            eprintln!(
                "[kajit_validate_utf8_range] invalid utf8 len={} bytes={:02x?}",
                len, bytes
            );
        }
        ctx.error.code = ErrorCode::InvalidUtf8 as u32;
    }
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

/// Allocate raw persistent heap memory with the requested layout.
///
/// This is the low-level primitive used by generated decoders that build
/// heap-backed host values in place before a later materialization step.
///
/// # Safety
/// - `ctx` must be a valid, aligned, non-null pointer to a `DeserContext`
/// - `align` must be accepted by `Layout::from_size_align`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_alloc_persistent(
    ctx: *mut DeserContext,
    size: usize,
    align: usize,
) -> *mut u8 {
    if size == 0 {
        return core::ptr::null_mut();
    }
    let layout = match std::alloc::Layout::from_size_align(size, align) {
        Ok(layout) => layout,
        Err(_) => {
            let ctx = unsafe { &mut *ctx };
            ctx.error.code = ErrorCode::AllocError as u32;
            return core::ptr::null_mut();
        }
    };
    let buf = unsafe { std::alloc::alloc(layout) };
    if buf.is_null() {
        let ctx = unsafe { &mut *ctx };
        ctx.error.code = ErrorCode::AllocError as u32;
    }
    buf
}

// --- Context-free allocation intrinsics (for scalar/VixenTypedFunction code) ---

/// Allocate heap memory without a runtime context.
///
/// Returns a heap pointer owned by the caller. Free with `kajit_free_transient`.
///
/// # Zero-length contract
/// `size == 0` returns `align` as a non-null aligned sentinel. Must NOT be freed.
///
/// # Safety
/// - `align` must be a power of two and accepted by `Layout::from_size_align`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_alloc_transient(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        return align as *mut u8;
    }
    let layout = match std::alloc::Layout::from_size_align(size, align) {
        Ok(layout) => layout,
        Err(_) => return core::ptr::null_mut(),
    };
    unsafe { std::alloc::alloc(layout) }
}

/// Copy `len` bytes from `src` to `dst` (non-overlapping).
///
/// Returns `dst.add(len)` — the pointer just past the copied bytes.
/// This return convention creates a data dependency chain that preserves
/// ordering of successive copies in the RVSDG.
///
/// # Safety
/// - `dst` and `src` must be valid for `len` bytes
/// - Regions must not overlap
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_memcpy(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
    if len > 0 {
        unsafe { core::ptr::copy_nonoverlapping(src, dst, len) };
    }
    unsafe { dst.add(len) }
}

/// Free heap memory allocated by `kajit_alloc_transient`.
///
/// # Zero-length contract
/// `size == 0` or `ptr.is_null()` is a no-op.
///
/// # Safety
/// - `ptr` must have been returned by `kajit_alloc_transient(size, align)`
/// - Must not be called on the zero-length sentinel
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kajit_free_transient(ptr: *mut u8, size: usize, align: usize) {
    if size == 0 || ptr.is_null() {
        return;
    }
    let layout = match std::alloc::Layout::from_size_align(size, align) {
        Ok(layout) => layout,
        Err(_) => return,
    };
    unsafe { std::alloc::dealloc(ptr, layout) };
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

/// Returns all known postcard intrinsics as `(name, IntrinsicFn)` pairs.
pub fn known_intrinsics() -> Vec<(&'static str, crate::ir::FnPtr)> {
    use crate::ir::FnPtr;
    vec![
        (
            "kajit_option_init_none",
            FnPtr(kajit_option_init_none as *const () as usize),
        ),
        (
            "kajit_option_init_some",
            FnPtr(kajit_option_init_some as *const () as usize),
        ),
        (
            "kajit_option_init_none_ctx",
            FnPtr(kajit_option_init_none_ctx as *const () as usize),
        ),
        (
            "kajit_option_init_some_ctx",
            FnPtr(kajit_option_init_some_ctx as *const () as usize),
        ),
        (
            "kajit_pointer_new_into",
            FnPtr(kajit_pointer_new_into as *const () as usize),
        ),
        (
            "kajit_validate_and_alloc_string",
            FnPtr(kajit_validate_and_alloc_string as *const () as usize),
        ),
        (
            "kajit_validate_utf8_range",
            FnPtr(kajit_validate_utf8_range as *const () as usize),
        ),
        (
            "kajit_string_validate_alloc_copy",
            FnPtr(kajit_string_validate_alloc_copy as *const () as usize),
        ),
        (
            "kajit_alloc_persistent",
            FnPtr(kajit_alloc_persistent as *const () as usize),
        ),
        (
            "kajit_alloc_transient",
            FnPtr(kajit_alloc_transient as *const () as usize),
        ),
        ("kajit_memcpy", FnPtr(kajit_memcpy as *const () as usize)),
        (
            "kajit_free_transient",
            FnPtr(kajit_free_transient as *const () as usize),
        ),
        (
            "kajit_vec_alloc",
            FnPtr(kajit_vec_alloc as *const () as usize),
        ),
        (
            "kajit_vec_grow",
            FnPtr(kajit_vec_grow as *const () as usize),
        ),
        (
            "kajit_vec_free",
            FnPtr(kajit_vec_free as *const () as usize),
        ),
        (
            "kajit_field_default_trait",
            FnPtr(kajit_field_default_trait as *const () as usize),
        ),
        (
            "kajit_field_default_custom",
            FnPtr(kajit_field_default_custom as *const () as usize),
        ),
        (
            "kajit_field_default_indirect",
            FnPtr(kajit_field_default_indirect as *const () as usize),
        ),
        (
            "kajit_map_build",
            FnPtr(kajit_map_build as *const () as usize),
        ),
    ]
}
