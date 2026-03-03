pub use kajit_ir::ErrorCode;

// r[impl callconv.deser-context]
/// The runtime context passed to every emitted deserializer function.
/// Layout is `#[repr(C)]` so field offsets are stable and can be used from JIT code.
#[repr(C)]
pub struct DeserContext {
    /// Current read position in the input buffer.
    pub input_ptr: *const u8,
    /// One-past-the-end of the input buffer.
    pub input_end: *const u8,
    /// Error slot — checked after each intrinsic call.
    pub error: ErrorSlot,
    /// Scratch buffer for unescaped JSON keys. Allocated on first use.
    /// Only accessed by intrinsics (never by JIT code).
    pub key_scratch_ptr: *mut u8,
    pub key_scratch_cap: usize,
    /// True when input came from `&str` and selected intrinsics may skip UTF-8 revalidation.
    pub trusted_utf8: bool,
}

// r[impl error.slot]
/// Error information written by intrinsics when something goes wrong.
#[repr(C)]
pub struct ErrorSlot {
    /// Non-zero means an error occurred.
    pub code: u32,
    /// Byte offset in the input where the error was detected.
    pub offset: u32,
}

// Field offset constants for use from JIT code.
pub const CTX_INPUT_PTR: u32 = core::mem::offset_of!(DeserContext, input_ptr) as u32;
pub const CTX_INPUT_END: u32 = core::mem::offset_of!(DeserContext, input_end) as u32;
pub const CTX_ERROR_CODE: u32 = core::mem::offset_of!(DeserContext, error.code) as u32;
pub const CTX_ERROR_OFFSET: u32 = core::mem::offset_of!(DeserContext, error.offset) as u32;

impl DeserContext {
    /// Create a new context pointing at the given input slice.
    pub fn new(input: &[u8]) -> Self {
        Self::from_bytes(input)
    }

    /// Create a context for untrusted raw bytes.
    pub fn from_bytes(input: &[u8]) -> Self {
        Self::new_with_trust(input, false)
    }

    /// Create a context for already-validated UTF-8 text input.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Self {
        Self::new_with_trust(input.as_bytes(), true)
    }

    fn new_with_trust(input: &[u8], trusted_utf8: bool) -> Self {
        let ptr = input.as_ptr();
        DeserContext {
            input_ptr: ptr,
            input_end: unsafe { ptr.add(input.len()) },
            error: ErrorSlot { code: 0, offset: 0 },
            key_scratch_ptr: core::ptr::null_mut(),
            key_scratch_cap: 0,
            trusted_utf8,
        }
    }

    /// Returns the number of bytes remaining.
    pub fn remaining(&self) -> usize {
        unsafe { self.input_end.offset_from(self.input_ptr) as usize }
    }

    /// Set an error code, recording the current offset from the start of the original input.
    ///
    /// # Safety
    /// `input_start` must point to the beginning of the same allocation that `self.input_ptr`
    /// was derived from, and `self.input_ptr` must be at or after `input_start`.
    pub unsafe fn set_error(&mut self, code: ErrorCode, input_start: *const u8) {
        self.error.code = code as u32;
        self.error.offset = unsafe { self.input_ptr.offset_from(input_start) as u32 };
    }

    /// Free the key scratch buffer, replacing it with the given Vec's allocation.
    /// The old scratch buffer (if any) is freed.
    ///
    /// # Safety
    /// The Vec must have been created with the global allocator.
    pub unsafe fn replace_key_scratch(&mut self, buf: Vec<u8>) -> (*const u8, usize) {
        // Free old scratch if any
        if self.key_scratch_cap > 0 {
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(self.key_scratch_cap, 1);
                std::alloc::dealloc(self.key_scratch_ptr, layout);
            }
        }
        let len = buf.len();
        let cap = buf.capacity();
        let ptr = buf.as_ptr();
        core::mem::forget(buf);
        self.key_scratch_ptr = ptr as *mut u8;
        self.key_scratch_cap = cap;
        (ptr, len)
    }
}

impl Drop for DeserContext {
    fn drop(&mut self) {
        if self.key_scratch_cap > 0 {
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(self.key_scratch_cap, 1);
                std::alloc::dealloc(self.key_scratch_ptr, layout);
            }
        }
    }
}

// =============================================================================
// EncodeContext — runtime state for serialization
// =============================================================================

const ENCODE_INITIAL_CAP: usize = 256;

/// The runtime context passed to every emitted serializer function.
/// Layout is `#[repr(C)]` so field offsets are stable and can be used from JIT code.
///
/// The JIT function signature is:
///   `unsafe extern "C" fn(input: *const u8, ctx: *mut EncodeContext)`
/// where `input` points to the typed struct being serialized.
#[repr(C)]
pub struct EncodeContext {
    /// Current write position in the output buffer.
    /// Cached in a callee-saved register by JIT code, flushed before intrinsic calls.
    pub output_ptr: *mut u8,
    /// One-past-the-end of the allocated output buffer.
    /// Cached in a callee-saved register by JIT code.
    pub output_end: *mut u8,
    /// Error slot — checked after each intrinsic call.
    pub error: ErrorSlot,
    /// Start of the allocated output buffer. Only accessed by the growth intrinsic
    /// and for final result extraction. Not cached in a register.
    pub output_start: *mut u8,
}

// Field offset constants for use from JIT code.
pub const ENC_OUTPUT_PTR: u32 = core::mem::offset_of!(EncodeContext, output_ptr) as u32;
pub const ENC_OUTPUT_END: u32 = core::mem::offset_of!(EncodeContext, output_end) as u32;
pub const ENC_ERROR_CODE: u32 = core::mem::offset_of!(EncodeContext, error.code) as u32;
pub const ENC_ERROR_OFFSET: u32 = core::mem::offset_of!(EncodeContext, error.offset) as u32;
pub const ENC_OUTPUT_START: u32 = core::mem::offset_of!(EncodeContext, output_start) as u32;

impl EncodeContext {
    /// Create a new encode context with default initial capacity.
    pub fn new() -> Self {
        Self::with_capacity(ENCODE_INITIAL_CAP)
    }

    /// Create a new encode context with the given initial capacity.
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(1); // avoid zero-sized allocations
        let layout = std::alloc::Layout::from_size_align(cap, 1).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        EncodeContext {
            output_ptr: ptr,
            output_end: unsafe { ptr.add(cap) },
            error: ErrorSlot { code: 0, offset: 0 },
            output_start: ptr,
        }
    }

    /// Returns the number of bytes written so far.
    pub fn written(&self) -> usize {
        unsafe { self.output_ptr.offset_from(self.output_start) as usize }
    }

    /// Returns the remaining capacity before the buffer needs to grow.
    pub fn remaining(&self) -> usize {
        unsafe { self.output_end.offset_from(self.output_ptr) as usize }
    }

    /// Returns the total allocated capacity.
    pub fn capacity(&self) -> usize {
        unsafe { self.output_end.offset_from(self.output_start) as usize }
    }

    /// Consume the context and return the output as a `Vec<u8>`.
    pub fn into_vec(self) -> Vec<u8> {
        let len = self.written();
        let cap = self.capacity();
        let ptr = self.output_start;
        core::mem::forget(self); // skip Drop — Vec takes ownership
        unsafe { Vec::from_raw_parts(ptr, len, cap) }
    }

    /// Set an error code, recording the current output position.
    pub fn set_error(&mut self, code: ErrorCode) {
        self.error.code = code as u32;
        self.error.offset = self.written() as u32;
    }

    /// Grow the output buffer to accommodate at least `needed` additional bytes.
    /// Updates output_start, output_ptr, and output_end.
    ///
    /// # Safety
    /// The output_ptr must be within [output_start, output_end].
    pub unsafe fn grow(&mut self, needed: usize) {
        let len = self.written();
        let old_cap = self.capacity();
        let min_cap = old_cap.checked_add(needed).expect("capacity overflow");
        // Double, but at least enough for the request.
        let new_cap = min_cap.max(old_cap.saturating_mul(2)).max(1);

        unsafe {
            let old_layout = std::alloc::Layout::from_size_align_unchecked(old_cap, 1);
            let new_ptr = std::alloc::realloc(self.output_start, old_layout, new_cap);
            if new_ptr.is_null() {
                self.set_error(ErrorCode::AllocError);
                return;
            }
            self.output_start = new_ptr;
            self.output_ptr = new_ptr.add(len);
            self.output_end = new_ptr.add(new_cap);
        }
    }
}

impl Default for EncodeContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for EncodeContext {
    fn drop(&mut self) {
        let cap = self.capacity();
        if cap > 0 {
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(cap, 1);
                std::alloc::dealloc(self.output_start, layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_context_new() {
        let ctx = EncodeContext::new();
        assert_eq!(ctx.written(), 0);
        assert_eq!(ctx.capacity(), 256);
        assert_eq!(ctx.remaining(), 256);
        assert_eq!(ctx.error.code, 0);
    }

    #[test]
    fn encode_context_with_capacity() {
        let ctx = EncodeContext::with_capacity(1024);
        assert_eq!(ctx.capacity(), 1024);
        assert_eq!(ctx.written(), 0);
    }

    #[test]
    fn encode_context_write_and_into_vec() {
        let mut ctx = EncodeContext::with_capacity(16);
        // Simulate JIT writing bytes
        unsafe {
            *ctx.output_ptr = 0x42;
            ctx.output_ptr = ctx.output_ptr.add(1);
            *ctx.output_ptr = 0xFF;
            ctx.output_ptr = ctx.output_ptr.add(1);
        }
        assert_eq!(ctx.written(), 2);
        assert_eq!(ctx.remaining(), 14);

        let vec = ctx.into_vec();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.capacity(), 16);
        assert_eq!(vec[0], 0x42);
        assert_eq!(vec[1], 0xFF);
    }

    #[test]
    fn encode_context_grow() {
        let mut ctx = EncodeContext::with_capacity(4);
        // Write 3 bytes
        unsafe {
            for i in 0..3u8 {
                *ctx.output_ptr = i;
                ctx.output_ptr = ctx.output_ptr.add(1);
            }
        }
        assert_eq!(ctx.written(), 3);
        assert_eq!(ctx.remaining(), 1);

        // Grow to fit 10 more bytes
        unsafe { ctx.grow(10) };
        assert_eq!(ctx.error.code, 0); // no error
        assert_eq!(ctx.written(), 3); // data preserved
        assert!(ctx.remaining() >= 10); // enough space

        // Verify existing data survived realloc
        let vec = ctx.into_vec();
        assert_eq!(&vec[..3], &[0, 1, 2]);
    }

    #[test]
    fn encode_context_grow_doubles() {
        let mut ctx = EncodeContext::with_capacity(8);
        unsafe { ctx.grow(1) };
        // Should at least double: max(8+1, 8*2) = 16
        assert!(ctx.capacity() >= 16);
    }

    #[test]
    fn encode_context_drop_frees() {
        // Just ensure no leak/crash — run under miri or valgrind for verification.
        let ctx = EncodeContext::with_capacity(1024);
        drop(ctx);
    }

    #[test]
    fn encode_context_empty_into_vec() {
        let ctx = EncodeContext::new();
        let vec = ctx.into_vec();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 256);
    }

    #[test]
    fn encode_context_field_offsets() {
        // Verify the offset constants match actual layout.
        assert_eq!(ENC_OUTPUT_PTR, 0);
        assert_eq!(ENC_OUTPUT_END, 8);
        // error.code is at offset 16 (after two 8-byte pointers)
        assert_eq!(ENC_ERROR_CODE, 16);
        assert_eq!(ENC_ERROR_OFFSET, 20);
        // output_start after the error slot (16 + 8 = 24)
        assert_eq!(ENC_OUTPUT_START, 24);
    }
}
