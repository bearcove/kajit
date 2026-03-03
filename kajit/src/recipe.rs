// r[impl no-ir.micro-ops]

use crate::context::ErrorCode;
use crate::malum::StringOffsets;

/// A virtual register slot for values flowing through a recipe.
/// Maps to platform-specific scratch registers during emission:
/// - Slot::A → aarch64: w9/x9,  x64: r10d/r10
/// - Slot::B → aarch64: w10/x10, x64: r11d/r11
#[derive(Debug, Clone, Copy)]
pub enum Slot {
    A,
    B,
}

/// Width of a memory operation.
#[derive(Debug, Clone, Copy)]
pub enum Width {
    W1,
    W2,
    W4,
    W8,
}

/// Target for branch-on-error operations.
#[derive(Debug, Clone, Copy)]
pub enum ErrorTarget {
    /// Branch to the recipe's EOF error path (writes UnexpectedEof, jumps to error_exit).
    Eof,
    /// Branch to error_exit directly (error code already set).
    ErrorExit,
}

/// A primitive operation in a recipe.
#[derive(Debug, Clone)]
pub enum Op {
    /// Check remaining >= count bytes. Branches to EOF error on failure.
    BoundsCheck { count: u32 },
    /// Load 1 byte from cursor into slot (zero-extended).
    LoadByte { dst: Slot },
    /// Load `width` bytes from cursor into slot.
    LoadFromCursor { dst: Slot, width: Width },
    /// Store `width` bytes from slot to out+offset.
    StoreToOut {
        src: Slot,
        offset: u32,
        width: Width,
    },
    /// Store 1 byte from slot to stack.
    StoreByteToStack { src: Slot, sp_offset: u32 },
    /// Store slot to stack.
    StoreToStack {
        src: Slot,
        sp_offset: u32,
        width: Width,
    },
    /// Load from stack into slot.
    LoadFromStack {
        dst: Slot,
        sp_offset: u32,
        width: Width,
    },
    /// Advance cursor by `count` bytes.
    AdvanceCursor { count: u32 },
    /// Advance cursor by the value in slot.
    AdvanceCursorBySlot { slot: Slot },
    /// Zigzag decode: slot = (slot >> 1) ^ -(slot & 1)
    ZigzagDecode { slot: Slot },
    /// Validate slot <= max_val, branch to error on failure.
    ValidateMax {
        slot: Slot,
        max_val: u32,
        error: ErrorCode,
    },
    /// Test bit 7 of slot; if set, branch to target label.
    TestBit7Branch { slot: Slot, target: usize },
    /// Unconditional branch to label.
    Branch { target: usize },
    /// Bind a label at the current position.
    BindLabel { index: usize },
    /// Flush cursor, call intrinsic(ctx, out+field_offset), reload, check error.
    CallIntrinsic {
        fn_ptr: *const u8,
        field_offset: u32,
    },
    /// Flush cursor, call intrinsic(ctx, sp+sp_offset), reload, check error.
    CallIntrinsicStackOut { fn_ptr: *const u8, sp_offset: u32 },
    /// dst = input_end - input_ptr (remaining bytes).
    ComputeRemaining { dst: Slot },
    /// If lhs < rhs (unsigned), branch to error target.
    CmpBranchLo {
        lhs: Slot,
        rhs: Slot,
        on_fail: ErrorTarget,
    },
    /// Save cursor pointer into slot.
    SaveCursor { dst: Slot },
    /// Flush cursor, call validate_alloc_copy(ctx, data_src, len_src).
    /// Returns buf pointer in return register (x0/rax).
    CallValidateAllocCopy {
        fn_ptr: *const u8,
        data_src: Slot,
        len_src: Slot,
    },
    /// Write String fields directly: ptr from return register, len/cap from slot.
    WriteMalumString {
        base_offset: u32,
        ptr_off: u32,
        len_off: u32,
        cap_off: u32,
        len_slot: Slot,
    },

    // ── Encode-direction ops ──────────────────────────────────────────
    /// Load `width` bytes from input struct at offset into slot.
    LoadFromInput {
        dst: Slot,
        offset: u32,
        width: Width,
    },
    /// Store `width` bytes from slot to output buffer at current position, advancing output.
    StoreToOutput { src: Slot, width: Width },
    /// Write a literal byte to the output buffer, advancing by 1.
    WriteByte { value: u8 },
    /// Advance the output pointer by `count` bytes (no write).
    AdvanceOutput { count: u32 },
    /// Advance the output pointer by the value in slot.
    AdvanceOutputBySlot { slot: Slot },
    /// Ensure at least `count` bytes are available in the output buffer.
    /// Calls kajit_output_grow if needed.
    OutputBoundsCheck { count: u32 },
    /// Sign-extend a narrower value in the slot to full register width.
    /// `from` is the source width (W1 or W2); the value is sign-extended
    /// to 32-bit (or 64-bit if `wide` is true in the subsequent zigzag).
    SignExtend { slot: Slot, from: Width },
    /// Zigzag encode: slot = (slot << 1) ^ (slot >> (bits-1))
    /// When `wide` is true, operates on the full 64-bit register.
    ZigzagEncode { slot: Slot, wide: bool },
    /// Encode a value as a varint into the output buffer.
    /// When `wide` is true, uses 64-bit register (up to 10 bytes).
    /// When false, uses 32-bit register (up to 5 bytes).
    EncodeVarint { slot: Slot, wide: bool },
}

/// A recipe: a sequence of ops with a declared number of labels.
pub struct Recipe {
    pub ops: Vec<Op>,
    pub label_count: usize,
}

impl Recipe {
    fn new() -> Self {
        Recipe {
            ops: Vec::new(),
            label_count: 0,
        }
    }

    fn label(&mut self) -> usize {
        let idx = self.label_count;
        self.label_count += 1;
        idx
    }
}

// ── Simple scalar recipes ────────────────────────────────────────────

/// Read 1 byte (u8/i8) from input, store to out+offset.
pub fn read_byte(offset: u32) -> Recipe {
    let mut r = Recipe::new();
    r.ops.extend([
        Op::BoundsCheck { count: 1 },
        Op::LoadByte { dst: Slot::A },
        Op::StoreToOut {
            src: Slot::A,
            offset,
            width: Width::W1,
        },
        Op::AdvanceCursor { count: 1 },
    ]);
    r
}

/// Read 1 byte from input, store to stack slot.
pub fn read_byte_to_stack(sp_offset: u32) -> Recipe {
    let mut r = Recipe::new();
    r.ops.extend([
        Op::BoundsCheck { count: 1 },
        Op::LoadByte { dst: Slot::A },
        Op::StoreByteToStack {
            src: Slot::A,
            sp_offset,
        },
        Op::AdvanceCursor { count: 1 },
    ]);
    r
}

/// Read 1 byte, validate 0 or 1, store to out+offset.
pub fn read_bool(offset: u32) -> Recipe {
    let mut r = Recipe::new();
    r.ops.extend([
        Op::BoundsCheck { count: 1 },
        Op::LoadByte { dst: Slot::A },
        Op::ValidateMax {
            slot: Slot::A,
            max_val: 1,
            error: ErrorCode::InvalidBool,
        },
        Op::StoreToOut {
            src: Slot::A,
            offset,
            width: Width::W1,
        },
        Op::AdvanceCursor { count: 1 },
    ]);
    r
}

/// Read 4 LE bytes (f32), store to out+offset.
pub fn read_f32(offset: u32) -> Recipe {
    let mut r = Recipe::new();
    r.ops.extend([
        Op::BoundsCheck { count: 4 },
        Op::LoadFromCursor {
            dst: Slot::A,
            width: Width::W4,
        },
        Op::StoreToOut {
            src: Slot::A,
            offset,
            width: Width::W4,
        },
        Op::AdvanceCursor { count: 4 },
    ]);
    r
}

/// Read 8 LE bytes (f64), store to out+offset.
pub fn read_f64(offset: u32) -> Recipe {
    let mut r = Recipe::new();
    r.ops.extend([
        Op::BoundsCheck { count: 8 },
        Op::LoadFromCursor {
            dst: Slot::A,
            width: Width::W8,
        },
        Op::StoreToOut {
            src: Slot::A,
            offset,
            width: Width::W8,
        },
        Op::AdvanceCursor { count: 8 },
    ]);
    r
}

// ── Varint recipe ────────────────────────────────────────────────────

/// Varint fast path: single-byte inline, multi-byte calls intrinsic.
pub fn varint_fast_path(
    offset: u32,
    store_width: Width,
    zigzag: bool,
    intrinsic: *const u8,
) -> Recipe {
    let mut r = Recipe::new();
    let slow_path = r.label();
    let done = r.label();

    r.ops.extend([
        Op::BoundsCheck { count: 1 },
        Op::LoadByte { dst: Slot::A },
        Op::TestBit7Branch {
            slot: Slot::A,
            target: slow_path,
        },
        Op::AdvanceCursor { count: 1 },
    ]);
    if zigzag {
        r.ops.push(Op::ZigzagDecode { slot: Slot::A });
    }
    r.ops.extend([
        Op::StoreToOut {
            src: Slot::A,
            offset,
            width: store_width,
        },
        Op::Branch { target: done },
        Op::BindLabel { index: slow_path },
        Op::CallIntrinsic {
            fn_ptr: intrinsic,
            field_offset: offset,
        },
        Op::BindLabel { index: done },
    ]);
    r
}

// ── Postcard string recipe ───────────────────────────────────────────

/// Postcard string with malum: inline varint decode + bounds check,
/// call intrinsic for UTF-8 validation + allocation, write (ptr, len, cap).
pub fn postcard_string_malum(
    offset: u32,
    string_offsets: &StringOffsets,
    slow_varint_intrinsic: *const u8,
    validate_alloc_copy_intrinsic: *const u8,
) -> Recipe {
    let mut r = Recipe::new();
    let varint_slow = r.label();
    let have_length = r.label();
    let done = r.label();

    // Step 1: Varint length decode
    r.ops.extend([
        Op::BoundsCheck { count: 1 },
        Op::LoadByte { dst: Slot::A },
        Op::TestBit7Branch {
            slot: Slot::A,
            target: varint_slow,
        },
        Op::AdvanceCursor { count: 1 },
        Op::Branch {
            target: have_length,
        },
        // Slow path: multi-byte varint
        Op::BindLabel { index: varint_slow },
        Op::CallIntrinsicStackOut {
            fn_ptr: slow_varint_intrinsic,
            sp_offset: 48,
        },
        Op::LoadFromStack {
            dst: Slot::A,
            sp_offset: 48,
            width: Width::W4,
        },
    ]);

    // Step 2: Bounds check + validate_alloc_copy + write fields
    r.ops.extend([
        Op::BindLabel { index: have_length },
        // Save length to stack (survives call)
        Op::StoreToStack {
            src: Slot::A,
            sp_offset: 48,
            width: Width::W4,
        },
        // remaining = end - ptr
        Op::ComputeRemaining { dst: Slot::B },
        // if remaining < length → EOF
        Op::CmpBranchLo {
            lhs: Slot::B,
            rhs: Slot::A,
            on_fail: ErrorTarget::Eof,
        },
        // Save cursor as data_ptr
        Op::SaveCursor { dst: Slot::B },
        // Call validate_alloc_copy(ctx, data_ptr, len) → buf in rax/x0
        Op::CallValidateAllocCopy {
            fn_ptr: validate_alloc_copy_intrinsic,
            data_src: Slot::B,
            len_src: Slot::A,
        },
        // Reload length from stack
        Op::LoadFromStack {
            dst: Slot::A,
            sp_offset: 48,
            width: Width::W4,
        },
        // Write (ptr, len, cap) at discovered offsets
        Op::WriteMalumString {
            base_offset: offset,
            ptr_off: string_offsets.ptr_offset,
            len_off: string_offsets.len_offset,
            cap_off: string_offsets.cap_offset,
            len_slot: Slot::A,
        },
        // Advance cursor by string length
        Op::AdvanceCursorBySlot { slot: Slot::A },
        Op::Branch { target: done },
        Op::BindLabel { index: done },
    ]);
    r
}
