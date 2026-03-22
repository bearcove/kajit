// r[impl seq.malum]

use facet::{ListDef, PtrConst, PtrMut, PtrUninit, Shape};
use std::mem::MaybeUninit;
use std::sync::OnceLock;

/// Returns true if the malum (direct Vec layout) path is enabled.
///
/// Enabled by default. Set `KAJIT_NO_MALUM=1` to disable.
pub fn malum_enabled() -> bool {
    std::env::var("KAJIT_NO_MALUM").map_or(true, |v| v != "1")
}

/// Discovered Vec field offsets (in bytes from base of the Vec).
///
/// These are probed at JIT-compile time by constructing a real `Vec<T>` through
/// facet vtable functions. No hardcoded assumptions about layout.
#[derive(Debug, Clone, Copy)]
pub struct VecOffsets {
    pub ptr_offset: u32,
    pub len_offset: u32,
    pub cap_offset: u32,
}

/// Discover the memory layout of `Vec<T>` by probing a real instance through
/// the facet vtable.
///
/// This runs at JIT-compile time (not `const`) — it allocates a real `Vec<T>`
/// and inspects the raw bytes to find where `ptr`, `len`, and `cap` live.
///
/// Panics if the layout cannot be determined or if the required vtable
/// functions are missing.
pub fn discover_vec_offsets(list_def: &ListDef, shape: &'static Shape) -> VecOffsets {
    let init_fn = list_def
        .init_in_place_with_capacity()
        .expect("Vec<T> must provide init_in_place_with_capacity for malum");
    let capacity_fn = list_def
        .capacity()
        .expect("Vec<T> must provide capacity() for malum");
    let as_mut_ptr_fn = list_def
        .as_mut_ptr_typed()
        .expect("Vec<T> must provide as_mut_ptr_typed() for malum");
    let set_len_fn = list_def
        .set_len()
        .expect("Vec<T> must provide set_len() for malum");

    // Phase 1: Create a Vec<T> with capacity 7 (distinctive value), len = 0.
    let mut storage: MaybeUninit<[u8; 24]> = MaybeUninit::zeroed();
    let base_ptr = storage.as_mut_ptr() as *mut u8;

    let vec_ptr: PtrMut = unsafe {
        let uninit = PtrUninit::new_sized(base_ptr);
        (init_fn)(uninit, 7)
    };

    // Read back values through the vtable.
    let data_ptr = unsafe { (as_mut_ptr_fn)(vec_ptr) } as usize;
    let cap = unsafe { (capacity_fn)(PtrConst::new_sized(base_ptr)) };
    let len = unsafe { (list_def.vtable.len)(PtrConst::new_sized(base_ptr)) };

    assert_eq!(len, 0, "freshly created Vec should have len=0");
    assert!(
        cap >= 7,
        "Vec::with_capacity(7) should have cap >= 7, got {cap}"
    );
    assert_ne!(data_ptr, 0, "Vec data pointer should be non-null");

    // Read the raw bytes as three usize words.
    let words: [usize; 3] = unsafe { core::ptr::read(base_ptr as *const [usize; 3]) };

    // Identify each field by matching against known values.
    let mut ptr_offset: Option<u32> = None;
    let mut len_offset: Option<u32> = None;
    let mut cap_offset: Option<u32> = None;

    for (i, &word) in words.iter().enumerate() {
        let offset = (i * 8) as u32;

        if word == data_ptr && ptr_offset.is_none() {
            ptr_offset = Some(offset);
        } else if word == cap && cap_offset.is_none() && word != 0 {
            cap_offset = Some(offset);
        } else if word == 0 && len_offset.is_none() {
            len_offset = Some(offset);
        }
    }

    // If cap == data_ptr (unlikely but possible), disambiguate with set_len.
    if ptr_offset.is_none() || len_offset.is_none() || cap_offset.is_none() {
        // Set len to 3 to disambiguate (safe: we don't actually have elements,
        // but we never read them — we only inspect the raw words).
        unsafe { (set_len_fn)(vec_ptr, 3) };
        let words2: [usize; 3] = unsafe { core::ptr::read(base_ptr as *const [usize; 3]) };

        // The word that changed from 0 to 3 is len_offset.
        // Re-scan with the new information.
        ptr_offset = None;
        len_offset = None;
        cap_offset = None;

        for (i, (&word, &word2)) in words.iter().zip(words2.iter()).enumerate() {
            let offset = (i * 8) as u32;
            if word != word2 && word2 == 3 {
                len_offset = Some(offset);
            }
        }

        // Now assign ptr and cap from the remaining two.
        for (i, &word) in words.iter().enumerate() {
            let offset = (i * 8) as u32;
            if Some(offset) == len_offset {
                continue;
            }
            if word == data_ptr && ptr_offset.is_none() {
                ptr_offset = Some(offset);
            } else if cap_offset.is_none() {
                cap_offset = Some(offset);
            }
        }

        // Reset len back to 0 before drop.
        unsafe { (set_len_fn)(vec_ptr, 0) };
    }

    let ptr_offset = ptr_offset.expect("could not identify Vec ptr field");
    let len_offset = len_offset.expect("could not identify Vec len field");
    let cap_offset = cap_offset.expect("could not identify Vec cap field");

    assert_ne!(ptr_offset, len_offset, "ptr and len at same offset");
    assert_ne!(ptr_offset, cap_offset, "ptr and cap at same offset");
    assert_ne!(len_offset, cap_offset, "len and cap at same offset");

    // Drop the probing Vec.
    unsafe {
        shape
            .call_drop_in_place(PtrMut::new_sized(base_ptr))
            .expect("Vec<T> must have drop_in_place");
    }

    let offsets = VecOffsets {
        ptr_offset,
        len_offset,
        cap_offset,
    };

    // Phase 2: Validation — create a second Vec and verify the offsets work.
    validate_offsets(
        list_def,
        shape,
        &offsets,
        init_fn,
        capacity_fn,
        as_mut_ptr_fn,
        set_len_fn,
    );

    offsets
}

/// Validate discovered offsets by creating a second Vec and mutating through
/// the vtable, then checking raw memory matches.
fn validate_offsets(
    list_def: &ListDef,
    shape: &'static Shape,
    offsets: &VecOffsets,
    init_fn: facet::ListInitInPlaceWithCapacityFn,
    capacity_fn: facet::ListCapacityFn,
    as_mut_ptr_fn: facet::ListAsMutPtrTypedFn,
    set_len_fn: facet::ListSetLenFn,
) {
    let reserve_fn = list_def
        .reserve()
        .expect("Vec<T> must provide reserve() for malum validation");

    let mut storage: MaybeUninit<[u8; 24]> = MaybeUninit::zeroed();
    let base_ptr = storage.as_mut_ptr() as *mut u8;

    let vec_ptr = unsafe {
        let uninit = PtrUninit::new_sized(base_ptr);
        (init_fn)(uninit, 5)
    };

    // Read through discovered offsets and verify against vtable.
    let read_word = |offset: u32| -> usize {
        unsafe { core::ptr::read((base_ptr as *const u8).add(offset as usize) as *const usize) }
    };

    let vtable_ptr = unsafe { (as_mut_ptr_fn)(vec_ptr) } as usize;
    let vtable_cap = unsafe { (capacity_fn)(PtrConst::new_sized(base_ptr)) };
    let vtable_len = unsafe { (list_def.vtable.len)(PtrConst::new_sized(base_ptr)) };

    assert_eq!(
        read_word(offsets.ptr_offset),
        vtable_ptr,
        "ptr_offset validation failed"
    );
    assert_eq!(
        read_word(offsets.cap_offset),
        vtable_cap,
        "cap_offset validation failed"
    );
    assert_eq!(
        read_word(offsets.len_offset),
        vtable_len,
        "len_offset validation failed"
    );

    // Mutate len through vtable, verify the raw word changed.
    unsafe { (set_len_fn)(vec_ptr, 5) };
    assert_eq!(
        read_word(offsets.len_offset),
        5,
        "len_offset mutation validation failed"
    );
    // Reset before further checks.
    unsafe { (set_len_fn)(vec_ptr, 0) };

    // Reserve more capacity, verify cap word changed.
    let old_cap = read_word(offsets.cap_offset);
    unsafe { (reserve_fn)(vec_ptr, 1000) };
    let new_cap = read_word(offsets.cap_offset);
    assert!(
        new_cap >= 1000,
        "cap_offset should reflect reserve(): expected >= 1000, got {new_cap}"
    );
    assert_ne!(old_cap, new_cap, "cap_offset should change after reserve()");

    // Also verify ptr may have changed (reallocation).
    let new_vtable_ptr = unsafe { (as_mut_ptr_fn)(vec_ptr) } as usize;
    assert_eq!(
        read_word(offsets.ptr_offset),
        new_vtable_ptr,
        "ptr_offset should track reallocation"
    );

    // Drop.
    unsafe {
        shape
            .call_drop_in_place(PtrMut::new_sized(base_ptr))
            .expect("Vec<T> must have drop_in_place");
    }
}

// ── String layout discovery ──────────────────────────────────────────────

/// Discovered String field offsets (in bytes from base of the String).
///
/// String has the same `(ptr, len, cap)` layout as `Vec<u8>`, but the field
/// order is not guaranteed by `repr(Rust)`. These are probed at JIT-compile
/// time by constructing a real `String`.
#[derive(Debug, Clone, Copy)]
pub struct StringOffsets {
    pub ptr_offset: u32,
    pub len_offset: u32,
    pub cap_offset: u32,
}

/// Cached String offsets — String layout is the same for all instances.
static STRING_OFFSETS: OnceLock<StringOffsets> = OnceLock::new();

/// Discover the memory layout of `String` by probing a real instance.
///
/// Cached after first call — String's layout is fixed for the lifetime of
/// the process.
pub fn discover_string_offsets() -> StringOffsets {
    *STRING_OFFSETS.get_or_init(discover_string_offsets_inner)
}

fn discover_string_offsets_inner() -> StringOffsets {
    // Phase 1: Create a String with capacity 7 (distinctive value), len = 0.
    let s = String::with_capacity(7);
    let data_ptr = s.as_ptr() as usize;
    let cap = s.capacity();
    let len = s.len();

    assert_eq!(len, 0);
    assert!(
        cap >= 7,
        "String::with_capacity(7) should have cap >= 7, got {cap}"
    );
    assert_ne!(data_ptr, 0, "String data pointer should be non-null");

    // Read the raw bytes as three usize words.
    let base_ptr = &s as *const String as *const u8;
    let words: [usize; 3] = unsafe { core::ptr::read(base_ptr as *const [usize; 3]) };

    // Identify each field by matching against known values.
    let mut ptr_offset: Option<u32> = None;
    let mut len_offset: Option<u32> = None;
    let mut cap_offset: Option<u32> = None;

    for (i, &word) in words.iter().enumerate() {
        let offset = (i * 8) as u32;

        if word == data_ptr && ptr_offset.is_none() {
            ptr_offset = Some(offset);
        } else if word == cap && cap_offset.is_none() && word != 0 {
            cap_offset = Some(offset);
        } else if word == 0 && len_offset.is_none() {
            len_offset = Some(offset);
        }
    }

    // If cap == data_ptr (unlikely but possible), disambiguate with set_len.
    if ptr_offset.is_none() || len_offset.is_none() || cap_offset.is_none() {
        let mut s = s;
        // Safety: we set len to 3 but never read the uninitialized bytes.
        // We only inspect the raw words to find which field changed.
        unsafe { s.as_mut_vec().set_len(3) };
        let base_ptr2 = &s as *const String as *const u8;
        let words2: [usize; 3] = unsafe { core::ptr::read(base_ptr2 as *const [usize; 3]) };

        ptr_offset = None;
        len_offset = None;
        cap_offset = None;

        for (i, (&word, &word2)) in words.iter().zip(words2.iter()).enumerate() {
            let offset = (i * 8) as u32;
            if word != word2 && word2 == 3 {
                len_offset = Some(offset);
            }
        }

        for (i, &word) in words.iter().enumerate() {
            let offset = (i * 8) as u32;
            if Some(offset) == len_offset {
                continue;
            }
            if word == data_ptr && ptr_offset.is_none() {
                ptr_offset = Some(offset);
            } else if cap_offset.is_none() {
                cap_offset = Some(offset);
            }
        }

        // Reset len back to 0 before drop.
        unsafe { s.as_mut_vec().set_len(0) };
    } else {
        drop(s);
    }

    let ptr_offset = ptr_offset.expect("could not identify String ptr field");
    let len_offset = len_offset.expect("could not identify String len field");
    let cap_offset = cap_offset.expect("could not identify String cap field");

    assert_ne!(ptr_offset, len_offset, "ptr and len at same offset");
    assert_ne!(ptr_offset, cap_offset, "ptr and cap at same offset");
    assert_ne!(len_offset, cap_offset, "len and cap at same offset");

    let offsets = StringOffsets {
        ptr_offset,
        len_offset,
        cap_offset,
    };

    // Phase 2: Validation — create a second String and verify offsets.
    validate_string_offsets(&offsets);

    offsets
}

fn validate_string_offsets(offsets: &StringOffsets) {
    let mut s = String::with_capacity(5);
    let base_ptr = &s as *const String as *const u8;

    let read_word = |offset: u32| -> usize {
        unsafe { core::ptr::read(base_ptr.add(offset as usize) as *const usize) }
    };

    assert_eq!(read_word(offsets.ptr_offset), s.as_ptr() as usize);
    assert_eq!(read_word(offsets.cap_offset), s.capacity());
    assert_eq!(read_word(offsets.len_offset), s.len());

    // Push some content and verify len changes.
    s.push_str("hello");
    // Re-read base_ptr since s may not have moved (capacity was 5, we wrote 5 bytes).
    let base_ptr = &s as *const String as *const u8;
    let read_word = |offset: u32| -> usize {
        unsafe { core::ptr::read(base_ptr.add(offset as usize) as *const usize) }
    };
    assert_eq!(read_word(offsets.len_offset), 5);
    assert_eq!(read_word(offsets.ptr_offset), s.as_ptr() as usize);

    // Reserve more capacity, verify cap changes.
    let old_cap = read_word(offsets.cap_offset);
    s.reserve(1000);
    let base_ptr = &s as *const String as *const u8;
    let read_word = |offset: u32| -> usize {
        unsafe { core::ptr::read(base_ptr.add(offset as usize) as *const usize) }
    };
    let new_cap = read_word(offsets.cap_offset);
    assert!(new_cap >= 1000);
    assert_ne!(old_cap, new_cap);
    assert_eq!(read_word(offsets.ptr_offset), s.as_ptr() as usize);
}
