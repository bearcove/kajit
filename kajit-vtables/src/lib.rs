//! Exported vtable function wrappers for the standalone harness.
//!
//! The JIT compiler bakes vtable function pointers (e.g. `Option<T>::init_some`)
//! as raw constants. When building a standalone harness object file, these pointers
//! need to be relocated. This crate exports `#[no_mangle]` wrappers that the
//! linker can resolve.
//!
//! Symbol naming convention: `kajit_vtable_{entry}__{mangled_type}`
//! where `{mangled_type}` is produced by `kajit_format::mangle_type_id`.

macro_rules! export_option_vtable {
    ($T:ty, $mangled:ident) => {
        paste::paste! {
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [< kajit_vtable_option_init_some__ $mangled >](
                option: facet::PtrUninit,
                value: facet::PtrMut,
            ) -> facet::PtrMut {
                let shape = <Option<$T> as facet::Facet>::SHAPE;
                let opt_def = match &shape.def {
                    facet::Def::Option(d) => d,
                    _ => unreachable!(),
                };
                unsafe { (opt_def.vtable.init_some)(option, value) }
            }

            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [< kajit_vtable_option_init_none__ $mangled >](
                option: facet::PtrUninit,
            ) -> facet::PtrMut {
                let shape = <Option<$T> as facet::Facet>::SHAPE;
                let opt_def = match &shape.def {
                    facet::Def::Option(d) => d,
                    _ => unreachable!(),
                };
                unsafe { (opt_def.vtable.init_none)(option) }
            }
        }
    };
}

// Export vtable wrappers for all Option<T> types used in the corpus.
// The mangled names must match what `kajit_format::mangle_type_id` produces
// when given the Shape's Display output (e.g. "Option<u32>" → "Option_Lu32_R").
export_option_vtable!(u32, Option_Lu32_R);
export_option_vtable!(String, Option_LString_R);
export_option_vtable!(u8, Option_Lu8_R);
export_option_vtable!(u16, Option_Lu16_R);
export_option_vtable!(u64, Option_Lu64_R);
export_option_vtable!(i8, Option_Li8_R);
export_option_vtable!(i16, Option_Li16_R);
export_option_vtable!(i32, Option_Li32_R);
export_option_vtable!(i64, Option_Li64_R);
export_option_vtable!(bool, Option_Lbool_R);

#[cfg(test)]
mod tests {
    use kajit_format::{VtableEntry, vtable_symbol_name};

    /// Verify that the mangled symbol names match what `vtable_symbol_name` produces,
    /// and that the exported wrappers call through to the correct vtable entry.
    #[test]
    fn symbol_names_match() {
        fn check<T: facet::Facet<'static>>() {
            let shape = T::SHAPE;
            let some_sym = vtable_symbol_name(shape, VtableEntry::OptionInitSome);
            let none_sym = vtable_symbol_name(shape, VtableEntry::OptionInitNone);
            eprintln!("  init_some: {some_sym}");
            eprintln!("  init_none: {none_sym}");
        }

        check::<Option<u32>>();
        check::<Option<String>>();
        check::<Option<u8>>();
        check::<Option<bool>>();

        // Verify the wrapper for Option<u32> actually works
        unsafe {
            let mut storage = std::mem::MaybeUninit::<Option<u32>>::uninit();
            let ptr_uninit = facet::PtrUninit::new(storage.as_mut_ptr());
            let _result = super::kajit_vtable_option_init_none__Option_Lu32_R(ptr_uninit);
            let val = storage.assume_init();
            assert_eq!(val, None);
        }
    }
}
