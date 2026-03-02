//! Deser/ser cases

use crate::Case;
use proc_macro2::TokenStream;
use quote::quote;

pub(crate) fn types_rs() -> TokenStream {
    quote! {
        use serde::{Serialize, Deserialize};

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Friend {
            age: u32,
            name: String,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Address {
            city: String,
            zip: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Person {
            name: String,
            age: u32,
            address: Address,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Inner {
            x: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Middle {
            inner: Inner,
            y: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Outer {
            middle: Middle,
            z: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct AllIntegers {
            a_u8: u8,
            a_u16: u16,
            a_u32: u32,
            a_u64: u64,
            a_i8: i8,
            a_i16: i16,
            a_i32: i32,
            a_i64: i64,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct BoolField {
            value: bool,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct ScalarVec {
            values: Vec<u32>,
        }

        type Pair = (u32, String);
    }
}

pub(crate) fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "flat_struct",
            value: quote!(Friend {
                age: 42,
                name: "Alice".into()
            }),
        },
        Case {
            name: "nested_struct",
            value: quote!(Person {
                name: "Alice".into(),
                age: 30,
                address: Address {
                    city: "Portland".into(),
                    zip: 97201
                }
            }),
        },
        Case {
            name: "deep_struct",
            value: quote!(Outer {
                middle: Middle {
                    inner: Inner { x: 1 },
                    y: 2
                },
                z: 3
            }),
        },
        Case {
            name: "all_integers",
            value: quote!(AllIntegers {
                a_u8: 255,
                a_u16: 65535,
                a_u32: 1_000_000,
                a_u64: 1_000_000_000_000,
                a_i8: -128,
                a_i16: -32768,
                a_i32: -1_000_000,
                a_i64: -1_000_000_000_000
            }),
        },
        Case {
            name: "bool_field",
            value: quote!(BoolField { value: true }),
        },
        Case {
            name: "tuple_pair",
            value: quote!((42u32, "Alice".to_string())),
        },
        Case {
            name: "vec_scalar_small",
            value: quote!(ScalarVec {
                values: (0..16).map(|i| i as u32).collect()
            }),
        },
        Case {
            name: "vec_scalar_large",
            value: quote!(ScalarVec {
                values: (0..2048).map(|i| i as u32).collect()
            }),
        },
    ]
}
