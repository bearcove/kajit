//! Deser/ser cases

use crate::{Case, CaseBuilder, DecodeExpectation, WireFormat};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{LitByteStr, LitStr};

struct PanicCase {
    name: &'static str,
    expected: &'static str,
    body: TokenStream,
}

pub(crate) fn types_rs() -> TokenStream {
    quote! {
        use serde::{Serialize, Deserialize};
        use std::collections::{BTreeMap, HashMap};

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Friend {
            age: u32,
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            name: String,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Address {
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            city: String,
            zip: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Person {
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            name: String,
            age: u32,
            address: Address,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Metadata {
            version: u32,
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            author: String,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Document {
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            title: String,
            #[facet(flatten)]
            meta: Metadata,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        #[repr(u8)]
        enum Animal {
            Cat,
            Dog { name: String, good_boy: bool },
            Parrot(String),
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Zoo {
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            name: String,
            star: Animal,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        #[facet(tag = "type", content = "data")]
        #[repr(u8)]
        enum AdjAnimal {
            Cat,
            Dog { name: String, good_boy: bool },
            Parrot(String),
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        #[facet(tag = "type")]
        #[repr(u8)]
        enum IntAnimal {
            Cat,
            Dog { name: String, good_boy: bool },
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        #[facet(untagged)]
        #[repr(u8)]
        enum UntaggedAnimal {
            Cat,
            Dog { name: String, good_boy: bool },
            Parrot(String),
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        #[facet(untagged)]
        #[repr(u8)]
        enum UntaggedConfig {
            Database { host: String, port: u32 },
            Redis { host: String, db: u32 },
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct SuccessPayload {
            items: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct ErrorPayload {
            message: String,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        #[facet(untagged)]
        #[repr(u8)]
        enum ApiResponse {
            Success { status: u32, data: SuccessPayload },
            Error { status: u32, data: ErrorPayload },
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Inner {
            x: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Middle {
            inner: Inner,
            y: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Outer {
            middle: Middle,
            z: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct AllIntegers {
            a_u8: u8,
            a_u16: u16,
            a_u32: u32,
            a_u64: u64,
            a_u128: u128,
            a_usize: usize,
            a_i8: i8,
            a_i16: i16,
            a_i32: i32,
            a_i64: i64,
            a_i128: i128,
            a_isize: isize,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct AllScalars {
            a_bool: bool,
            a_u8: u8,
            a_u16: u16,
            a_u32: u32,
            a_u64: u64,
            a_u128: u128,
            a_usize: usize,
            a_i8: i8,
            a_i16: i16,
            a_i32: i32,
            a_i64: i64,
            a_i128: i128,
            a_isize: isize,
            #[proptest(strategy = "-1000000.0f32..1000000.0f32")]
            a_f32: f32,
            #[proptest(strategy = "-1000000.0f64..1000000.0f64")]
            a_f64: f64,
            a_char: char,
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            a_name: String,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Bools {
            a: bool,
            b: bool,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Boundaries {
            u8_max: u8,
            u16_max: u16,
            i8_min: i8,
            i8_max: i8,
            i16_min: i16,
            i32_min: i32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct BoolField {
            value: bool,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Tiny {
            val: u8,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Floats {
            #[proptest(strategy = "-1000000.0f64..1000000.0f64")]
            a: f64,
            #[proptest(strategy = "-1000000.0f64..1000000.0f64")]
            b: f64,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct WithOptU32 {
            value: Option<u32>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct WithOptStr {
            #[proptest(strategy = "proptest::option::of(proptest::string::string_regex(\"(?s).{0,64}\").unwrap())")]
            name: Option<String>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct WithOptAddr {
            addr: Option<Address>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct MultiOpt {
            a: Option<u32>,
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            b: String,
            #[proptest(strategy = "proptest::option::of(proptest::string::string_regex(\"(?s).{0,64}\").unwrap())")]
            c: Option<String>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct RenameField {
            #[facet(rename = "user_name")]
            name: String,
            age: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        #[facet(rename_all = "camelCase")]
        struct CamelCaseStruct {
            user_name: String,
            birth_year: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        #[facet(transparent)]
        struct Wrapper(u32);

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        #[facet(transparent)]
        struct StringWrapper(
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")] String,
        );

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        #[facet(transparent)]
        struct StructWrapper(Friend);

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        #[facet(deny_unknown_fields)]
        struct Strict {
            x: u32,
            y: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct WithDefault {
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            name: String,
            #[facet(default)]
            score: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct WithDefaultString {
            #[facet(default)]
            label: String,
            value: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        #[facet(default)]
        struct AllDefault {
            x: u32,
            y: u32,
        }

        impl Default for AllDefault {
            fn default() -> Self {
                AllDefault { x: 10, y: 20 }
            }
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct WithSkip {
            name: String,
            #[facet(skip, default)]
            cached: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct WithSkipDeser {
            name: String,
            #[facet(skip_deserializing, default)]
            internal: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct SkipWithCustomDefault {
            value: u32,
            #[facet(skip, default = 42)]
            magic: u32,
        }

        #[derive(Debug, PartialEq, Facet)]
        struct RcScalar {
            value: std::rc::Rc<u32>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct BoxedScalar {
            value: Box<u32>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct BoxedString {
            #[allow(clippy::box_collection)]
            name: Box<String>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct BoxedNested {
            inner: Box<Friend>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct OptionBox {
            value: Option<Box<u32>>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct VecBox {
            #[allow(clippy::vec_box)]
            items: Vec<Box<u32>>,
        }

        #[derive(Debug, PartialEq, Facet)]
        struct ArcScalar {
            value: std::sync::Arc<u32>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct UnitField {
            geo: (),
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            name: String,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct ScalarVec {
            #[proptest(strategy = "proptest::collection::vec(proptest::arbitrary::any::<u32>(), 0..256)")]
            values: Vec<u32>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Nums {
            #[proptest(strategy = "proptest::collection::vec(proptest::arbitrary::any::<u32>(), 0..256)")]
            vals: Vec<u32>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Names {
            #[proptest(strategy = "proptest::collection::vec(proptest::string::string_regex(\"(?s).{0,32}\").unwrap(), 0..128)")]
            items: Vec<String>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct AddressList {
            #[proptest(strategy = "proptest::collection::vec(proptest::arbitrary::any::<Address>(), 0..128)")]
            addrs: Vec<Address>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct TwoAddresses {
            home: Address,
            work: Address,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct ConfigMap {
            #[proptest(strategy = "proptest::collection::hash_map(proptest::string::string_regex(\"[a-z]{1,8}\").unwrap(), proptest::arbitrary::any::<u32>(), 0..32)")]
            scores: HashMap<String, u32>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct EnvMap {
            #[proptest(strategy = "proptest::collection::hash_map(proptest::string::string_regex(\"[A-Z_]{1,8}\").unwrap(), proptest::string::string_regex(\"(?s).{0,16}\").unwrap(), 0..32)")]
            vars: HashMap<String, String>,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct BTreeConfigMap {
            #[proptest(strategy = "proptest::collection::btree_map(proptest::string::string_regex(\"[a-z]{1,8}\").unwrap(), proptest::arbitrary::any::<u32>(), 0..32)")]
            scores: BTreeMap<String, u32>,
        }

        #[allow(dead_code)]
        type Pair = (u32, String);
    }
}

pub(crate) fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "flat_struct",
            ty: quote!(Friend),
            values: vec![quote!(Friend {
                age: 42,
                name: "Alice".into()
            })],
            inputs: vec![],
        },
        Case {
            name: "nested_struct",
            ty: quote!(Person),
            values: vec![quote!(Person {
                name: "Alice".into(),
                age: 30,
                address: Address {
                    city: "Portland".into(),
                    zip: 97201
                }
            })],
            inputs: vec![],
        },
        Case {
            name: "deep_struct",
            ty: quote!(Outer),
            values: vec![quote!(Outer {
                middle: Middle {
                    inner: Inner { x: 1 },
                    y: 2
                },
                z: 3
            })],
            inputs: vec![],
        },
        Case {
            name: "all_integers",
            ty: quote!(AllIntegers),
            values: vec![quote!(AllIntegers {
                a_u8: 255,
                a_u16: 65535,
                a_u32: 1_000_000,
                a_u64: 1_000_000_000_000,
                a_u128: 340282366920938463463374607431768211455u128,
                a_usize: 123_456usize,
                a_i8: -128,
                a_i16: -32768,
                a_i32: -1_000_000,
                a_i64: -1_000_000_000_000,
                a_i128: -170141183460469231731687303715884105728i128,
                a_isize: -123_456isize
            })],
            inputs: vec![],
        },
        Case {
            name: "all_scalars",
            ty: quote!(AllScalars),
            values: vec![quote!(AllScalars {
                a_bool: true,
                a_u8: 200,
                a_u16: 1000,
                a_u32: 70000,
                a_u64: 1_000_000_000_000,
                a_u128: 18_446_744_073_709_551_621u128,
                a_usize: 12345,
                a_i8: -42,
                a_i16: -1000,
                a_i32: -70000,
                a_i64: -1_000_000_000_000,
                a_i128: -18_446_744_073_709_551_621i128,
                a_isize: -12345,
                a_f32: 3.14,
                a_f64: 2.718281828459045,
                a_char: 'ß',
                a_name: "hello".into()
            })],
            inputs: vec![],
        },
        Case {
            name: "bool_true_false",
            ty: quote!(Bools),
            values: vec![quote!(Bools { a: true, b: false })],
            inputs: vec![],
        },
        Case {
            name: "integer_boundaries",
            ty: quote!(Boundaries),
            values: vec![quote!(Boundaries {
                u8_max: 255,
                u16_max: 65535,
                i8_min: -128,
                i8_max: 127,
                i16_min: -32768,
                i32_min: -2147483648
            })],
            inputs: vec![],
        },
        Case {
            name: "scalar_u16",
            ty: quote!(u16),
            values: vec![
                quote!(0u16),
                quote!(1u16),
                quote!(127u16),
                quote!(128u16),
                quote!(255u16),
                quote!(16383u16),
                quote!(16384u16),
                quote!(u16::MAX),
            ],
            inputs: vec![],
        },
        Case {
            name: "scalar_u32",
            ty: quote!(u32),
            values: vec![
                quote!(0u32),
                quote!(1u32),
                quote!(127u32),
                quote!(128u32),
                quote!(16383u32),
                quote!(16384u32),
                quote!((1u32 << 21) - 1),
                quote!(1u32 << 21),
                quote!(u32::MAX),
            ],
            inputs: vec![],
        },
        Case {
            name: "scalar_u64",
            ty: quote!(u64),
            values: vec![
                quote!(0u64),
                quote!(1u64),
                quote!(127u64),
                quote!(128u64),
                quote!(16383u64),
                quote!(16384u64),
                quote!((1u64 << 21) - 1),
                quote!(1u64 << 21),
                quote!(u64::MAX),
            ],
            inputs: vec![],
        },
        Case {
            name: "scalar_i16",
            ty: quote!(i16),
            values: vec![
                quote!(i16::MIN),
                quote!(-16384i16),
                quote!(-129i16),
                quote!(-128i16),
                quote!(-1i16),
                quote!(0i16),
                quote!(1i16),
                quote!(127i16),
                quote!(128i16),
                quote!(i16::MAX),
            ],
            inputs: vec![],
        },
        Case {
            name: "scalar_i32",
            ty: quote!(i32),
            values: vec![
                quote!(i32::MIN),
                quote!(-1_000_000i32),
                quote!(-16384i32),
                quote!(-129i32),
                quote!(-128i32),
                quote!(-1i32),
                quote!(0i32),
                quote!(1i32),
                quote!(127i32),
                quote!(128i32),
                quote!(16384i32),
                quote!(1_000_000i32),
                quote!(i32::MAX),
            ],
            inputs: vec![],
        },
        Case {
            name: "scalar_i64",
            ty: quote!(i64),
            values: vec![
                quote!(i64::MIN),
                quote!(-1_000_000_000_000i64),
                quote!(-16384i64),
                quote!(-129i64),
                quote!(-128i64),
                quote!(-1i64),
                quote!(0i64),
                quote!(1i64),
                quote!(127i64),
                quote!(128i64),
                quote!(16384i64),
                quote!(1_000_000_000_000i64),
                quote!(i64::MAX),
            ],
            inputs: vec![],
        },
        Case {
            name: "bool_field",
            ty: quote!(BoolField),
            values: vec![quote!(BoolField { value: true })],
            inputs: vec![],
        },
        Case {
            name: "enum_external",
            ty: quote!(Animal),
            values: vec![
                quote!(Animal::Cat),
                quote!(Animal::Dog {
                    name: "Rex".into(),
                    good_boy: true
                }),
                quote!(Animal::Parrot("Polly".into())),
            ],
            inputs: vec![],
        },
        Case {
            name: "enum_as_struct_field",
            ty: quote!(Zoo),
            values: vec![quote!(Zoo {
                name: "City Zoo".into(),
                star: Animal::Dog {
                    name: "Rex".into(),
                    good_boy: true
                }
            })],
            inputs: vec![],
        },
        Case {
            name: "tuple_pair",
            ty: quote!(Pair),
            values: vec![quote!((42u32, "Alice".to_string()))],
            inputs: vec![],
        },
        Case {
            name: "tuple_triple",
            ty: quote!((u32, u32, u32)),
            values: vec![quote!((1u32, 2u32, 3u32))],
            inputs: vec![],
        },
        Case {
            name: "array_u32_4",
            ty: quote!([u32; 4]),
            values: vec![quote!([10u32, 20u32, 30u32, 40u32])],
            inputs: vec![],
        },
        Case {
            name: "tuple_nested",
            ty: quote!(((u32, u32), u32)),
            values: vec![quote!(((1u32, 2u32), 3u32))],
            inputs: vec![],
        },
        Case {
            name: "vec_scalar_small",
            ty: quote!(ScalarVec),
            values: vec![quote!(ScalarVec {
                values: (0..16).map(|i| i as u32).collect()
            })],
            inputs: vec![],
        },
        Case {
            name: "box_scalar",
            ty: quote!(BoxedScalar),
            values: vec![quote!(BoxedScalar {
                value: Box::new(42)
            })],
            inputs: vec![],
        },
        Case {
            name: "box_string",
            ty: quote!(BoxedString),
            values: vec![quote!(BoxedString {
                name: Box::new("hello".to_string())
            })],
            inputs: vec![],
        },
        Case {
            name: "box_nested",
            ty: quote!(BoxedNested),
            values: vec![quote!(BoxedNested {
                inner: Box::new(Friend {
                    age: 30,
                    name: "Bob".to_string()
                })
            })],
            inputs: vec![],
        },
        Case {
            name: "option_box",
            ty: quote!(OptionBox),
            values: vec![
                quote!(OptionBox {
                    value: Some(Box::new(7))
                }),
                quote!(OptionBox { value: None }),
            ],
            inputs: vec![],
        },
        Case {
            name: "vec_box",
            ty: quote!(VecBox),
            values: vec![quote!(VecBox {
                items: vec![Box::new(1), Box::new(2), Box::new(3)]
            })],
            inputs: vec![],
        },
        Case {
            name: "unit_field",
            ty: quote!(UnitField),
            values: vec![quote!(UnitField {
                geo: (),
                name: "test".into()
            })],
            inputs: vec![],
        },
        Case {
            name: "vec_scalar_large",
            ty: quote!(ScalarVec),
            values: vec![quote!(ScalarVec {
                values: (0..2048).map(|i| i as u32).collect()
            })],
            inputs: vec![],
        },
        Case {
            name: "flatten",
            ty: quote!(Document),
            values: vec![quote!(Document {
                title: "Hello".into(),
                meta: Metadata {
                    version: 1,
                    author: "Amos".into()
                }
            })],
            inputs: vec![],
        },
        Case {
            name: "option_u32",
            ty: quote!(WithOptU32),
            values: vec![
                quote!(WithOptU32 { value: Some(42) }),
                quote!(WithOptU32 { value: None }),
            ],
            inputs: vec![],
        },
        Case {
            name: "option_string",
            ty: quote!(WithOptStr),
            values: vec![
                quote!(WithOptStr {
                    name: Some("Alice".into())
                }),
                quote!(WithOptStr { name: None }),
            ],
            inputs: vec![],
        },
        Case {
            name: "option_struct",
            ty: quote!(WithOptAddr),
            values: vec![
                quote!(WithOptAddr {
                    addr: Some(Address {
                        city: "Portland".into(),
                        zip: 97201
                    })
                }),
                quote!(WithOptAddr { addr: None }),
            ],
            inputs: vec![],
        },
        Case {
            name: "multi_options",
            ty: quote!(MultiOpt),
            values: vec![quote!(MultiOpt {
                a: Some(7),
                b: "hello".into(),
                c: None
            })],
            inputs: vec![],
        },
        Case {
            name: "vec_u32",
            ty: quote!(Nums),
            values: vec![
                quote!(Nums {
                    vals: vec![1, 2, 3]
                }),
                quote!(Nums { vals: vec![] }),
            ],
            inputs: vec![],
        },
        Case {
            name: "vec_string",
            ty: quote!(Names),
            values: vec![
                quote!(Names {
                    items: vec!["hello".into(), "world".into()]
                }),
                quote!(Names { items: vec![] }),
            ],
            inputs: vec![],
        },
        Case {
            name: "vec_nested_struct",
            ty: quote!(AddressList),
            values: vec![quote!(AddressList {
                addrs: vec![
                    Address {
                        city: "Portland".into(),
                        zip: 97201
                    },
                    Address {
                        city: "Seattle".into(),
                        zip: 98101
                    },
                ]
            })],
            inputs: vec![],
        },
        Case {
            name: "rename_field",
            ty: quote!(RenameField),
            values: vec![quote!(RenameField {
                name: "Alice".into(),
                age: 30
            })],
            inputs: vec![],
        },
        Case {
            name: "deny_unknown_fields",
            ty: quote!(Strict),
            values: vec![
                quote!(Strict { x: 10, y: 20 }),
                quote!(Strict { x: 128, y: 0 }),
                quote!(Strict { x: 0, y: 128 }),
                quote!(Strict { x: 16384, y: 16384 }),
            ],
            inputs: vec![],
        },
        Case {
            name: "default_field",
            ty: quote!(WithDefault),
            values: vec![quote!(WithDefault {
                name: "hello".into(),
                score: 7
            })],
            inputs: vec![],
        },
        Case {
            name: "transparent_scalar",
            ty: quote!(Wrapper),
            values: vec![quote!(Wrapper(42))],
            inputs: vec![],
        },
        Case {
            name: "transparent_string",
            ty: quote!(StringWrapper),
            values: vec![quote!(StringWrapper("hello".into()))],
            inputs: vec![],
        },
        Case {
            name: "transparent_composite",
            ty: quote!(StructWrapper),
            values: vec![quote!(StructWrapper(Friend {
                age: 25,
                name: "Eve".into()
            }))],
            inputs: vec![],
        },
        Case {
            name: "shared_inner_type",
            ty: quote!(TwoAddresses),
            values: vec![quote!(TwoAddresses {
                home: Address {
                    city: "Portland".into(),
                    zip: 97201
                },
                work: Address {
                    city: "Seattle".into(),
                    zip: 98101
                }
            })],
            inputs: vec![],
        },
    ]
}

fn all_cases() -> Vec<Case> {
    let mut out = cases();
    out.extend(input_cases());
    out.push(
        CaseBuilder::new("from_str_entrypoint", quote!(Friend))
            .input_ok(
                WireFormat::Postcard,
                "postcard",
                b"*\x05Alice",
                quote!(Friend {
                    age: 42,
                    name: "Alice".into()
                }),
            )
            .build(),
    );
    out.push(
        CaseBuilder::new("enum_unknown_discriminant", quote!(Animal))
            .input_err(
                WireFormat::Postcard,
                "postcard",
                b"\x63",
                quote!(kajit::context::ErrorCode::UnknownVariant),
            )
            .build(),
    );
    out
}

fn normalized_case_ty(case: &Case) -> String {
    case.ty
        .to_string()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

fn is_pointer_case_type(ty: &str) -> bool {
    matches!(
        ty,
        "RcScalar"
            | "ArcScalar"
            | "BoxedScalar"
            | "BoxedString"
            | "BoxedNested"
            | "OptionBox"
            | "VecBox"
    )
}

fn is_map_case_type(ty: &str) -> bool {
    matches!(ty, "ConfigMap" | "EnvMap" | "BTreeConfigMap")
}

fn is_default_or_skip_case_type(ty: &str) -> bool {
    matches!(
        ty,
        "WithDefault"
            | "WithDefaultString"
            | "AllDefault"
            | "WithSkip"
            | "WithSkipDeser"
            | "SkipWithCustomDefault"
    )
}

fn is_json_enum_case_type(ty: &str) -> bool {
    matches!(
        ty,
        "Animal"
            | "AdjAnimal"
            | "IntAnimal"
            | "UntaggedAnimal"
            | "UntaggedConfig"
            | "ApiResponse"
            | "Zoo"
    )
}

fn is_json_option_case_type(ty: &str) -> bool {
    matches!(ty, "WithOptU32" | "WithOptStr" | "WithOptAddr" | "MultiOpt")
}

fn is_array_case_type(ty: &str) -> bool {
    ty.starts_with('[') && ty.ends_with(']') && ty.contains(';')
}

fn is_json_flatten_case(case: &Case) -> bool {
    case.name == "flatten"
}

fn is_json_rename_case(case: &Case) -> bool {
    case.name.starts_with("rename_field")
}

fn is_json_sequence_case(case: &Case) -> bool {
    case.name.starts_with("vec_") || case.name.starts_with("tuple_") || case.name == "unit_field"
}

fn is_postcard_wide_scalar_case(case: &Case) -> bool {
    matches!(
        case.name,
        "all_integers" | "all_scalars" | "integer_boundaries"
    )
}

fn unsupported_reason_for_format(case: &Case, format: WireFormat) -> Option<String> {
    let ty = normalized_case_ty(case);
    let reason = match format {
        WireFormat::Json => {
            if is_json_enum_case_type(&ty) {
                Some("json enum lowering is not implemented yet")
            } else if is_json_option_case_type(&ty) {
                Some("json option lowering is not implemented yet")
            } else if is_map_case_type(&ty) {
                Some("map lowering is not implemented in IR path yet")
            } else if is_pointer_case_type(&ty) {
                Some("pointer lowering is not implemented in IR path yet")
            } else if is_default_or_skip_case_type(&ty) {
                Some("default/skip field lowering is not implemented in IR path yet")
            } else if is_array_case_type(&ty) {
                Some("json array lowering is not implemented in IR path yet")
            } else if is_json_sequence_case(case) {
                Some("json sequence lowering is not implemented in IR path yet")
            } else if is_json_flatten_case(case) {
                Some("json flatten lowering is not implemented in IR path yet")
            } else if is_json_rename_case(case) {
                Some("json rename lowering is not implemented in IR path yet")
            } else {
                None
            }
        }
        WireFormat::Postcard => {
            if is_map_case_type(&ty) {
                Some("map lowering is not implemented in IR path yet")
            } else if is_pointer_case_type(&ty) {
                Some("pointer lowering is not implemented in IR path yet")
            } else if is_default_or_skip_case_type(&ty) {
                Some("default/skip field lowering is not implemented in IR path yet")
            } else if is_postcard_wide_scalar_case(case) {
                Some("postcard wide scalar aggregate lowering is not implemented in IR path yet")
            } else {
                None
            }
        }
    }?;

    Some(format!("{reason}; case={}, type={ty}", case.name))
}

fn ignore_attr_for_format(case: &Case, format: WireFormat) -> Option<TokenStream> {
    unsupported_reason_for_format(case, format).map(|msg| {
        let lit = LitStr::new(&msg, Span::call_site());
        quote!(#[ignore = #lit])
    })
}

fn ignore_attr_for_prop_case(case: &Case) -> Option<TokenStream> {
    let mut reasons = Vec::new();
    if let Some(reason) = unsupported_reason_for_format(case, WireFormat::Json) {
        reasons.push(format!("json: {reason}"));
    }
    if let Some(reason) = unsupported_reason_for_format(case, WireFormat::Postcard) {
        reasons.push(format!("postcard: {reason}"));
    }
    if reasons.is_empty() {
        None
    } else {
        let msg = reasons.join(" | ");
        let lit = LitStr::new(&msg, Span::call_site());
        Some(quote!(#[ignore = #lit]))
    }
}

macro_rules! json_case_spec {
    (
        name: $name:expr,
        ty: $ty:expr,
        input: $input:expr,
        expected: Some($expected:expr),
        expected_error_code: None $(,)?
    ) => {
        CaseBuilder::new($name, $ty)
            .json_ok($name, $input, $expected)
            .build()
    };
    (
        name: $name:expr,
        ty: $ty:expr,
        input: $input:expr,
        expected: None,
        expected_error_code: Some($err:expr) $(,)?
    ) => {
        CaseBuilder::new($name, $ty)
            .json_err($name, $input, $err)
            .build()
    };
    (
        name: $name:expr,
        ty: $ty:expr,
        input: $input:expr,
        expected: None,
        expected_error_code: None $(,)?
    ) => {
        CaseBuilder::new($name, $ty)
            .json_err_any($name, $input)
            .build()
    };
}

fn input_cases() -> Vec<Case> {
    vec![
        json_case_spec! {
            name: "reversed_key_order",
            ty: quote!(Friend),
            input: r#"{"name": "Alice", "age": 42}"#,
            expected: Some(quote!(Friend {
                age: 42,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "unknown_keys_skipped",
            ty: quote!(Friend),
            input: r#"{"age": 42, "extra": true, "name": "Alice"}"#,
            expected: Some(quote!(Friend {
                age: 42,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "empty_object_missing_fields",
            ty: quote!(Friend),
            input: r#"{}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::MissingRequiredField)),
        },
        json_case_spec! {
            name: "nested_struct_reversed_keys",
            ty: quote!(Person),
            input: r#"{"address": {"zip": 97201, "city": "Portland"}, "age": 30, "name": "Alice"}"#,
            expected: Some(quote!(Person {
                name: "Alice".into(),
                age: 30,
                address: Address {
                    city: "Portland".into(),
                    zip: 97201
                }
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "flatten_reversed_keys",
            ty: quote!(Document),
            input: r#"{"author": "Amos", "version": 1, "title": "Hello"}"#,
            expected: Some(quote!(Document {
                title: "Hello".into(),
                meta: Metadata {
                    version: 1,
                    author: "Amos".into()
                }
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "enum_struct_variant_reversed_keys",
            ty: quote!(Animal),
            input: r#"{"Dog": {"good_boy": true, "name": "Rex"}}"#,
            expected: Some(quote!(Animal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "enum_unit_as_string",
            ty: quote!(Animal),
            input: r#""Cat""#,
            expected: Some(quote!(Animal::Cat)),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "enum_struct_variant",
            ty: quote!(Animal),
            input: r#"{"Dog": {"name": "Rex", "good_boy": true}}"#,
            expected: Some(quote!(Animal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "enum_tuple_variant",
            ty: quote!(Animal),
            input: r#"{"Parrot": "Polly"}"#,
            expected: Some(quote!(Animal::Parrot("Polly".into()))),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "enum_unit_in_object",
            ty: quote!(Animal),
            input: r#"{"Cat": null}"#,
            expected: Some(quote!(Animal::Cat)),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "enum_unknown_variant",
            ty: quote!(Animal),
            input: r#""Snake""#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::UnknownVariant)),
        },
        json_case_spec! {
            name: "adjacent_unit_no_content",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Cat"}"#,
            expected: Some(quote!(AdjAnimal::Cat)),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "adjacent_unit_with_null_content",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Cat", "data": null}"#,
            expected: Some(quote!(AdjAnimal::Cat)),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "adjacent_struct_variant",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Dog", "data": {"name": "Rex", "good_boy": true}}"#,
            expected: Some(quote!(AdjAnimal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "adjacent_struct_variant_reversed_fields",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Dog", "data": {"good_boy": true, "name": "Rex"}}"#,
            expected: Some(quote!(AdjAnimal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "adjacent_tuple_variant",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Parrot", "data": "Polly"}"#,
            expected: Some(quote!(AdjAnimal::Parrot("Polly".into()))),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "adjacent_unknown_variant",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Snake", "data": null}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::UnknownVariant)),
        },
        json_case_spec! {
            name: "adjacent_wrong_first_key",
            ty: quote!(AdjAnimal),
            input: r#"{"data": null, "type": "Cat"}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::ExpectedTagKey)),
        },
        json_case_spec! {
            name: "internal_struct_variant_reversed_fields",
            ty: quote!(IntAnimal),
            input: r#"{"type": "Dog", "good_boy": true, "name": "Rex"}"#,
            expected: Some(quote!(IntAnimal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "internal_unit_variant",
            ty: quote!(IntAnimal),
            input: r#"{"type": "Cat"}"#,
            expected: Some(quote!(IntAnimal::Cat)),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "internal_struct_variant",
            ty: quote!(IntAnimal),
            input: r#"{"type": "Dog", "name": "Rex", "good_boy": true}"#,
            expected: Some(quote!(IntAnimal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "internal_unknown_variant",
            ty: quote!(IntAnimal),
            input: r#"{"type": "Snake"}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::UnknownVariant)),
        },
        json_case_spec! {
            name: "internal_wrong_first_key",
            ty: quote!(IntAnimal),
            input: r#"{"name": "Rex", "type": "Dog", "good_boy": true}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::ExpectedTagKey)),
        },
        json_case_spec! {
            name: "untagged_struct_reversed_keys",
            ty: quote!(UntaggedAnimal),
            input: r#"{"good_boy": false, "name": "Rex"}"#,
            expected: Some(quote!(UntaggedAnimal::Dog {
                name: "Rex".into(),
                good_boy: false
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "untagged_solver_key_order_independent",
            ty: quote!(UntaggedConfig),
            input: r#"{"db": 0, "host": "localhost"}"#,
            expected: Some(quote!(UntaggedConfig::Redis {
                host: "localhost".into(),
                db: 0
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "untagged_nested_key_order_independent",
            ty: quote!(ApiResponse),
            input: r#"{"data": {"items": 5}, "status": 200}"#,
            expected: Some(quote!(ApiResponse::Success {
                status: 200,
                data: SuccessPayload { items: 5 }
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "option_some_scalar",
            ty: quote!(WithOptU32),
            input: r#"{"value": 42}"#,
            expected: Some(quote!(WithOptU32 { value: Some(42) })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "option_some_string",
            ty: quote!(WithOptStr),
            input: r#"{"name": "Alice"}"#,
            expected: Some(quote!(WithOptStr {
                name: Some("Alice".into())
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "option_some_struct",
            ty: quote!(WithOptAddr),
            input: r#"{"addr": {"city": "Portland", "zip": 97201}}"#,
            expected: Some(quote!(WithOptAddr {
                addr: Some(Address {
                    city: "Portland".into(),
                    zip: 97201
                })
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "option_none_struct",
            ty: quote!(WithOptAddr),
            input: r#"{"addr": null}"#,
            expected: Some(quote!(WithOptAddr { addr: None })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "option_none_scalar",
            ty: quote!(WithOptU32),
            input: r#"{"value": null}"#,
            expected: Some(quote!(WithOptU32 { value: None })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "option_none_string",
            ty: quote!(WithOptStr),
            input: r#"{"name": null}"#,
            expected: Some(quote!(WithOptStr { name: None })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "option_reversed_keys",
            ty: quote!(MultiOpt),
            input: r#"{"c": "world", "b": "hello", "a": null}"#,
            expected: Some(quote!(MultiOpt {
                a: None,
                b: "hello".into(),
                c: Some("world".into())
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "rc_scalar",
            ty: quote!(RcScalar),
            input: r#"{"value": 77}"#,
            expected: Some(quote!(RcScalar {
                value: std::rc::Rc::new(77)
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "rename_field",
            ty: quote!(RenameField),
            input: r#"{"user_name": "Alice", "age": 30}"#,
            expected: Some(quote!(RenameField {
                name: "Alice".into(),
                age: 30
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "rename_field_original_name_rejected",
            ty: quote!(RenameField),
            input: r#"{"name": "Alice", "age": 30}"#,
            expected: None,
            expected_error_code: None,
        },
        json_case_spec! {
            name: "rename_all_camel_case",
            ty: quote!(CamelCaseStruct),
            input: r#"{"userName": "Bob", "birthYear": 1990}"#,
            expected: Some(quote!(CamelCaseStruct {
                user_name: "Bob".into(),
                birth_year: 1990
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "deny_unknown_fields_rejects",
            ty: quote!(Strict),
            input: r#"{"x": 1, "y": 2, "z": 3}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::UnknownField)),
        },
        json_case_spec! {
            name: "deny_unknown_fields_allows_known",
            ty: quote!(Strict),
            input: r#"{"x": 1, "y": 2}"#,
            expected: Some(quote!(Strict { x: 1, y: 2 })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "default_field_missing",
            ty: quote!(WithDefault),
            input: r#"{"name": "Alice"}"#,
            expected: Some(quote!(WithDefault {
                name: "Alice".into(),
                score: 0
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "default_field_present",
            ty: quote!(WithDefault),
            input: r#"{"name": "Alice", "score": 99}"#,
            expected: Some(quote!(WithDefault {
                name: "Alice".into(),
                score: 99
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "default_field_required_still_errors",
            ty: quote!(WithDefault),
            input: r#"{"score": 50}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::MissingRequiredField)),
        },
        json_case_spec! {
            name: "default_string_field",
            ty: quote!(WithDefaultString),
            input: r#"{"value": 42}"#,
            expected: Some(quote!(WithDefaultString {
                label: String::new(),
                value: 42
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "container_default_empty_object",
            ty: quote!(AllDefault),
            input: r#"{}"#,
            expected: Some(quote!(AllDefault { x: 0, y: 0 })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "container_default_partial",
            ty: quote!(AllDefault),
            input: r#"{"x": 5}"#,
            expected: Some(quote!(AllDefault { x: 5, y: 0 })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "skip_field",
            ty: quote!(WithSkip),
            input: r#"{"name": "Alice"}"#,
            expected: Some(quote!(WithSkip {
                name: "Alice".into(),
                cached: 0
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "skip_field_in_input_treated_as_unknown",
            ty: quote!(WithSkip),
            input: r#"{"name": "Alice", "cached": 99}"#,
            expected: Some(quote!(WithSkip {
                name: "Alice".into(),
                cached: 0
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "skip_deserializing_field",
            ty: quote!(WithSkipDeser),
            input: r#"{"name": "Bob"}"#,
            expected: Some(quote!(WithSkipDeser {
                name: "Bob".into(),
                internal: 0
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "skip_with_custom_default",
            ty: quote!(SkipWithCustomDefault),
            input: r#"{"value": 10}"#,
            expected: Some(quote!(SkipWithCustomDefault {
                value: 10,
                magic: 42
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "arc_scalar",
            ty: quote!(ArcScalar),
            input: r#"{"value": 99}"#,
            expected: Some(quote!(ArcScalar {
                value: std::sync::Arc::new(99)
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "u8_out_of_range",
            ty: quote!(Tiny),
            input: r#"{"val": 256}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::NumberOutOfRange)),
        },
        json_case_spec! {
            name: "float_scientific",
            ty: quote!(Floats),
            input: r#"{"a": 1.5e2, "b": -3.14}"#,
            expected: Some(quote!(Floats { a: 150.0, b: -3.14 })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "string_escape_newline",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "hello\nworld"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "hello\nworld".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "string_escape_tab",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "hello\tworld"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "hello\tworld".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "string_escape_backslash",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "hello\\world"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "hello\\world".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "string_escape_quote",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "hello\"world"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "hello\"world".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "string_escape_all_simple",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "a\"b\\c\/d\be\ff\ng\rh\ti"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "a\"b\\c/d\x08e\x0Cf\ng\rh\ti".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "string_unicode_escape_bmp",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "\u0041lice"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "string_unicode_escape_non_ascii",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "caf\u00E9"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "caf\u{00E9}".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "string_unicode_surrogate_pair",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "\uD83D\uDE00"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "\u{1F600}".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "key_with_unicode_escape",
            ty: quote!(Friend),
            input: r#"{"age": 42, "na\u006De": "Alice"}"#,
            expected: Some(quote!(Friend {
                age: 42,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "string_invalid_escape",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "hello\xworld"}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::InvalidEscapeSequence)),
        },
        json_case_spec! {
            name: "string_lone_high_surrogate",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "\uD800"}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::InvalidEscapeSequence)),
        },
        json_case_spec! {
            name: "string_truncated_unicode",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "\u00"}"#,
            expected: None,
            expected_error_code: None,
        },
        json_case_spec! {
            name: "skip_value_with_unicode_escape",
            ty: quote!(Friend),
            input: r#"{"age": 42, "extra": "test\uD83D\uDE00end", "name": "Alice"}"#,
            expected: Some(quote!(Friend {
                age: 42,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "skip_value_with_backslash_escape",
            ty: quote!(Friend),
            input: r#"{"age": 42, "extra": "test\n\t\\end", "name": "Alice"}"#,
            expected: Some(quote!(Friend {
                age: 42,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "map_string_to_u32",
            ty: quote!(ConfigMap),
            input: r#"{"scores": {"alice": 42, "bob": 7}}"#,
            expected: Some(quote!(ConfigMap {
                scores: std::collections::HashMap::from([
                    ("alice".to_string(), 42u32),
                    ("bob".to_string(), 7u32),
                ])
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "map_empty",
            ty: quote!(ConfigMap),
            input: r#"{"scores": {}}"#,
            expected: Some(quote!(ConfigMap {
                scores: std::collections::HashMap::new()
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "map_string_to_string",
            ty: quote!(EnvMap),
            input: r#"{"vars": {"HOME": "/root", "PATH": "/usr/bin"}}"#,
            expected: Some(quote!(EnvMap {
                vars: std::collections::HashMap::from([
                    ("HOME".to_string(), "/root".to_string()),
                    ("PATH".to_string(), "/usr/bin".to_string()),
                ])
            })),
            expected_error_code: None,
        },
        json_case_spec! {
            name: "map_growth",
            ty: quote!(ConfigMap),
            input: r#"{"scores": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}"#,
            expected: Some(quote!(ConfigMap {
                scores: std::collections::HashMap::from([
                    ("a".to_string(), 1u32),
                    ("b".to_string(), 2u32),
                    ("c".to_string(), 3u32),
                    ("d".to_string(), 4u32),
                    ("e".to_string(), 5u32),
                    ("f".to_string(), 6u32),
                ])
            })),
            expected_error_code: None,
        },
    ]
}

fn panic_cases() -> Vec<PanicCase> {
    vec![PanicCase {
        name: "flatten_name_collision",
        expected: "field name collision",
        body: quote! {
            #[derive(Facet)]
            struct Collider {
                x: u32,
            }

            #[derive(Facet)]
            struct HasCollision {
                x: u32,
                #[facet(flatten)]
                inner: Collider,
            }

            kajit::compile_decoder(HasCollision::SHAPE, &kajit::json::KajitJson);
        },
    }]
}

pub(crate) fn render_bench_file() -> String {
    let cases = all_cases();
    let types = types_rs();
    let bench_calls: Vec<TokenStream> = cases
        .iter()
        .filter_map(|case| {
            let value = case.values.first()?.clone();
            let sample_name = case.name.to_string();
            let enable_json_kajit = unsupported_reason_for_format(case, WireFormat::Json).is_none();
            let enable_postcard_kajit =
                unsupported_reason_for_format(case, WireFormat::Postcard).is_none();
            Some(quote! {
                register_bench_case(
                    &mut v,
                    #sample_name,
                    #value,
                    #enable_json_kajit,
                    #enable_postcard_kajit,
                );
            })
        })
        .collect();
    let file_tokens = quote! {
        #[path = "harness.rs"]
        mod harness;

        use facet::Facet;
        use std::hint::black_box;
        use std::panic::{catch_unwind, AssertUnwindSafe};
        use std::sync::Arc;

        #types

        fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
            if let Some(msg) = payload.downcast_ref::<&str>() {
                (*msg).to_owned()
            } else if let Some(msg) = payload.downcast_ref::<String>() {
                msg.clone()
            } else {
                "non-string panic payload".to_owned()
            }
        }

        fn register_bench_case<T>(
            v: &mut Vec<harness::Bench>,
            group: &str,
            value: T,
            enable_json_kajit: bool,
            enable_postcard_kajit: bool,
        )
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + 'static,
        {
            let json_data = Arc::new(serde_json::to_string(&value).unwrap());
            let postcard_data = Arc::new(postcard::to_allocvec(&value).unwrap());
            let value = Arc::new(value);

            let json_prefix = format!("{group}/json");
            let postcard_prefix = format!("{group}/postcard");

            v.push(harness::Bench {
                name: format!("{json_prefix}/serde_deser"),
                func: Box::new({
                    let data = Arc::clone(&json_data);
                    move |runner| {
                        runner.run(|| {
                            black_box(serde_json::from_str::<T>(black_box(data.as_str())).unwrap());
                        });
                    }
                }),
            });
            if enable_json_kajit {
                let json_decoder =
                    match catch_unwind(AssertUnwindSafe(|| {
                        kajit::compile_decoder(T::SHAPE, &kajit::json::KajitJson)
                    })) {
                        Ok(decoder) => Some(Arc::new(decoder)),
                        Err(payload) => {
                            eprintln!(
                                "skipping {json_prefix}/kajit_deser: compile unsupported ({})",
                                panic_payload_to_string(payload)
                            );
                            None
                        }
                    };

                if let Some(decoder) = json_decoder {
                    match catch_unwind(AssertUnwindSafe(|| {
                        kajit::from_str::<T>(decoder.as_ref(), json_data.as_str())
                    })) {
                        Ok(Ok(_)) => {
                            v.push(harness::Bench {
                                name: format!("{json_prefix}/kajit_deser"),
                        func: Box::new({
                            let data = Arc::clone(&json_data);
                            let decoder = Arc::clone(&decoder);
                            move |runner| {
                                let decoder = decoder.as_ref();
                                runner.run(|| {
                                    black_box(
                                        kajit::from_str::<T>(
                                            decoder,
                                            black_box(data.as_str()),
                                        )
                                        .unwrap(),
                                    );
                                });
                                    }
                                }),
                            });
                        }
                        Ok(Err(err)) => {
                            eprintln!(
                                "skipping {json_prefix}/kajit_deser: preflight decode failed ({err:?})"
                            );
                        }
                        Err(payload) => {
                            eprintln!(
                                "skipping {json_prefix}/kajit_deser: preflight panic ({})",
                                panic_payload_to_string(payload)
                            );
                        }
                    }
                }
            }
            v.push(harness::Bench {
                name: format!("{json_prefix}/serde_ser"),
                func: Box::new({
                    let value = Arc::clone(&value);
                    move |runner| {
                        runner.run(|| {
                            black_box(serde_json::to_vec(black_box(&*value)).unwrap());
                        });
                    }
                }),
            });

            v.push(harness::Bench {
                name: format!("{postcard_prefix}/serde_deser"),
                func: Box::new({
                    let data = Arc::clone(&postcard_data);
                    move |runner| {
                        runner.run(|| {
                            black_box(postcard::from_bytes::<T>(black_box(&data[..])).unwrap());
                        });
                    }
                }),
            });
            if enable_postcard_kajit {
                let postcard_decoder =
                    match catch_unwind(AssertUnwindSafe(|| {
                        kajit::compile_decoder(T::SHAPE, &kajit::postcard::KajitPostcard)
                    })) {
                        Ok(decoder) => Some(Arc::new(decoder)),
                        Err(payload) => {
                            eprintln!(
                                "skipping {postcard_prefix}/kajit_deser: compile unsupported ({})",
                                panic_payload_to_string(payload)
                            );
                            None
                        }
                    };

                if let Some(decoder) = postcard_decoder {
                    match catch_unwind(AssertUnwindSafe(|| {
                        kajit::deserialize::<T>(decoder.as_ref(), &postcard_data[..])
                    })) {
                        Ok(Ok(_)) => {
                            v.push(harness::Bench {
                                name: format!("{postcard_prefix}/kajit_deser"),
                        func: Box::new({
                            let data = Arc::clone(&postcard_data);
                            let decoder = Arc::clone(&decoder);
                            move |runner| {
                                let decoder = decoder.as_ref();
                                runner.run(|| {
                                    black_box(
                                        kajit::deserialize::<T>(
                                            decoder,
                                            black_box(&data[..]),
                                        )
                                        .unwrap(),
                                    );
                                });
                                    }
                                }),
                            });
                        }
                        Ok(Err(err)) => {
                            eprintln!(
                                "skipping {postcard_prefix}/kajit_deser: preflight decode failed ({err:?})"
                            );
                        }
                        Err(payload) => {
                            eprintln!(
                                "skipping {postcard_prefix}/kajit_deser: preflight panic ({})",
                                panic_payload_to_string(payload)
                            );
                        }
                    }
                }
            }
            v.push(harness::Bench {
                name: format!("{postcard_prefix}/serde_ser"),
                func: Box::new({
                    let value = Arc::clone(&value);
                    move |runner| {
                        runner.run(|| {
                            black_box(postcard::to_allocvec(black_box(&*value)).unwrap());
                        });
                    }
                }),
            });
        }

        fn main() {
            let mut v: Vec<harness::Bench> = Vec::new();
            #(#bench_calls)*
            harness::run_benchmarks(v);
        }
    };
    let file: syn::File =
        syn::parse2(file_tokens).expect("generated synthetic bench file should parse");
    format!(
        "// @generated by xtask generate-synthetic. Do not edit manually.\n{}",
        prettyplease::unparse(&file)
    )
}

pub(crate) fn render_test_file() -> String {
    let cases = all_cases();
    let panic_cases = panic_cases();
    let types = types_rs();
    let json_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values.iter().enumerate().map(|(sample_idx, value)| {
                let test_name = if case.values.len() == 1 {
                    format_ident!("{}", case.name)
                } else {
                    format_ident!("{}_v{}", case.name, sample_idx)
                };
                let value = value.clone();
                let ignore_attr = ignore_attr_for_format(case, WireFormat::Json);
                quote! {
                    #[test]
                    #ignore_attr
                    fn #test_name() {
                        let value = #value;
                        assert_json_case(value);
                    }
                }
            })
        })
        .collect();
    let json_rvsdg_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values
                .iter()
                .enumerate()
                .map(|(sample_idx, value)| {
                    let test_name = if case.values.len() == 1 {
                        format_ident!("{}", case.name)
                    } else {
                        format_ident!("{}_v{}", case.name, sample_idx)
                    };
                    let case_name = if case.values.len() == 1 {
                        case.name.to_string()
                    } else {
                        format!("{}__v{}", case.name, sample_idx)
                    };
                    let value = value.clone();
                    let ignore_attr = ignore_attr_for_format(case, WireFormat::Json);
                    quote! {
                        #[test]
                        #ignore_attr
                        fn #test_name() {
                            let value = #value;
                            assert_codegen_rvsdg_snapshot("json", #case_name, &kajit::json::KajitJson, &value);
                        }
                    }
                })
        })
        .collect();
    let json_ra_mir_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values
                .iter()
                .enumerate()
                .map(|(sample_idx, value)| {
                    let test_name = if case.values.len() == 1 {
                        format_ident!("{}", case.name)
                    } else {
                        format_ident!("{}_v{}", case.name, sample_idx)
                    };
                    let case_name = if case.values.len() == 1 {
                        case.name.to_string()
                    } else {
                        format!("{}__v{}", case.name, sample_idx)
                    };
                    let value = value.clone();
                    let ignore_attr = ignore_attr_for_format(case, WireFormat::Json);
                    quote! {
                        #[test]
                        #ignore_attr
                        fn #test_name() {
                            let value = #value;
                            assert_codegen_ra_mir_snapshot("json", #case_name, &kajit::json::KajitJson, &value);
                        }
                    }
                })
        })
        .collect();
    let json_postreg_edits_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values
                .iter()
                .enumerate()
                .map(|(sample_idx, value)| {
                    let test_name = if case.values.len() == 1 {
                        format_ident!("{}", case.name)
                    } else {
                        format_ident!("{}_v{}", case.name, sample_idx)
                    };
                    let case_name = if case.values.len() == 1 {
                        case.name.to_string()
                    } else {
                        format!("{}__v{}", case.name, sample_idx)
                    };
                    let value = value.clone();
                    let ignore_attr = ignore_attr_for_format(case, WireFormat::Json);
                    quote! {
                        #[test]
                        #ignore_attr
                        fn #test_name() {
                            let value = #value;
                            assert_codegen_edits_snapshot("json", #case_name, &kajit::json::KajitJson, &value);
                        }
                    }
                })
        })
        .collect();
    let postcard_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values.iter().enumerate().map(|(sample_idx, value)| {
                let test_name = if case.values.len() == 1 {
                    format_ident!("{}", case.name)
                } else {
                    format_ident!("{}_v{}", case.name, sample_idx)
                };
                let value = value.clone();
                let ignore_attr = ignore_attr_for_format(case, WireFormat::Postcard);
                quote! {
                    #[test]
                    #ignore_attr
                    fn #test_name() {
                        let value = #value;
                        assert_postcard_case(value);
                    }
                }
            })
        })
        .collect();
    let postcard_rvsdg_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values
                .iter()
                .enumerate()
                .map(|(sample_idx, value)| {
                    let test_name = if case.values.len() == 1 {
                        format_ident!("{}", case.name)
                    } else {
                        format_ident!("{}_v{}", case.name, sample_idx)
                    };
                    let case_name = if case.values.len() == 1 {
                        case.name.to_string()
                    } else {
                        format!("{}__v{}", case.name, sample_idx)
                    };
                    let value = value.clone();
                    let ignore_attr = ignore_attr_for_format(case, WireFormat::Postcard);
                    quote! {
                        #[test]
                        #ignore_attr
                        fn #test_name() {
                            let value = #value;
                            assert_codegen_rvsdg_snapshot("postcard", #case_name, &kajit::postcard::KajitPostcard, &value);
                        }
                    }
                })
        })
        .collect();
    let postcard_ra_mir_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values
                .iter()
                .enumerate()
                .map(|(sample_idx, value)| {
                    let test_name = if case.values.len() == 1 {
                        format_ident!("{}", case.name)
                    } else {
                        format_ident!("{}_v{}", case.name, sample_idx)
                    };
                    let case_name = if case.values.len() == 1 {
                        case.name.to_string()
                    } else {
                        format!("{}__v{}", case.name, sample_idx)
                    };
                    let value = value.clone();
                    let ignore_attr = ignore_attr_for_format(case, WireFormat::Postcard);
                    quote! {
                        #[test]
                        #ignore_attr
                        fn #test_name() {
                            let value = #value;
                            assert_codegen_ra_mir_snapshot("postcard", #case_name, &kajit::postcard::KajitPostcard, &value);
                        }
                    }
                })
        })
        .collect();
    let postcard_postreg_edits_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values
                .iter()
                .enumerate()
                .map(|(sample_idx, value)| {
                    let test_name = if case.values.len() == 1 {
                        format_ident!("{}", case.name)
                    } else {
                        format_ident!("{}_v{}", case.name, sample_idx)
                    };
                    let case_name = if case.values.len() == 1 {
                        case.name.to_string()
                    } else {
                        format!("{}__v{}", case.name, sample_idx)
                    };
                    let value = value.clone();
                    let ignore_attr = ignore_attr_for_format(case, WireFormat::Postcard);
                    quote! {
                        #[test]
                        #ignore_attr
                        fn #test_name() {
                            let value = #value;
                            assert_codegen_edits_snapshot("postcard", #case_name, &kajit::postcard::KajitPostcard, &value);
                        }
                    }
                })
        })
        .collect();
    let prop_tests: Vec<TokenStream> = cases
        .iter()
        .filter(|case| !case.values.is_empty())
        .map(|case| {
            let test_name = format_ident!("{}", case.name);
            let value = case
                .values
                .first()
                .cloned()
                .expect("each case should define at least one sample value");
            let ignore_attr = ignore_attr_for_prop_case(case);
            quote! {
                #[test]
                #ignore_attr
                fn #test_name() {
                    let marker = #value;
                    assert_prop_case(&marker);
                }
            }
        })
        .collect();
    let json_input_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.inputs
                .iter()
                .filter(|input| input.format == WireFormat::Json)
                .map(|input| {
                    let test_name = if case.inputs.len() == 1 {
                        format_ident!("{}", case.name)
                    } else {
                        format_ident!("{}_{}", case.name, input.name)
                    };
                    let ty = case.ty.clone();
                    let input_bytes = LitByteStr::new(input.input, Span::call_site());
                    let ignore_attr = ignore_attr_for_format(case, WireFormat::Json);
                    match &input.expect {
                        DecodeExpectation::Ok(expected) => {
                            let expected = expected.clone();
                            quote! {
                                #[test]
                                #ignore_attr
                                fn #test_name() {
                                    assert_json_input_case::<#ty>(#input_bytes, #expected);
                                }
                            }
                        }
                        DecodeExpectation::Err(expected_error_code) => {
                            let expected_error_code = expected_error_code.clone();
                            quote! {
                                #[test]
                                #ignore_attr
                                fn #test_name() {
                                    assert_json_input_err_code::<#ty>(#input_bytes, #expected_error_code);
                                }
                            }
                        }
                        DecodeExpectation::AnyErr => {
                            quote! {
                                #[test]
                                #ignore_attr
                                fn #test_name() {
                                    assert_json_input_err::<#ty>(#input_bytes);
                                }
                            }
                        }
                    }
                })
        })
        .collect();
    let postcard_input_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.inputs
                .iter()
                .filter(|input| input.format == WireFormat::Postcard)
                .map(|input| {
                    let test_name = if case.inputs.len() == 1 {
                        format_ident!("{}", case.name)
                    } else {
                        format_ident!("{}_{}", case.name, input.name)
                    };
                    let ty = case.ty.clone();
                    let input_bytes = LitByteStr::new(input.input, Span::call_site());
                    let ignore_attr = ignore_attr_for_format(case, WireFormat::Postcard);
                    match &input.expect {
                        DecodeExpectation::Ok(expected) => {
                            let expected = expected.clone();
                            quote! {
                                #[test]
                                #ignore_attr
                                fn #test_name() {
                                    assert_postcard_input_case::<#ty>(#input_bytes, #expected);
                                }
                            }
                        }
                        DecodeExpectation::Err(expected_error_code) => {
                            let expected_error_code = expected_error_code.clone();
                            quote! {
                                #[test]
                                #ignore_attr
                                fn #test_name() {
                                    assert_postcard_input_err_code::<#ty>(#input_bytes, #expected_error_code);
                                }
                            }
                        }
                        DecodeExpectation::AnyErr => {
                            quote! {
                                #[test]
                                #ignore_attr
                                fn #test_name() {
                                    assert_postcard_input_err::<#ty>(#input_bytes);
                                }
                            }
                        }
                    }
                })
        })
        .collect();
    let panic_tests: Vec<TokenStream> = panic_cases
        .iter()
        .map(|case| {
            let test_name = format_ident!("{}", case.name);
            let expected = case.expected;
            let body = case.body.clone();
            quote! {
                #[test]
                #[should_panic(expected = #expected)]
                fn #test_name() {
                    #body
                }
            }
        })
        .collect();
    let file_tokens = quote! {
        use facet::Facet;
        use proptest::arbitrary::Arbitrary;

        #types

        fn assert_json_case<T>(value: T)
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
        {
            let encoded = serde_json::to_string(&value).unwrap();
            let expected: T = serde_json::from_str(&encoded).unwrap();
            let decoder = kajit::compile_decoder(T::SHAPE, &kajit::json::KajitJson);
            let got: T = kajit::from_str(&decoder, &encoded).unwrap();
            assert_eq!(got, expected);
        }

        fn assert_postcard_case<T>(value: T)
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
        {
            let encoded = ::postcard::to_allocvec(&value).unwrap();
            let expected: T = ::postcard::from_bytes(&encoded).unwrap();
            let decoder = kajit::compile_decoder(T::SHAPE, &kajit::postcard::KajitPostcard);
            let got: T = kajit::deserialize(&decoder, &encoded).unwrap();
            assert_eq!(got, expected);
        }

        fn assert_json_input_case<T>(input: &[u8], expected: T)
        where
            for<'input> T: Facet<'input> + PartialEq + std::fmt::Debug,
        {
            let decoder = kajit::compile_decoder(T::SHAPE, &kajit::json::KajitJson);
            let got: T = kajit::deserialize(&decoder, input).unwrap();
            assert_eq!(got, expected);
        }

        fn assert_json_input_err<T>(input: &[u8])
        where
            for<'input> T: Facet<'input>,
        {
            let decoder = kajit::compile_decoder(T::SHAPE, &kajit::json::KajitJson);
            let out = kajit::deserialize::<T>(&decoder, input);
            assert!(out.is_err(), "expected json decode failure");
        }

        fn assert_json_input_err_code<T>(input: &[u8], expected_code: kajit::context::ErrorCode)
        where
            for<'input> T: Facet<'input>,
        {
            let decoder = kajit::compile_decoder(T::SHAPE, &kajit::json::KajitJson);
            let out = kajit::deserialize::<T>(&decoder, input);
            let err = match out {
                Ok(_) => panic!("expected json decode failure"),
                Err(err) => err,
            };
            assert_eq!(err.code, expected_code);
        }

        fn assert_postcard_input_case<T>(input: &[u8], expected: T)
        where
            for<'input> T: Facet<'input> + PartialEq + std::fmt::Debug,
        {
            let decoder = kajit::compile_decoder(T::SHAPE, &kajit::postcard::KajitPostcard);
            let input = core::str::from_utf8(input).expect("postcard input must be valid utf-8 for from_str path");
            let got: T = kajit::from_str(&decoder, input).unwrap();
            assert_eq!(got, expected);
        }

        #[allow(dead_code)]
        fn assert_postcard_input_err<T>(input: &[u8])
        where
            for<'input> T: Facet<'input>,
        {
            let decoder = kajit::compile_decoder(T::SHAPE, &kajit::postcard::KajitPostcard);
            let input = core::str::from_utf8(input).expect("postcard input must be valid utf-8 for from_str path");
            let out = kajit::from_str::<T>(&decoder, input);
            assert!(out.is_err(), "expected postcard decode failure");
        }

        fn assert_postcard_input_err_code<T>(input: &[u8], expected_code: kajit::context::ErrorCode)
        where
            for<'input> T: Facet<'input>,
        {
            let decoder = kajit::compile_decoder(T::SHAPE, &kajit::postcard::KajitPostcard);
            let input = core::str::from_utf8(input).expect("postcard input must be valid utf-8 for from_str path");
            let out = kajit::from_str::<T>(&decoder, input);
            let err = match out {
                Ok(_) => panic!("expected postcard decode failure"),
                Err(err) => err,
            };
            assert_eq!(err.code, expected_code);
        }

        fn assert_prop_case<T>(_marker: &T)
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug + Arbitrary + 'static,
        {
            let json_decoder = kajit::compile_decoder(T::SHAPE, &kajit::json::KajitJson);
            let postcard_decoder =
                kajit::compile_decoder(T::SHAPE, &kajit::postcard::KajitPostcard);
            let mut runner = proptest::test_runner::TestRunner::new(proptest::test_runner::Config {
                cases: 64,
                ..proptest::test_runner::Config::default()
            });
            let strategy = proptest::arbitrary::any::<T>();
            runner
                .run(&strategy, |value| {
                    let json_encoded = serde_json::to_string(&value).unwrap();
                    let json_expected: T = serde_json::from_str(&json_encoded).unwrap();
                    let json_got: T = kajit::from_str(&json_decoder, &json_encoded).unwrap();
                    assert_eq!(json_got, json_expected);

                    let postcard_encoded = ::postcard::to_allocvec(&value).unwrap();
                    let postcard_expected: T = ::postcard::from_bytes(&postcard_encoded).unwrap();
                    let postcard_got: T =
                        kajit::deserialize(&postcard_decoder, &postcard_encoded).unwrap();
                    assert_eq!(postcard_got, postcard_expected);
                    Ok(())
                })
                .unwrap();
        }

        const DUMP_STAGES_ENV: &str = "KAJIT_DUMP_STAGES";
        const DUMP_FILTER_ENV: &str = "KAJIT_DUMP_FILTER";
        const DUMP_DIR_ENV: &str = "KAJIT_DUMP_DIR";
        const ASSERT_SNAPSHOTS_ENV: &str = "KAJIT_ASSERT_CODEGEN_SNAPSHOTS";

        struct CodegenArtifacts {
            ir_text: String,
            linear_text: String,
            ra_text: String,
            edits: usize,
            opt_timeline: Vec<(String, String)>,
        }

        fn codegen_artifacts<T, F>(decoder: &F) -> CodegenArtifacts
        where
            for<'input> T: Facet<'input>,
            F: kajit::format::Decoder,
        {
            let shape = T::SHAPE;
            let (ir_text, ra_text) = kajit::debug_ir_and_ra_mir_text(shape, decoder);
            let linear_text = kajit::debug_linear_ir_text(shape, decoder);
            let edits = kajit::regalloc_edit_count(shape, decoder);
            let opt_timeline = kajit::debug_ir_opt_timeline_text(shape, decoder);
            CodegenArtifacts {
                ir_text,
                linear_text,
                ra_text,
                edits,
                opt_timeline,
            }
        }

        fn env_truthy(name: &str) -> bool {
            let Some(raw) = std::env::var_os(name) else {
                return false;
            };
            let raw = raw.to_string_lossy();
            matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        }

        fn snapshot_assertions_enabled() -> bool {
            env_truthy(ASSERT_SNAPSHOTS_ENV)
        }

        fn dump_stages() -> Option<Vec<String>> {
            let raw = std::env::var_os(DUMP_STAGES_ENV)?;
            let raw = raw.to_string_lossy();
            let stages: Vec<String> = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_ascii_lowercase())
                .collect();
            if stages.is_empty() {
                None
            } else {
                Some(stages)
            }
        }

        fn should_dump_stage(stage: &str) -> bool {
            let Some(stages) = dump_stages() else {
                return false;
            };
            stages.iter().any(|s| s == "all" || s == stage)
        }

        fn dump_filter_matches(target: &str) -> bool {
            let Some(raw) = std::env::var_os(DUMP_FILTER_ENV) else {
                return true;
            };
            let raw = raw.to_string_lossy();
            let filters: Vec<&str> = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();
            if filters.is_empty() {
                return true;
            }
            filters.iter().any(|needle| target.contains(needle))
        }

        fn dumps_enabled_for_case(format_label: &str, case: &str) -> bool {
            if dump_stages().is_none() {
                return false;
            }
            dump_filter_matches(&format!("{format_label}::{case}"))
        }

        fn default_dump_dir() -> std::path::PathBuf {
            let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            manifest_dir
                .parent()
                .map(|workspace_root| workspace_root.join("target/kajit-stage-dumps"))
                .unwrap_or_else(|| manifest_dir.join("target/kajit-stage-dumps"))
        }

        fn dump_dir() -> std::path::PathBuf {
            match std::env::var_os(DUMP_DIR_ENV) {
                Some(raw) if !raw.is_empty() => std::path::PathBuf::from(raw),
                _ => default_dump_dir(),
            }
        }

        fn dump_stage(format_label: &str, case: &str, stage: &str, content: &str) {
            let dir = dump_dir();
            std::fs::create_dir_all(&dir).expect("failed to create dump directory");
            let path = dir.join(format!(
                "{format_label}__{case}__{}__{stage}.txt",
                std::env::consts::ARCH
            ));
            std::fs::write(&path, content).unwrap_or_else(|error| {
                panic!("failed writing dump to {}: {error}", path.display());
            });
        }

        fn maybe_dump_codegen_artifacts(format_label: &str, case: &str, artifacts: &CodegenArtifacts) {
            if !dumps_enabled_for_case(format_label, case) {
                return;
            }
            if should_dump_stage("ir") {
                dump_stage(format_label, case, "ir", &artifacts.ir_text);
            }
            if should_dump_stage("linear") {
                dump_stage(format_label, case, "linear", &artifacts.linear_text);
            }
            if should_dump_stage("ra") {
                dump_stage(format_label, case, "ra", &artifacts.ra_text);
            }
            if should_dump_stage("edits") {
                dump_stage(format_label, case, "edits", &format!("{}", artifacts.edits));
            }
            if should_dump_stage("opts") {
                for (index, (pass_name, ir_text)) in artifacts.opt_timeline.iter().enumerate() {
                    dump_stage(
                        format_label,
                        case,
                        &format!("opts_{index:02}_{pass_name}"),
                        ir_text,
                    );
                }
            }
        }

        fn assert_codegen_rvsdg_snapshot<T, F>(
            format_label: &str,
            case: &str,
            decoder: &F,
            _marker: &T,
        )
        where
            for<'input> T: Facet<'input>,
            F: kajit::format::Decoder,
        {
            let should_assert = snapshot_assertions_enabled();
            let should_dump = dumps_enabled_for_case(format_label, case);
            if !should_assert && !should_dump {
                return;
            }
            let artifacts = codegen_artifacts::<T, F>(decoder);
            maybe_dump_codegen_artifacts(format_label, case, &artifacts);
            if should_assert {
                insta::assert_snapshot!(
                    format!(
                        "generated_rvsdg_{}_{}_{}",
                        format_label,
                        case,
                        std::env::consts::ARCH
                    ),
                    artifacts.ir_text
                );
            }
        }

        fn assert_codegen_ra_mir_snapshot<T, F>(
            format_label: &str,
            case: &str,
            decoder: &F,
            _marker: &T,
        )
        where
            for<'input> T: Facet<'input>,
            F: kajit::format::Decoder,
        {
            let should_assert = snapshot_assertions_enabled();
            let should_dump = dumps_enabled_for_case(format_label, case);
            if !should_assert && !should_dump {
                return;
            }
            let artifacts = codegen_artifacts::<T, F>(decoder);
            maybe_dump_codegen_artifacts(format_label, case, &artifacts);
            if should_assert {
                insta::assert_snapshot!(
                    format!(
                        "generated_ra_mir_{}_{}_{}",
                        format_label,
                        case,
                        std::env::consts::ARCH
                    ),
                    artifacts.ra_text
                );
            }
        }

        fn assert_codegen_edits_snapshot<T, F>(
            format_label: &str,
            case: &str,
            decoder: &F,
            _marker: &T,
        )
        where
            for<'input> T: Facet<'input>,
            F: kajit::format::Decoder,
        {
            let should_assert = snapshot_assertions_enabled();
            let should_dump = dumps_enabled_for_case(format_label, case);
            if !should_assert && !should_dump {
                return;
            }
            let artifacts = codegen_artifacts::<T, F>(decoder);
            maybe_dump_codegen_artifacts(format_label, case, &artifacts);
            if should_assert {
                insta::assert_snapshot!(
                    format!(
                        "generated_postreg_edits_{}_{}_{}",
                        format_label,
                        case,
                        std::env::consts::ARCH
                    ),
                    format!("{}", artifacts.edits)
                );
            }
        }

        mod json {
            use super::*;
            #(#json_tests)*
        }

        mod postcard {
            use super::*;
            #(#postcard_tests)*
        }

        mod rvsdg_json {
            use super::*;
            #(#json_rvsdg_tests)*
        }

        mod rvsdg_postcard {
            use super::*;
            #(#postcard_rvsdg_tests)*
        }

        mod ra_mir_json {
            use super::*;
            #(#json_ra_mir_tests)*
        }

        mod ra_mir_postcard {
            use super::*;
            #(#postcard_ra_mir_tests)*
        }

        mod postreg_edits_json {
            use super::*;
            #(#json_postreg_edits_tests)*
        }

        mod postreg_edits_postcard {
            use super::*;
            #(#postcard_postreg_edits_tests)*
        }

        mod prop {
            use super::*;
            #(#prop_tests)*
        }

        mod json_input {
            use super::*;
            #(#json_input_tests)*
        }

        mod postcard_input {
            use super::*;
            #(#postcard_input_tests)*
        }

        mod panics {
            use super::*;
            #(#panic_tests)*
        }

        mod postreg {
            use super::*;

            #[test]
            fn vec_scalar_large_hotpath_asserts() {
                let artifacts = codegen_artifacts::<ScalarVec, _>(&kajit::postcard::KajitPostcard);

                assert!(
                    artifacts.ir_text.contains("theta") || artifacts.ir_text.contains("apply @"),
                    "expected loop form (`theta`) or outlined loop body (`apply`) in IR"
                );
                assert!(
                    artifacts.ra_text.contains("branch_if"),
                    "expected loop backedge in RA-MIR"
                );
                assert!(
                    artifacts.ra_text.contains("call_intrinsic"),
                    "expected intrinsic-heavy vec decode path in RA-MIR"
                );
                assert!(
                    artifacts.edits <= 128,
                    "expected edit budget <= 128, got {}",
                    artifacts.edits
                );
            }
        }
    };
    let file: syn::File =
        syn::parse2(file_tokens).expect("generated synthetic test file should parse");
    format!(
        "// @generated by xtask generate-synthetic. Do not edit manually.\n{}",
        prettyplease::unparse(&file)
    )
}
