//! Deser/ser cases

use crate::{Case, CaseBuilder, DecodeExpectation, WireFormat};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::LitByteStr;

struct JsonInputCase {
    name: &'static str,
    ty: TokenStream,
    input: &'static str,
    expected: Option<TokenStream>,
    expected_error_code: Option<TokenStream>,
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

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        #[repr(u8)]
        enum Animal {
            Cat,
            Dog { name: String, good_boy: bool },
            Parrot(String),
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
        struct BoolField {
            value: bool,
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

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        #[facet(deny_unknown_fields)]
        struct Strict {
            x: u32,
            y: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct WithDefault {
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
        struct ScalarVec {
            #[proptest(strategy = "proptest::collection::vec(proptest::arbitrary::any::<u32>(), 0..256)")]
            values: Vec<u32>,
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
            name: "bool_field",
            ty: quote!(BoolField),
            values: vec![quote!(BoolField { value: true })],
            inputs: vec![],
        },
        Case {
            name: "tuple_pair",
            ty: quote!(Pair),
            values: vec![quote!((42u32, "Alice".to_string()))],
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
            name: "vec_scalar_large",
            ty: quote!(ScalarVec),
            values: vec![quote!(ScalarVec {
                values: (0..2048).map(|i| i as u32).collect()
            })],
            inputs: vec![],
        },
    ]
}

fn all_cases() -> Vec<Case> {
    let mut out = cases();
    for case in json_input_cases() {
        let builder = CaseBuilder::new(case.name, case.ty);
        let case = if let Some(expected) = case.expected {
            builder.json_ok(case.name, case.input, expected).build()
        } else if let Some(err_code) = case.expected_error_code {
            builder.json_err(case.name, case.input, err_code).build()
        } else {
            builder.json_err_any(case.name, case.input).build()
        };
        out.push(case);
    }
    out
}

fn json_input_cases() -> Vec<JsonInputCase> {
    vec![
        JsonInputCase {
            name: "reversed_key_order",
            ty: quote!(Friend),
            input: r#"{"name": "Alice", "age": 42}"#,
            expected: Some(quote!(Friend {
                age: 42,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "unknown_keys_skipped",
            ty: quote!(Friend),
            input: r#"{"age": 42, "extra": true, "name": "Alice"}"#,
            expected: Some(quote!(Friend {
                age: 42,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "empty_object_missing_fields",
            ty: quote!(Friend),
            input: r#"{}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::MissingRequiredField)),
        },
        JsonInputCase {
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
        JsonInputCase {
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
        JsonInputCase {
            name: "enum_struct_variant_reversed_keys",
            ty: quote!(Animal),
            input: r#"{"Dog": {"good_boy": true, "name": "Rex"}}"#,
            expected: Some(quote!(Animal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "enum_unit_as_string",
            ty: quote!(Animal),
            input: r#""Cat""#,
            expected: Some(quote!(Animal::Cat)),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "enum_struct_variant",
            ty: quote!(Animal),
            input: r#"{"Dog": {"name": "Rex", "good_boy": true}}"#,
            expected: Some(quote!(Animal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "enum_tuple_variant",
            ty: quote!(Animal),
            input: r#"{"Parrot": "Polly"}"#,
            expected: Some(quote!(Animal::Parrot("Polly".into()))),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "enum_unit_in_object",
            ty: quote!(Animal),
            input: r#"{"Cat": null}"#,
            expected: Some(quote!(Animal::Cat)),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "enum_unknown_variant",
            ty: quote!(Animal),
            input: r#""Snake""#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::UnknownVariant)),
        },
        JsonInputCase {
            name: "adjacent_unit_no_content",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Cat"}"#,
            expected: Some(quote!(AdjAnimal::Cat)),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "adjacent_unit_with_null_content",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Cat", "data": null}"#,
            expected: Some(quote!(AdjAnimal::Cat)),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "adjacent_struct_variant",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Dog", "data": {"name": "Rex", "good_boy": true}}"#,
            expected: Some(quote!(AdjAnimal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "adjacent_struct_variant_reversed_fields",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Dog", "data": {"good_boy": true, "name": "Rex"}}"#,
            expected: Some(quote!(AdjAnimal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "adjacent_tuple_variant",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Parrot", "data": "Polly"}"#,
            expected: Some(quote!(AdjAnimal::Parrot("Polly".into()))),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "adjacent_unknown_variant",
            ty: quote!(AdjAnimal),
            input: r#"{"type": "Snake", "data": null}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::UnknownVariant)),
        },
        JsonInputCase {
            name: "adjacent_wrong_first_key",
            ty: quote!(AdjAnimal),
            input: r#"{"data": null, "type": "Cat"}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::ExpectedTagKey)),
        },
        JsonInputCase {
            name: "internal_struct_variant_reversed_fields",
            ty: quote!(IntAnimal),
            input: r#"{"type": "Dog", "good_boy": true, "name": "Rex"}"#,
            expected: Some(quote!(IntAnimal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "internal_unit_variant",
            ty: quote!(IntAnimal),
            input: r#"{"type": "Cat"}"#,
            expected: Some(quote!(IntAnimal::Cat)),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "internal_struct_variant",
            ty: quote!(IntAnimal),
            input: r#"{"type": "Dog", "name": "Rex", "good_boy": true}"#,
            expected: Some(quote!(IntAnimal::Dog {
                name: "Rex".into(),
                good_boy: true
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "internal_unknown_variant",
            ty: quote!(IntAnimal),
            input: r#"{"type": "Snake"}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::UnknownVariant)),
        },
        JsonInputCase {
            name: "internal_wrong_first_key",
            ty: quote!(IntAnimal),
            input: r#"{"name": "Rex", "type": "Dog", "good_boy": true}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::ExpectedTagKey)),
        },
        JsonInputCase {
            name: "untagged_struct_reversed_keys",
            ty: quote!(UntaggedAnimal),
            input: r#"{"good_boy": false, "name": "Rex"}"#,
            expected: Some(quote!(UntaggedAnimal::Dog {
                name: "Rex".into(),
                good_boy: false
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "untagged_solver_key_order_independent",
            ty: quote!(UntaggedConfig),
            input: r#"{"db": 0, "host": "localhost"}"#,
            expected: Some(quote!(UntaggedConfig::Redis {
                host: "localhost".into(),
                db: 0
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "untagged_nested_key_order_independent",
            ty: quote!(ApiResponse),
            input: r#"{"data": {"items": 5}, "status": 200}"#,
            expected: Some(quote!(ApiResponse::Success {
                status: 200,
                data: SuccessPayload { items: 5 }
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "option_some_scalar",
            ty: quote!(WithOptU32),
            input: r#"{"value": 42}"#,
            expected: Some(quote!(WithOptU32 { value: Some(42) })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "option_some_string",
            ty: quote!(WithOptStr),
            input: r#"{"name": "Alice"}"#,
            expected: Some(quote!(WithOptStr {
                name: Some("Alice".into())
            })),
            expected_error_code: None,
        },
        JsonInputCase {
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
        JsonInputCase {
            name: "option_none_struct",
            ty: quote!(WithOptAddr),
            input: r#"{"addr": null}"#,
            expected: Some(quote!(WithOptAddr { addr: None })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "option_none_scalar",
            ty: quote!(WithOptU32),
            input: r#"{"value": null}"#,
            expected: Some(quote!(WithOptU32 { value: None })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "option_none_string",
            ty: quote!(WithOptStr),
            input: r#"{"name": null}"#,
            expected: Some(quote!(WithOptStr { name: None })),
            expected_error_code: None,
        },
        JsonInputCase {
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
        JsonInputCase {
            name: "rc_scalar",
            ty: quote!(RcScalar),
            input: r#"{"value": 77}"#,
            expected: Some(quote!(RcScalar {
                value: std::rc::Rc::new(77)
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "rename_field",
            ty: quote!(RenameField),
            input: r#"{"user_name": "Alice", "age": 30}"#,
            expected: Some(quote!(RenameField {
                name: "Alice".into(),
                age: 30
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "rename_field_original_name_rejected",
            ty: quote!(RenameField),
            input: r#"{"name": "Alice", "age": 30}"#,
            expected: None,
            expected_error_code: None,
        },
        JsonInputCase {
            name: "rename_all_camel_case",
            ty: quote!(CamelCaseStruct),
            input: r#"{"userName": "Bob", "birthYear": 1990}"#,
            expected: Some(quote!(CamelCaseStruct {
                user_name: "Bob".into(),
                birth_year: 1990
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "deny_unknown_fields_rejects",
            ty: quote!(Strict),
            input: r#"{"x": 1, "y": 2, "z": 3}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::UnknownField)),
        },
        JsonInputCase {
            name: "deny_unknown_fields_allows_known",
            ty: quote!(Strict),
            input: r#"{"x": 1, "y": 2}"#,
            expected: Some(quote!(Strict { x: 1, y: 2 })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "default_field_missing",
            ty: quote!(WithDefault),
            input: r#"{"name": "Alice"}"#,
            expected: Some(quote!(WithDefault {
                name: "Alice".into(),
                score: 0
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "default_field_present",
            ty: quote!(WithDefault),
            input: r#"{"name": "Alice", "score": 99}"#,
            expected: Some(quote!(WithDefault {
                name: "Alice".into(),
                score: 99
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "default_field_required_still_errors",
            ty: quote!(WithDefault),
            input: r#"{"score": 50}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::MissingRequiredField)),
        },
        JsonInputCase {
            name: "default_string_field",
            ty: quote!(WithDefaultString),
            input: r#"{"value": 42}"#,
            expected: Some(quote!(WithDefaultString {
                label: String::new(),
                value: 42
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "container_default_empty_object",
            ty: quote!(AllDefault),
            input: r#"{}"#,
            expected: Some(quote!(AllDefault { x: 0, y: 0 })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "container_default_partial",
            ty: quote!(AllDefault),
            input: r#"{"x": 5}"#,
            expected: Some(quote!(AllDefault { x: 5, y: 0 })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "skip_field",
            ty: quote!(WithSkip),
            input: r#"{"name": "Alice"}"#,
            expected: Some(quote!(WithSkip {
                name: "Alice".into(),
                cached: 0
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "skip_field_in_input_treated_as_unknown",
            ty: quote!(WithSkip),
            input: r#"{"name": "Alice", "cached": 99}"#,
            expected: Some(quote!(WithSkip {
                name: "Alice".into(),
                cached: 0
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "skip_deserializing_field",
            ty: quote!(WithSkipDeser),
            input: r#"{"name": "Bob"}"#,
            expected: Some(quote!(WithSkipDeser {
                name: "Bob".into(),
                internal: 0
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "skip_with_custom_default",
            ty: quote!(SkipWithCustomDefault),
            input: r#"{"value": 10}"#,
            expected: Some(quote!(SkipWithCustomDefault {
                value: 10,
                magic: 42
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "string_escape_newline",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "hello\nworld"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "hello\nworld".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "string_escape_tab",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "hello\tworld"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "hello\tworld".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "string_escape_backslash",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "hello\\world"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "hello\\world".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "string_escape_quote",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "hello\"world"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "hello\"world".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "string_escape_all_simple",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "a\"b\\c\/d\be\ff\ng\rh\ti"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "a\"b\\c/d\x08e\x0Cf\ng\rh\ti".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "string_unicode_escape_bmp",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "\u0041lice"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "string_unicode_escape_non_ascii",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "caf\u00E9"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "caf\u{00E9}".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "string_unicode_surrogate_pair",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "\uD83D\uDE00"}"#,
            expected: Some(quote!(Friend {
                age: 1,
                name: "\u{1F600}".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "key_with_unicode_escape",
            ty: quote!(Friend),
            input: r#"{"age": 42, "na\u006De": "Alice"}"#,
            expected: Some(quote!(Friend {
                age: 42,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "string_invalid_escape",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "hello\xworld"}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::InvalidEscapeSequence)),
        },
        JsonInputCase {
            name: "string_lone_high_surrogate",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "\uD800"}"#,
            expected: None,
            expected_error_code: Some(quote!(kajit::context::ErrorCode::InvalidEscapeSequence)),
        },
        JsonInputCase {
            name: "string_truncated_unicode",
            ty: quote!(Friend),
            input: r#"{"age": 1, "name": "\u00"}"#,
            expected: None,
            expected_error_code: None,
        },
        JsonInputCase {
            name: "skip_value_with_unicode_escape",
            ty: quote!(Friend),
            input: r#"{"age": 42, "extra": "test\uD83D\uDE00end", "name": "Alice"}"#,
            expected: Some(quote!(Friend {
                age: 42,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
            name: "skip_value_with_backslash_escape",
            ty: quote!(Friend),
            input: r#"{"age": 42, "extra": "test\n\t\\end", "name": "Alice"}"#,
            expected: Some(quote!(Friend {
                age: 42,
                name: "Alice".into()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
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
        JsonInputCase {
            name: "map_empty",
            ty: quote!(ConfigMap),
            input: r#"{"scores": {}}"#,
            expected: Some(quote!(ConfigMap {
                scores: std::collections::HashMap::new()
            })),
            expected_error_code: None,
        },
        JsonInputCase {
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
        JsonInputCase {
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

pub(crate) fn render_bench_file() -> String {
    let cases = all_cases();
    let types = types_rs();
    let bench_calls: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values.iter().enumerate().map(|(sample_idx, value)| {
                let sample_name = if case.values.len() == 1 {
                    case.name.to_string()
                } else {
                    format!("{}__v{}", case.name, sample_idx)
                };
                let value = value.clone();
                quote! {
                    register_bench_case(&mut v, #sample_name, #value);
                }
            })
        })
        .collect();
    let file_tokens = quote! {
        #[path = "harness.rs"]
        mod harness;

        use facet::Facet;
        use std::hint::black_box;
        use std::sync::Arc;

        #types

        fn register_bench_case<T>(v: &mut Vec<harness::Bench>, group: &str, value: T)
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + 'static,
        {
            let json_data = Arc::new(serde_json::to_string(&value).unwrap());
            let postcard_data = Arc::new(postcard::to_allocvec(&value).unwrap());
            let value = Arc::new(value);

            let json_decoder =
                Arc::new(kajit::compile_decoder_legacy(T::SHAPE, &kajit::json::KajitJson));
            let json_encoder =
                Arc::new(kajit::compile_encoder(T::SHAPE, &kajit::json::KajitJsonEncoder));

            let postcard_decoder =
                Arc::new(kajit::compile_decoder_legacy(T::SHAPE, &kajit::postcard::KajitPostcard));
            let postcard_ir_decoder =
                Arc::new(kajit::compile_decoder_via_ir(T::SHAPE, &kajit::postcard::KajitPostcard));
            let postcard_encoder =
                Arc::new(kajit::compile_encoder(T::SHAPE, &kajit::postcard::KajitPostcard));

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
            v.push(harness::Bench {
                name: format!("{json_prefix}/kajit_dynasm_deser"),
                func: Box::new({
                    let data = Arc::clone(&json_data);
                    let decoder = Arc::clone(&json_decoder);
                    move |runner| {
                        let decoder = &*decoder;
                        runner.run(|| {
                            black_box(kajit::from_str::<T>(decoder, black_box(data.as_str())).unwrap());
                        });
                    }
                }),
            });
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
                name: format!("{json_prefix}/kajit_dynasm_ser"),
                func: Box::new({
                    let value = Arc::clone(&value);
                    let encoder = Arc::clone(&json_encoder);
                    move |runner| {
                        let encoder = &*encoder;
                        runner.run(|| {
                            black_box(kajit::serialize(encoder, black_box(&*value)));
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
            v.push(harness::Bench {
                name: format!("{postcard_prefix}/kajit_dynasm_deser"),
                func: Box::new({
                    let data = Arc::clone(&postcard_data);
                    let decoder = Arc::clone(&postcard_decoder);
                    move |runner| {
                        let decoder = &*decoder;
                        runner.run(|| {
                            black_box(kajit::deserialize::<T>(decoder, black_box(&data[..])).unwrap());
                        });
                    }
                }),
            });
            v.push(harness::Bench {
                name: format!("{postcard_prefix}/kajit_ir_deser"),
                func: Box::new({
                    let data = Arc::clone(&postcard_data);
                    let decoder = Arc::clone(&postcard_ir_decoder);
                    move |runner| {
                        let decoder = &*decoder;
                        runner.run(|| {
                            black_box(kajit::deserialize::<T>(decoder, black_box(&data[..])).unwrap());
                        });
                    }
                }),
            });
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
            v.push(harness::Bench {
                name: format!("{postcard_prefix}/kajit_dynasm_ser"),
                func: Box::new({
                    let value = Arc::clone(&value);
                    let encoder = Arc::clone(&postcard_encoder);
                    move |runner| {
                        let encoder = &*encoder;
                        runner.run(|| {
                            black_box(kajit::serialize(encoder, black_box(&*value)));
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
    let types = types_rs();
    let json_tests: Vec<TokenStream> = cases
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
                    quote! {
                        #[test]
                        fn #test_name() {
                            let value = #value;
                            assert_codegen_snapshots("json", #case_name, &kajit::json::KajitJson, &value);
                            assert_json_case(value);
                        }
                    }
                })
        })
        .collect();
    let postcard_tests: Vec<TokenStream> = cases
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
                    quote! {
                        #[test]
                        fn #test_name() {
                            let value = #value;
                            assert_codegen_snapshots("postcard", #case_name, &kajit::postcard::KajitPostcard, &value);
                            assert_postcard_case(value);
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
            quote! {
                #[test]
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
                    match &input.expect {
                        DecodeExpectation::Ok(expected) => {
                            let expected = expected.clone();
                            quote! {
                                #[test]
                                fn #test_name() {
                                    assert_json_input_case::<#ty>(#input_bytes, #expected);
                                }
                            }
                        }
                        DecodeExpectation::Err(expected_error_code) => {
                            let expected_error_code = expected_error_code.clone();
                            quote! {
                                #[test]
                                fn #test_name() {
                                    assert_json_input_err_code::<#ty>(#input_bytes, #expected_error_code);
                                }
                            }
                        }
                        DecodeExpectation::AnyErr => {
                            quote! {
                                #[test]
                                fn #test_name() {
                                    assert_json_input_err::<#ty>(#input_bytes);
                                }
                            }
                        }
                    }
                })
        })
        .collect();
    let file_tokens = quote! {
        use facet::Facet;
        use proptest::arbitrary::Arbitrary;
        use std::fmt::Write;
        #[cfg(target_arch = "x86_64")]
        use yaxpeax_arch::LengthedInstruction;
        use yaxpeax_arch::{Decoder, U8Reader};

        #types

        fn assert_json_case<T>(value: T)
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
        {
            let encoded = serde_json::to_string(&value).unwrap();
            let expected: T = serde_json::from_str(&encoded).unwrap();
            let decoder = kajit::compile_decoder_legacy(T::SHAPE, &kajit::json::KajitJson);
            let got: T = kajit::from_str(&decoder, &encoded).unwrap();
            assert_eq!(got, expected);
        }

        fn assert_postcard_case<T>(value: T)
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
        {
            let encoded = ::postcard::to_allocvec(&value).unwrap();
            let expected: T = ::postcard::from_bytes(&encoded).unwrap();
            let legacy = kajit::compile_decoder_legacy(T::SHAPE, &kajit::postcard::KajitPostcard);
            let legacy_out: T = kajit::deserialize(&legacy, &encoded).unwrap();
            assert_eq!(legacy_out, expected);
            let ir = kajit::compile_decoder_via_ir(T::SHAPE, &kajit::postcard::KajitPostcard);
            let ir_out: T = kajit::deserialize(&ir, &encoded).unwrap();
            assert_eq!(ir_out, expected);
        }

        fn assert_json_input_case<T>(input: &[u8], expected: T)
        where
            for<'input> T: Facet<'input> + PartialEq + std::fmt::Debug,
        {
            let decoder = kajit::compile_decoder_legacy(T::SHAPE, &kajit::json::KajitJson);
            let got: T = kajit::deserialize(&decoder, input).unwrap();
            assert_eq!(got, expected);
        }

        fn assert_json_input_err<T>(input: &[u8])
        where
            for<'input> T: Facet<'input>,
        {
            let decoder = kajit::compile_decoder_legacy(T::SHAPE, &kajit::json::KajitJson);
            let out = kajit::deserialize::<T>(&decoder, input);
            assert!(out.is_err(), "expected json decode failure");
        }

        fn assert_json_input_err_code<T>(input: &[u8], expected_code: kajit::context::ErrorCode)
        where
            for<'input> T: Facet<'input>,
        {
            let decoder = kajit::compile_decoder_legacy(T::SHAPE, &kajit::json::KajitJson);
            let out = kajit::deserialize::<T>(&decoder, input);
            let err = match out {
                Ok(_) => panic!("expected json decode failure"),
                Err(err) => err,
            };
            assert_eq!(err.code, expected_code);
        }

        fn assert_prop_case<T>(_marker: &T)
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug + Arbitrary + 'static,
        {
            let json_decoder = kajit::compile_decoder_legacy(T::SHAPE, &kajit::json::KajitJson);
            let postcard_legacy =
                kajit::compile_decoder_legacy(T::SHAPE, &kajit::postcard::KajitPostcard);
            let postcard_ir =
                kajit::compile_decoder_via_ir(T::SHAPE, &kajit::postcard::KajitPostcard);
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
                    let postcard_legacy_out: T =
                        kajit::deserialize(&postcard_legacy, &postcard_encoded).unwrap();
                    assert_eq!(postcard_legacy_out, postcard_expected);

                    let postcard_ir_out: T =
                        kajit::deserialize(&postcard_ir, &postcard_encoded).unwrap();
                    assert_eq!(postcard_ir_out, postcard_expected);
                    Ok(())
                })
                .unwrap();
        }

        fn disasm_bytes(code: &[u8], marker_offset: Option<usize>) -> String {
            let mut out = String::new();

            #[cfg(target_arch = "aarch64")]
            {
                use yaxpeax_arm::armv8::a64::InstDecoder;

                let decoder = InstDecoder::default();
                let mut reader = U8Reader::new(code);
                let mut offset = 0usize;
                let mut ret_count = 0u32;

                while offset + 4 <= code.len() {
                    let prefix = if marker_offset == Some(offset) { "> " } else { "  " };
                    match decoder.decode(&mut reader) {
                        Ok(inst) => {
                            let text = kajit::disasm_normalize::normalize_inst(&format!("{inst}"));
                            writeln!(&mut out, "{prefix}{text}").unwrap();
                            if text.trim() == "ret" {
                                ret_count += 1;
                                if ret_count >= 2 {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let word = u32::from_le_bytes(code[offset..offset + 4].try_into().unwrap());
                            writeln!(&mut out, "{prefix}<{e}> (0x{word:08x})").unwrap();
                        }
                    }
                    offset += 4;
                }
            }

            #[cfg(target_arch = "x86_64")]
            {
                use yaxpeax_x86::amd64::InstDecoder;

                let decoder = InstDecoder::default();
                let mut reader = U8Reader::new(code);
                let mut offset = 0usize;
                let mut ret_count = 0u32;

                while offset < code.len() {
                    let prefix = if marker_offset == Some(offset) { "> " } else { "  " };
                    match decoder.decode(&mut reader) {
                        Ok(inst) => {
                            let len = inst.len().to_const() as usize;
                            let text = kajit::disasm_normalize::normalize_inst(&format!("{inst}"));
                            writeln!(&mut out, "{prefix}{text}").unwrap();
                            if text.trim() == "ret" {
                                ret_count += 1;
                                if ret_count >= 2 {
                                    break;
                                }
                            }
                            offset += len;
                        }
                        Err(_) => {
                            writeln!(&mut out, "{prefix}<decode error> (0x{:02x})", code[offset]).unwrap();
                            offset += 1;
                        }
                    }
                }
            }

            out
        }

        fn codegen_artifacts<T, F>(decoder: &F) -> (String, String, usize, String)
        where
            for<'input> T: Facet<'input>,
            F: kajit::format::Decoder + kajit::format::IrDecoder,
        {
            let shape = T::SHAPE;
            let (ir_text, ra_text) = kajit::debug_ir_and_ra_mir_text(shape, decoder);
            let edits = kajit::regalloc_edit_count_via_ir(shape, decoder);
            let compiled = kajit::compile_decoder_with_backend(shape, decoder, kajit::DecoderBackend::Ir);
            let disasm = disasm_bytes(compiled.code(), Some(compiled.entry_offset()));
            (ir_text, ra_text, edits, disasm)
        }

        fn assert_codegen_snapshots<T, F>(
            format_label: &str,
            case: &str,
            decoder: &F,
            _marker: &T,
        )
        where
            for<'input> T: Facet<'input>,
            F: kajit::format::Decoder + kajit::format::IrDecoder,
        {
            let (ir_text, ra_text, edits, disasm) = codegen_artifacts::<T, F>(decoder);
            insta::assert_snapshot!(
                format!(
                    "generated_rvsdg_{}_{}_{}",
                    format_label,
                    case,
                    std::env::consts::ARCH
                ),
                ir_text
            );
            insta::assert_snapshot!(
                format!(
                    "generated_ra_mir_{}_{}_{}",
                    format_label,
                    case,
                    std::env::consts::ARCH
                ),
                ra_text
            );
            insta::assert_snapshot!(
                format!(
                    "generated_postreg_disasm_{}_{}_{}",
                    format_label,
                    case,
                    std::env::consts::ARCH
                ),
                disasm
            );
            insta::assert_snapshot!(
                format!(
                    "generated_postreg_edits_{}_{}_{}",
                    format_label,
                    case,
                    std::env::consts::ARCH
                ),
                format!("{edits}")
            );
        }

        mod json {
            use super::*;
            #(#json_tests)*
        }

        mod postcard {
            use super::*;
            #(#postcard_tests)*
        }

        mod prop {
            use super::*;
            #(#prop_tests)*
        }

        mod json_input {
            use super::*;
            #(#json_input_tests)*
        }

        mod postreg {
            use super::*;

            #[test]
            fn vec_scalar_large_hotpath_asserts() {
                let (ir_text, ra_text, edits, disasm) =
                    codegen_artifacts::<ScalarVec, _>(&kajit::postcard::KajitPostcard);

                assert!(
                    ir_text.contains("theta") || ir_text.contains("apply @"),
                    "expected loop form (`theta`) or outlined loop body (`apply`) in IR"
                );
                assert!(
                    ra_text.contains("branch_if"),
                    "expected loop backedge in RA-MIR"
                );
                assert!(
                    ra_text.contains("call_intrinsic"),
                    "expected intrinsic-heavy vec decode path in RA-MIR"
                );
                assert!(edits <= 128, "expected edit budget <= 128, got {edits}");

                assert!(!disasm.is_empty(), "expected non-empty disassembly artifact");
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
