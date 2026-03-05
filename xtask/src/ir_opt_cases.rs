//! IR optimization test cases.

use crate::IrOptCase;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::LitStr;

pub(crate) const IR_OPT_CASES: &[IrOptCase] = &[
    IrOptCase {
        name: "theta_invariant_tree_hoist",
        ir: r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = Const(0x4) [] -> [v0]
    n1 = Const(0x1) [] -> [v1]
    n2 = theta [v0, v1, %cs:arg, %os:arg] {
      region {
        args: [arg0, arg1, %cs, %os]
        n3 = Const(0x7) [] -> [v2]
        n4 = Const(0x3) [] -> [v3]
        n5 = Add [v2, v3] -> [v4]
        n6 = Xor [v4, v3] -> [v5]
        n7 = Sub [arg0, arg1] -> [v6]
        n8 = Add [v5, v6] -> [v7]
        results: [v6, v6, arg1, %cs:arg, %os:arg]
      }
    } -> [v8, v9, %cs, %os]
    n9 = WriteToField(offset=0, W1) [v8, %os:n2] -> [%os]
    results: [%cs:n2, %os:n9]
  }
}
"#,
        must_not_contain_after: &[
            "n3 = Const(0x7) [] -> [v2]",
            "n4 = Const(0x3) [] -> [v3]",
            "n5 = Add [v2, v3] -> [v4]",
            "n6 = Xor [v4, v3] -> [v5]",
        ],
        must_contain_after: &[],
        input: &[],
    },
    IrOptCase {
        name: "theta_loop_variant_not_hoisted",
        ir: r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = Const(0x4) [] -> [v0]
    n1 = Const(0x1) [] -> [v1]
    n2 = theta [v0, v1, %cs:arg, %os:arg] {
      region {
        args: [arg0, arg1, %cs, %os]
        n3 = Add [arg0, arg1] -> [v2]
        n4 = Sub [arg0, arg1] -> [v3]
        results: [v2, v3, arg1, %cs:arg, %os:arg]
      }
    } -> [v4, v5, %cs, %os]
    n5 = WriteToField(offset=0, W1) [v4, %os:n2] -> [%os]
    results: [%cs:n2, %os:n5]
  }
}
"#,
        must_not_contain_after: &[],
        must_contain_after: &["n3 = Add [arg0, arg1] -> [v2]"],
        input: &[],
    },
    IrOptCase {
        name: "bounds_check_chain_coalesce",
        ir: r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = BoundsCheck(1) [%cs:arg] -> [%cs]
    n1 = PeekByte [%cs:n0] -> [v0, %cs]
    n2 = BoundsCheck(1) [%cs:n1] -> [%cs]
    n3 = ReadBytes(1) [%cs:n2] -> [v1, %cs]
    n4 = WriteToField(offset=0, W1) [v1, %os:arg] -> [%os]
    results: [%cs:n3, %os:n4]
  }
}
"#,
        must_not_contain_after: &["n2 = BoundsCheck(1) [%cs:n1] -> [%cs]"],
        must_contain_after: &["BoundsCheck(1) [%cs:arg] -> [%cs]"],
        input: &[7],
    },
];

pub(crate) fn render_ir_opt_test_file() -> String {
    let tests = IR_OPT_CASES.iter().flat_map(|case| {
        let ir = LitStr::new(case.ir, Span::call_site());
        let name = case.name;
        let snapshot = format_ident!("ir_opt_snapshot_{}", name);
        let asserts = format_ident!("ir_opt_asserts_{}", name);
        let exec = format_ident!("ir_opt_exec_{}", name);
        let must_not = case.must_not_contain_after.iter().map(|needle| {
            let needle = LitStr::new(needle, Span::call_site());
            quote! {
                assert!(!after.contains(#needle), "expected to hoist/remove: {}", #needle);
            }
        });
        let must = case.must_contain_after.iter().map(|needle| {
            let needle = LitStr::new(needle, Span::call_site());
            quote! {
                assert!(after.contains(#needle), "expected to keep/preserve: {}", #needle);
            }
        });
        let input: Vec<u8> = case.input.to_vec();
        let input_before = input.clone();
        let input_after = input;
        vec![
            quote! {
                #[test]
                fn #snapshot() {
                    let (before, after) = run_pass(#ir);
                    insta::assert_snapshot!(concat!("generated_ir_opt_before_", #name), before);
                    insta::assert_snapshot!(concat!("generated_ir_opt_after_", #name), after);
                }
            },
            quote! {
                #[test]
                fn #asserts() {
                    let (_before, after) = run_pass(#ir);
                    #(#must_not)*
                    #(#must)*
                }
            },
            quote! {
                #[test]
                fn #exec() {
                    let before_out = run_exec(#ir, &[#(#input_before),*]);
                    let mut optimized = parse_case(#ir);
                    kajit::ir_passes::run_default_passes(&mut optimized);
                    let linear = kajit::linearize::linearize(&mut optimized);
                    let dec = kajit::compile_decoder_linear_ir(&linear, false);
                    let after_out = kajit::deserialize::<u8>(&dec, &[#(#input_after),*])
                        .expect("optimized decoder should execute");
                    assert_eq!(after_out, before_out);
                }
            },
        ]
    });
    let file_tokens = quote! {
        use facet::Facet;

        fn parse_case(ir: &str) -> kajit::ir::IrFunc {
            let registry = kajit::ir::IntrinsicRegistry::empty();
            kajit::ir_parse::parse_ir(ir, <u8 as Facet>::SHAPE, &registry)
                .expect("text IR should parse")
        }

        fn run_pass(ir: &str) -> (String, String) {
            let mut func = parse_case(ir);
            let before = format!("{func}");
            kajit::ir_passes::run_default_passes(&mut func);
            let after = format!("{func}");
            (before, after)
        }

        fn run_exec(ir: &str, input: &[u8]) -> u8 {
            let mut func = parse_case(ir);
            let linear = kajit::linearize::linearize(&mut func);
            let dec = kajit::compile_decoder_linear_ir(&linear, false);
            kajit::deserialize::<u8>(&dec, input).expect("decoder should execute")
        }

        #(#tests)*
    };
    let file: syn::File =
        syn::parse2(file_tokens).expect("generated ir opt corpus test file should parse");
    format!(
        "// @generated by xtask generate-ir-opt-corpus. Do not edit manually.\n{}",
        prettyplease::unparse(&file)
    )
}
