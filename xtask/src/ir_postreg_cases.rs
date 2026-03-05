// IR post-register allocation cases

use crate::IrPostRegCase;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::LitStr;

pub(crate) const IR_POSTREG_CASES: &[IrPostRegCase] = &[
    IrPostRegCase {
        name: "cmpne_gamma_branch",
        ir: r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = BoundsCheck(1) [%cs:arg] -> [%cs]
    n1 = ReadBytes(1) [%cs:n0] -> [v0, %cs]
    n2 = Const(0x0) [] -> [v1]
    n3 = CmpNe [v0, v1] -> [v2]
    n4 = gamma [
      pred: v2
      in0: %cs:n1
      in1: %os:arg
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          n5 = Const(0x3) [] -> [v3]
          results: [v3, %cs:arg, %os:arg]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n6 = Const(0x7) [] -> [v4]
          results: [v4, %cs:arg, %os:arg]
        }
    } -> [v5, %cs, %os]
    n7 = WriteToField(offset=0, W1) [v5, %os:n4] -> [%os]
    results: [%cs:n4, %os:n7]
  }
}
"#,
        must_contain_linear: &["CmpNe", "br_zero"],
        max_total_edits: 64,
        input: &[1],
        expected: 7,
    },
    IrPostRegCase {
        name: "and_cmpne_gamma_branch",
        ir: r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = BoundsCheck(2) [%cs:arg] -> [%cs]
    n1 = ReadBytes(1) [%cs:n0] -> [v0, %cs]
    n2 = ReadBytes(1) [%cs:n1] -> [v1, %cs]
    n3 = Const(0x0) [] -> [v2]
    n4 = CmpNe [v0, v2] -> [v3]
    n5 = CmpNe [v1, v2] -> [v4]
    n6 = And [v3, v4] -> [v5]
    n7 = gamma [
      pred: v5
      in0: %cs:n2
      in1: %os:arg
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          n8 = Const(0x4) [] -> [v6]
          results: [v6, %cs:arg, %os:arg]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n9 = Const(0x9) [] -> [v7]
          results: [v7, %cs:arg, %os:arg]
        }
    } -> [v8, %cs, %os]
    n10 = WriteToField(offset=0, W1) [v8, %os:n7] -> [%os]
    results: [%cs:n7, %os:n10]
  }
}
"#,
        must_contain_linear: &["And", "CmpNe", "br_zero"],
        max_total_edits: 96,
        input: &[1, 1],
        expected: 9,
    },
    IrPostRegCase {
        name: "theta_then_gamma_edit_budget",
        ir: r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = Const(0x3) [] -> [v0]
    n1 = Const(0x1) [] -> [v1]
    n2 = theta [v0, v1, %cs:arg, %os:arg] {
      region {
        args: [arg0, arg1, %cs, %os]
        n3 = Sub [arg0, arg1] -> [v2]
        results: [v2, v2, arg1, %cs:arg, %os:arg]
      }
    } -> [v3, v4, %cs, %os]
    n4 = gamma [
      pred: v3
      in0: %cs:n2
      in1: %os:n2
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          n5 = Const(0xb) [] -> [v5]
          results: [v5, %cs:arg, %os:arg]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n6 = Const(0x16) [] -> [v6]
          results: [v6, %cs:arg, %os:arg]
        }
    } -> [v7, %cs, %os]
    n8 = WriteToField(offset=0, W1) [v7, %os:n4] -> [%os]
    results: [%cs:n4, %os:n8]
  }
}
"#,
        must_contain_linear: &["br_if", "br_zero"],
        max_total_edits: 128,
        input: &[],
        expected: 11,
    },
];

pub(crate) fn render_ir_postreg_test_file() -> String {
    let tests = IR_POSTREG_CASES.iter().flat_map(|case| {
        let name = case.name;
        let ir = LitStr::new(case.ir, Span::call_site());
        let snap = format_ident!("ir_postreg_snapshot_{}", name);
        let asserts = format_ident!("ir_postreg_asserts_{}", name);
        let exec = format_ident!("ir_postreg_exec_{}", name);
        let max_edits = case.max_total_edits;
        let expected = case.expected;
        let input: Vec<u8> = case.input.to_vec();
        let input_before = input.clone();
        let input_after = input;
        let must_contain = case.must_contain_linear.iter().map(|needle| {
            let needle = LitStr::new(needle, Span::call_site());
            quote! {
                assert!(linear.contains(#needle), "expected linear artifact to contain: {}", #needle);
            }
        });
        vec![
            quote! {
                #[test]
                fn #snap() {
                    let (linear, cfg, edits) = postreg_artifacts(#ir);
                    insta::assert_snapshot!(concat!("generated_ir_postreg_linear_", #name), linear);
                    insta::assert_snapshot!(concat!("generated_ir_postreg_cfg_", #name), cfg);
                    insta::assert_snapshot!(concat!("generated_ir_postreg_edits_", #name), format!("{edits}"));
                }
            },
            quote! {
                #[test]
                fn #asserts() {
                    let (linear, _cfg, edits) = postreg_artifacts(#ir);
                    assert!(edits <= #max_edits, "expected edit budget <= {}, got {edits}", #max_edits);
                    #(#must_contain)*
                }
            },
            quote! {
                #[test]
                fn #exec() {
                    let before = run_exec(#ir, &[#(#input_before),*], false);
                    let after = run_exec(#ir, &[#(#input_after),*], true);
                    assert_eq!(after, #expected, "optimized output mismatch against expected");
                    assert_eq!(after, before, "optimized and baseline outputs diverged");
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

        fn run_exec(ir: &str, input: &[u8], with_passes: bool) -> u8 {
            let mut func = parse_case(ir);
            if with_passes {
                kajit::ir_passes::run_default_passes(&mut func);
            }
            let linear = kajit::linearize::linearize(&mut func);
            let dec = kajit::compile_decoder_linear_ir(&linear, false);
            kajit::deserialize::<u8>(&dec, input).expect("decoder should execute")
        }

        fn postreg_artifacts(ir: &str) -> (String, String, usize) {
            let mut func = parse_case(ir);
            kajit::ir_passes::run_default_passes(&mut func);
            let linear = kajit::linearize::linearize(&mut func);
            let linear_text = format!("{linear}");
            let cfg = kajit::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
            let cfg_text = format!("{cfg}");
            let alloc = kajit::regalloc_engine::allocate_cfg_program(&cfg)
                .expect("regalloc should allocate post-reg corpus case");
            let edits: usize = alloc.functions.iter().map(|f| f.edits.len()).sum();
            (linear_text, cfg_text, edits)
        }

        #(#tests)*
    };
    let file: syn::File =
        syn::parse2(file_tokens).expect("generated ir postreg corpus test file should parse");
    format!(
        "// @generated by xtask generate-ir-postreg-corpus. Do not edit manually.\n{}",
        prettyplease::unparse(&file)
    )
}
