//! IR behavior cases

use crate::{BehaviorVector, IrBehaviorCase};
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::LitStr;

pub(crate) const IR_BEHAVIOR_CASES: &[IrBehaviorCase] = &[
    IrBehaviorCase {
        name: "enum_like_gamma_tag_branch",
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
          n5 = Const(0x2a) [] -> [v3]
          results: [v3, %cs:arg, %os:arg]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n6 = Const(0x63) [] -> [v4]
          results: [v4, %cs:arg, %os:arg]
        }
    } -> [v5, %cs, %os]
    n7 = WriteToField(offset=0, W1) [v5, %os:n4] -> [%os]
    results: [%cs:n4, %os:n7]
  }
}
"#,
        vectors: &[
            BehaviorVector {
                input: &[0],
                expected: 42,
            },
            BehaviorVector {
                input: &[1],
                expected: 99,
            },
            BehaviorVector {
                input: &[255],
                expected: 99,
            },
        ],
    },
    IrBehaviorCase {
        name: "theta_countdown_accumulate",
        ir: r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = BoundsCheck(1) [%cs:arg] -> [%cs]
    n1 = ReadBytes(1) [%cs:n0] -> [v0, %cs]
    n2 = Const(0x0) [] -> [v1]
    n3 = Const(0x1) [] -> [v2]
    n4 = theta [v0, v1, v2, %cs:n1, %os:arg] {
      region {
        args: [arg0, arg1, arg2, %cs, %os]
        n5 = Add [arg1, arg2] -> [v3]
        n6 = Sub [arg0, arg2] -> [v4]
        results: [v4, v4, v3, arg2, %cs:arg, %os:arg]
      }
    } -> [v5, v6, v7, %cs, %os]
    n8 = WriteToField(offset=0, W1) [v6, %os:n4] -> [%os]
    results: [%cs:n4, %os:n8]
  }
}
"#,
        vectors: &[
            BehaviorVector {
                input: &[1],
                expected: 1,
            },
            BehaviorVector {
                input: &[3],
                expected: 3,
            },
            BehaviorVector {
                input: &[7],
                expected: 7,
            },
        ],
    },
    IrBehaviorCase {
        name: "and_cmpne_branch_surface",
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
        vectors: &[
            BehaviorVector {
                input: &[1, 1],
                expected: 9,
            },
            BehaviorVector {
                input: &[1, 0],
                expected: 4,
            },
            BehaviorVector {
                input: &[0, 1],
                expected: 4,
            },
            BehaviorVector {
                input: &[0, 0],
                expected: 4,
            },
        ],
    },
    IrBehaviorCase {
        name: "boundscheck_peek_read_chain",
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
        vectors: &[
            BehaviorVector {
                input: &[7],
                expected: 7,
            },
            BehaviorVector {
                input: &[200],
                expected: 200,
            },
        ],
    },
];

pub(crate) fn render_ir_behavior_test_file() -> String {
    let tests = IR_BEHAVIOR_CASES.iter().map(|case| {
        let test_name = format_ident!("ir_behavior_{}", case.name);
        let ir = LitStr::new(case.ir, Span::call_site());
        let vectors = case.vectors.iter().map(|vec| {
            let input = vec.input.iter().copied();
            let expected = vec.expected;
            quote! {
                {
                    let input: &[u8] = &[#(#input),*];
                    let expected: u8 = #expected;
                    let baseline = run_exec(ir, input, false);
                    let optimized = run_exec(ir, input, true);
                    assert_eq!(baseline, expected, "baseline output mismatch");
                    assert_eq!(optimized, expected, "optimized output mismatch");
                    assert_eq!(optimized, baseline, "optimized and baseline outputs diverged");
                }
            }
        });
        quote! {
            #[test]
            fn #test_name() {
                let ir = #ir;
                #(#vectors)*
            }
        }
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

        #(#tests)*
    };
    let file: syn::File =
        syn::parse2(file_tokens).expect("generated ir behavior corpus test file should parse");
    format!(
        "// @generated by xtask generate-ir-behavior-corpus. Do not edit manually.\n{}",
        prettyplease::unparse(&file)
    )
}
