use facet::Shape;
use facet_testhelpers::test;
use kajit_hir_text::parse_hir;

use super::{
    build_structural_hir_ir, compile_linear_ir_decoder, compile_structural_hir_decoder,
    run_default_passes_from_env,
};

#[test]
fn structural_hir_ir_path_decodes_constant_output() {
    let module = parse_hir(
        r#"
hir_module {
  regions [
  ]
  stores [
  ]
  types [
    type t0 "ConstantNumber" size=4 = struct {
      "value": u32 @0
    }
  ]
  callables [
  ]
  functions [
    function f0 "const_number" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t0
      ]
      locals [
      ]
      return unit
      scopes [
        scope sc0 parent none comment "constant structural HIR"
      ]
      body @sc0 {
        stmt0: init field(l1, "value") = 0x2a
        stmt1: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}

#[test]
fn structural_hir_ir_path_preserves_local_scalar_across_empty_else_if() {
    let module = parse_hir(
        r#"
hir_module {
  regions []
  stores []
  types [
    type t0 "ConstantNumber" size=4 = struct {
      "value": u32 @0
    }
  ]
  callables []
  functions [
    function f0 "local_across_if" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t0
      ]
      locals [
        l2 temp "tmp": u32
      ]
      return unit
      scopes [
        scope sc0 parent none comment "local scalar across empty else"
      ]
      body @sc0 {
        stmt0: init l2 = 0x2
        stmt1: if false @sc0 {
          stmt2: fail InvalidBool
        } else @sc0 {
        }
        stmt3: assign l2 = binary bitor(l2, 0x4)
        stmt4: init field(l1, "value") = l2
        stmt5: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}

#[test]
fn structural_hir_ir_path_preserves_temp_after_cursor_sync() {
    let module = parse_hir(
        r#"
hir_module {
  regions [
    r0 "input"
  ]
  stores []
  types [
    type t0 "Cursor" <region "r_input"> size=24 = struct {
      "bytes": Slice<r0, u8> @0
      "pos": u64 @16
    }
    type t1 "ConstantNumber" size=4 = struct {
      "value": u32 @0
    }
  ]
  callables []
  functions [
    function f0 "temp_after_cursor_sync" {
      regions [r0]
      stores []
      params [
        l0 param "cursor": t0<r0>
        l1 destination "out": t1
      ]
      locals [
        l2 temp "raw": u8
      ]
      return unit
      scopes [
        scope sc0 parent none comment "temp survives cursor sync"
      ]
      body @sc0 {
        stmt0: init l2 = load w1(slice_data(field(l0, "bytes")))
        stmt1: assign field(l0, "pos") = 0x1
        stmt2: init field(l1, "value") = l2
        stmt3: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}

#[test]
fn structural_hir_ir_path_executes_loop_break_and_continue() {
    let module = parse_hir(
        r#"
hir_module {
  regions []
  stores []
  types [
    type t0 "ConstantNumber" size=4 = struct {
      "value": u32 @0
    }
  ]
  callables []
  functions [
    function f0 "loop_break_continue" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t0
      ]
      locals [
        l2 temp "i": u64
        l3 temp "sum": u64
      ]
      return unit
      scopes [
        scope sc0 parent none comment "loop break/continue kernel"
      ]
      body @sc0 {
        stmt0: init l2 = 0x0
        stmt1: init l3 = 0x0
        stmt2: loop @sc0 {
          stmt3: if binary eq(l2, 0x5) @sc0 {
            stmt4: break
          } else @sc0 {
          }
          stmt5: assign l2 = binary add(l2, 0x1)
          stmt6: if binary eq(binary bitand(l2, 0x1), 0x0) @sc0 {
            stmt7: continue
          } else @sc0 {
          }
          stmt8: assign l3 = binary add(l3, l2)
        }
        stmt9: init field(l1, "value") = l3
        stmt10: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}

#[test]
fn structural_hir_ir_path_decodes_if_and_match() {
    let module = parse_hir(
        r#"
hir_module {
  regions []
  stores []
  types [
    type t0 "UnitAnimal" size=1 = enum disc_width=1 {
      "Cat" =0 {}
      "Dog" =1 {}
      "Parrot" =2 {}
    }
    type t1 "BranchyAnimal" size=8 = struct {
      "animal": t0 @0
      "value": u32 @4
    }
  ]
  callables []
  functions [
    function f0 "branchy_animal" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t1
      ]
      locals [
        l2 let "flag": bool
        l3 let "tag": u32
      ]
      return unit
      scopes [
        scope sc0 parent none comment "structural if/match HIR"
      ]
      body @sc0 {
        stmt0: init l2 = true
        stmt1: if l2 @sc0 {
          stmt2: init field(l1, "animal") = variant t0::"Dog" {}
        } else @sc0 {
          stmt3: init field(l1, "animal") = variant t0::"Cat" {}
        }
        stmt4: init l3 = 0x1
        stmt5: match l3 {
          arm 0x0 @sc0 {
            stmt6: init field(l1, "value") = 0x7
          }
          arm 0x1 @sc0 {
            stmt7: init field(l1, "value") = 0x2a
          }
        }
        stmt8: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}

#[test]
fn structural_hir_ir_path_computes_bit_masks() {
    let module = parse_hir(
        r#"
hir_module {
  regions []
  stores []
  types [
    type t0 "MaskSummary" size=16 = struct {
      "masked": u32 @0
      "shifted": u32 @4
      "toggled": u32 @8
      "combined": u32 @12
    }
  ]
  callables []
  functions [
    function f0 "mask_summary" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t0
      ]
      locals [
        l2 let "mask": u32
        l3 let "masked": u32
        l4 let "shifted": u32
        l5 let "toggled": u32
        l6 let "combined": u32
      ]
      return unit
      scopes [
        scope sc0 parent none comment "structural bit-mask HIR"
      ]
      body @sc0 {
        stmt0: init l2 = 0xf
        stmt1: init l3 = binary bitand(l2, 0xb)
        stmt2: init l4 = binary shr(l3, 0x1)
        stmt3: init l5 = binary xor(l3, 0x3)
        stmt4: init l6 = binary bitor(binary shl(0x1, 0x3), 0x1)
        stmt5: init field(l1, "masked") = l3
        stmt6: init field(l1, "shifted") = l4
        stmt7: init field(l1, "toggled") = l5
        stmt8: init field(l1, "combined") = l6
        stmt9: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}

#[test]
fn structural_hir_ir_path_updates_local_scratch_struct_fields() {
    let module = parse_hir(
        r#"
hir_module {
  regions []
  stores []
  types [
    type t0 "ScratchState" size=8 = struct {
      "mask": u32 @0
      "done": u32 @4
    }
    type t1 "ScratchSummary" size=8 = struct {
      "mask": u32 @0
      "done": u32 @4
    }
  ]
  callables []
  functions [
    function f0 "scratch_summary" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t1
      ]
      locals [
        l2 let "scratch": t0
      ]
      return unit
      scopes [
        scope sc0 parent none comment "structural local scratch-state HIR"
      ]
      body @sc0 {
        stmt0: init field(l2, "mask") = 0xf
        stmt1: init field(l2, "done") = binary bitand(field(l2, "mask"), 0x3)
        stmt2: init field(l1, "mask") = field(l2, "mask")
        stmt3: init field(l1, "done") = field(l2, "done")
        stmt4: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}

#[test]
fn structural_hir_ir_path_updates_dynamic_local_array_elements() {
    let module = parse_hir(
        r#"
hir_module {
  regions []
  stores []
  types [
    type t0 "DynamicIndexSummary" size=4 = struct {
      "selected": u32 @0
    }
  ]
  callables []
  functions [
    function f0 "dynamic_index_summary" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t0
      ]
      locals [
        l2 let "scratch": Array<u32, 4>
        l3 let "idx": u32
      ]
      return unit
      scopes [
        scope sc0 parent none comment "structural dynamic indexed scratch-array HIR"
      ]
      body @sc0 {
        stmt0: init l3 = 0x2
        stmt1: assign index(l2, l3) = 0x2a
        stmt2: init field(l1, "selected") = index(l2, l3)
        stmt3: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}

#[test]
fn structural_hir_ir_path_updates_dynamic_destination_array_elements() {
    let module = parse_hir(
        r#"
hir_module {
  regions []
  stores []
  types [
    type t0 "DynamicDestinationSummary" size=20 = struct {
      "values": Array<u32, 4> @0
      "selected": u32 @16
    }
  ]
  callables []
  functions [
    function f0 "dynamic_destination_summary" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t0
      ]
      locals [
        l2 let "idx": u32
      ]
      return unit
      scopes [
        scope sc0 parent none comment "structural dynamic indexed destination-array HIR"
      ]
      body @sc0 {
        stmt0: init l2 = 0x1
        stmt1: init index(field(l1, "values"), 0x0) = 0x5
        stmt2: assign index(field(l1, "values"), l2) = 0x7
        stmt3: init index(field(l1, "values"), 0x2) = 0xb
        stmt4: init index(field(l1, "values"), 0x3) = 0xd
        stmt5: init field(l1, "selected") = index(field(l1, "values"), l2)
        stmt6: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}

#[test]
fn structural_hir_ir_path_reads_dynamic_local_aggregate_elements() {
    let module = parse_hir(
        r#"
hir_module {
  regions []
  stores []
  types [
    type t0 "Pair" size=16 = struct {
      "lo": u64 @0
      "hi": u64 @8
    }
    type t1 "DynamicAggregateSummary" size=16 = struct {
      "pair": t0 @0
    }
  ]
  callables []
  functions [
    function f0 "dynamic_aggregate_summary" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t1
      ]
      locals [
        l2 let "pairs": Array<t0, 2>
        l3 let "idx": u32
      ]
      return unit
      scopes [
        scope sc0 parent none comment "structural dynamic indexed aggregate-array HIR"
      ]
      body @sc0 {
        stmt0: init field(index(l2, 0x0), "lo") = 0x1
        stmt1: init field(index(l2, 0x0), "hi") = 0x2
        stmt2: init field(index(l2, 0x1), "lo") = 0x3
        stmt3: init field(index(l2, 0x1), "hi") = 0x4
        stmt4: init l3 = 0x1
        stmt5: init field(l1, "pair") = index(l2, l3)
        stmt6: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}

#[test]
fn structural_hir_ir_path_writes_dynamic_local_aggregate_elements() {
    let module = parse_hir(
        r#"
hir_module {
  regions []
  stores []
  types [
    type t0 "Pair" size=16 = struct {
      "lo": u64 @0
      "hi": u64 @8
    }
    type t1 "DynamicAggregateSummary" size=16 = struct {
      "pair": t0 @0
    }
  ]
  callables []
  functions [
    function f0 "dynamic_aggregate_write_summary" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t1
      ]
      locals [
        l2 let "pairs": Array<t0, 2>
        l3 let "pair": t0
        l4 let "idx": u32
      ]
      return unit
      scopes [
        scope sc0 parent none comment "structural dynamic indexed aggregate-array write HIR"
      ]
      body @sc0 {
        stmt0: init field(l3, "lo") = 0x9
        stmt1: init field(l3, "hi") = 0xa
        stmt2: init l4 = 0x1
        stmt3: assign index(l2, l4) = l3
        stmt4: init field(l1, "pair") = index(l2, l4)
        stmt5: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}

#[test]
fn structural_hir_ir_path_writes_dynamic_destination_aggregate_elements() {
    let module = parse_hir(
        r#"
hir_module {
  regions []
  stores []
  types [
    type t0 "Pair" size=16 = struct {
      "lo": u64 @0
      "hi": u64 @8
    }
    type t1 "DynamicAggregateDestinationSummary" size=48 = struct {
      "pairs": Array<t0, 2> @0
      "selected": t0 @32
    }
  ]
  callables []
  functions [
    function f0 "dynamic_aggregate_destination_summary" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t1
      ]
      locals [
        l2 let "pair": t0
        l3 let "idx": u32
      ]
      return unit
      scopes [
        scope sc0 parent none comment "structural dynamic indexed destination aggregate-array HIR"
      ]
      body @sc0 {
        stmt0: init field(l2, "lo") = 0x15
        stmt1: init field(l2, "hi") = 0x16
        stmt2: init l3 = 0x1
        stmt3: assign index(field(l1, "pairs"), l3) = l2
        stmt4: init field(index(field(l1, "pairs"), 0x0), "lo") = 0x1
        stmt5: init field(index(field(l1, "pairs"), 0x0), "hi") = 0x2
        stmt6: init field(l1, "selected") = index(field(l1, "pairs"), l3)
        stmt7: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = lower_hir_module(&module);
    insta::assert_snapshot!(format!(
        "{}",
        ir.display_with_registry(&crate::ir::IntrinsicRegistry::empty())
    ));
}
