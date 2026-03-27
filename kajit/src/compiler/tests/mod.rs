use facet::Facet;
use facet::Shape;
use facet_testhelpers::test;
use kajit_format::instantiated_shape_symbol_key;
use kajit_hir as hir;
use kajit_hir_text::parse_hir;
use serde::Serialize;

use super::{
    CompiledDecoder, build_decoder_ir_via_hir, build_jit_debug_info_from_source_map,
    build_json_decoder_hir, build_postcard_decoder_hir, build_postcard_decoder_ir_via_hir,
    build_structural_hir_ir, cfg_mir_dwarf_variables, cfg_semantic_field_dwarf_variables,
    cfg_semantic_named_dwarf_variables, cfg_value_dwarf_variables, compile_linear_ir_decoder,
    deser_dwarf_variables, dwarf_expr_for_out_field, format_allocated_regalloc_edits,
    jit_dwarf_target_arch, lower_hir_module, materialize_backend_result,
    run_default_passes_from_env,
};

#[derive(Facet)]
struct Wrapper<T> {
    inner: T,
}

#[derive(Facet)]
struct ConstWrapper<const N: usize> {
    inner: [u8; N],
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct BorrowedHeader<'a> {
    len: u32,
    name: &'a str,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct OwnedHeader {
    len: u32,
    name: String,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct OwnedAddress {
    city: String,
    zip: u32,
}

#[derive(Debug, PartialEq, Facet)]
struct FloatHeader {
    a: f32,
    b: f64,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct CharHeader {
    ch: char,
}

#[derive(Debug, PartialEq, Eq, Facet, Serialize)]
struct BigUnsigned {
    value: u128,
}

#[derive(Debug, PartialEq, Eq, Facet, Serialize)]
struct BigSigned {
    value: i128,
}

#[derive(Debug, PartialEq, Eq, Facet, Serialize)]
struct MaybeBigUnsigned {
    value: Option<u128>,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct MaybeBorrowedName<'a> {
    name: Option<&'a str>,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct MaybeCount {
    count: Option<u32>,
}

#[derive(Debug, PartialEq, Eq, Facet)]
#[repr(u8)]
enum UnitAnimal {
    Cat,
    Dog,
    Parrot,
}

#[derive(Debug, PartialEq, Eq, Facet)]
#[repr(u8)]
enum PayloadAnimal<'a> {
    Cat,
    Count(u32),
    Name(&'a str),
}

#[derive(Debug, PartialEq, Eq, Facet)]
#[repr(u8)]
enum OwnedAnimal {
    Cat,
    Dog { name: String, good_boy: bool },
    Parrot(String),
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct OwnedZoo {
    name: String,
    star: OwnedAnimal,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct ConstantNumber {
    value: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
pub(crate) struct ScalarNumber {
    pub(crate) value: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
pub(crate) struct BoolHeader {
    pub(crate) value: bool,
}

#[derive(Debug, PartialEq, Eq, Facet)]
pub(crate) struct ScalarArrayHolder {
    pub(crate) values: [u32; 4],
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct BorrowedArrayHolder<'a> {
    values: [&'a str; 2],
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct BranchyAnimal {
    animal: UnitAnimal,
    value: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct MaskSummary {
    masked: u32,
    shifted: u32,
    toggled: u32,
    combined: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct ScratchSummary {
    mask: u32,
    done: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct DynamicIndexSummary {
    selected: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct DynamicDestinationSummary {
    values: [u32; 4],
    selected: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
pub(crate) struct PersistentBufferSummary {
    pub(crate) ptr: usize,
    pub(crate) len: usize,
}

#[derive(Debug, PartialEq, Eq, Facet)]
pub(crate) struct VecHolder {
    values: Vec<u32>,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct Pair {
    lo: u64,
    hi: u64,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct DynamicAggregateSummary {
    pair: Pair,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct DynamicAggregateDestinationSummary {
    pairs: [Pair; 2],
    selected: Pair,
}

pub(crate) fn compile_structural_hir_decoder(
    shape: &'static Shape,
    module: &hir::Module,
) -> CompiledDecoder {
    let mut func = build_structural_hir_ir(shape, module);
    run_default_passes_from_env(&mut func);
    let linear = crate::linearize::linearize(&mut func);
    compile_linear_ir_decoder(&linear, false)
}

fn compile_postcard_decoder_via_structural_hir(shape: &'static Shape) -> CompiledDecoder {
    let module = build_postcard_decoder_hir(shape);
    compile_structural_hir_decoder(shape, &module)
}

#[test]
fn instantiated_shape_symbol_key_distinguishes_generic_instantiations() {
    let u32_key = instantiated_shape_symbol_key(<Wrapper<u32>>::SHAPE);
    let string_key = instantiated_shape_symbol_key(<Wrapper<String>>::SHAPE);

    assert_ne!(u32_key, string_key);
    assert!(u32_key.starts_with('d'));
    assert!(u32_key.contains("__t_0_d"));
    assert_eq!(
        <Wrapper<u32>>::SHAPE.decl_id,
        <Wrapper<String>>::SHAPE.decl_id
    );
}

#[test]
fn instantiated_shape_symbol_key_includes_const_params() {
    let n4_key = instantiated_shape_symbol_key(<ConstWrapper<4>>::SHAPE);
    let n8_key = instantiated_shape_symbol_key(<ConstWrapper<8>>::SHAPE);

    assert_ne!(n4_key, n8_key);
    assert!(n4_key.contains("__c_0_u4"));
    assert!(n8_key.contains("__c_0_u8"));
}

#[test]
fn postcard_hir_models_borrowed_output_structs() {
    let module = build_postcard_decoder_hir(<BorrowedHeader<'static>>::SHAPE);
    assert_eq!(module.functions.len(), 1);

    let (_, function) = module.functions.iter().next().unwrap();
    assert_eq!(function.params.len(), 2);
    assert_eq!(function.region_params.len(), 1);

    let destination = function.destination_param().unwrap();
    let hir::Type::Named { def, args } = &destination.ty else {
        panic!("expected named root output type");
    };
    assert_eq!(
        module.type_defs[*def].name,
        <BorrowedHeader<'static>>::SHAPE.type_identifier
    );
    assert_eq!(
        args,
        &vec![hir::GenericArg::Region(function.region_params[0])]
    );

    let statements = &function.body.statements;
    assert!(
        module
            .callable_named("runtime.validate_utf8_range")
            .is_some(),
        "borrowed string lowering should install runtime UTF-8 validation"
    );
    assert!(
        module.callable_named("postcard.read_str").is_none(),
        "borrowed string lowering should not use postcard.read_str"
    );
    assert!(
        module.callable_named("postcard.read_u32").is_none(),
        "borrowed header lowering should not use postcard.read_u32"
    );
    assert!(
        statements.iter().any(|stmt| matches!(
            &stmt.kind,
            hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(callable_id),
                ..
            })) if module.callables[*callable_id].name == "runtime.validate_utf8_range"
        )),
        "borrowed string lowering should validate UTF-8 explicitly"
    );
    assert!(
        statements.iter().any(|stmt| matches!(
            &stmt.kind,
            hir::StmtKind::Init {
                place: hir::Place::Field { field, .. },
                value: hir::Expr::Str { .. },
            } if field == "name"
        )),
        "borrowed string lowering should materialize a str value directly into the destination"
    );
}

#[test]
fn json_hir_models_root_bool_without_reader_calls() {
    let module = build_json_decoder_hir(<bool>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    assert!(
        module.callable_named("postcard.read_bool").is_none(),
        "json root bool HIR should not mention postcard bool readers"
    );
    assert!(
        module.callable_named("kajit_json_read_bool").is_none(),
        "json root bool HIR should not call the old JSON bool intrinsic"
    );
    assert!(
        module.callables.is_empty(),
        "root bool HIR should be leaf-free"
    );
    assert!(
        function
            .body
            .statements
            .iter()
            .any(|stmt| matches!(&stmt.kind, hir::StmtKind::Loop { .. })),
        "json root bool HIR should spell out whitespace skipping as control flow"
    );
    assert!(
        function
            .body
            .statements
            .iter()
            .any(|stmt| matches!(&stmt.kind, hir::StmtKind::If { .. })),
        "json root bool HIR should spell out token dispatch as control flow"
    );
}

#[test]
fn json_hir_models_root_u32_without_reader_calls() {
    let module = build_json_decoder_hir(<u32>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    assert!(
        module.callable_named("postcard.read_u32").is_none(),
        "json root u32 HIR should not mention postcard integer readers"
    );
    assert!(
        module.callable_named("kajit_json_read_u32").is_none(),
        "json root u32 HIR should not call the old JSON u32 intrinsic"
    );
    assert!(
        module.callables.is_empty(),
        "root u32 HIR should be leaf-free"
    );
    assert!(
        function
            .body
            .statements
            .iter()
            .any(|stmt| matches!(&stmt.kind, hir::StmtKind::Loop { .. })),
        "json root u32 HIR should spell out whitespace and digit scanning as control flow"
    );
}

#[test]
fn json_hir_models_root_u64_without_reader_calls() {
    let module = build_json_decoder_hir(<u64>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    assert!(
        module.callable_named("postcard.read_u64").is_none(),
        "json root u64 HIR should not mention postcard integer readers"
    );
    assert!(
        module.callable_named("kajit_json_read_u64").is_none(),
        "json root u64 HIR should not call the old JSON u64 intrinsic"
    );
    assert!(
        module.callables.is_empty(),
        "root u64 HIR should be leaf-free"
    );
    assert!(
        function
            .body
            .statements
            .iter()
            .any(|stmt| matches!(&stmt.kind, hir::StmtKind::Loop { .. })),
        "json root u64 HIR should spell out whitespace and digit scanning as control flow"
    );
}

#[test]
fn compile_decoder_prefers_hir_for_supported_json_root_u32() {
    let decoder = crate::compile_decoder(<u32>::SHAPE, crate::DecoderKind::Json);
    let listing = decoder.cfg_mir_line_text_by_line.join("\n");

    assert!(
        !listing.contains("kajit_json_read_u32"),
        "supported JSON shapes should compile through the HIR path"
    );
    assert!(
        listing.contains("branch_if_zero") || listing.contains("branch "),
        "HIR JSON lowering should still produce explicit control flow"
    );
}

#[test]
fn postcard_hir_models_owned_output_strings() {
    let module = build_postcard_decoder_hir(<OwnedHeader>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    assert!(
        module
            .callable_named("runtime.string_validate_alloc_copy")
            .is_some(),
        "owned string lowering should install the raw string allocation helper"
    );
    assert!(
        module.callable_named("postcard.read_str").is_none(),
        "owned string lowering should not use postcard.read_str"
    );
    assert!(
        function
            .locals
            .iter()
            .any(|local| matches!(local.ty, hir::Type::Address { .. })),
        "owned string lowering should allocate a persistent data pointer local"
    );
    assert!(
        function.body.statements.iter().any(|stmt| matches!(
            &stmt.kind,
            hir::StmtKind::Init {
                value: hir::Expr::Call(hir::CallExpr {
                    target: hir::CallTarget::Callable(callable_id),
                    ..
                }),
                ..
            } if module.callables[*callable_id].name == "runtime.string_validate_alloc_copy"
        )),
        "owned string lowering should compute string storage through the lean helper"
    );
}

#[test]
fn compile_decoder_prefers_hir_for_supported_postcard_bool_field() {
    let decoder = crate::compile_decoder(<BoolHeader>::SHAPE, crate::DecoderKind::Postcard);
    let listing = decoder.cfg_mir_line_text_by_line.join("\n");

    assert!(
        !listing.contains("kajit_read_bool"),
        "supported postcard shapes should compile through the HIR path"
    );
}

#[test]
fn postcard_hir_models_float_scalars_without_reader_calls() {
    let module = build_postcard_decoder_hir(<FloatHeader>::SHAPE);

    assert!(
        module.callable_named("postcard.read_f32").is_none(),
        "float lowering should not use postcard.read_f32"
    );
    assert!(
        module.callable_named("postcard.read_f64").is_none(),
        "float lowering should not use postcard.read_f64"
    );
}

#[test]
fn postcard_hir_models_char_without_reader_calls() {
    let module = build_postcard_decoder_hir(<CharHeader>::SHAPE);

    assert!(
        module.callable_named("postcard.read_char").is_none(),
        "char lowering should not use postcard.read_char"
    );
    assert!(
        module
            .callable_named("runtime.validate_utf8_range")
            .is_some(),
        "char lowering should validate UTF-8 explicitly"
    );
}

#[test]
fn postcard_hir_models_128bit_scalars_without_reader_calls() {
    let unsigned = build_postcard_decoder_hir(<BigUnsigned>::SHAPE);
    let signed = build_postcard_decoder_hir(<BigSigned>::SHAPE);
    let optional = build_postcard_decoder_hir(<MaybeBigUnsigned>::SHAPE);

    assert!(
        unsigned.callable_named("postcard.read_u128").is_none(),
        "u128 lowering should not use postcard.read_u128"
    );
    assert!(
        signed.callable_named("postcard.read_i128").is_none(),
        "i128 lowering should not use postcard.read_i128"
    );
    assert!(
        optional.callable_named("postcard.read_u128").is_none(),
        "Option<u128> lowering should not use postcard.read_u128"
    );
}

#[test]
fn postcard_hir_models_option_borrowed_fields() {
    let module = build_postcard_decoder_hir(<MaybeBorrowedName<'static>>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();
    let input_region = function.region_params[0];

    assert!(
        module.callable_named("postcard.read_option_tag").is_none(),
        "option lowering should not use postcard.read_option_tag"
    );

    assert!(function.locals.len() >= 4);
    assert!(
        function
            .locals
            .iter()
            .any(|local| local.ty == hir::Type::bool())
    );
    assert!(
        function
            .locals
            .iter()
            .any(|local| local.ty == hir::Type::u(8))
    );
    assert!(
        function
            .locals
            .iter()
            .any(|local| local.ty == hir::Type::str(input_region))
    );

    let (_, then_block, else_block) = function
        .body
        .statements
        .iter()
        .find_map(|stmt| match &stmt.kind {
            hir::StmtKind::If {
                condition,
                then_block,
                else_block,
            } if matches!(condition, hir::Expr::Local(_)) => {
                Some((condition, then_block, else_block))
            }
            _ => None,
        })
        .expect("expected option if statement");

    let Some(else_block) = else_block else {
        panic!("expected explicit option else block");
    };
    assert!(then_block.statements.len() >= 2);
    assert_eq!(else_block.statements.len(), 1);
    assert!(
        then_block.statements.iter().any(|stmt| matches!(
            &stmt.kind,
            hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(callable_id),
                ..
            })) if module.callables[*callable_id].name == "runtime.validate_utf8_range"
        )),
        "borrowed option payload should validate UTF-8 explicitly"
    );

    let value = then_block
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "Some") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected Some variant init");
    let hir::Expr::Variant {
        def,
        variant,
        fields,
    } = value
    else {
        panic!("expected Some variant expression");
    };
    assert_eq!(variant, "Some");
    assert_eq!(fields.len(), 1);
    let hir::Expr::Local(payload_local) = &fields[0].1 else {
        panic!("expected Some variant payload local");
    };
    assert_eq!(
        function
            .locals
            .iter()
            .find(|local| local.ty == hir::Type::str(input_region))
            .expect("expected borrowed payload local")
            .local,
        *payload_local
    );

    let hir::TypeDefKind::Enum { variants, .. } = &module.type_defs[*def].kind else {
        panic!("expected Option HIR enum type");
    };
    assert_eq!(variants[0].name, "None");
    assert_eq!(variants[1].name, "Some");
    assert_eq!(variants[1].fields[0].ty, hir::Type::str(input_region));

    let value = else_block
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "None") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected None variant init");
    let hir::Expr::Variant {
        variant, fields, ..
    } = value
    else {
        panic!("expected None variant expression");
    };
    assert_eq!(variant, "None");
    assert!(fields.is_empty());
}

#[test]
fn postcard_hir_text_round_trips() {
    std::thread::Builder::new()
        .name("postcard_hir_text_round_trips".to_owned())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let module = build_postcard_decoder_hir(<MaybeBorrowedName<'static>>::SHAPE);
            let text = module.to_string();
            let reparsed = parse_hir(&text).expect("postcard HIR text should parse");

            assert_eq!(reparsed, module);
        })
        .expect("thread should spawn")
        .join()
        .expect("round-trip thread should succeed");
}

#[test]
fn postcard_hir_ir_path_decodes_option_borrowed_fields() {
    let decoder = crate::compile_postcard_decoder_via_hir(<MaybeBorrowedName<'static>>::SHAPE);

    let some = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[1, 2, b'h', b'i'])
        .expect("HIR->RVSDG postcard decoder should decode Some(&str)");
    assert_eq!(some, MaybeBorrowedName { name: Some("hi") });

    let none = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[0])
        .expect("HIR->RVSDG postcard decoder should decode None");
    assert_eq!(none, MaybeBorrowedName { name: None });
}

#[test]
fn postcard_structural_hir_ir_path_decodes_float_fields() {
    let decoder = compile_postcard_decoder_via_structural_hir(<FloatHeader>::SHAPE);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&3.14f32.to_le_bytes());
    bytes.extend_from_slice(&2.718281828459045f64.to_le_bytes());

    let value = crate::deserialize::<FloatHeader>(&decoder, &bytes)
        .expect("structural HIR postcard decoder should decode float fields");
    assert_eq!(value.a.to_bits(), 3.14f32.to_bits());
    assert_eq!(value.b.to_bits(), 2.718281828459045f64.to_bits());
}

#[test]
fn postcard_structural_hir_ir_path_decodes_char_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<CharHeader>::SHAPE);

    let value = crate::deserialize::<CharHeader>(&decoder, &[2, 0xC3, 0x9F])
        .expect("structural HIR postcard decoder should decode char fields");
    assert_eq!(value, CharHeader { ch: 'ß' });
}

#[test]
fn postcard_structural_hir_ir_path_decodes_128bit_fields() {
    let unsigned = BigUnsigned {
        value: (1_u128 << 100) | 0x1234_5678_9abc_def0_u128,
    };
    let signed = BigSigned {
        value: -((1_i128 << 97) - 0x1234_5678_9abc_i128),
    };

    let unsigned_decoder = compile_postcard_decoder_via_structural_hir(<BigUnsigned>::SHAPE);
    let unsigned_bytes =
        postcard::to_allocvec(&unsigned).expect("postcard should encode unsigned 128-bit sample");
    let unsigned_value = crate::deserialize::<BigUnsigned>(&unsigned_decoder, &unsigned_bytes)
        .expect("structural HIR postcard decoder should decode u128 fields");
    assert_eq!(unsigned_value, unsigned);

    let signed_decoder = compile_postcard_decoder_via_structural_hir(<BigSigned>::SHAPE);
    let signed_bytes =
        postcard::to_allocvec(&signed).expect("postcard should encode signed 128-bit sample");
    let signed_value = crate::deserialize::<BigSigned>(&signed_decoder, &signed_bytes)
        .expect("structural HIR postcard decoder should decode i128 fields");
    assert_eq!(signed_value, signed);
}

#[test]
fn postcard_structural_hir_ir_path_decodes_option_u128_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<MaybeBigUnsigned>::SHAPE);
    let sample = MaybeBigUnsigned {
        value: Some((1_u128 << 72) | 0x55aa_33cc_77ee_u128),
    };
    let bytes = postcard::to_allocvec(&sample).expect("postcard should encode Option<u128>");
    let value = crate::deserialize::<MaybeBigUnsigned>(&decoder, &bytes)
        .expect("structural HIR postcard decoder should decode Option<u128>");
    assert_eq!(value, sample);
}

#[test]
fn postcard_hir_models_unit_enums() {
    let module = build_postcard_decoder_hir(<UnitAnimal>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    assert!(
        module
            .callable_named("postcard.read_discriminant")
            .is_none(),
        "unit enum lowering should not use postcard.read_discriminant"
    );

    assert!(
        function
            .locals
            .iter()
            .any(|local| local.ty == hir::Type::u(32)),
        "unit enum lowering should use a scalar discriminant local"
    );
    let (scrutinee, arms) = function
        .body
        .statements
        .iter()
        .find_map(|stmt| match &stmt.kind {
            hir::StmtKind::Match { scrutinee, arms } => Some((scrutinee, arms)),
            _ => None,
        })
        .expect("expected enum match statement");
    let hir::Expr::Local(disc_local) = scrutinee else {
        panic!("expected discriminant local");
    };
    assert!(
        function
            .locals
            .iter()
            .any(|local| local.local == *disc_local),
        "match should scrutinee the decoded discriminant local"
    );
    assert_eq!(arms.len(), 3);
    assert!(matches!(arms[0].pattern, hir::Pattern::Integer(0)));
    assert!(matches!(arms[1].pattern, hir::Pattern::Integer(1)));
    assert!(matches!(arms[2].pattern, hir::Pattern::Integer(2)));

    let hir::StmtKind::Init { value, .. } = &arms[1].body.statements[0].kind else {
        panic!("expected unit variant init");
    };
    let hir::Expr::Variant {
        variant, fields, ..
    } = value
    else {
        panic!("expected unit variant expression");
    };
    assert_eq!(variant, "Dog");
    assert!(fields.is_empty());
}

#[test]
fn postcard_hir_models_payload_enums() {
    let module = build_postcard_decoder_hir(<PayloadAnimal<'static>>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    let arms = function
        .body
        .statements
        .iter()
        .find_map(|stmt| match &stmt.kind {
            hir::StmtKind::Match { arms, .. } => Some(arms),
            _ => None,
        })
        .expect("expected enum match statement");
    assert_eq!(arms.len(), 3);

    let count_arm = &arms[1];
    let value = count_arm
            .body
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "Count") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected Count variant init");
    let hir::Expr::Variant {
        variant, fields, ..
    } = value
    else {
        panic!("expected Count variant expression");
    };
    assert_eq!(variant, "Count");
    assert_eq!(fields.len(), 1);

    let name_arm = &arms[2];
    assert!(name_arm.body.statements.len() >= 2);
    let value = name_arm
            .body
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "Name") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected Name variant init");
    let hir::Expr::Variant {
        variant, fields, ..
    } = value
    else {
        panic!("expected Name variant expression");
    };
    assert_eq!(variant, "Name");
    assert_eq!(fields.len(), 1);
    assert!(
        name_arm.body.statements.iter().any(|stmt| matches!(
            &stmt.kind,
            hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(callable_id),
                ..
            })) if module.callables[*callable_id].name == "runtime.validate_utf8_range"
        )),
        "borrowed enum payload should validate UTF-8 explicitly"
    );
}

#[test]
fn postcard_hir_scalar_array_u32_4() {
    let module = build_postcard_decoder_hir(<ScalarArrayHolder>::SHAPE);
    insta::assert_snapshot!(module.to_string());
}

#[test]
fn postcard_hir_models_arrays() {
    let module = build_postcard_decoder_hir(<BorrowedArrayHolder<'static>>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    // Array elements should be decoded in a loop, not unrolled
    let has_loop = function
        .body
        .statements
        .iter()
        .any(|stmt| matches!(stmt.kind, hir::StmtKind::Loop { .. }));
    assert!(has_loop, "array decoding should use a loop");
    assert!(
        module.callable_named("postcard.read_str").is_none(),
        "borrowed array lowering should not use postcard.read_str"
    );
}

#[test]
fn postcard_hir_ir_path_decodes_unit_enums() {
    let decoder = crate::compile_postcard_decoder_via_hir(<UnitAnimal>::SHAPE);

    let cat = crate::deserialize::<UnitAnimal>(&decoder, &[0])
        .expect("HIR->RVSDG postcard decoder should decode Cat");
    assert_eq!(cat, UnitAnimal::Cat);

    let dog = crate::deserialize::<UnitAnimal>(&decoder, &[1])
        .expect("HIR->RVSDG postcard decoder should decode Dog");
    assert_eq!(dog, UnitAnimal::Dog);

    let parrot = crate::deserialize::<UnitAnimal>(&decoder, &[2])
        .expect("HIR->RVSDG postcard decoder should decode Parrot");
    assert_eq!(parrot, UnitAnimal::Parrot);
}

#[test]
fn postcard_structural_hir_ir_path_decodes_scalar_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<ScalarNumber>::SHAPE);

    let value = crate::deserialize::<ScalarNumber>(&decoder, &[42])
        .expect("structural HIR postcard decoder should decode a scalar field");
    assert_eq!(value, ScalarNumber { value: 42 });
}

#[test]
fn postcard_structural_hir_ir_path_decodes_borrowed_header() {
    let decoder = compile_postcard_decoder_via_structural_hir(<BorrowedHeader<'static>>::SHAPE);

    let value = crate::deserialize::<BorrowedHeader<'_>>(&decoder, &[7, 2, b'h', b'i'])
        .expect("structural HIR postcard decoder should decode direct borrowed fields");
    assert_eq!(value, BorrowedHeader { len: 7, name: "hi" });
}

#[test]
fn postcard_structural_hir_ir_path_decodes_owned_header() {
    let decoder = compile_postcard_decoder_via_structural_hir(<OwnedHeader>::SHAPE);

    let value = crate::deserialize::<OwnedHeader>(&decoder, &[7, 2, b'h', b'i'])
        .expect("structural HIR postcard decoder should decode direct owned string fields");
    assert_eq!(
        value,
        OwnedHeader {
            len: 7,
            name: "hi".to_owned(),
        }
    );
}

#[test]
fn postcard_structural_hir_ir_path_decodes_root_vec_u32() {
    let decoder = compile_postcard_decoder_via_structural_hir(<Vec<u32>>::SHAPE);

    let value = crate::deserialize::<Vec<u32>>(&decoder, &[3, 1, 2, 3])
        .expect("structural HIR postcard decoder should decode root Vec<u32>");
    assert_eq!(value, vec![1, 2, 3]);
}

#[test]
fn postcard_structural_hir_ir_path_decodes_root_vec_string() {
    let decoder = compile_postcard_decoder_via_structural_hir(<Vec<String>>::SHAPE);

    let value = crate::deserialize::<Vec<String>>(&decoder, &[2, 2, b'h', b'i', 2, b'b', b'y'])
        .expect("structural HIR postcard decoder should decode root Vec<String>");
    assert_eq!(value, vec!["hi".to_owned(), "by".to_owned()]);
}

#[test]
fn postcard_structural_hir_ir_path_decodes_root_vec_structs() {
    let decoder = compile_postcard_decoder_via_structural_hir(<Vec<OwnedAddress>>::SHAPE);

    let value = crate::deserialize::<Vec<OwnedAddress>>(
        &decoder,
        &[2, 2, b'P', b'A', 75, 2, b'L', b'Y', 13],
    )
    .expect("structural HIR postcard decoder should decode root Vec<struct>");
    assert_eq!(
        value,
        vec![
            OwnedAddress {
                city: "PA".to_owned(),
                zip: 75,
            },
            OwnedAddress {
                city: "LY".to_owned(),
                zip: 13,
            },
        ]
    );
}

#[test]
fn postcard_structural_hir_ir_path_decodes_unit_enums() {
    let decoder = compile_postcard_decoder_via_structural_hir(<UnitAnimal>::SHAPE);

    let cat = crate::deserialize::<UnitAnimal>(&decoder, &[0])
        .expect("structural HIR postcard decoder should decode Cat");
    assert_eq!(cat, UnitAnimal::Cat);

    let dog = crate::deserialize::<UnitAnimal>(&decoder, &[1])
        .expect("structural HIR postcard decoder should decode Dog");
    assert_eq!(dog, UnitAnimal::Dog);

    let parrot = crate::deserialize::<UnitAnimal>(&decoder, &[2])
        .expect("structural HIR postcard decoder should decode Parrot");
    assert_eq!(parrot, UnitAnimal::Parrot);
}

#[test]
fn postcard_structural_hir_ir_path_decodes_option_scalar_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<MaybeCount>::SHAPE);

    let some = crate::deserialize::<MaybeCount>(&decoder, &[1, 42])
        .expect("structural HIR postcard decoder should decode Some(u32)");
    assert_eq!(some, MaybeCount { count: Some(42) });

    let none = crate::deserialize::<MaybeCount>(&decoder, &[0])
        .expect("structural HIR postcard decoder should decode None");
    assert_eq!(none, MaybeCount { count: None });
}

#[test]
fn postcard_structural_hir_ir_path_decodes_option_borrowed_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<MaybeBorrowedName<'static>>::SHAPE);

    let some = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[1, 2, b'h', b'i'])
        .expect("structural HIR postcard decoder should decode Some(&str)");
    assert_eq!(some, MaybeBorrowedName { name: Some("hi") });

    let none = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[0])
        .expect("structural HIR postcard decoder should decode None");
    assert_eq!(none, MaybeBorrowedName { name: None });
}

mod hir_to_ir;

#[test]
fn postcard_structural_hir_ir_path_decodes_scalar_arrays() {
    let decoder = compile_postcard_decoder_via_structural_hir(<ScalarArrayHolder>::SHAPE);

    let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
        .expect("structural HIR postcard decoder should decode scalar arrays");
    assert_eq!(
        value,
        ScalarArrayHolder {
            values: [1, 2, 3, 4],
        }
    );
}

#[test]
fn postcard_structural_hir_ir_path_decodes_borrowed_arrays() {
    let decoder =
        compile_postcard_decoder_via_structural_hir(<BorrowedArrayHolder<'static>>::SHAPE);

    let value =
        crate::deserialize::<BorrowedArrayHolder<'_>>(&decoder, &[2, b'h', b'i', 2, b'o', b'k'])
            .expect("structural HIR postcard decoder should decode borrowed arrays");
    assert_eq!(
        value,
        BorrowedArrayHolder {
            values: ["hi", "ok"],
        }
    );
}

#[test]
fn postcard_structural_hir_ir_path_decodes_payload_enums() {
    let decoder = compile_postcard_decoder_via_structural_hir(<PayloadAnimal<'static>>::SHAPE);

    let cat = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[0])
        .expect("structural HIR postcard decoder should decode unit enum variant");
    assert_eq!(cat, PayloadAnimal::Cat);

    let count = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[1, 42])
        .expect("structural HIR postcard decoder should decode scalar payload enum variant");
    assert_eq!(count, PayloadAnimal::Count(42));

    let name = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[2, 2, b'h', b'i'])
        .expect("structural HIR postcard decoder should decode borrowed payload enum variant");
    assert_eq!(name, PayloadAnimal::Name("hi"));
}

#[test]
fn postcard_structural_hir_ir_path_decodes_enum_in_struct_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<OwnedZoo>::SHAPE);

    let value = crate::deserialize::<OwnedZoo>(
        &decoder,
        &[
            8, b'C', b'i', b't', b'y', b' ', b'Z', b'o', b'o', // zoo name
            1,    // Dog discriminant
            3, b'R', b'e', b'x', // dog name
            1,    // good_boy
        ],
    )
    .expect("structural HIR postcard decoder should decode nested enum payloads");
    assert_eq!(
        value,
        OwnedZoo {
            name: "City Zoo".to_owned(),
            star: OwnedAnimal::Dog {
                name: "Rex".to_owned(),
                good_boy: true,
            },
        }
    );
}

#[test]
fn postcard_structural_hir_array_path_matches_jit_differential_harness() {
    let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
    let linear = crate::linearize::linearize(&mut func);
    let output_size = std::mem::size_of::<ScalarArrayHolder>();
    let report = crate::differential_check_linear_ir_vs_jit_with_output_size(
        &linear,
        &[1, 2, 3, 4],
        output_size,
    )
    .expect("differential harness should execute structural HIR postcard array decoder");
    assert!(
        report.is_match(),
        "unexpected differential mismatch: {:?}",
        report.mismatch
    );
}

#[test]
fn postcard_structural_hir_array_path_matches_post_regalloc_simulation() {
    let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
    let linear = crate::linearize::linearize(&mut func);
    let hints = Default::default();
    let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear, hints);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
        .expect("regalloc should allocate structural HIR postcard array cfg");
    let result = crate::regalloc_engine::differential_check_cfg(&cfg, &alloc, &[1, 2, 3, 4]);
    assert!(
        matches!(
            result,
            crate::regalloc_engine::DifferentialCheckResult::Match { .. }
        ),
        "unexpected interpreter/post-regalloc mismatch: {result:?}"
    );
}

#[test]
fn postcard_structural_hir_array_path_without_backend_edit_emission() {
    let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
    let linear = crate::linearize::linearize(&mut func);
    let hints = Default::default();
    let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear, hints);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
        .expect("regalloc should allocate structural HIR postcard array cfg");
    let result = crate::ir_backend::compile_linear_ir_with_alloc_and_mode(
        &linear, &cfg, &alloc, false, None,
    );
    let (buf, entry, _source_map, _backend_debug_info, _asm_program) =
        materialize_backend_result(result);
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
    let decoder = CompiledDecoder {
        buf,
        cfg_mir_line_text_by_line: Default::default(),
        entry,
        func,
        trusted_utf8_input: false,
        _jit_registration: None,
        asm_program: None,
    };

    let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
        .expect("structural HIR postcard array decoder should execute without backend edits");
    assert_eq!(
        value,
        ScalarArrayHolder {
            values: [1, 2, 3, 4]
        }
    );
}

#[test]
fn debug_scalar_array_regalloc_edits() {
    let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
    let linear = crate::linearize::linearize(&mut func);
    let hints = Default::default();
    let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear, hints);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
        .expect("regalloc should allocate structural HIR postcard array cfg");
    println!("{}", format_allocated_regalloc_edits(&alloc));
}

#[test]
fn debug_scalar_array_emission_trace() {
    let decoder = compile_postcard_decoder_via_structural_hir(<ScalarArrayHolder>::SHAPE);
    println!(
        "{}",
        decoder
            .emission_trace_text()
            .expect("emission trace should render")
    );
}

#[test]
#[ignore = "json struct HIR lowering not implemented yet"]
fn json_bool_true_false_matches_post_regalloc_simulation() {
    #[derive(Debug, PartialEq, Eq, Facet, serde::Serialize, serde::Deserialize)]
    struct Bools {
        a: bool,
        b: bool,
    }

    let value = Bools { a: true, b: false };
    let input = serde_json::to_vec(&value).expect("serialize json input");
    let mut func = build_decoder_ir_via_hir(<Bools>::SHAPE, crate::DecoderKind::Json);
    run_default_passes_from_env(&mut func);
    let linear = crate::linearize::linearize(&mut func);
    let hints = Default::default();
    let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear, hints);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
        .expect("regalloc should allocate json bool cfg");
    let result = crate::regalloc_engine::differential_check_cfg(&cfg, &alloc, &input);
    assert!(
        matches!(
            result,
            crate::regalloc_engine::DifferentialCheckResult::Match { .. }
        ),
        "unexpected interpreter/post-regalloc mismatch: {result:?}"
    );
}

#[test]
#[ignore = "json struct HIR lowering not implemented yet"]
fn json_bool_true_false_without_backend_edit_emission() {
    #[derive(Debug, PartialEq, Eq, Facet, serde::Serialize, serde::Deserialize)]
    struct Bools {
        a: bool,
        b: bool,
    }

    let value = Bools { a: true, b: false };
    let input = serde_json::to_vec(&value).expect("serialize json input");
    let expected: Bools = serde_json::from_slice(&input).expect("decode reference json");
    let mut func = build_decoder_ir_via_hir(<Bools>::SHAPE, crate::DecoderKind::Json);
    run_default_passes_from_env(&mut func);
    let linear = crate::linearize::linearize(&mut func);
    let hints = Default::default();
    let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear, hints);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
        .expect("regalloc should allocate json bool cfg");
    let result = crate::ir_backend::compile_linear_ir_with_alloc_and_mode(
        &linear, &cfg, &alloc, false, None,
    );
    let (buf, entry, _source_map, _backend_debug_info, _asm_program) =
        materialize_backend_result(result);
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
    let decoder = CompiledDecoder {
        buf,
        cfg_mir_line_text_by_line: Default::default(),
        entry,
        func,
        trusted_utf8_input: false,
        _jit_registration: None,
        asm_program: None,
    };

    let got = crate::from_str::<Bools>(&decoder, core::str::from_utf8(&input).unwrap())
        .expect("json bool decoder should execute without backend edits");
    assert_eq!(got, expected);
}

#[test]
#[ignore = "non-HIR path disabled"]
fn json_bool_true_false_with_backend_edit_emission() {
    #[derive(Debug, PartialEq, Eq, Facet, serde::Serialize, serde::Deserialize)]
    struct Bools {
        a: bool,
        b: bool,
    }

    let value = Bools { a: true, b: false };
    let input = serde_json::to_vec(&value).expect("serialize json input");
    let expected: Bools = serde_json::from_slice(&input).expect("decode reference json");
    let decoder = crate::compile_decoder(<Bools>::SHAPE, crate::DecoderKind::Json);
    let got = crate::from_str::<Bools>(&decoder, core::str::from_utf8(&input).unwrap())
        .expect("json bool decoder should execute with backend edits");
    assert_eq!(got, expected);
    assert!(
        crate::regalloc_edit_count(<Bools>::SHAPE, crate::DecoderKind::Json) > 0,
        "expected this regression test to exercise backend edit emission"
    );
}

#[test]
fn builds_dwarf_sections_from_source_map_lines() {
    let source_map = vec![
        kajit_emit::SourceMapEntry {
            offset: 0,
            location: kajit_emit::SourceLocation {
                file: 0,
                line: 1,
                column: 1,
            },
        },
        kajit_emit::SourceMapEntry {
            offset: 8,
            location: kajit_emit::SourceLocation {
                file: 0,
                line: 2,
                column: 1,
            },
        },
    ];

    let listing_path = std::env::temp_dir()
        .join(format!("kajit-debug-test-{}", std::process::id()))
        .join("sample.cfg-mir");
    std::fs::create_dir_all(listing_path.parent().expect("temp listing dir")).unwrap();
    std::fs::write(&listing_path, "inst0\ninst1\n").unwrap();

    let debug_info = build_jit_debug_info_from_source_map(
        0x1000 as *const u8,
        32,
        Some(&source_map),
        &listing_path,
        crate::jit_dwarf::JitDebugSubprogram {
            name: "kajit::decode::test".to_string(),
            frame_base_expression: crate::jit_dwarf::expr_breg(
                crate::jit_dwarf::frame_base_register(jit_dwarf_target_arch()),
                0,
            ),
            variables: Vec::new(),
            lexical_blocks: Vec::new(),
        },
    )
    .expect("expected debug info");
    let dwarf = crate::jit_dwarf::build_jit_dwarf_sections_from_debug_info(&debug_info)
        .expect("expected dwarf sections");
    assert!(!dwarf.debug_line.is_empty());
}

#[test]
fn deser_dwarf_variables_cover_fixed_runtime_state() {
    let vars = deser_dwarf_variables(jit_dwarf_target_arch());
    let names = vars.iter().map(|var| var.name.as_str()).collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "input_ptr",
            "input_end",
            "out_ptr",
            "ctx",
            "error_code",
            "error_offset",
        ]
    );
    for var in vars {
        match var.location {
            crate::jit_dwarf::DwarfVariableLocation::Expr(expr) => {
                assert!(!expr.is_empty());
            }
            crate::jit_dwarf::DwarfVariableLocation::List(_) => {
                panic!("deserializer runtime-state vars should use inline exprloc")
            }
        }
    }
}

#[test]
fn cfg_value_dwarf_variables_cover_def_vregs() {
    let v0 = crate::ir::VReg::new(0);
    let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let inst_id_2 = crate::regalloc_engine::cfg_mir::InstId::new(1);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![crate::regalloc_engine::cfg_mir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst_id, inst_id_2],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
            dead: false,
        }],
        edges: Vec::new(),
        insts: vec![
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id_2,
                op: crate::linearize::LinearOp::Copy {
                    dst: crate::ir::VReg::new(1),
                    src: v0,
                },
                operands: vec![
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: crate::ir::VReg::new(1),
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                ],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
    };
    let root_scope = crate::ir::DebugScopeId::new(0);
    let block_scope = crate::ir::DebugScopeId::new(1);
    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    scopes.push(crate::ir::DebugScope {
        parent: Some(root_scope),
        kind: crate::ir::DebugScopeKind::ThetaBody,
    });
    let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
    let op_id_2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id_2);
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values: crate::ir::Arena::new(),
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_id), block_scope),
                ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                (
                    (
                        crate::ir::LambdaId::new(0),
                        crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                    ),
                    block_scope,
                ),
            ]),
            op_values: std::collections::HashMap::new(),
            vreg_scopes: vec![Some(block_scope), Some(root_scope)],
            vreg_values: vec![None, None],
        },
        hints: Default::default(),
        extra_excluded_regs: vec![],
    };
    #[cfg(target_arch = "aarch64")]
    let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
    #[cfg(target_arch = "aarch64")]
    let reg_2 = regalloc2::PReg::new(20, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg_2 = regalloc2::PReg::new(13, regalloc2::RegClass::Int);
    let alloc = crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: program.clone(),
        functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: crate::ir::LambdaId::new(0),
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::from([
                (op_id, vec![regalloc2::Allocation::reg(reg)]),
                (
                    op_id_2,
                    vec![
                        regalloc2::Allocation::reg(reg),
                        regalloc2::Allocation::reg(reg_2),
                    ],
                ),
            ]),
            op_operands: std::collections::HashMap::from([
                (
                    op_id,
                    vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (
                    op_id_2,
                    vec![
                        (v0, crate::regalloc_engine::cfg_mir::OperandKind::Use),
                        (
                            crate::ir::VReg::new(1),
                            crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        ),
                    ],
                ),
            ]),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        }],
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id,
                line: 1,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 4,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op_id_2,
                line: 2,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 4,
                    end_offset: 8,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                line: 3,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 8,
                    end_offset: 12,
                }],
            },
        ],
    };

    let vars = cfg_value_dwarf_variables(
        &program,
        &alloc,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
        true,
        false,
    );

    assert_eq!(vars.len(), 1);
    assert_eq!(vars[0].variable.name, "v0");
    assert_eq!(
        vars[0].lexical_ranges,
        vec![crate::jit_dwarf::JitDebugRange {
            low_pc: 0x1000,
            high_pc: 0x1008,
        }]
    );
    match &vars[0].variable.location {
        crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].start, 0x1004);
            assert_eq!(locations[0].end, 0x1008);
            let dwarf_reg = crate::jit_dwarf::dwarf_register_from_hw_encoding(
                jit_dwarf_target_arch(),
                reg.hw_enc() as u8,
            )
            .unwrap();
            assert_eq!(
                locations[0].expression,
                crate::jit_dwarf::expr_reg(dwarf_reg)
            );
        }
        crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
            panic!("cfg def vregs should use ranged locations")
        }
    }
}

#[test]
fn cfg_value_dwarf_variables_keep_edge_carried_defs_live() {
    let v0 = crate::ir::VReg::new(0);
    let v1 = crate::ir::VReg::new(1);
    let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let inst_id_2 = crate::regalloc_engine::cfg_mir::InstId::new(1);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let return_term_id = crate::regalloc_engine::cfg_mir::TermId::new(1);
    let entry_block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let exit_block_id = crate::regalloc_engine::cfg_mir::BlockId::new(1);
    let edge_id = crate::regalloc_engine::cfg_mir::EdgeId::new(0);
    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: entry_block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![
            crate::regalloc_engine::cfg_mir::Block {
                id: entry_block_id,
                params: Vec::new(),
                insts: vec![inst_id, inst_id_2],
                term: term_id,
                preds: Vec::new(),
                succs: vec![edge_id],
                dead: false,
            },
            crate::regalloc_engine::cfg_mir::Block {
                id: exit_block_id,
                params: vec![v0],
                insts: Vec::new(),
                term: return_term_id,
                preds: vec![edge_id],
                succs: Vec::new(),
                dead: false,
            },
        ],
        edges: vec![crate::regalloc_engine::cfg_mir::Edge {
            id: edge_id,
            from: entry_block_id,
            to: exit_block_id,
            args: vec![crate::regalloc_engine::cfg_mir::EdgeArg {
                target: v0,
                source: v0,
            }],
        }],
        insts: vec![
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id_2,
                op: crate::linearize::LinearOp::Const { dst: v1, value: 9 },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v1,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
        ],
        terms: vec![
            crate::regalloc_engine::cfg_mir::Terminator::Branch { edge: edge_id },
            crate::regalloc_engine::cfg_mir::Terminator::Return,
        ],
    };
    let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
    let op_id_2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id_2);
    let root_scope = crate::ir::DebugScopeId::new(0);
    let block_scope = crate::ir::DebugScopeId::new(1);
    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    scopes.push(crate::ir::DebugScope {
        parent: Some(root_scope),
        kind: crate::ir::DebugScopeKind::ThetaBody,
    });
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values: crate::ir::Arena::new(),
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_id), block_scope),
                ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                (
                    (
                        crate::ir::LambdaId::new(0),
                        crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                    ),
                    block_scope,
                ),
            ]),
            op_values: std::collections::HashMap::new(),
            vreg_scopes: vec![Some(block_scope), Some(root_scope)],
            vreg_values: vec![None, None],
        },
        hints: Default::default(),
        extra_excluded_regs: vec![],
    };
    let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
    #[cfg(target_arch = "aarch64")]
    let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
    #[cfg(target_arch = "aarch64")]
    let reg_2 = regalloc2::PReg::new(20, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg_2 = regalloc2::PReg::new(13, regalloc2::RegClass::Int);
    let alloc = crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: program.clone(),
        functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: crate::ir::LambdaId::new(0),
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::from([
                (op_id, vec![regalloc2::Allocation::reg(reg)]),
                (op_id_2, vec![regalloc2::Allocation::reg(reg_2)]),
            ]),
            op_operands: std::collections::HashMap::from([
                (
                    op_id,
                    vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (
                    op_id_2,
                    vec![(v1, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (term_op, Vec::new()),
            ]),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        }],
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id,
                line: 1,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 4,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op_id_2,
                line: 2,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 4,
                    end_offset: 8,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: term_op,
                line: 3,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 8,
                    end_offset: 12,
                }],
            },
        ],
    };

    let vars = cfg_value_dwarf_variables(
        &program,
        &alloc,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
        true,
        false,
    );

    assert_eq!(vars.len(), 1);
    assert_eq!(vars[0].variable.name, "v0");
    assert_eq!(
        vars[0].lexical_ranges,
        vec![crate::jit_dwarf::JitDebugRange {
            low_pc: 0x1000,
            high_pc: 0x100c,
        }]
    );
    match &vars[0].variable.location {
        crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].start, 0x1004);
            assert_eq!(locations[0].end, 0x100c);
            let dwarf_reg = crate::jit_dwarf::dwarf_register_from_hw_encoding(
                jit_dwarf_target_arch(),
                reg.hw_enc() as u8,
            )
            .unwrap();
            assert_eq!(
                locations[0].expression,
                crate::jit_dwarf::expr_reg(dwarf_reg)
            );
        }
        crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
            panic!("cfg edge-carried vregs should use ranged locations")
        }
    }
}

#[test]
fn cfg_mir_dwarf_variables_place_block_local_vregs_in_lexical_blocks() {
    let v0 = crate::ir::VReg::new(0);
    let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let inst_id_2 = crate::regalloc_engine::cfg_mir::InstId::new(1);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![crate::regalloc_engine::cfg_mir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst_id, inst_id_2],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
            dead: false,
        }],
        edges: Vec::new(),
        insts: vec![
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id_2,
                op: crate::linearize::LinearOp::Copy {
                    dst: crate::ir::VReg::new(1),
                    src: v0,
                },
                operands: vec![
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: crate::ir::VReg::new(1),
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                ],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
    };
    let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
    let op_id_2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id_2);
    let root_scope = crate::ir::DebugScopeId::new(0);
    let block_scope = crate::ir::DebugScopeId::new(1);
    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    scopes.push(crate::ir::DebugScope {
        parent: Some(root_scope),
        kind: crate::ir::DebugScopeKind::ThetaBody,
    });
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values: crate::ir::Arena::new(),
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_id), block_scope),
                ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                (
                    (
                        crate::ir::LambdaId::new(0),
                        crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                    ),
                    block_scope,
                ),
            ]),
            op_values: std::collections::HashMap::new(),
            vreg_scopes: vec![Some(block_scope), Some(root_scope)],
            vreg_values: vec![None, None],
        },
        hints: Default::default(),
        extra_excluded_regs: vec![],
    };
    #[cfg(target_arch = "aarch64")]
    let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
    #[cfg(target_arch = "aarch64")]
    let reg_2 = regalloc2::PReg::new(20, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg_2 = regalloc2::PReg::new(13, regalloc2::RegClass::Int);
    let alloc = crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: program.clone(),
        functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: crate::ir::LambdaId::new(0),
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::from([
                (op_id, vec![regalloc2::Allocation::reg(reg)]),
                (
                    op_id_2,
                    vec![
                        regalloc2::Allocation::reg(reg),
                        regalloc2::Allocation::reg(reg_2),
                    ],
                ),
            ]),
            op_operands: std::collections::HashMap::from([
                (
                    op_id,
                    vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (
                    op_id_2,
                    vec![
                        (v0, crate::regalloc_engine::cfg_mir::OperandKind::Use),
                        (
                            crate::ir::VReg::new(1),
                            crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        ),
                    ],
                ),
            ]),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        }],
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id,
                line: 1,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 4,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op_id_2,
                line: 2,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 4,
                    end_offset: 8,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                line: 3,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 8,
                    end_offset: 12,
                }],
            },
        ],
    };

    let subprogram = cfg_mir_dwarf_variables(
        None,
        &program,
        &alloc,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
        true,
    );

    assert!(
        !subprogram
            .variables
            .iter()
            .any(|variable| variable.name == "v0")
    );
    assert_eq!(subprogram.lexical_blocks.len(), 1);
    assert_eq!(subprogram.lexical_blocks[0].ranges.len(), 1);
    assert_eq!(subprogram.lexical_blocks[0].ranges[0].low_pc, 0x1000);
    assert_eq!(subprogram.lexical_blocks[0].ranges[0].high_pc, 0x100c);
    assert!(subprogram.lexical_blocks[0].variables.is_empty());
    assert_eq!(subprogram.lexical_blocks[0].lexical_blocks.len(), 1);
    assert_eq!(
        subprogram.lexical_blocks[0].lexical_blocks[0].ranges.len(),
        1
    );
    assert_eq!(
        subprogram.lexical_blocks[0].lexical_blocks[0].ranges[0].low_pc,
        0x1000
    );
    assert_eq!(
        subprogram.lexical_blocks[0].lexical_blocks[0].ranges[0].high_pc,
        0x1008
    );
    assert_eq!(
        subprogram.lexical_blocks[0].lexical_blocks[0]
            .variables
            .len(),
        1
    );
    assert_eq!(
        subprogram.lexical_blocks[0].lexical_blocks[0].variables[0].name,
        "v0"
    );
}

#[test]
fn cfg_semantic_field_dwarf_variables_follow_field_debug_values() {
    #[derive(Facet)]
    struct Bools {
        a: bool,
        b: bool,
    }

    let inst_a = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let inst_b = crate::regalloc_engine::cfg_mir::InstId::new(1);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let op_a = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_a);
    let op_b = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_b);
    let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
    let root_scope = crate::ir::DebugScopeId::new(0);

    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![crate::regalloc_engine::cfg_mir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst_a, inst_b],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
            dead: false,
        }],
        edges: Vec::new(),
        insts: vec![
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_a,
                op: crate::linearize::LinearOp::CallIntrinsic {
                    func: crate::ir::IntrinsicFn(
                        crate::json_intrinsics::kajit_json_read_bool as *const () as usize,
                    ),
                    args: Vec::new(),
                    dst: None,
                    field_offset: 0,
                },
                operands: Vec::new(),
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_b,
                op: crate::linearize::LinearOp::CallIntrinsic {
                    func: crate::ir::IntrinsicFn(
                        crate::json_intrinsics::kajit_json_read_bool as *const () as usize,
                    ),
                    args: Vec::new(),
                    dst: None,
                    field_offset: 1,
                },
                operands: Vec::new(),
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
    };

    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    let mut values = crate::ir::Arena::new();
    let debug_a = values.push(crate::ir::DebugValue {
        name: "a".to_string(),
        kind: crate::ir::DebugValueKind::Field { offset: 0 },
    });
    let debug_b = values.push(crate::ir::DebugValue {
        name: "b".to_string(),
        kind: crate::ir::DebugValueKind::Field { offset: 1 },
    });
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 0,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values,
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::new(),
            op_values: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_a), debug_a),
                ((crate::ir::LambdaId::new(0), op_b), debug_b),
            ]),
            vreg_scopes: Vec::new(),
            vreg_values: Vec::new(),
        },
        hints: Default::default(),
        extra_excluded_regs: vec![],
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op_a,
                line: 10,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 8,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op_b,
                line: 20,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 16,
                    end_offset: 24,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: term_op,
                line: 30,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 32,
                    end_offset: 40,
                }],
            },
        ],
    };

    let vars = cfg_semantic_field_dwarf_variables(
        <Bools as Facet>::SHAPE,
        &program,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
    );

    assert_eq!(vars.len(), 2);
    assert_eq!(vars[0].scope, Some(root_scope));
    assert_eq!(vars[0].variable.name, "a");
    assert_eq!(
        vars[0].lexical_ranges,
        vec![crate::jit_dwarf::JitDebugRange {
            low_pc: 0x1000,
            high_pc: 0x1028,
        }]
    );
    assert_eq!(vars[1].scope, Some(root_scope));
    assert_eq!(vars[1].variable.name, "b");
    assert_eq!(
        vars[1].lexical_ranges,
        vec![crate::jit_dwarf::JitDebugRange {
            low_pc: 0x1010,
            high_pc: 0x1028,
        }]
    );

    let expected_expr_a = dwarf_expr_for_out_field(jit_dwarf_target_arch(), 0, 1);
    let expected_expr_b = dwarf_expr_for_out_field(jit_dwarf_target_arch(), 1, 1);
    match &vars[0].variable.location {
        crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].start, 0x1008);
            assert_eq!(locations[0].end, 0x1028);
            assert_eq!(locations[0].expression, expected_expr_a);
        }
        crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
            panic!("semantic field vars should use ranged locations")
        }
    }
    match &vars[1].variable.location {
        crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].start, 0x1018);
            assert_eq!(locations[0].end, 0x1028);
            assert_eq!(locations[0].expression, expected_expr_b);
        }
        crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
            panic!("semantic field vars should use ranged locations")
        }
    }
}

#[test]
fn cfg_value_dwarf_variables_can_hide_semantic_owned_vregs() {
    let v0 = crate::ir::VReg::new(0);
    let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
    let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
    let root_scope = crate::ir::DebugScopeId::new(0);
    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![crate::regalloc_engine::cfg_mir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst_id],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
            dead: false,
        }],
        edges: Vec::new(),
        insts: vec![crate::regalloc_engine::cfg_mir::Inst {
            id: inst_id,
            op: crate::linearize::LinearOp::Const { dst: v0, value: 1 },
            operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                vreg: v0,
                kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                fixed: None,
            }],
            clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
        }],
        terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
    };
    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    let mut values = crate::ir::Arena::new();
    let debug_a = values.push(crate::ir::DebugValue {
        name: "a".to_string(),
        kind: crate::ir::DebugValueKind::Field { offset: 0 },
    });
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 1,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values,
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_id), root_scope),
                ((crate::ir::LambdaId::new(0), term_op), root_scope),
            ]),
            op_values: std::collections::HashMap::new(),
            vreg_scopes: vec![Some(root_scope)],
            vreg_values: vec![Some(debug_a)],
        },
        hints: Default::default(),
        extra_excluded_regs: vec![],
    };
    #[cfg(target_arch = "aarch64")]
    let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
    let alloc = crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: program.clone(),
        functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: crate::ir::LambdaId::new(0),
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::from([(
                op_id,
                vec![regalloc2::Allocation::reg(reg)],
            )]),
            op_operands: std::collections::HashMap::from([
                (
                    op_id,
                    vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (term_op, Vec::new()),
            ]),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        }],
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id,
                line: 1,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 4,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: term_op,
                line: 2,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 4,
                    end_offset: 8,
                }],
            },
        ],
    };

    let vars = cfg_value_dwarf_variables(
        &program,
        &alloc,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
        true,
        true,
    );

    assert!(vars.is_empty(), "semantic-owned vregs should be hidden");
}

#[test]
fn cfg_semantic_named_dwarf_variables_merge_shared_vregs() {
    let v0 = crate::ir::VReg::new(0);
    let v1 = crate::ir::VReg::new(1);
    let inst0 = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let inst1 = crate::regalloc_engine::cfg_mir::InstId::new(1);
    let inst2 = crate::regalloc_engine::cfg_mir::InstId::new(2);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let op0 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst0);
    let op1 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst1);
    let op2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst2);
    let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
    let root_scope = crate::ir::DebugScopeId::new(0);
    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![crate::regalloc_engine::cfg_mir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst0, inst1, inst2],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
            dead: false,
        }],
        edges: Vec::new(),
        insts: vec![
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst0,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 1 },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst1,
                op: crate::linearize::LinearOp::Copy { dst: v1, src: v0 },
                operands: vec![
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v1,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                ],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst2,
                op: crate::linearize::LinearOp::WriteToField {
                    src: v1,
                    offset: 0,
                    width: crate::ir::Width::W1,
                },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v1,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
    };
    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    let mut values = crate::ir::Arena::new();
    let debug_flag = values.push(crate::ir::DebugValue {
        name: "flag".to_string(),
        kind: crate::ir::DebugValueKind::Named,
    });
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values,
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op0), root_scope),
                ((crate::ir::LambdaId::new(0), op1), root_scope),
                ((crate::ir::LambdaId::new(0), op2), root_scope),
                ((crate::ir::LambdaId::new(0), term_op), root_scope),
            ]),
            op_values: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op0), debug_flag),
                ((crate::ir::LambdaId::new(0), op1), debug_flag),
            ]),
            vreg_scopes: vec![Some(root_scope), Some(root_scope)],
            vreg_values: vec![Some(debug_flag), Some(debug_flag)],
        },
        hints: Default::default(),
        extra_excluded_regs: vec![],
    };
    #[cfg(target_arch = "aarch64")]
    let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
    let alloc = crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: program.clone(),
        functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: crate::ir::LambdaId::new(0),
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::from([
                (op0, vec![regalloc2::Allocation::reg(reg)]),
                (
                    op1,
                    vec![
                        regalloc2::Allocation::reg(reg),
                        regalloc2::Allocation::reg(reg),
                    ],
                ),
                (op2, vec![regalloc2::Allocation::reg(reg)]),
            ]),
            op_operands: std::collections::HashMap::from([
                (
                    op0,
                    vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (
                    op1,
                    vec![
                        (v0, crate::regalloc_engine::cfg_mir::OperandKind::Use),
                        (v1, crate::regalloc_engine::cfg_mir::OperandKind::Def),
                    ],
                ),
                (
                    op2,
                    vec![(v1, crate::regalloc_engine::cfg_mir::OperandKind::Use)],
                ),
                (term_op, Vec::new()),
            ]),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        }],
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op0,
                line: 1,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 4,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op1,
                line: 2,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 4,
                    end_offset: 8,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op2,
                line: 3,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 8,
                    end_offset: 12,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: term_op,
                line: 4,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 12,
                    end_offset: 16,
                }],
            },
        ],
    };

    let vars = cfg_semantic_named_dwarf_variables(
        &program,
        &alloc,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
        true,
    );

    assert_eq!(vars.len(), 1);
    assert_eq!(vars[0].scope, Some(root_scope));
    assert_eq!(vars[0].variable.name, "flag");
    assert_eq!(
        vars[0].lexical_ranges,
        vec![crate::jit_dwarf::JitDebugRange {
            low_pc: 0x1000,
            high_pc: 0x100c,
        }]
    );
    match &vars[0].variable.location {
        crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].start, 0x1004);
            assert_eq!(locations[0].end, 0x100c);
        }
        crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
            panic!("semantic named vars should use ranged locations")
        }
    }
}

#[test]
fn postcard_option_scalar_matches_differential_harness() {
    let module = build_postcard_decoder_hir(<MaybeCount>::SHAPE);
    let mut func = build_structural_hir_ir(<MaybeCount>::SHAPE, &module);
    run_default_passes_from_env(&mut func);
    let linear = crate::linearize::linearize(&mut func);
    let output_size = std::mem::size_of::<MaybeCount>();
    let report = crate::differential_check_linear_ir_vs_jit_with_output_size(
        &linear,
        &[1, 42], // Some(42) in postcard
        output_size,
    )
    .expect("differential harness should execute option decoder");
    assert!(
        report.is_match(),
        "interpreter vs JIT mismatch for option Some(42): {:?}",
        report.mismatch
    );
}
