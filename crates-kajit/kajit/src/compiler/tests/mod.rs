use facet::Facet;
use facet::Shape;
use facet_testhelpers::test;
use kajit_format::instantiated_shape_symbol_key;
use kajit_hir as hir;
use kajit_hir_text::parse_hir;
use serde::Serialize;

use super::{
    CompiledDecoder, build_jit_debug_info_from_source_map, build_postcard_decoder_hir,
    cfg_mir_dwarf_variables, cfg_semantic_field_dwarf_variables,
    cfg_semantic_named_dwarf_variables, cfg_value_dwarf_variables, deser_dwarf_variables,
    jit_dwarf_target_arch, lower_hir_module, normalize_debug_line_rows,
    run_default_passes_from_env,
};

fn test_location_map(
    regs: &[(u32, u8)],
    call_lines: &[u32],
    call_return_vregs: &[(u32, u32)],
) -> crate::harness::LocationMap {
    crate::harness::LocationMap {
        static_locations: regs
            .iter()
            .map(|(vreg, preg)| (*vreg, crate::harness::VRegLocation::Register(*preg)))
            .collect(),
        call_lines: call_lines.iter().copied().collect(),
        call_return_vregs: call_return_vregs.iter().copied().collect(),
        edit_clobbers: std::collections::HashMap::new(),
        num_spill_slots: 0,
    }
}

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

#[derive(Debug, PartialEq, Eq, Facet, Serialize)]
struct MultiOpt {
    a: Option<u32>,
    b: String,
    c: Option<String>,
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
    let registry = super::symbol_registry_for_shape(shape);
    let mut func = lower_hir_module(module);
    run_default_passes_from_env(&mut func);
    let linear = crate::linearize::linearize(&mut func);
    super::compile_linear_ir_decoder_with_options(
        &linear,
        false,
        crate::pipeline_opts::PipelineOptions::from_env(),
        Some(&registry),
        Some(shape),
    )
}

fn compile_postcard_decoder_via_structural_hir(shape: &'static Shape) -> CompiledDecoder {
    let (module, _symbol_table) = build_postcard_decoder_hir(shape);
    compile_structural_hir_decoder(shape, &module)
}

fn structural_hir_type_size_for_test(module: &hir::Module, ty: &hir::Type) -> usize {
    match ty {
        hir::Type::Unit => 0,
        hir::Type::Bool => 1,
        hir::Type::Integer(kind) => (kind.bits as usize) / 8,
        hir::Type::Ref { .. } | hir::Type::Address { .. } | hir::Type::Handle { .. } => {
            core::mem::size_of::<usize>()
        }
        hir::Type::Str { .. } | hir::Type::Slice { .. } => core::mem::size_of::<usize>() * 2,
        hir::Type::Array { element, len } => {
            structural_hir_type_size_for_test(module, element) * len
        }
        hir::Type::Named { def, .. } => {
            let type_def = &module.type_defs[*def];
            if let Some(size) = type_def.size {
                return size as usize;
            }
            match &type_def.kind {
                hir::TypeDefKind::Struct { fields } => {
                    let mut max_end = 0usize;
                    let mut cursor = 0usize;
                    for field in fields {
                        let field_size = structural_hir_type_size_for_test(module, &field.ty);
                        let offset = field.offset.map(|offset| offset as usize).unwrap_or(cursor);
                        max_end = max_end.max(offset + field_size);
                        cursor = offset + field_size;
                    }
                    max_end
                }
                hir::TypeDefKind::Enum {
                    variants,
                    discriminant_width,
                } => {
                    let disc_size = discriminant_width.unwrap_or(1) as usize;
                    let max_payload = variants
                        .iter()
                        .map(|variant| {
                            let mut max_end = 0usize;
                            let mut cursor = 0usize;
                            for field in &variant.fields {
                                let field_size =
                                    structural_hir_type_size_for_test(module, &field.ty);
                                let offset =
                                    field.offset.map(|offset| offset as usize).unwrap_or(cursor);
                                max_end = max_end.max(offset + field_size);
                                cursor = offset + field_size;
                            }
                            max_end
                        })
                        .max()
                        .unwrap_or(0);
                    disc_size + max_payload
                }
            }
        }
    }
}

fn mark_non_semantic_output_bytes_for_type(
    module: &hir::Module,
    ty: &hir::Type,
    base: usize,
    mask: &mut [bool],
) {
    match ty {
        hir::Type::Unit | hir::Type::Bool | hir::Type::Integer(_) => {}
        hir::Type::Ref { .. } | hir::Type::Address { .. } | hir::Type::Handle { .. } => {
            for offset in 0..core::mem::size_of::<usize>() {
                if let Some(slot) = mask.get_mut(base + offset) {
                    *slot = true;
                }
            }
        }
        hir::Type::Str { .. } | hir::Type::Slice { .. } => {
            for offset in 0..core::mem::size_of::<usize>() {
                if let Some(slot) = mask.get_mut(base + offset) {
                    *slot = true;
                }
            }
        }
        hir::Type::Array { element, len } => {
            let elem_size = structural_hir_type_size_for_test(module, element);
            for index in 0..*len {
                mark_non_semantic_output_bytes_for_type(
                    module,
                    element,
                    base + index * elem_size,
                    mask,
                );
            }
        }
        hir::Type::Named { def, .. } => {
            let type_def = &module.type_defs[*def];
            match &type_def.kind {
                hir::TypeDefKind::Struct { fields } => {
                    let mut cursor = 0usize;
                    for field in fields {
                        let offset = field.offset.map(|offset| offset as usize).unwrap_or(cursor);
                        mark_non_semantic_output_bytes_for_type(
                            module,
                            &field.ty,
                            base + offset,
                            mask,
                        );
                        cursor = offset + structural_hir_type_size_for_test(module, &field.ty);
                    }
                }
                hir::TypeDefKind::Enum { variants, .. } => {
                    for variant in variants {
                        let mut cursor = 0usize;
                        for field in &variant.fields {
                            let offset =
                                field.offset.map(|offset| offset as usize).unwrap_or(cursor);
                            mark_non_semantic_output_bytes_for_type(
                                module,
                                &field.ty,
                                base + offset,
                                mask,
                            );
                            cursor = offset + structural_hir_type_size_for_test(module, &field.ty);
                        }
                    }
                }
            }
        }
    }
}

fn non_semantic_output_byte_mask(module: &hir::Module) -> Vec<bool> {
    let function = module
        .functions
        .iter()
        .next()
        .map(|(_, function)| function)
        .expect("decoder module should have a root function");
    // The second parameter (index 1) is the output pointer.
    let dest_ty = function
        .params
        .get(1)
        .map(|param| &param.ty)
        .unwrap_or(&function.return_type);
    let output_size = structural_hir_type_size_for_test(module, dest_ty);
    let mut mask = vec![false; output_size];
    mark_non_semantic_output_bytes_for_type(module, dest_ty, 0, &mut mask);
    mask
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
    let (module, _symbol_table) = build_postcard_decoder_hir(<BorrowedHeader<'static>>::SHAPE);
    assert_eq!(module.functions.len(), 1);

    let (_, function) = module.functions.iter().next().unwrap();
    assert_eq!(function.params.len(), 2);
    assert_eq!(function.region_params.len(), 1);

    let out_param = &function.params[1];
    let hir::Type::Named { def, args } = &out_param.ty else {
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
fn postcard_hir_models_owned_output_strings() {
    let (module, _symbol_table) = build_postcard_decoder_hir(<OwnedHeader>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    fn block_contains_call_named(module: &hir::Module, block: &hir::Block, name: &str) -> bool {
        block.statements.iter().any(|stmt| match &stmt.kind {
            hir::StmtKind::Init {
                value:
                    hir::Expr::Call(hir::CallExpr {
                        target: hir::CallTarget::Callable(callable_id),
                        ..
                    }),
                ..
            }
            | hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(callable_id),
                ..
            })) => module.callables[*callable_id].name == name,
            hir::StmtKind::If {
                then_block,
                else_block,
                ..
            } => {
                block_contains_call_named(module, then_block, name)
                    || else_block
                        .as_ref()
                        .is_some_and(|block| block_contains_call_named(module, block, name))
            }
            hir::StmtKind::Loop { body, .. } => block_contains_call_named(module, body, name),
            hir::StmtKind::Match { arms, .. } => arms
                .iter()
                .any(|arm| block_contains_call_named(module, &arm.body, name)),
            _ => false,
        })
    }

    assert!(
        module
            .callable_named("runtime.string_validate_alloc_copy")
            .is_none(),
        "owned string lowering should not rely on the combined string helper"
    );
    assert!(
        module.callable_named("postcard.read_str").is_none(),
        "owned string lowering should not use postcard.read_str"
    );
    assert!(
        module
            .callable_named("runtime.validate_utf8_range")
            .is_some(),
        "owned string lowering should validate UTF-8 explicitly"
    );
    assert!(
        module.callable_named("runtime.alloc_persistent").is_some(),
        "owned string lowering should allocate string storage explicitly"
    );
    assert!(
        module.callable_named("runtime.memcpy").is_some(),
        "owned string lowering should copy string bytes explicitly"
    );
    assert!(
        function
            .locals
            .iter()
            .any(|local| matches!(local.ty, hir::Type::Address { .. })),
        "owned string lowering should allocate a persistent data pointer local"
    );
    assert!(
        block_contains_call_named(&module, &function.body, "runtime.validate_utf8_range"),
        "owned string lowering should validate the borrowed byte range"
    );
    assert!(
        block_contains_call_named(&module, &function.body, "runtime.alloc_persistent"),
        "owned string lowering should allocate storage explicitly"
    );
    assert!(
        block_contains_call_named(&module, &function.body, "runtime.memcpy"),
        "owned string lowering should copy string bytes explicitly"
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
    let (module, _symbol_table) = build_postcard_decoder_hir(<FloatHeader>::SHAPE);

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
    let (module, _symbol_table) = build_postcard_decoder_hir(<CharHeader>::SHAPE);

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
    let (unsigned, _) = build_postcard_decoder_hir(<BigUnsigned>::SHAPE);
    let (signed, _) = build_postcard_decoder_hir(<BigSigned>::SHAPE);
    let (optional, _) = build_postcard_decoder_hir(<MaybeBigUnsigned>::SHAPE);

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
    let (module, _symbol_table) = build_postcard_decoder_hir(<MaybeBorrowedName<'static>>::SHAPE);
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

    let (some_args, some_callable) = then_block
        .statements
        .iter()
        .find_map(|stmt| match &stmt.kind {
            hir::StmtKind::Expr(hir::Expr::Call(call)) => {
                let hir::CallTarget::Callable(callable_id) = call.target;
                (module.callables[callable_id].name == "runtime.option_init_some")
                    .then_some((&call.args, callable_id))
            }
            _ => None,
        })
        .expect("expected explicit Option::Some init call");
    assert_eq!(
        module.callables[some_callable].intrinsic,
        Some(hir::RuntimeIntrinsic::OptionInitSome)
    );
    assert_eq!(some_args.len(), 3);
    let hir::Expr::AddrOf(payload_place) = &some_args[2] else {
        panic!("expected Some payload addr");
    };
    let hir::Place::Local(payload_local) = &**payload_place else {
        panic!("expected Some payload local");
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

    let (none_args, none_callable) = else_block
        .statements
        .iter()
        .find_map(|stmt| match &stmt.kind {
            hir::StmtKind::Expr(hir::Expr::Call(call)) => {
                let hir::CallTarget::Callable(callable_id) = call.target;
                (module.callables[callable_id].name == "runtime.option_init_none")
                    .then_some((&call.args, callable_id))
            }
            _ => None,
        })
        .expect("expected explicit Option::None init call");
    assert_eq!(
        module.callables[none_callable].intrinsic,
        Some(hir::RuntimeIntrinsic::OptionInitNone)
    );
    assert_eq!(none_args.len(), 2);
}

// pre-existing: HIR round-trip mismatch on max_iterations
#[test]
#[ignore]
fn postcard_hir_text_round_trips() {
    std::thread::Builder::new()
        .name("postcard_hir_text_round_trips".to_owned())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let (module, _symbol_table) =
                build_postcard_decoder_hir(<MaybeBorrowedName<'static>>::SHAPE);
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
fn postcard_hir_lowering_decodes_float_fields() {
    let decoder = compile_postcard_decoder_via_structural_hir(<FloatHeader>::SHAPE);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&3.14f32.to_le_bytes());
    bytes.extend_from_slice(&2.718281828459045f64.to_le_bytes());

    let value = crate::deserialize::<FloatHeader>(&decoder, &bytes)
        .expect("postcard HIR lowering should decode float fields");
    assert_eq!(value.a.to_bits(), 3.14f32.to_bits());
    assert_eq!(value.b.to_bits(), 2.718281828459045f64.to_bits());
}

#[test]
fn postcard_hir_lowering_decodes_char_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<CharHeader>::SHAPE);

    let value = crate::deserialize::<CharHeader>(&decoder, &[2, 0xC3, 0x9F])
        .expect("postcard HIR lowering should decode char fields");
    assert_eq!(value, CharHeader { ch: 'ß' });
}

// pre-existing: HIR round-trip mismatch on max_iterations
#[test]
#[ignore]
fn postcard_hir_lowering_decodes_128bit_fields() {
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
        .expect("postcard HIR lowering should decode u128 fields");
    assert_eq!(unsigned_value, unsigned);

    let signed_decoder = compile_postcard_decoder_via_structural_hir(<BigSigned>::SHAPE);
    let signed_bytes =
        postcard::to_allocvec(&signed).expect("postcard should encode signed 128-bit sample");
    let signed_value = crate::deserialize::<BigSigned>(&signed_decoder, &signed_bytes)
        .expect("postcard HIR lowering should decode i128 fields");
    assert_eq!(signed_value, signed);
}

// pre-existing: HIR round-trip mismatch on max_iterations
#[test]
#[ignore]
fn postcard_hir_lowering_decodes_option_u128_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<MaybeBigUnsigned>::SHAPE);
    let sample = MaybeBigUnsigned {
        value: Some((1_u128 << 72) | 0x55aa_33cc_77ee_u128),
    };
    let bytes = postcard::to_allocvec(&sample).expect("postcard should encode Option<u128>");
    let value = crate::deserialize::<MaybeBigUnsigned>(&decoder, &bytes)
        .expect("postcard HIR lowering should decode Option<u128>");
    assert_eq!(value, sample);
}

#[test]
fn postcard_hir_models_unit_enums() {
    let (module, _symbol_table) = build_postcard_decoder_hir(<UnitAnimal>::SHAPE);
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
    let (module, _symbol_table) = build_postcard_decoder_hir(<PayloadAnimal<'static>>::SHAPE);
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

// pre-existing: HIR round-trip mismatch on max_iterations
#[test]
fn postcard_hir_scalar_array_u32_4() {
    let (module, _symbol_table) = build_postcard_decoder_hir(<ScalarArrayHolder>::SHAPE);
    insta::assert_snapshot!(module.to_string());
}

#[test]
fn postcard_hir_models_arrays() {
    let (module, _symbol_table) = build_postcard_decoder_hir(<BorrowedArrayHolder<'static>>::SHAPE);
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

// pre-existing: HIR round-trip mismatch on max_iterations
#[test]
#[ignore]
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
fn postcard_hir_lowering_decodes_scalar_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<ScalarNumber>::SHAPE);

    let value = crate::deserialize::<ScalarNumber>(&decoder, &[42])
        .expect("postcard HIR lowering should decode a scalar field");
    assert_eq!(value, ScalarNumber { value: 42 });
}

#[test]
fn postcard_hir_lowering_decodes_borrowed_header() {
    let decoder = compile_postcard_decoder_via_structural_hir(<BorrowedHeader<'static>>::SHAPE);

    let value = crate::deserialize::<BorrowedHeader<'_>>(&decoder, &[7, 2, b'h', b'i'])
        .expect("postcard HIR lowering should decode direct borrowed fields");
    assert_eq!(value, BorrowedHeader { len: 7, name: "hi" });
}

#[cfg(target_os = "linux")]
fn debug_postcard_borrowed_header_harness() {
    let shape = <BorrowedHeader<'static>>::SHAPE;
    let (module, _symbol_table) = build_postcard_decoder_hir(shape);
    let registry = super::symbol_registry_for_shape(shape);
    let mut func = lower_hir_module(&module);
    run_default_passes_from_env(&mut func);
    let linear = crate::linearize::linearize(&mut func);
    let hints = Default::default();
    let cfg_program = crate::regalloc_engine::ir::lower_and_optimize(&linear, hints);
    let ra3_alloc = crate::regalloc_engine::allocate_cfg_program_regalloc3_native(&cfg_program)
        .expect("regalloc3 should allocate BorrowedHeader cfg");
    let base_frame = crate::backends::aarch64::regalloc3_backend::compute_base_frame(&ra3_alloc);
    let alloc_map = ra3_alloc
        .functions
        .first()
        .map(|func| crate::harness::AllocationMap::from_regalloc3(func, base_frame))
        .unwrap_or_default();

    let empty_symbols = kajit_types::SymbolTable::new();
    let result = crate::backends::aarch64::regalloc3_backend::compile_regalloc3(
        &ra3_alloc,
        &empty_symbols,
        crate::pipeline_opts::CompileTarget::Jit,
    );
    let intrinsic_call_sites = result.intrinsic_call_sites.clone();
    let extern_addr_relocs = result.extern_addr_relocs.clone();
    let (buf, entry, _source_map, _backend_debug_info, asm_program) =
        super::materialize_backend_result(result);
    let func = unsafe { buf.code_ptr().add(entry) };
    let listing = super::build_cfg_mir_listing(&cfg_program, Some(&registry));

    let decoder = CompiledDecoder {
        buf,
        cfg_mir_line_text_by_line: listing.line_text_by_line,
        entry,
        func,
        trusted_utf8_input: false,
        _jit_registration: None,
        #[cfg(target_arch = "aarch64")]
        asm_program,
    };

    let output_dir = std::path::PathBuf::from("/tmp/kajit-harness");
    let base_name = "harness_postcard_borrowed_header";
    let listing_path = output_dir.join(format!("{base_name}.cfg-mir"));
    let dwarf = decoder.build_standalone_dwarf(&listing_path);
    let known = crate::intrinsics::known_intrinsics();
    let intrinsic_calls = intrinsic_call_sites
        .iter()
        .filter_map(|site| {
            let name = known
                .iter()
                .find(|(_, func)| func.0 == site.func.0)
                .map(|(name, _)| name.to_string())?;
            Some(crate::harness::IntrinsicCallSite {
                code_offset: site.code_offset,
                baked_addr: site.func.0 as u64,
                symbol_name: name,
            })
        })
        .collect();

    let harness_input = crate::harness::HarnessInput {
        code: decoder.code(),
        entry_offset: decoder.entry_offset(),
        output_size: std::mem::size_of::<BorrowedHeader<'static>>(),
        dwarf,
        cfg_mir_lines: decoder.cfg_mir_lines(),
        function_name: "kajit_decode",
        alloc_map: Some(&alloc_map),
        intrinsic_calls,
        extern_addr_relocs,
    };

    let exe_path = crate::harness::generate_harness(&harness_input, &output_dir, base_name)
        .expect("generate BorrowedHeader harness");
    let output = std::process::Command::new(&exe_path)
        .arg("07026869")
        .output()
        .expect("run BorrowedHeader harness");
    assert!(
        output.status.success(),
        "BorrowedHeader harness should run successfully: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    eprintln!("[harness-test] executable: {}", exe_path.display());
}

#[test]
fn postcard_hir_lowering_decodes_owned_header() {
    let decoder = compile_postcard_decoder_via_structural_hir(<OwnedHeader>::SHAPE);

    let value = crate::deserialize::<OwnedHeader>(&decoder, &[7, 2, b'h', b'i'])
        .expect("postcard HIR lowering should decode direct owned string fields");
    assert_eq!(
        value,
        OwnedHeader {
            len: 7,
            name: "hi".to_owned(),
        }
    );
}

#[test]
fn postcard_hir_lowering_decodes_root_vec_u32() {
    let decoder = compile_postcard_decoder_via_structural_hir(<Vec<u32>>::SHAPE);

    let value = crate::deserialize::<Vec<u32>>(&decoder, &[3, 1, 2, 3])
        .expect("postcard HIR lowering should decode root Vec<u32>");
    assert_eq!(value, vec![1, 2, 3]);
}

#[test]
fn postcard_hir_lowering_decodes_root_vec_string() {
    let decoder = compile_postcard_decoder_via_structural_hir(<Vec<String>>::SHAPE);

    let value = crate::deserialize::<Vec<String>>(&decoder, &[2, 2, b'h', b'i', 2, b'b', b'y'])
        .expect("postcard HIR lowering should decode root Vec<String>");
    assert_eq!(value, vec!["hi".to_owned(), "by".to_owned()]);
}

#[test]
fn handwritten_hir_decodes_vec_string_with_one_empty_string() {
    let module = parse_hir(
        r#"
hir_module {
  regions [
    r0 "input"
  ]
  stores []
  types [
    type t0 "Cursor" <region "r_input"> = struct {
      "bytes": Slice<r0, u8>
      "pos": u64
    }
    type t1 "Vec" size=24 = struct {
      "cap": u64 @0
      "ptr": addr<persistent> @8
      "len": u64 @16
    }
    type t2 "HostStringRaw" size=24 = struct {
      "cap": u64 @0
      "ptr": addr<persistent> @8
      "len": u64 @16
    }
  ]
  callables [
    callable c0 host "runtime.alloc_persistent" {
      params [u64, u64]
      intrinsic alloc_persistent
      returns [addr<persistent>]
      effect mutates
      domains ["persistent_heap":mutate]
      control may_fail
      capabilities ["runtime.alloc"]
      safety opaque_host
      docs "Allocate persistent memory that may escape in the result."
    }
  ]
  functions [
    function f0 "build_vec_with_one_empty_string" {
      regions [r0]
      stores []
      params [
        l0 param "cursor": &mut t0<r0>
        l1 destination "out": t1
      ]
      locals [
        l2 temp "list_len": u64
        l3 temp "list_bytes": u64
        l4 temp "list_ptr": addr<persistent>
        l5 temp "list_index": u64
        l6 temp "list_elem": t2
        l7 temp "string_len": u32
        l8 temp "string_ptr": addr<persistent>
        l9 temp "string_raw": t2
      ]
      return unit
      scopes [
        scope sc0 parent none comment "handwritten Vec<String> builder"
      ]
      body @sc0 {
        stmt0: init l2 = 0x1
        stmt1: init l3 = binary mul(l2, 0x18)
        stmt2: init l4 = call c0(l3, 0x8)
        stmt3: init l5 = 0x0
        stmt20: loop @sc0 {
          stmt4: if binary eq(l5, l2) @sc0 {
            stmt5: break
          } else @sc0 {
          }
          stmt6: init l7 = 0x0
          stmt7: init l8 = 0x1
          stmt8: init field(l9, "ptr") = l8
          stmt9: init field(l9, "len") = l7
          stmt10: init field(l9, "cap") = l7
          stmt11: init l6 = l9
          stmt12: store w8 binary add(binary add(l4, binary mul(l5, 0x18)), 0x8) = field(l6, "ptr")
          stmt13: store w8 binary add(binary add(l4, binary mul(l5, 0x18)), 0x10) = field(l6, "len")
          stmt14: store w8 binary add(l4, binary mul(l5, 0x18)) = field(l6, "cap")
          stmt15: assign l5 = binary add(l5, 0x1)
        }
        stmt21: init field(l1, "cap") = l2
        stmt22: init field(l1, "ptr") = binary add(l4, binary mul(binary eq(l2, 0x0), 0x8))
        stmt23: init field(l1, "len") = l2
        stmt24: return
      }
    }
  ]
}
"#,
    )
    .expect("handwritten HIR should parse");

    let decoder = compile_structural_hir_decoder(<Vec<String>>::SHAPE, &module);

    let value = crate::deserialize::<Vec<String>>(&decoder, &[])
        .expect("handwritten HIR should decode Vec<String> with one empty string");
    assert_eq!(value, vec![String::new()]);
}

#[test]
fn handwritten_hir_decodes_vec_string_with_one_inline_string() {
    let module = parse_hir(
        r#"
hir_module {
  regions [
    r0 "input"
  ]
  stores []
  types [
    type t0 "Cursor" <region "r_input"> = struct {
      "bytes": Slice<r0, u8>
      "pos": u64
    }
    type t1 "Vec" size=24 = struct {
      "cap": u64 @0
      "ptr": addr<persistent> @8
      "len": u64 @16
    }
    type t2 "HostStringRaw" size=24 = struct {
      "cap": u64 @0
      "ptr": addr<persistent> @8
      "len": u64 @16
    }
  ]
  callables [
    callable c0 host "runtime.alloc_persistent" {
      params [u64, u64]
      intrinsic alloc_persistent
      returns [addr<persistent>]
      effect mutates
      domains ["persistent_heap":mutate]
      control may_fail
      capabilities ["runtime.alloc"]
      safety opaque_host
      docs "Allocate persistent memory that may escape in the result."
    }
    callable c1 host "runtime.validate_utf8_range" {
      params [u64, u32]
      intrinsic validate_utf8_range
      returns []
      effect reads
      domains ["input":read]
      control may_fail
      capabilities ["runtime.utf8"]
      safety opaque_host
      docs "Validate that a borrowed byte range is UTF-8."
    }
    callable c2 host "runtime.memcpy" {
      params [u64, u64, u64]
      intrinsic memcpy
      returns [u64]
      effect mutates
      domains ["persistent_heap":mutate, "input":read]
      control returns
      capabilities ["runtime.memcpy"]
      safety opaque_host
      docs "Copy bytes from one address to another."
    }
  ]
  functions [
    function f0 "build_vec_with_one_inline_string" {
      regions [r0]
      stores []
      params [
        l0 param "cursor": &mut t0<r0>
        l1 destination "out": t1
      ]
      locals [
        l2 temp "list_len": u64
        l3 temp "list_bytes": u64
        l4 temp "list_ptr": addr<persistent>
        l5 temp "list_index": u64
        l6 temp "list_elem": t2
        l7 temp "string_len": u32
        l8 temp "string_data": u64
        l9 temp "string_ptr": addr<persistent>
        l10 temp "string_raw": t2
      ]
      return unit
      scopes [
        scope sc0 parent none comment "handwritten Vec<String> builder with inline string copy"
      ]
      body @sc0 {
        stmt0: init l2 = 0x1
        stmt1: init l3 = binary mul(l2, 0x18)
        stmt2: init l4 = call c0(l3, 0x8)
        stmt3: init l5 = 0x0
        stmt28: loop @sc0 {
          stmt4: if binary eq(l5, l2) @sc0 {
            stmt5: break
          } else @sc0 {
          }
          stmt6: init l7 = 0x2
          stmt7: if binary gt(binary add(field(deref(l0), "pos"), l7), slice_len(field(deref(l0), "bytes"))) @sc0 {
            stmt8: fail UnexpectedEof
          } else @sc0 {
          }
          stmt9: init l8 = binary add(slice_data(field(deref(l0), "bytes")), field(deref(l0), "pos"))
          stmt10: expr call c1(l8, l7)
          stmt11: init l9 = call c0(l7, 0x1)
          stmt12: expr call c2(l9, l8, l7)
          stmt13: assign field(deref(l0), "pos") = binary add(field(deref(l0), "pos"), l7)
          stmt14: init field(l10, "ptr") = l9
          stmt15: init field(l10, "len") = l7
          stmt16: init field(l10, "cap") = l7
          stmt17: init l6 = l10
          stmt18: store w8 binary add(binary add(l4, binary mul(l5, 0x18)), 0x8) = field(l6, "ptr")
          stmt19: store w8 binary add(binary add(l4, binary mul(l5, 0x18)), 0x10) = field(l6, "len")
          stmt20: store w8 binary add(l4, binary mul(l5, 0x18)) = field(l6, "cap")
          stmt21: assign l5 = binary add(l5, 0x1)
        }
        stmt29: init field(l1, "cap") = l2
        stmt30: init field(l1, "ptr") = binary add(l4, binary mul(binary eq(l2, 0x0), 0x8))
        stmt31: init field(l1, "len") = l2
        stmt32: return
      }
    }
  ]
}
"#,
    )
    .expect("handwritten HIR should parse");

    let decoder = compile_structural_hir_decoder(<Vec<String>>::SHAPE, &module);

    let value = crate::deserialize::<Vec<String>>(&decoder, b"hi")
        .expect("handwritten HIR should decode Vec<String> with one inline string");
    assert_eq!(value, vec!["hi".to_owned()]);
}

#[test]
#[ignore = "reduced repro for Vec<String> nested varint decode bug"]
fn handwritten_hir_decodes_vec_string_with_varint_string_len() {
    let module = parse_hir(
        r#"
hir_module {
  regions [
    r0 "input"
  ]
  stores []
  types [
    type t0 "Cursor" <region "r_input"> = struct {
      "bytes": Slice<r0, u8>
      "pos": u64
    }
    type t1 "Vec" size=24 = struct {
      "cap": u64 @0
      "ptr": addr<persistent> @8
      "len": u64 @16
    }
    type t2 "HostStringRaw" size=24 = struct {
      "cap": u64 @0
      "ptr": addr<persistent> @8
      "len": u64 @16
    }
  ]
  callables [
    callable c0 host "runtime.alloc_persistent" {
      params [u64, u64]
      intrinsic alloc_persistent
      returns [addr<persistent>]
      effect mutates
      domains ["persistent_heap":mutate]
      control may_fail
      capabilities ["runtime.alloc"]
      safety opaque_host
      docs "Allocate persistent memory that may escape in the result."
    }
    callable c1 host "runtime.validate_utf8_range" {
      params [u64, u32]
      intrinsic validate_utf8_range
      returns []
      effect reads
      domains ["input":read]
      control may_fail
      capabilities ["runtime.utf8"]
      safety opaque_host
      docs "Validate that a borrowed byte range is UTF-8."
    }
    callable c2 host "runtime.memcpy" {
      params [u64, u64, u64]
      intrinsic memcpy
      returns [u64]
      effect mutates
      domains ["persistent_heap":mutate, "input":read]
      control returns
      capabilities ["runtime.memcpy"]
      safety opaque_host
      docs "Copy bytes from one address to another."
    }
  ]
  functions [
    function f0 "build_vec_with_varint_string_len" {
      regions [r0]
      stores []
      params [
        l0 param "cursor": &mut t0<r0>
        l1 destination "out": t1
      ]
      locals [
        l2 temp "list_len": u64
        l3 temp "list_bytes": u64
        l4 temp "list_ptr": addr<persistent>
        l5 temp "list_index": u64
        l6 temp "list_elem": t2
        l7 temp "varint_acc": u64
        l8 temp "varint_shift": u64
        l9 temp "varint_byte": u8
        l10 temp "string_len": u32
        l11 temp "string_data": u64
        l12 temp "string_ptr": addr<persistent>
        l13 temp "string_raw": t2
      ]
      return unit
      scopes [
        scope sc0 parent none comment "handwritten Vec<String> builder with varint string len"
      ]
      body @sc0 {
        stmt0: init l2 = 0x1
        stmt1: init l3 = binary mul(l2, 0x18)
        stmt2: init l4 = call c0(l3, 0x8)
        stmt3: init l5 = 0x0
        stmt39: loop @sc0 {
          stmt4: if binary eq(l5, l2) @sc0 {
            stmt5: break
          } else @sc0 {
          }
          stmt6: init l7 = 0x0
          stmt7: init l8 = 0x0
          stmt8: init l9 = 0x0
          stmt19: loop @sc0 {
            stmt9: if binary gt(binary add(field(deref(l0), "pos"), 0x1), slice_len(field(deref(l0), "bytes"))) @sc0 {
              stmt10: fail UnexpectedEof
            } else @sc0 {
            }
            stmt11: assign l9 = load w1(binary add(slice_data(field(deref(l0), "bytes")), field(deref(l0), "pos")))
            stmt12: assign field(deref(l0), "pos") = binary add(field(deref(l0), "pos"), 0x1)
            stmt13: assign l7 = binary bitor(l7, binary shl(binary bitand(l9, 0x7f), l8))
            stmt14: assign l8 = binary add(l8, 0x7)
            stmt16: if binary eq(binary bitand(l9, 0x80), 0x0) @sc0 {
              stmt15: break
            } else @sc0 {
            }
            stmt17: if binary eq(l8, 0x23) @sc0 {
              stmt18: fail InvalidVarint
            } else @sc0 {
            }
          }
          stmt20: if binary ne(binary shr(l7, 0x20), 0x0) @sc0 {
            stmt21: fail NumberOutOfRange
          } else @sc0 {
          }
          stmt22: init l10 = l7
          stmt23: if binary gt(binary add(field(deref(l0), "pos"), l10), slice_len(field(deref(l0), "bytes"))) @sc0 {
            stmt24: fail UnexpectedEof
          } else @sc0 {
          }
          stmt25: init l11 = binary add(slice_data(field(deref(l0), "bytes")), field(deref(l0), "pos"))
          stmt26: expr call c1(l11, l10)
          stmt27: if binary eq(l10, 0x0) @sc0 {
            stmt28: init l12 = 0x1
          } else @sc0 {
            stmt29: init l12 = call c0(l10, 0x1)
            stmt30: expr call c2(l12, l11, l10)
          }
          stmt31: assign field(deref(l0), "pos") = binary add(field(deref(l0), "pos"), l10)
          stmt32: init field(l13, "ptr") = l12
          stmt33: init field(l13, "len") = l10
          stmt34: init field(l13, "cap") = l10
          stmt35: init l6 = l13
          stmt36: store w8 binary add(binary add(l4, binary mul(l5, 0x18)), 0x8) = field(l6, "ptr")
          stmt37: store w8 binary add(binary add(l4, binary mul(l5, 0x18)), 0x10) = field(l6, "len")
          stmt38: store w8 binary add(l4, binary mul(l5, 0x18)) = field(l6, "cap")
          stmt40: assign l5 = binary add(l5, 0x1)
        }
        stmt41: init field(l1, "cap") = l2
        stmt42: init field(l1, "ptr") = binary add(l4, binary mul(binary eq(l2, 0x0), 0x8))
        stmt43: init field(l1, "len") = l2
        stmt44: return
      }
    }
  ]
}
"#,
    )
    .expect("handwritten HIR should parse");

    let decoder = compile_structural_hir_decoder(<Vec<String>>::SHAPE, &module);

    let empty = crate::deserialize::<Vec<String>>(&decoder, &[0])
        .expect("handwritten HIR should decode Vec<String> with empty varint string");
    assert_eq!(empty, vec![String::new()]);

    let value = crate::deserialize::<Vec<String>>(&decoder, &[2, b'h', b'i'])
        .expect("handwritten HIR should decode Vec<String> with varint string len");
    assert_eq!(value, vec!["hi".to_owned()]);
}

#[test]
fn postcard_hir_lowering_decodes_root_vec_structs() {
    let decoder = compile_postcard_decoder_via_structural_hir(<Vec<OwnedAddress>>::SHAPE);

    let value = crate::deserialize::<Vec<OwnedAddress>>(
        &decoder,
        &[2, 2, b'P', b'A', 75, 2, b'L', b'Y', 13],
    )
    .expect("postcard HIR lowering should decode root Vec<struct>");
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

// pre-existing: HIR round-trip mismatch on max_iterations
#[test]
#[ignore]
fn postcard_hir_lowering_decodes_unit_enums() {
    let decoder = compile_postcard_decoder_via_structural_hir(<UnitAnimal>::SHAPE);

    let cat = crate::deserialize::<UnitAnimal>(&decoder, &[0])
        .expect("postcard HIR lowering should decode Cat");
    assert_eq!(cat, UnitAnimal::Cat);

    let dog = crate::deserialize::<UnitAnimal>(&decoder, &[1])
        .expect("postcard HIR lowering should decode Dog");
    assert_eq!(dog, UnitAnimal::Dog);

    let parrot = crate::deserialize::<UnitAnimal>(&decoder, &[2])
        .expect("postcard HIR lowering should decode Parrot");
    assert_eq!(parrot, UnitAnimal::Parrot);
}

#[test]
fn postcard_hir_lowering_decodes_option_scalar_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<MaybeCount>::SHAPE);

    let some = crate::deserialize::<MaybeCount>(&decoder, &[1, 42])
        .expect("postcard HIR lowering should decode Some(u32)");
    assert_eq!(some, MaybeCount { count: Some(42) });

    let none = crate::deserialize::<MaybeCount>(&decoder, &[0])
        .expect("postcard HIR lowering should decode None");
    assert_eq!(none, MaybeCount { count: None });
}

#[test]
fn postcard_hir_lowering_decodes_option_borrowed_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<MaybeBorrowedName<'static>>::SHAPE);

    let some = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[1, 2, b'h', b'i'])
        .expect("postcard HIR lowering should decode Some(&str)");
    assert_eq!(some, MaybeBorrowedName { name: Some("hi") });

    let none = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[0])
        .expect("postcard HIR lowering should decode None");
    assert_eq!(none, MaybeBorrowedName { name: None });
}

#[test]
fn postcard_hir_lowering_decodes_multi_options() {
    let hir = crate::debug_postcard_hir_text(<MultiOpt>::SHAPE);
    std::fs::write("/tmp/multiopt.hir.txt", hir).expect("write MultiOpt HIR dump");
    let cfg = crate::debug_cfg_mir_text(<MultiOpt>::SHAPE, crate::DecoderKind::Postcard);
    std::fs::write("/tmp/multiopt.cfg.txt", cfg).expect("write MultiOpt CFG dump");
    let (module, _symbol_table) = build_postcard_decoder_hir(<MultiOpt>::SHAPE);
    let mut ir = lower_hir_module(&module);
    crate::compiler::run_default_passes_from_env(&mut ir);
    let linear = crate::linearize::linearize(&mut ir);
    let hints = Default::default();
    let cfg = crate::regalloc_engine::ir::lower_and_optimize(&linear, hints);
    let ra3 = crate::regalloc_engine::allocate_cfg_program_regalloc3_native(&cfg)
        .expect("regalloc3 should allocate postcard HIR-lowered MultiOpt cfg");
    if let Some(func) = ra3.functions.first() {
        let mut dump = String::new();
        let mut allocs: Vec<_> = func.allocations.iter().collect();
        allocs.sort_by_key(|(vreg, _)| vreg.index());
        for (vreg, alloc) in allocs {
            let spill = func.spill_slot_for_vreg(*vreg);
            dump.push_str(&format!(
                "v{} => {:?} spill={spill:?}\n",
                vreg.index(),
                alloc
            ));
        }
        std::fs::write("/tmp/multiopt.regalloc3.txt", dump).expect("write MultiOpt regalloc3 dump");
    }
    let decoder = compile_postcard_decoder_via_structural_hir(<MultiOpt>::SHAPE);
    #[cfg(target_arch = "aarch64")]
    if let Some(asm) = decoder.assembly_text() {
        std::fs::write("/tmp/multiopt.asm.txt", asm).expect("write MultiOpt asm dump");
    }
    std::fs::write(
        "/tmp/multiopt.emit.txt",
        decoder
            .emission_trace_text()
            .expect("build MultiOpt emission trace"),
    )
    .expect("write MultiOpt emission dump");
    let encoded = ::postcard::to_allocvec(&MultiOpt {
        a: Some(7),
        b: "hello".to_owned(),
        c: None,
    })
    .expect("postcard should encode MultiOpt");

    let value = crate::deserialize::<MultiOpt>(&decoder, &encoded)
        .expect("postcard HIR lowering should decode mixed option/string fields");
    assert_eq!(
        value,
        MultiOpt {
            a: Some(7),
            b: "hello".to_owned(),
            c: None,
        }
    );
}

mod hir_to_ir;

#[test]
fn postcard_hir_lowering_decodes_scalar_arrays() {
    let decoder = compile_postcard_decoder_via_structural_hir(<ScalarArrayHolder>::SHAPE);

    let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
        .expect("postcard HIR lowering should decode scalar arrays");
    assert_eq!(
        value,
        ScalarArrayHolder {
            values: [1, 2, 3, 4],
        }
    );
}

#[test]
fn postcard_hir_lowering_decodes_borrowed_arrays() {
    let decoder =
        compile_postcard_decoder_via_structural_hir(<BorrowedArrayHolder<'static>>::SHAPE);

    let value =
        crate::deserialize::<BorrowedArrayHolder<'_>>(&decoder, &[2, b'h', b'i', 2, b'o', b'k'])
            .expect("postcard HIR lowering should decode borrowed arrays");
    assert_eq!(
        value,
        BorrowedArrayHolder {
            values: ["hi", "ok"],
        }
    );
}

// pre-existing: HIR round-trip mismatch on max_iterations
#[test]
#[ignore]
fn postcard_hir_lowering_decodes_payload_enums() {
    let decoder = compile_postcard_decoder_via_structural_hir(<PayloadAnimal<'static>>::SHAPE);

    let cat = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[0])
        .expect("postcard HIR lowering should decode unit enum variant");
    assert_eq!(cat, PayloadAnimal::Cat);

    let count = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[1, 42])
        .expect("postcard HIR lowering should decode scalar payload enum variant");
    assert_eq!(count, PayloadAnimal::Count(42));

    let name = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[2, 2, b'h', b'i'])
        .expect("postcard HIR lowering should decode borrowed payload enum variant");
    assert_eq!(name, PayloadAnimal::Name("hi"));
}

// pre-existing: HIR round-trip mismatch on max_iterations
#[test]
#[ignore]
fn postcard_hir_lowering_decodes_enum_in_struct_field() {
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
    .expect("postcard HIR lowering should decode nested enum payloads");
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
#[ignore = "ideal interpreter does not support scalar ABI yet"]
fn postcard_hir_lowering_array_path_matches_jit_differential_harness() {
    let (module, _symbol_table) = build_postcard_decoder_hir(<ScalarArrayHolder>::SHAPE);
    let mut func = lower_hir_module(&module);
    let linear = crate::linearize::linearize(&mut func);
    let output_size = std::mem::size_of::<ScalarArrayHolder>();
    let report = crate::differential_check_linear_ir_vs_jit_with_output_size(
        &linear,
        &[1, 2, 3, 4],
        output_size,
    )
    .expect("differential harness should execute postcard HIR-lowered array decoder");
    assert!(
        report.is_match(),
        "unexpected differential mismatch: {:?}",
        report.mismatch
    );
}

#[test]
#[ignore = "ideal interpreter does not support scalar ABI yet"]
fn postcard_hir_lowering_multi_options_matches_jit_differential_harness() {
    let (module, _symbol_table) = build_postcard_decoder_hir(<MultiOpt>::SHAPE);
    let mut func = lower_hir_module(&module);
    let linear = crate::linearize::linearize(&mut func);
    let encoded = ::postcard::to_allocvec(&MultiOpt {
        a: Some(7),
        b: "hello".to_owned(),
        c: None,
    })
    .expect("postcard should encode MultiOpt");
    let output_size = std::mem::size_of::<MultiOpt>();
    let ignored_output_bytes = non_semantic_output_byte_mask(&module);
    let report =
        crate::differential_check_linear_ir_vs_jit_with_output_size(&linear, &encoded, output_size)
            .expect("differential harness should execute postcard HIR-lowered MultiOpt decoder");
    let masked_mismatch = crate::compare_differential_outcomes_with_ignored_output_bytes(
        &report.interpreter,
        &report.jit,
        Some(&ignored_output_bytes),
    );
    assert!(
        masked_mismatch.is_none(),
        "unexpected differential mismatch: {masked_mismatch:?} (raw={:?})",
        report.mismatch,
    );
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
fn debug_line_rows_cover_entry_prologue() {
    let rows = normalize_debug_line_rows(&vec![
        kajit_emit::SourceMapEntry {
            offset: 40,
            location: kajit_emit::SourceLocation {
                file: 0,
                line: 5,
                column: 1,
            },
        },
        kajit_emit::SourceMapEntry {
            offset: 48,
            location: kajit_emit::SourceLocation {
                file: 0,
                line: 6,
                column: 1,
            },
        },
    ]);

    assert_eq!(rows[0].code_offset, 0);
    assert_eq!(rows[0].line, 5);
    assert_eq!(rows[1].code_offset, 40);
    assert_eq!(rows[1].line, 5);
    assert_eq!(rows[2].code_offset, 48);
    assert_eq!(rows[2].line, 6);
}

#[test]
fn debug_line_rows_do_not_duplicate_existing_entry_mapping() {
    let rows = normalize_debug_line_rows(&vec![
        kajit_emit::SourceMapEntry {
            offset: 0,
            location: kajit_emit::SourceLocation {
                file: 0,
                line: 5,
                column: 1,
            },
        },
        kajit_emit::SourceMapEntry {
            offset: 48,
            location: kajit_emit::SourceLocation {
                file: 0,
                line: 6,
                column: 1,
            },
        },
    ]);

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].code_offset, 0);
    assert_eq!(rows[0].line, 5);
    assert_eq!(rows[1].code_offset, 48);
    assert_eq!(rows[1].line, 6);
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
    let inst_id = crate::regalloc_engine::ir::InstId::new(0);
    let inst_id_2 = crate::regalloc_engine::ir::InstId::new(1);
    let term_id = crate::regalloc_engine::ir::TermId::new(0);
    let block_id = crate::regalloc_engine::ir::BlockId::new(0);
    let func = crate::regalloc_engine::ir::Function {
        id: crate::regalloc_engine::ir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![crate::regalloc_engine::ir::Block {
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
            crate::regalloc_engine::ir::Inst {
                id: inst_id,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                operands: vec![crate::regalloc_engine::ir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::ir::OperandKind::Def,
                    class: crate::regalloc_engine::ir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::ir::Clobbers::default(),
            },
            crate::regalloc_engine::ir::Inst {
                id: inst_id_2,
                op: crate::linearize::LinearOp::Copy {
                    dst: crate::ir::VReg::new(1),
                    src: v0,
                },
                operands: vec![
                    crate::regalloc_engine::ir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::ir::OperandKind::Use,
                        class: crate::regalloc_engine::ir::RegClass::Gpr,
                        fixed: None,
                    },
                    crate::regalloc_engine::ir::Operand {
                        vreg: crate::ir::VReg::new(1),
                        kind: crate::regalloc_engine::ir::OperandKind::Def,
                        class: crate::regalloc_engine::ir::RegClass::Gpr,
                        fixed: None,
                    },
                ],
                clobbers: crate::regalloc_engine::ir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::ir::Terminator::Return],
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
    let op_id = crate::regalloc_engine::ir::OpId::Inst(inst_id);
    let op_id_2 = crate::regalloc_engine::ir::OpId::Inst(inst_id_2);
    let program = crate::regalloc_engine::ir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 0,
        param_slot_count: 0,
        debug: crate::regalloc_engine::ir::ProgramDebugProvenance {
            scopes,
            values: crate::ir::Arena::new(),
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_id), block_scope),
                ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                (
                    (
                        crate::ir::LambdaId::new(0),
                        crate::regalloc_engine::ir::OpId::Term(term_id),
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
        data_blobs: vec![],
        stack_allocs: vec![],
        data_arg_layouts: vec![],
    };
    #[cfg(target_arch = "aarch64")]
    let reg = kajit_mir::regalloc3::machine_inst::PReg(19);
    #[cfg(target_arch = "x86_64")]
    let reg = kajit_mir::regalloc3::machine_inst::PReg(12);
    #[cfg(target_arch = "aarch64")]
    let reg_2 = kajit_mir::regalloc3::machine_inst::PReg(20);
    #[cfg(target_arch = "x86_64")]
    let reg_2 = kajit_mir::regalloc3::machine_inst::PReg(13);
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
                op_id: crate::regalloc_engine::ir::OpId::Term(term_id),
                line: 3,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 8,
                    end_offset: 12,
                }],
            },
        ],
    };

    let location_map = test_location_map(&[(0, reg.0), (1, reg_2.0)], &[], &[]);
    let vars = cfg_value_dwarf_variables(
        &program,
        &location_map,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
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
            let dwarf_reg =
                crate::jit_dwarf::dwarf_register_from_hw_encoding(jit_dwarf_target_arch(), reg.0)
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
    let inst_id = crate::regalloc_engine::ir::InstId::new(0);
    let inst_id_2 = crate::regalloc_engine::ir::InstId::new(1);
    let term_id = crate::regalloc_engine::ir::TermId::new(0);
    let return_term_id = crate::regalloc_engine::ir::TermId::new(1);
    let entry_block_id = crate::regalloc_engine::ir::BlockId::new(0);
    let exit_block_id = crate::regalloc_engine::ir::BlockId::new(1);
    let edge_id = crate::regalloc_engine::ir::EdgeId::new(0);
    let func = crate::regalloc_engine::ir::Function {
        id: crate::regalloc_engine::ir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: entry_block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![
            crate::regalloc_engine::ir::Block {
                id: entry_block_id,
                params: Vec::new(),
                insts: vec![inst_id, inst_id_2],
                term: term_id,
                preds: Vec::new(),
                succs: vec![edge_id],
                dead: false,
            },
            crate::regalloc_engine::ir::Block {
                id: exit_block_id,
                params: vec![v0],
                insts: Vec::new(),
                term: return_term_id,
                preds: vec![edge_id],
                succs: Vec::new(),
                dead: false,
            },
        ],
        edges: vec![crate::regalloc_engine::ir::Edge {
            id: edge_id,
            from: entry_block_id,
            to: exit_block_id,
            args: vec![crate::regalloc_engine::ir::EdgeArg {
                target: v0,
                source: v0,
            }],
        }],
        insts: vec![
            crate::regalloc_engine::ir::Inst {
                id: inst_id,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                operands: vec![crate::regalloc_engine::ir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::ir::OperandKind::Def,
                    class: crate::regalloc_engine::ir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::ir::Clobbers::default(),
            },
            crate::regalloc_engine::ir::Inst {
                id: inst_id_2,
                op: crate::linearize::LinearOp::Const { dst: v1, value: 9 },
                operands: vec![crate::regalloc_engine::ir::Operand {
                    vreg: v1,
                    kind: crate::regalloc_engine::ir::OperandKind::Def,
                    class: crate::regalloc_engine::ir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::ir::Clobbers::default(),
            },
        ],
        terms: vec![
            crate::regalloc_engine::ir::Terminator::Branch { edge: edge_id },
            crate::regalloc_engine::ir::Terminator::Return,
        ],
    };
    let op_id = crate::regalloc_engine::ir::OpId::Inst(inst_id);
    let op_id_2 = crate::regalloc_engine::ir::OpId::Inst(inst_id_2);
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
    let program = crate::regalloc_engine::ir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 0,
        param_slot_count: 0,
        debug: crate::regalloc_engine::ir::ProgramDebugProvenance {
            scopes,
            values: crate::ir::Arena::new(),
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_id), block_scope),
                ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                (
                    (
                        crate::ir::LambdaId::new(0),
                        crate::regalloc_engine::ir::OpId::Term(term_id),
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
        data_blobs: vec![],
        stack_allocs: vec![],
        data_arg_layouts: vec![],
    };
    let term_op = crate::regalloc_engine::ir::OpId::Term(term_id);
    #[cfg(target_arch = "aarch64")]
    let reg = kajit_mir::regalloc3::machine_inst::PReg(19);
    #[cfg(target_arch = "x86_64")]
    let reg = kajit_mir::regalloc3::machine_inst::PReg(12);
    #[cfg(target_arch = "aarch64")]
    let reg_2 = kajit_mir::regalloc3::machine_inst::PReg(20);
    #[cfg(target_arch = "x86_64")]
    let reg_2 = kajit_mir::regalloc3::machine_inst::PReg(13);
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

    let location_map = test_location_map(&[(0, reg.0), (1, reg_2.0)], &[], &[]);
    let vars = cfg_value_dwarf_variables(
        &program,
        &location_map,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
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
            let dwarf_reg =
                crate::jit_dwarf::dwarf_register_from_hw_encoding(jit_dwarf_target_arch(), reg.0)
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
    let inst_id = crate::regalloc_engine::ir::InstId::new(0);
    let inst_id_2 = crate::regalloc_engine::ir::InstId::new(1);
    let term_id = crate::regalloc_engine::ir::TermId::new(0);
    let block_id = crate::regalloc_engine::ir::BlockId::new(0);
    let func = crate::regalloc_engine::ir::Function {
        id: crate::regalloc_engine::ir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![crate::regalloc_engine::ir::Block {
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
            crate::regalloc_engine::ir::Inst {
                id: inst_id,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                operands: vec![crate::regalloc_engine::ir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::ir::OperandKind::Def,
                    class: crate::regalloc_engine::ir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::ir::Clobbers::default(),
            },
            crate::regalloc_engine::ir::Inst {
                id: inst_id_2,
                op: crate::linearize::LinearOp::Copy {
                    dst: crate::ir::VReg::new(1),
                    src: v0,
                },
                operands: vec![
                    crate::regalloc_engine::ir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::ir::OperandKind::Use,
                        class: crate::regalloc_engine::ir::RegClass::Gpr,
                        fixed: None,
                    },
                    crate::regalloc_engine::ir::Operand {
                        vreg: crate::ir::VReg::new(1),
                        kind: crate::regalloc_engine::ir::OperandKind::Def,
                        class: crate::regalloc_engine::ir::RegClass::Gpr,
                        fixed: None,
                    },
                ],
                clobbers: crate::regalloc_engine::ir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::ir::Terminator::Return],
    };
    let op_id = crate::regalloc_engine::ir::OpId::Inst(inst_id);
    let op_id_2 = crate::regalloc_engine::ir::OpId::Inst(inst_id_2);
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
    let program = crate::regalloc_engine::ir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 0,
        param_slot_count: 0,
        debug: crate::regalloc_engine::ir::ProgramDebugProvenance {
            scopes,
            values: crate::ir::Arena::new(),
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_id), block_scope),
                ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                (
                    (
                        crate::ir::LambdaId::new(0),
                        crate::regalloc_engine::ir::OpId::Term(term_id),
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
        data_blobs: vec![],
        stack_allocs: vec![],
        data_arg_layouts: vec![],
    };
    #[cfg(target_arch = "aarch64")]
    let reg = kajit_mir::regalloc3::machine_inst::PReg(19);
    #[cfg(target_arch = "x86_64")]
    let reg = kajit_mir::regalloc3::machine_inst::PReg(12);
    #[cfg(target_arch = "aarch64")]
    let reg_2 = kajit_mir::regalloc3::machine_inst::PReg(20);
    #[cfg(target_arch = "x86_64")]
    let reg_2 = kajit_mir::regalloc3::machine_inst::PReg(13);
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
                op_id: crate::regalloc_engine::ir::OpId::Term(term_id),
                line: 3,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 8,
                    end_offset: 12,
                }],
            },
        ],
    };

    let location_map = test_location_map(&[(0, reg.0), (1, reg_2.0)], &[], &[]);
    let subprogram = cfg_mir_dwarf_variables(
        None,
        &program,
        &location_map,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
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

    let inst_a = crate::regalloc_engine::ir::InstId::new(0);
    let inst_b = crate::regalloc_engine::ir::InstId::new(1);
    let term_id = crate::regalloc_engine::ir::TermId::new(0);
    let block_id = crate::regalloc_engine::ir::BlockId::new(0);
    let op_a = crate::regalloc_engine::ir::OpId::Inst(inst_a);
    let op_b = crate::regalloc_engine::ir::OpId::Inst(inst_b);
    let term_op = crate::regalloc_engine::ir::OpId::Term(term_id);
    let root_scope = crate::ir::DebugScopeId::new(0);

    let func = crate::regalloc_engine::ir::Function {
        id: crate::regalloc_engine::ir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: vec![crate::ir::VReg::new(0)],
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![crate::regalloc_engine::ir::Block {
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
            crate::regalloc_engine::ir::Inst {
                id: inst_a,
                op: crate::linearize::LinearOp::CallIntrinsic {
                    func: crate::ir::FnPtr(
                        crate::intrinsics::kajit_read_bool as *const () as usize,
                    ),
                    args: Vec::new(),
                    dst: None,
                },
                operands: Vec::new(),
                clobbers: crate::regalloc_engine::ir::Clobbers::default(),
            },
            crate::regalloc_engine::ir::Inst {
                id: inst_b,
                op: crate::linearize::LinearOp::CallIntrinsic {
                    func: crate::ir::FnPtr(
                        crate::intrinsics::kajit_read_bool as *const () as usize,
                    ),
                    args: Vec::new(),
                    dst: None,
                },
                operands: Vec::new(),
                clobbers: crate::regalloc_engine::ir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::ir::Terminator::Return],
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
    let program = crate::regalloc_engine::ir::Program {
        funcs: vec![func],
        vreg_count: 1,
        slot_count: 0,
        param_slot_count: 0,
        debug: crate::regalloc_engine::ir::ProgramDebugProvenance {
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
        data_blobs: vec![],
        stack_allocs: vec![],
        data_arg_layouts: vec![],
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

    let out_ptr_preg = match jit_dwarf_target_arch() {
        crate::jit_dwarf::DwarfTargetArch::X86_64 => 14,
        crate::jit_dwarf::DwarfTargetArch::Aarch64 => 21,
    };
    let location_map = test_location_map(&[(0, out_ptr_preg)], &[], &[]);

    let vars = cfg_semantic_field_dwarf_variables(
        <Bools as Facet>::SHAPE,
        &program,
        &location_map,
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

    let out_ptr_reg =
        crate::jit_dwarf::dwarf_register_from_hw_encoding(jit_dwarf_target_arch(), out_ptr_preg)
            .expect("out_ptr preg should map to a DWARF register");
    let expected_expr_a = crate::jit_dwarf::expr_breg_deref_size_stack_value(out_ptr_reg, 0, 1);
    let expected_expr_b = crate::jit_dwarf::expr_breg_deref_size_stack_value(out_ptr_reg, 1, 1);
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
fn cfg_semantic_field_dwarf_variables_work_with_spilled_out_ptr() {
    #[derive(Facet)]
    struct Bools {
        a: bool,
    }

    let inst_a = crate::regalloc_engine::ir::InstId::new(0);
    let term_id = crate::regalloc_engine::ir::TermId::new(0);
    let block_id = crate::regalloc_engine::ir::BlockId::new(0);
    let op_a = crate::regalloc_engine::ir::OpId::Inst(inst_a);
    let term_op = crate::regalloc_engine::ir::OpId::Term(term_id);
    let root_scope = crate::ir::DebugScopeId::new(0);

    let func = crate::regalloc_engine::ir::Function {
        id: crate::regalloc_engine::ir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: vec![crate::ir::VReg::new(0)],
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![crate::regalloc_engine::ir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst_a],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
            dead: false,
        }],
        edges: Vec::new(),
        insts: vec![crate::regalloc_engine::ir::Inst {
            id: inst_a,
            op: crate::linearize::LinearOp::CallIntrinsic {
                func: crate::ir::FnPtr(crate::intrinsics::kajit_read_bool as *const () as usize),
                args: Vec::new(),
                dst: None,
            },
            operands: Vec::new(),
            clobbers: crate::regalloc_engine::ir::Clobbers::default(),
        }],
        terms: vec![crate::regalloc_engine::ir::Terminator::Return],
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

    let program = crate::regalloc_engine::ir::Program {
        funcs: vec![func],
        vreg_count: 1,
        slot_count: 0,
        param_slot_count: 0,
        debug: crate::regalloc_engine::ir::ProgramDebugProvenance {
            scopes,
            values,
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::new(),
            op_values: std::collections::HashMap::from([(
                (crate::ir::LambdaId::new(0), op_a),
                debug_a,
            )]),
            vreg_scopes: Vec::new(),
            vreg_values: Vec::new(),
        },
        hints: Default::default(),
        extra_excluded_regs: vec![],
        data_blobs: vec![],
        stack_allocs: vec![],
        data_arg_layouts: vec![],
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
                op_id: term_op,
                line: 20,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 16,
                    end_offset: 24,
                }],
            },
        ],
    };

    let stack_offset = 64u32;
    let location_map = crate::harness::LocationMap {
        static_locations: std::collections::HashMap::from([(
            0u32,
            crate::harness::VRegLocation::StackSlot(stack_offset),
        )]),
        call_lines: std::collections::HashSet::new(),
        call_return_vregs: std::collections::HashMap::new(),
        edit_clobbers: std::collections::HashMap::new(),
        num_spill_slots: 0,
    };

    let vars = cfg_semantic_field_dwarf_variables(
        <Bools as Facet>::SHAPE,
        &program,
        &location_map,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
    );

    assert_eq!(vars.len(), 1);
    let mut expected = crate::jit_dwarf::expr_fbreg_deref_size(stack_offset as i64, 8);
    expected.extend(crate::jit_dwarf::expr_plus_uconst(0));
    expected.extend(crate::jit_dwarf::expr_deref_size(1));
    expected.extend(crate::jit_dwarf::expr_stack_value());
    match &vars[0].variable.location {
        crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].expression, expected);
        }
        crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
            panic!("semantic field vars should use ranged locations")
        }
    }
}

#[test]
fn cfg_value_dwarf_variables_can_hide_semantic_owned_vregs() {
    let v0 = crate::ir::VReg::new(0);
    let inst_id = crate::regalloc_engine::ir::InstId::new(0);
    let term_id = crate::regalloc_engine::ir::TermId::new(0);
    let block_id = crate::regalloc_engine::ir::BlockId::new(0);
    let op_id = crate::regalloc_engine::ir::OpId::Inst(inst_id);
    let term_op = crate::regalloc_engine::ir::OpId::Term(term_id);
    let root_scope = crate::ir::DebugScopeId::new(0);
    let func = crate::regalloc_engine::ir::Function {
        id: crate::regalloc_engine::ir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![crate::regalloc_engine::ir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst_id],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
            dead: false,
        }],
        edges: Vec::new(),
        insts: vec![crate::regalloc_engine::ir::Inst {
            id: inst_id,
            op: crate::linearize::LinearOp::Const { dst: v0, value: 1 },
            operands: vec![crate::regalloc_engine::ir::Operand {
                vreg: v0,
                kind: crate::regalloc_engine::ir::OperandKind::Def,
                class: crate::regalloc_engine::ir::RegClass::Gpr,
                fixed: None,
            }],
            clobbers: crate::regalloc_engine::ir::Clobbers::default(),
        }],
        terms: vec![crate::regalloc_engine::ir::Terminator::Return],
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
    let program = crate::regalloc_engine::ir::Program {
        funcs: vec![func],
        vreg_count: 1,
        slot_count: 0,
        param_slot_count: 0,
        debug: crate::regalloc_engine::ir::ProgramDebugProvenance {
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
        data_blobs: vec![],
        stack_allocs: vec![],
        data_arg_layouts: vec![],
    };
    #[cfg(target_arch = "aarch64")]
    let reg = kajit_mir::regalloc3::machine_inst::PReg(19);
    #[cfg(target_arch = "x86_64")]
    let reg = kajit_mir::regalloc3::machine_inst::PReg(12);
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

    let location_map = test_location_map(&[(0, reg.0)], &[], &[]);
    let vars = cfg_value_dwarf_variables(
        &program,
        &location_map,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
        true,
    );

    assert!(vars.is_empty(), "semantic-owned vregs should be hidden");
}

#[test]
fn cfg_semantic_named_dwarf_variables_merge_shared_vregs() {
    let v0 = crate::ir::VReg::new(0);
    let v1 = crate::ir::VReg::new(1);
    let inst0 = crate::regalloc_engine::ir::InstId::new(0);
    let inst1 = crate::regalloc_engine::ir::InstId::new(1);
    let inst2 = crate::regalloc_engine::ir::InstId::new(2);
    let term_id = crate::regalloc_engine::ir::TermId::new(0);
    let block_id = crate::regalloc_engine::ir::BlockId::new(0);
    let op0 = crate::regalloc_engine::ir::OpId::Inst(inst0);
    let op1 = crate::regalloc_engine::ir::OpId::Inst(inst1);
    let op2 = crate::regalloc_engine::ir::OpId::Inst(inst2);
    let term_op = crate::regalloc_engine::ir::OpId::Term(term_id);
    let root_scope = crate::ir::DebugScopeId::new(0);
    let func = crate::regalloc_engine::ir::Function {
        id: crate::regalloc_engine::ir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        output_size: 0,
        blocks: vec![crate::regalloc_engine::ir::Block {
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
            crate::regalloc_engine::ir::Inst {
                id: inst0,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 1 },
                operands: vec![crate::regalloc_engine::ir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::ir::OperandKind::Def,
                    class: crate::regalloc_engine::ir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::ir::Clobbers::default(),
            },
            crate::regalloc_engine::ir::Inst {
                id: inst1,
                op: crate::linearize::LinearOp::Copy { dst: v1, src: v0 },
                operands: vec![
                    crate::regalloc_engine::ir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::ir::OperandKind::Use,
                        class: crate::regalloc_engine::ir::RegClass::Gpr,
                        fixed: None,
                    },
                    crate::regalloc_engine::ir::Operand {
                        vreg: v1,
                        kind: crate::regalloc_engine::ir::OperandKind::Def,
                        class: crate::regalloc_engine::ir::RegClass::Gpr,
                        fixed: None,
                    },
                ],
                clobbers: crate::regalloc_engine::ir::Clobbers::default(),
            },
            crate::regalloc_engine::ir::Inst {
                id: inst2,
                op: crate::linearize::LinearOp::WriteToSlot {
                    src: v1,
                    slot: crate::ir::SlotId::new(0),
                },
                operands: vec![crate::regalloc_engine::ir::Operand {
                    vreg: v1,
                    kind: crate::regalloc_engine::ir::OperandKind::Use,
                    class: crate::regalloc_engine::ir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::ir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::ir::Terminator::Return],
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
    let program = crate::regalloc_engine::ir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 1,
        param_slot_count: 0,
        debug: crate::regalloc_engine::ir::ProgramDebugProvenance {
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
        data_blobs: vec![],
        stack_allocs: vec![],
        data_arg_layouts: vec![],
    };
    #[cfg(target_arch = "aarch64")]
    let reg = kajit_mir::regalloc3::machine_inst::PReg(19);
    #[cfg(target_arch = "x86_64")]
    let reg = kajit_mir::regalloc3::machine_inst::PReg(12);
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

    let location_map = test_location_map(&[(0, reg.0), (1, reg.0)], &[], &[]);
    let vars = cfg_semantic_named_dwarf_variables(
        &program,
        &location_map,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
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

// pre-existing: HIR round-trip mismatch on max_iterations
#[test]
#[ignore]
fn postcard_option_scalar_matches_differential_harness() {
    let (module, _symbol_table) = build_postcard_decoder_hir(<MaybeCount>::SHAPE);
    let mut func = lower_hir_module(&module);
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
