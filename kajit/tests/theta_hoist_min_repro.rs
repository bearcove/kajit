use facet::Facet;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
struct TwoFields {
    u8_max: u8,
    i32_min: i32,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
struct ThreeFields {
    u8_max: u8,
    i16_min: i16,
    i32_min: i32,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
struct FourFields {
    u8_max: u8,
    i8_min: i8,
    i8_max: i8,
    i32_min: i32,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
struct SixFields {
    u8_max: u8,
    u16_max: u16,
    i8_min: i8,
    i8_max: i8,
    i16_min: i16,
    i32_min: i32,
}

fn theta_opts(enabled: bool) -> kajit::PipelineOptions {
    let mut opts = kajit::PipelineOptions::default();
    opts.enable_option("all_opts")
        .expect("all_opts should exist");
    opts.enable_option("regalloc")
        .expect("regalloc should exist");
    if enabled {
        opts.enable_option("pass.theta_loop_invariant_hoist")
            .expect("theta pass should exist");
    } else {
        opts.disable_option("pass.theta_loop_invariant_hoist")
            .expect("theta pass should exist");
    }
    opts
}

fn decode_json_with_opts<T>(input: &[u8], opts: &kajit::PipelineOptions) -> T
where
    for<'a> T: Facet<'a> + DeserializeOwned,
{
    let decoder = kajit::compile_decoder_with_options(T::SHAPE, &kajit::json::KajitJson, opts);
    kajit::deserialize::<T>(&decoder, input).expect("decode should succeed")
}

fn miscompiles_with_theta_hoist<T>(input: &[u8]) -> bool
where
    for<'a> T: Facet<'a> + DeserializeOwned,
    T: PartialEq + core::fmt::Debug,
{
    let expected: T = serde_json::from_slice(input).expect("serde_json should decode baseline");
    let off = decode_json_with_opts::<T>(input, &theta_opts(false));
    assert_eq!(
        off, expected,
        "theta OFF should match serde baseline for this minimization harness"
    );
    let on = decode_json_with_opts::<T>(input, &theta_opts(true));
    on != expected
}

#[test]
#[cfg(target_arch = "aarch64")]
fn theta_hoist_smallest_known_struct_repro_is_three_fields() {
    // Keep this input aligned with `Boundaries` style values from corpus.
    let two_fields = br#"{"u8_max":255,"i32_min":-2147483648}"#;
    let three_fields = br#"{"u8_max":255,"i16_min":-32768,"i32_min":-2147483648}"#;
    let four_fields = br#"{"u8_max":255,"i8_min":-128,"i8_max":127,"i32_min":-2147483648}"#;
    let six_fields =
        br#"{"u8_max":255,"u16_max":65535,"i8_min":-128,"i8_max":127,"i16_min":-32768,"i32_min":-2147483648}"#;

    assert!(
        !miscompiles_with_theta_hoist::<TwoFields>(two_fields),
        "2-field case should not repro; if it does, update minimization"
    );
    assert!(
        miscompiles_with_theta_hoist::<ThreeFields>(three_fields),
        "expected 3-field case to repro theta-hoist miscompile"
    );
    assert!(
        !miscompiles_with_theta_hoist::<FourFields>(four_fields),
        "4-field variant currently does not repro; keep as guard for minimization assumptions"
    );
    assert!(
        miscompiles_with_theta_hoist::<SixFields>(six_fields),
        "expected 6-field boundaries case to repro theta-hoist miscompile"
    );
}
