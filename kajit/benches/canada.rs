//! Benchmark parsing canada.json (GeoJSON) from nativejson-benchmark.
//!
//! Tests deeply nested arrays of floating-point coordinates.

#[path = "harness.rs"]
mod harness;

use facet::Facet;
use serde::Deserialize;
use std::hint::black_box;
use std::sync::LazyLock;

// =============================================================================
// Types for canada.json (GeoJSON)
// =============================================================================

#[derive(Debug, Deserialize, Facet)]
struct FeatureCollection {
    #[serde(rename = "type")]
    #[facet(rename = "type")]
    type_: String,
    features: Vec<Feature>,
}

#[derive(Debug, Deserialize, Facet)]
struct Feature {
    #[serde(rename = "type")]
    #[facet(rename = "type")]
    type_: String,
    properties: Properties,
    geometry: Geometry,
}

#[derive(Debug, Deserialize, Facet)]
struct Properties {
    name: String,
}

#[derive(Debug, Deserialize, Facet)]
struct Geometry {
    #[serde(rename = "type")]
    #[facet(rename = "type")]
    type_: String,
    coordinates: Vec<Vec<Vec<f64>>>,
}

// =============================================================================
// Data loading
// =============================================================================

fn decompress(compressed: &[u8]) -> Vec<u8> {
    let mut decompressed = Vec::new();
    brotli::BrotliDecompress(&mut std::io::Cursor::new(compressed), &mut decompressed)
        .expect("Failed to decompress fixture");
    decompressed
}

static CANADA_JSON: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let compressed = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/canada.json.br"
    ));
    decompress(compressed)
});

static CANADA_STR: LazyLock<String> =
    LazyLock::new(|| String::from_utf8(CANADA_JSON.clone()).expect("canada.json is valid UTF-8"));

// =============================================================================
// Cached compiled deserializers
// =============================================================================

static KAJIT_CANADA: LazyLock<kajit::compiler::CompiledDecoder> =
    LazyLock::new(|| kajit::compile_decoder(FeatureCollection::SHAPE, &kajit::json::KajitJson));

// =============================================================================
// Benchmarks
// =============================================================================

fn main() {
    let mut v: Vec<harness::Bench> = Vec::new();

    v.push(harness::Bench {
        name: "canada/serde_deser".into(),
        func: Box::new(|runner| {
            let data = &*CANADA_STR;
            runner.run(|| {
                black_box(serde_json::from_str::<FeatureCollection>(black_box(data)).unwrap());
            });
        }),
    });

    let kajit_preflight = kajit::from_str::<FeatureCollection>(&KAJIT_CANADA, &CANADA_STR);
    match kajit_preflight {
        Ok(_) => {
            v.push(harness::Bench {
                name: "canada/kajit_dynasm_deser".into(),
                func: Box::new(|runner| {
                    let data = &*CANADA_STR;
                    let deser = &*KAJIT_CANADA;
                    runner.run(|| {
                        black_box(
                            kajit::from_str::<FeatureCollection>(deser, black_box(data)).unwrap(),
                        );
                    });
                }),
            });
        }
        Err(err) => {
            eprintln!(
                "skipping canada/kajit_dynasm_deser: fixture currently unsupported by kajit ({err:?})"
            );
        }
    }

    harness::run_benchmarks(v);
}
