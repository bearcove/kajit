//! Benchmark parsing citm_catalog.json from nativejson-benchmark.
//!
//! Tests HashMap-heavy JSON with camelCase field names, nested structs,
//! vectors, and optional fields. The fixture is a venue/ticketing catalog.
//!
//! # On the `()` fields
//!
//! Several fields (`description`, `subject_code`, `subtitle`, `name` in
//! Performance, `seat_map_image`) are always `null` in this fixture.
//! We use bare `()` for these, matching the serde-rs/json-benchmark
//! convention. See `benches/twitter.rs` for more context.

#[path = "harness.rs"]
mod harness;

use facet::Facet;
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::hint::black_box;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, LazyLock};

// =============================================================================
// Types for citm_catalog.json
// =============================================================================

#[derive(Debug, Deserialize, Facet)]
#[serde(rename_all = "camelCase")]
#[facet(rename_all = "camelCase")]
struct CitmCatalog<'a> {
    #[serde(borrow)]
    area_names: HashMap<String, &'a str>,
    #[serde(borrow)]
    audience_sub_category_names: HashMap<String, &'a str>,
    #[serde(borrow)]
    block_names: HashMap<String, &'a str>,
    #[serde(borrow)]
    events: HashMap<String, Event<'a>>,
    #[serde(borrow)]
    performances: Vec<Performance<'a>>,
    #[serde(borrow)]
    seat_category_names: HashMap<String, &'a str>,
    #[serde(borrow)]
    sub_topic_names: HashMap<String, &'a str>,
    #[serde(borrow)]
    subject_names: HashMap<String, &'a str>,
    #[serde(borrow)]
    topic_names: HashMap<String, &'a str>,
    topic_sub_topics: HashMap<String, Vec<u64>>,
    #[serde(borrow)]
    venue_names: HashMap<String, &'a str>,
}

#[derive(Debug, Deserialize, Facet)]
#[serde(rename_all = "camelCase")]
#[facet(rename_all = "camelCase")]
struct Event<'a> {
    description: (),
    id: u64,
    #[serde(borrow)]
    logo: Option<&'a str>,
    #[serde(borrow)]
    name: Cow<'a, str>,
    sub_topic_ids: Vec<u64>,
    subject_code: (),
    subtitle: (),
    topic_ids: Vec<u64>,
}

#[derive(Debug, Deserialize, Facet)]
#[serde(rename_all = "camelCase")]
#[facet(rename_all = "camelCase")]
struct Performance<'a> {
    event_id: u64,
    id: u64,
    #[serde(borrow)]
    logo: Option<&'a str>,
    name: (),
    prices: Vec<Price>,
    seat_categories: Vec<SeatCategory>,
    seat_map_image: (),
    start: u64,
    venue_code: &'a str,
}

#[derive(Debug, Deserialize, Facet)]
#[serde(rename_all = "camelCase")]
#[facet(rename_all = "camelCase")]
struct Price {
    amount: u64,
    audience_sub_category_id: u64,
    seat_category_id: u64,
}

#[derive(Debug, Deserialize, Facet)]
#[serde(rename_all = "camelCase")]
#[facet(rename_all = "camelCase")]
struct SeatCategory {
    areas: Vec<Area>,
    seat_category_id: u64,
}

#[derive(Debug, Deserialize, Facet)]
#[serde(rename_all = "camelCase")]
#[facet(rename_all = "camelCase")]
struct Area {
    area_id: u64,
    block_ids: Vec<u64>,
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

static CITM_JSON: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let compressed = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/citm_catalog.json.br"
    ));
    decompress(compressed)
});

static CITM_STR: LazyLock<String> = LazyLock::new(|| {
    String::from_utf8(CITM_JSON.clone()).expect("citm_catalog.json is valid UTF-8")
});

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_owned()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

// =============================================================================
// Benchmarks
// =============================================================================

fn main() {
    let mut v: Vec<harness::Bench> = Vec::new();

    v.push(harness::Bench {
        name: "citm_catalog/serde_deser".into(),
        func: Box::new(|runner| {
            let data = &*CITM_STR;
            runner.run(|| {
                black_box(serde_json::from_str::<CitmCatalog>(black_box(data)).unwrap());
            });
        }),
    });

    let kajit_decoder = match catch_unwind(AssertUnwindSafe(|| {
        kajit::compile_decoder(CitmCatalog::SHAPE, &kajit::json::KajitJson)
    })) {
        Ok(decoder) => Some(Arc::new(decoder)),
        Err(payload) => {
            eprintln!(
                "skipping citm_catalog/kajit_deser: compile unsupported ({})",
                panic_payload_to_string(payload)
            );
            None
        }
    };

    if let Some(decoder) = kajit_decoder {
        match kajit::from_str::<CitmCatalog>(decoder.as_ref(), &CITM_STR) {
            Ok(_) => {
                v.push(harness::Bench {
                    name: "citm_catalog/kajit_deser".into(),
                    func: Box::new({
                        let decoder = Arc::clone(&decoder);
                        move |runner| {
                            let data = &*CITM_STR;
                            let decoder = Arc::clone(&decoder);
                            runner.run(move || {
                                black_box(
                                    kajit::from_str::<CitmCatalog>(
                                        decoder.as_ref(),
                                        black_box(data),
                                    )
                                    .unwrap(),
                                );
                            });
                        }
                    }),
                });
            }
            Err(err) => {
                eprintln!(
                    "skipping citm_catalog/kajit_deser: fixture unsupported by kajit ({err:?})"
                );
            }
        }
    }

    harness::run_benchmarks(v);
}
