//! Benchmark parsing twitter.json from nativejson-benchmark.
//!
//! Tests a realistic JSON payload with nested objects, optional fields,
//! borrowed strings, escape sequences, recursive types, and mixed scalars.
//!
//! # On the `()` fields
//!
//! Several Status fields (`geo`, `coordinates`, `place`, `contributors`) are
//! always `null` in this fixture. We use bare `()` for these, matching the
//! serde-rs/json-benchmark convention. In JSON, `null` deserializes to `()`.
//! This is _not_ a cheat — both serde_json and kajit consume the `null` token.
//! If the fixture ever contained non-null values for these fields, the types
//! would need to be real structs (GeoJSON Point, etc.).

#[path = "harness.rs"]
mod harness;

use facet::Facet;
use serde::Deserialize;
use std::borrow::Cow;
use std::hint::black_box;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, LazyLock};

// =============================================================================
// Types for twitter.json — shared between serde and kajit
// =============================================================================

#[derive(Debug, Deserialize, Facet)]
struct Twitter<'a> {
    #[serde(borrow)]
    statuses: Vec<Status<'a>>,
    #[serde(borrow)]
    search_metadata: SearchMetadata<'a>,
}

#[derive(Debug, Deserialize, Facet)]
struct SearchMetadata<'a> {
    completed_in: f64,
    max_id: u64,
    max_id_str: &'a str,
    next_results: &'a str,
    query: &'a str,
    refresh_url: &'a str,
    count: u64,
    since_id: u64,
    since_id_str: &'a str,
}

#[derive(Debug, Deserialize, Facet)]
struct Status<'a> {
    #[serde(borrow)]
    metadata: Metadata<'a>,
    created_at: &'a str,
    id: u64,
    id_str: &'a str,
    #[serde(borrow)]
    text: Cow<'a, str>,
    #[serde(borrow)]
    source: Cow<'a, str>,
    truncated: bool,
    in_reply_to_status_id: Option<u64>,
    in_reply_to_status_id_str: Option<&'a str>,
    in_reply_to_user_id: Option<u64>,
    in_reply_to_user_id_str: Option<&'a str>,
    in_reply_to_screen_name: Option<&'a str>,
    #[serde(borrow)]
    user: User<'a>,
    // These fields are always null in the fixture. See module-level comment.
    geo: (),
    coordinates: (),
    place: (),
    contributors: (),
    #[serde(default, borrow)]
    #[facet(default)]
    retweeted_status: Option<Box<Status<'a>>>,
    retweet_count: u64,
    favorite_count: u64,
    #[serde(borrow)]
    entities: Entities<'a>,
    favorited: bool,
    retweeted: bool,
    #[serde(default)]
    #[facet(default)]
    possibly_sensitive: Option<bool>,
    lang: &'a str,
}

#[derive(Debug, Deserialize, Facet)]
struct Metadata<'a> {
    result_type: &'a str,
    iso_language_code: &'a str,
}

#[derive(Debug, Deserialize, Facet)]
struct User<'a> {
    id: u64,
    id_str: &'a str,
    #[serde(borrow)]
    name: Cow<'a, str>,
    screen_name: &'a str,
    #[serde(borrow)]
    location: Cow<'a, str>,
    #[serde(borrow)]
    description: Cow<'a, str>,
    url: Option<&'a str>,
    #[serde(borrow)]
    entities: UserEntities<'a>,
    protected: bool,
    followers_count: u64,
    friends_count: u64,
    listed_count: u64,
    created_at: &'a str,
    favourites_count: u64,
    utc_offset: Option<i64>,
    time_zone: Option<&'a str>,
    geo_enabled: bool,
    verified: bool,
    statuses_count: u64,
    lang: &'a str,
    contributors_enabled: bool,
    is_translator: bool,
    is_translation_enabled: bool,
    profile_background_color: &'a str,
    profile_background_image_url: &'a str,
    profile_background_image_url_https: &'a str,
    profile_background_tile: bool,
    profile_image_url: &'a str,
    profile_image_url_https: &'a str,
    #[serde(default)]
    #[facet(default)]
    profile_banner_url: Option<&'a str>,
    profile_link_color: &'a str,
    profile_sidebar_border_color: &'a str,
    profile_sidebar_fill_color: &'a str,
    profile_text_color: &'a str,
    profile_use_background_image: bool,
    default_profile: bool,
    default_profile_image: bool,
    following: bool,
    follow_request_sent: bool,
    notifications: bool,
}

#[derive(Debug, Deserialize, Facet)]
struct UserEntities<'a> {
    #[serde(default, borrow)]
    #[facet(default)]
    url: Option<EntityUrl<'a>>,
    #[serde(borrow)]
    description: EntityDescription<'a>,
}

#[derive(Debug, Deserialize, Facet)]
struct EntityUrl<'a> {
    #[serde(borrow)]
    urls: Vec<Url<'a>>,
}

#[derive(Debug, Deserialize, Facet)]
struct EntityDescription<'a> {
    #[serde(borrow)]
    urls: Vec<Url<'a>>,
}

#[derive(Debug, Deserialize, Facet)]
struct Entities<'a> {
    #[serde(borrow)]
    hashtags: Vec<Hashtag<'a>>,
    #[serde(borrow)]
    symbols: Vec<Symbol<'a>>,
    #[serde(borrow)]
    urls: Vec<Url<'a>>,
    #[serde(borrow)]
    user_mentions: Vec<UserMention<'a>>,
    #[serde(default, borrow)]
    #[facet(default)]
    media: Option<Vec<Media<'a>>>,
}

#[derive(Debug, Deserialize, Facet)]
struct Hashtag<'a> {
    #[serde(borrow)]
    text: Cow<'a, str>,
    indices: Vec<u64>,
}

#[derive(Debug, Deserialize, Facet)]
struct Symbol<'a> {
    #[serde(borrow)]
    text: Cow<'a, str>,
    indices: Vec<u64>,
}

#[derive(Debug, Deserialize, Facet)]
struct Url<'a> {
    url: &'a str,
    expanded_url: &'a str,
    display_url: &'a str,
    indices: Vec<u64>,
}

#[derive(Debug, Deserialize, Facet)]
struct UserMention<'a> {
    screen_name: &'a str,
    #[serde(borrow)]
    name: Cow<'a, str>,
    id: u64,
    id_str: &'a str,
    indices: Vec<u64>,
}

#[derive(Debug, Deserialize, Facet)]
struct Media<'a> {
    id: u64,
    id_str: &'a str,
    indices: Vec<u64>,
    media_url: &'a str,
    media_url_https: &'a str,
    url: &'a str,
    display_url: &'a str,
    expanded_url: &'a str,
    #[serde(rename = "type")]
    #[facet(rename = "type")]
    media_type: &'a str,
    sizes: Sizes,
    #[serde(default)]
    #[facet(default)]
    source_status_id: Option<u64>,
    #[serde(default)]
    #[facet(default)]
    source_status_id_str: Option<&'a str>,
}

#[derive(Debug, Deserialize, Facet)]
struct Sizes {
    medium: SizeInfo,
    small: SizeInfo,
    thumb: SizeInfo,
    large: SizeInfo,
}

#[derive(Debug, Deserialize, Facet)]
struct SizeInfo {
    w: u64,
    h: u64,
    resize: String,
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

static TWITTER_JSON: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let compressed = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/twitter.json.br"
    ));
    decompress(compressed)
});

static TWITTER_STR: LazyLock<String> =
    LazyLock::new(|| String::from_utf8(TWITTER_JSON.clone()).expect("twitter.json is valid UTF-8"));

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
        name: "twitter/serde_deser".into(),
        func: Box::new(|runner| {
            let data = &*TWITTER_STR;
            runner.run(|| {
                black_box(serde_json::from_str::<Twitter>(black_box(data)).unwrap());
            });
        }),
    });

    let kajit_decoder = match catch_unwind(AssertUnwindSafe(|| {
        kajit::compile_decoder(Twitter::SHAPE, &kajit::json::KajitJson)
    })) {
        Ok(decoder) => Some(Arc::new(decoder)),
        Err(payload) => {
            eprintln!(
                "skipping twitter/kajit_deser: compile unsupported ({})",
                panic_payload_to_string(payload)
            );
            None
        }
    };

    if let Some(decoder) = kajit_decoder {
        match kajit::from_str::<Twitter>(decoder.as_ref(), &TWITTER_STR) {
            Ok(_) => {
                v.push(harness::Bench {
                    name: "twitter/kajit_deser".into(),
                    func: Box::new({
                        let decoder = Arc::clone(&decoder);
                        move |runner| {
                            let data = &*TWITTER_STR;
                            let decoder = Arc::clone(&decoder);
                            runner.run(move || {
                                black_box(
                                    kajit::from_str::<Twitter>(decoder.as_ref(), black_box(data))
                                        .unwrap(),
                                );
                            });
                        }
                    }),
                });
            }
            Err(err) => {
                eprintln!("skipping twitter/kajit_deser: fixture unsupported by kajit ({err:?})");
            }
        }
    }

    harness::run_benchmarks(v);
}
