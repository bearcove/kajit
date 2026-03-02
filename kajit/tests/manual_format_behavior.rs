use facet::Facet;
use std::borrow::Cow;

#[derive(Facet, Debug, PartialEq)]
struct BorrowedFriend<'a> {
    age: u32,
    name: &'a str,
}

#[derive(Facet, Debug, PartialEq)]
struct CowFriend<'a> {
    age: u32,
    name: Cow<'a, str>,
}

#[test]
fn postcard_borrowed_str_zero_copy() {
    let input = [0x2A, 0x05, b'A', b'l', b'i', b'c', b'e'];
    let deser = kajit::compile_decoder(BorrowedFriend::SHAPE, &kajit::postcard::KajitPostcard);
    let result: BorrowedFriend<'_> = kajit::deserialize(&deser, &input).unwrap();
    assert_eq!(result.age, 42);
    assert_eq!(result.name, "Alice");
    assert_eq!(result.name.as_ptr(), unsafe { input.as_ptr().add(2) });
}

#[test]
fn postcard_cow_str_borrowed_zero_copy() {
    let input = [0x2A, 0x05, b'A', b'l', b'i', b'c', b'e'];
    let deser = kajit::compile_decoder(CowFriend::SHAPE, &kajit::postcard::KajitPostcard);
    let result: CowFriend<'_> = kajit::deserialize(&deser, &input).unwrap();
    assert_eq!(result.age, 42);
    assert!(matches!(result.name, Cow::Borrowed("Alice")));
}

#[test]
fn json_borrowed_str_zero_copy_fast_path() {
    let input = br#"{"age":42,"name":"Alice"}"#;
    let name_start = input.windows(5).position(|w| w == b"Alice").unwrap();
    let deser = kajit::compile_decoder(BorrowedFriend::SHAPE, &kajit::json::KajitJson);
    let result: BorrowedFriend<'_> = kajit::deserialize(&deser, input).unwrap();
    assert_eq!(result.age, 42);
    assert_eq!(result.name, "Alice");
    assert_eq!(result.name.as_ptr(), unsafe {
        input.as_ptr().add(name_start)
    });
}

#[test]
fn json_borrowed_str_escape_is_error() {
    let input = br#"{"age":42,"name":"A\nB"}"#;
    let deser = kajit::compile_decoder(BorrowedFriend::SHAPE, &kajit::json::KajitJson);
    let err = kajit::deserialize::<BorrowedFriend<'_>>(&deser, input).unwrap_err();
    assert_eq!(err.code, kajit::context::ErrorCode::InvalidEscapeSequence);
}

#[test]
fn json_cow_str_fast_path_borrowed() {
    let input = br#"{"age":42,"name":"Alice"}"#;
    let deser = kajit::compile_decoder(CowFriend::SHAPE, &kajit::json::KajitJson);
    let result: CowFriend<'_> = kajit::deserialize(&deser, input).unwrap();
    assert_eq!(result.age, 42);
    assert!(matches!(result.name, Cow::Borrowed("Alice")));
}

#[test]
fn json_cow_str_escape_slow_path_owned() {
    let input = br#"{"age":42,"name":"A\nB"}"#;
    let deser = kajit::compile_decoder(CowFriend::SHAPE, &kajit::json::KajitJson);
    let result: CowFriend<'_> = kajit::deserialize(&deser, input).unwrap();
    assert_eq!(result.age, 42);
    assert!(matches!(result.name, Cow::Owned(ref s) if s == "A\nB"));
}

#[test]
fn json_f64_edge_cases() {
    #[derive(Facet, Debug, PartialEq)]
    struct F {
        x: f64,
    }

    let cases: &[(&[u8], f64)] = &[
        (br#"{"x":0}"#, 0.0),
        (br#"{"x":1}"#, 1.0),
        (br#"{"x":42}"#, 42.0),
        (br#"{"x":9007199254740992}"#, 9007199254740992.0),
        (br#"{"x":0.5}"#, 0.5),
        (br#"{"x":1.0}"#, 1.0),
        (br#"{"x":3.14}"#, 3.14),
        (br#"{"x":2.718281828459045}"#, 2.718281828459045),
        (br#"{"x":-1.0}"#, -1.0),
        (br#"{"x":-0.0}"#, -0.0_f64),
        (br#"{"x":-3.14}"#, -3.14),
        (br#"{"x":1e10}"#, 1e10),
        (br#"{"x":1.5e2}"#, 150.0),
        (br#"{"x":1e-10}"#, 1e-10),
        (br#"{"x":1E10}"#, 1e10),
        (br#"{"x":1e+10}"#, 1e10),
        (br#"{"x":1e0}"#, 1.0),
        (br#"{"x":0.001}"#, 0.001),
        (br#"{"x":0.0000000000000000000001}"#, 1e-22),
        (br#"{"x":1e308}"#, 1e308),
        (br#"{"x":1.7976931348623157e308}"#, f64::MAX),
        (br#"{"x":1e-308}"#, 1e-308),
        (br#"{"x":2.2250738585072014e-308}"#, 2.2250738585072014e-308),
        (br#"{"x":1e309}"#, f64::INFINITY),
        (br#"{"x":-1e309}"#, f64::NEG_INFINITY),
        (br#"{"x":1e-400}"#, 0.0),
        (br#"{"x":12345678901234567890.0}"#, 12345678901234567890.0),
        (br#"{"x": 1.0}"#, 1.0),
        (br#"{"x":	1.0}"#, 1.0),
    ];

    let deser = kajit::compile_decoder(F::SHAPE, &kajit::json::KajitJson);
    for (input, expected) in cases {
        let result: F = kajit::deserialize(&deser, input).unwrap();
        assert_eq!(
            result.x.to_bits(),
            expected.to_bits(),
            "input={:?}: got {} ({:#018x}), expected {} ({:#018x})",
            std::str::from_utf8(input).unwrap(),
            result.x,
            result.x.to_bits(),
            expected,
            expected.to_bits(),
        );
    }
}

#[test]
fn json_f64_canada_roundtrip() {
    #[derive(Facet, Debug)]
    struct Coord {
        v: f64,
    }

    let compressed = include_bytes!("../fixtures/canada.json.br");
    let mut json_bytes = Vec::new();
    brotli::BrotliDecompress(&mut std::io::Cursor::new(compressed), &mut json_bytes).unwrap();

    let deser = kajit::compile_decoder(Coord::SHAPE, &kajit::json::KajitJson);

    let mut mismatches = 0;
    let mut total = 0;
    let mut i = 0;
    while i < json_bytes.len() {
        if json_bytes[i] == b'-' || json_bytes[i].is_ascii_digit() {
            let start = i;
            i += 1;
            while i < json_bytes.len() {
                match json_bytes[i] {
                    b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => i += 1,
                    _ => break,
                }
            }
            let s = &json_bytes[start..i];
            if s.contains(&b'.') || s.contains(&b'e') || s.contains(&b'E') {
                total += 1;
                let std_val: f64 = std::str::from_utf8(s).unwrap().parse().unwrap();
                let json_input = format!(r#"{{"v":{}}}"#, std::str::from_utf8(s).unwrap());
                let result: Coord = kajit::deserialize(&deser, json_input.as_bytes()).unwrap();
                if std_val.to_bits() != result.v.to_bits() {
                    if mismatches < 10 {
                        eprintln!(
                            "MISMATCH: {:?} -> std={std_val:?} jit={:?}",
                            std::str::from_utf8(s).unwrap(),
                            result.v,
                        );
                    }
                    mismatches += 1;
                }
            }
        } else {
            i += 1;
        }
    }
    eprintln!("{total} floats checked, {mismatches} mismatches");
    assert_eq!(mismatches, 0, "{mismatches}/{total} values differ from std");
}
