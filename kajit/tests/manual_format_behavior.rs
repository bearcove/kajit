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
    let deser = kajit::compile_decoder(BorrowedFriend::SHAPE, kajit::DecoderKind::Postcard);
    let result: BorrowedFriend<'_> = kajit::deserialize(&deser, &input).unwrap();
    assert_eq!(result.age, 42);
    assert_eq!(result.name, "Alice");
    assert_eq!(result.name.as_ptr(), unsafe { input.as_ptr().add(2) });
}

#[test]
#[ignore = "postcard Cow field HIR lowering not implemented yet"]
fn postcard_cow_str_borrowed_zero_copy() {
    let input = [0x2A, 0x05, b'A', b'l', b'i', b'c', b'e'];
    let deser = kajit::compile_decoder(CowFriend::SHAPE, kajit::DecoderKind::Postcard);
    let result: CowFriend<'_> = kajit::deserialize(&deser, &input).unwrap();
    assert_eq!(result.age, 42);
    assert!(matches!(result.name, Cow::Borrowed("Alice")));
}
