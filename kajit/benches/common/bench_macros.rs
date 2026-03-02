// Shared bench macro definitions for bench targets under benches/.

use std::hint::black_box;
use crate::harness;
use facet::Facet;
use std::sync::Arc;

pub(crate) fn add_json<T>(v: &mut Vec<harness::Bench>, group: &str, value: T)
where
    for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    let data = Arc::new(serde_json::to_string(&value).unwrap());
    let decoder = Arc::new(kajit::compile_decoder_legacy(T::SHAPE, &kajit::json::KajitJson));

    let prefix = format!("{group}/json");

    v.push(harness::Bench {
        name: format!("{prefix}/serde_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            move |runner| {
                runner.run(|| {
                    black_box(serde_json::from_str::<T>(black_box(data.as_str())).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/kajit_dynasm_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            let decoder = Arc::clone(&decoder);
            move |runner| {
                let deser = &*decoder;
                runner.run(|| {
                    black_box(kajit::from_str::<T>(deser, black_box(data.as_str())).unwrap());
                });
            }
        }),
    });
}

pub(crate) fn add_json_ser<T>(v: &mut Vec<harness::Bench>, group: &str, value: T)
where
    for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    let data = Arc::new(serde_json::to_string(&value).unwrap());
    let value = Arc::new(value);
    let decoder = Arc::new(kajit::compile_decoder_legacy(T::SHAPE, &kajit::json::KajitJson));
    let encoder = Arc::new(kajit::compile_encoder(T::SHAPE, &kajit::json::KajitJsonEncoder));

    let prefix = format!("{group}/json");

    v.push(harness::Bench {
        name: format!("{prefix}/serde_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            move |runner| {
                runner.run(|| {
                    black_box(serde_json::from_str::<T>(black_box(data.as_str())).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/kajit_dynasm_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            let decoder = Arc::clone(&decoder);
            move |runner| {
                let deser = &*decoder;
                runner.run(|| {
                    black_box(kajit::from_str::<T>(deser, black_box(data.as_str())).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/serde_ser"),
        func: Box::new({
            let value = Arc::clone(&value);
            move |runner| {
                runner.run(|| {
                    black_box(serde_json::to_vec(black_box(&*value)).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/kajit_dynasm_ser"),
        func: Box::new({
            let value = Arc::clone(&value);
            let encoder = Arc::clone(&encoder);
            move |runner| {
                let enc = &*encoder;
                runner.run(|| {
                    black_box(kajit::serialize(enc, black_box(&*value)));
                });
            }
        }),
    });
}

pub(crate) fn add_postcard<T>(v: &mut Vec<harness::Bench>, group: &str, value: T)
where
    for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    let data = Arc::new(::postcard::to_allocvec(&value).unwrap());
    let decoder = Arc::new(kajit::compile_decoder_legacy(T::SHAPE, &kajit::postcard::KajitPostcard));

    let prefix = format!("{group}/postcard");

    v.push(harness::Bench {
        name: format!("{prefix}/serde_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            move |runner| {
                runner.run(|| {
                    black_box(::postcard::from_bytes::<T>(black_box(&data[..])).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/kajit_dynasm_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            let decoder = Arc::clone(&decoder);
            move |runner| {
                let deser = &*decoder;
                runner.run(|| {
                    black_box(kajit::deserialize::<T>(deser, black_box(&data[..])).unwrap());
                });
            }
        }),
    });
}

pub(crate) fn add_postcard_ser<T>(v: &mut Vec<harness::Bench>, group: &str, value: T)
where
    for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    let data = Arc::new(::postcard::to_allocvec(&value).unwrap());
    let value = Arc::new(value);
    let decoder = Arc::new(kajit::compile_decoder_legacy(T::SHAPE, &kajit::postcard::KajitPostcard));
    let encoder = Arc::new(kajit::compile_encoder(T::SHAPE, &kajit::postcard::KajitPostcard));

    let prefix = format!("{group}/postcard");

    v.push(harness::Bench {
        name: format!("{prefix}/serde_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            move |runner| {
                runner.run(|| {
                    black_box(::postcard::from_bytes::<T>(black_box(&data[..])).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/kajit_dynasm_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            let decoder = Arc::clone(&decoder);
            move |runner| {
                let deser = &*decoder;
                runner.run(|| {
                    black_box(kajit::deserialize::<T>(deser, black_box(&data[..])).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/serde_ser"),
        func: Box::new({
            let value = Arc::clone(&value);
            move |runner| {
                runner.run(|| {
                    black_box(::postcard::to_allocvec(black_box(&*value)).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/kajit_dynasm_ser"),
        func: Box::new({
            let value = Arc::clone(&value);
            let encoder = Arc::clone(&encoder);
            move |runner| {
                let enc = &*encoder;
                runner.run(|| {
                    black_box(kajit::serialize(enc, black_box(&*value)));
                });
            }
        }),
    });
}

pub(crate) fn add_postcard_ser_ir<T>(v: &mut Vec<harness::Bench>, group: &str, value: T)
where
    for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    let data = Arc::new(::postcard::to_allocvec(&value).unwrap());
    let value = Arc::new(value);
    let legacy_decoder =
        Arc::new(kajit::compile_decoder_legacy(T::SHAPE, &kajit::postcard::KajitPostcard));
    let ir_decoder =
        Arc::new(kajit::compile_decoder_via_ir(T::SHAPE, &kajit::postcard::KajitPostcard));
    let encoder = Arc::new(kajit::compile_encoder(T::SHAPE, &kajit::postcard::KajitPostcard));

    let prefix = format!("{group}/postcard");

    v.push(harness::Bench {
        name: format!("{prefix}/serde_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            move |runner| {
                runner.run(|| {
                    black_box(::postcard::from_bytes::<T>(black_box(&data[..])).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/kajit_dynasm_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            let legacy_decoder = Arc::clone(&legacy_decoder);
            move |runner| {
                let deser = &*legacy_decoder;
                runner.run(|| {
                    black_box(kajit::deserialize::<T>(deser, black_box(&data[..])).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/kajit_ir_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            let ir_decoder = Arc::clone(&ir_decoder);
            move |runner| {
                let deser = &*ir_decoder;
                runner.run(|| {
                    black_box(kajit::deserialize::<T>(deser, black_box(&data[..])).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/serde_ser"),
        func: Box::new({
            let value = Arc::clone(&value);
            move |runner| {
                runner.run(|| {
                    black_box(::postcard::to_allocvec(black_box(&*value)).unwrap());
                });
            }
        }),
    });
    v.push(harness::Bench {
        name: format!("{prefix}/kajit_dynasm_ser"),
        func: Box::new({
            let value = Arc::clone(&value);
            let encoder = Arc::clone(&encoder);
            move |runner| {
                let enc = &*encoder;
                runner.run(|| {
                    black_box(kajit::serialize(enc, black_box(&*value)));
                });
            }
        }),
    });
}

pub(crate) fn add_postcard_legacy_ir<T>(v: &mut Vec<harness::Bench>, group: &str, value: T, with_compile: bool)
where
    for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    let data = Arc::new(::postcard::to_allocvec(&value).unwrap());
    let legacy_decoder =
        Arc::new(kajit::compile_decoder_legacy(T::SHAPE, &kajit::postcard::KajitPostcard));
    let ir_decoder =
        Arc::new(kajit::compile_decoder_via_ir(T::SHAPE, &kajit::postcard::KajitPostcard));

    let prefix = format!("{group}/postcard");

    v.push(harness::Bench {
        name: format!("{prefix}/serde_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            move |runner| {
                runner.run(|| {
                    black_box(::postcard::from_bytes::<T>(black_box(&data[..])).unwrap());
                });
            }
        }),
    });

    v.push(harness::Bench {
        name: format!("{prefix}/kajit_dynasm_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            let legacy_decoder = Arc::clone(&legacy_decoder);
            move |runner| {
                let deser = &*legacy_decoder;
                runner.run(|| {
                    black_box(kajit::deserialize::<T>(deser, black_box(&data[..])).unwrap());
                });
            }
        }),
    });

    v.push(harness::Bench {
        name: format!("{prefix}/kajit_ir_deser"),
        func: Box::new({
            let data = Arc::clone(&data);
            let ir_decoder = Arc::clone(&ir_decoder);
            move |runner| {
                let deser = &*ir_decoder;
                runner.run(|| {
                    black_box(kajit::deserialize::<T>(deser, black_box(&data[..])).unwrap());
                });
            }
        }),
    });

    if with_compile {
        v.push(harness::Bench {
            name: format!("{prefix}/kajit_dynasm_compile"),
            func: Box::new(move |runner| {
                runner.run(|| {
                    black_box(kajit::compile_decoder_legacy(
                        T::SHAPE,
                        &kajit::postcard::KajitPostcard,
                    ));
                });
            }),
        });

        v.push(harness::Bench {
            name: format!("{prefix}/kajit_ir_compile"),
            func: Box::new(move |runner| {
                runner.run(|| {
                    black_box(kajit::compile_decoder_via_ir(
                        T::SHAPE,
                        &kajit::postcard::KajitPostcard,
                    ));
                });
            }),
        });
    }
}

macro_rules! bench {
    ($v:ident, $name:ident, $value:expr) => {{
        let group = stringify!($name);
        crate::bench_macros::add_json(&mut $v, group, { $value });
        crate::bench_macros::add_postcard(&mut $v, group, { $value });
    }};

    ($v:ident, $name:ident, $value:expr, +ser) => {{
        let group = stringify!($name);
        crate::bench_macros::add_json_ser(&mut $v, group, { $value });
        crate::bench_macros::add_postcard_ser(&mut $v, group, { $value });
    }};

    ($v:ident, $name:ident, $value:expr, +ir) => {{
        let group = stringify!($name);
        crate::bench_macros::add_json(&mut $v, group, { $value });
        crate::bench_macros::add_postcard_legacy_ir(&mut $v, group, { $value }, false);
    }};

    ($v:ident, $name:ident, $value:expr, +ser, +ir) => {{
        let group = stringify!($name);
        crate::bench_macros::add_json_ser(&mut $v, group, { $value });
        crate::bench_macros::add_postcard_ser_ir(&mut $v, group, { $value });
    }};

    ($v:ident, $name:ident, $value:expr, +ir, +ser) => {{
        let group = stringify!($name);
        crate::bench_macros::add_json_ser(&mut $v, group, { $value });
        crate::bench_macros::add_postcard_ser_ir(&mut $v, group, { $value });
    }};

    ($v:ident, $name:ident, $value:expr, json_only) => {{
        let group = stringify!($name);
        crate::bench_macros::add_json(&mut $v, group, { $value });
    }};

    ($v:ident, $name:ident, $value:expr, postcard_legacy_ir) => {{
        let group = stringify!($name);
        crate::bench_macros::add_postcard_legacy_ir(&mut $v, group, { $value }, false);
    }};

    ($v:ident, $name:ident, $value:expr, postcard_legacy_ir_compile) => {{
        let group = stringify!($name);
        crate::bench_macros::add_postcard_legacy_ir(&mut $v, group, { $value }, true);
    }};

    ($v:ident, $name:ident, $Type:ty, $value:expr) => {{
        let _ = core::marker::PhantomData::<$Type>;
        bench!($v, $name, $value);
    }};

    ($v:ident, $name:ident, $Type:ty, $value:expr, +ser) => {{
        let _ = core::marker::PhantomData::<$Type>;
        bench!($v, $name, $value, +ser);
    }};

    ($v:ident, $name:ident, $Type:ty, $value:expr, +ir) => {{
        let _ = core::marker::PhantomData::<$Type>;
        bench!($v, $name, $value, +ir);
    }};

    ($v:ident, $name:ident, $Type:ty, $value:expr, +ser, +ir) => {{
        let _ = core::marker::PhantomData::<$Type>;
        bench!($v, $name, $value, +ser, +ir);
    }};

    ($v:ident, $name:ident, $Type:ty, $value:expr, +ir, +ser) => {{
        let _ = core::marker::PhantomData::<$Type>;
        bench!($v, $name, $value, +ir, +ser);
    }};

    ($v:ident, $name:ident, $Type:ty, $value:expr, json_only) => {{
        let _ = core::marker::PhantomData::<$Type>;
        bench!($v, $name, $value, json_only);
    }};

    ($v:ident, $name:ident, $Type:ty, $value:expr, postcard_legacy_ir) => {{
        let _ = core::marker::PhantomData::<$Type>;
        bench!($v, $name, $value, postcard_legacy_ir);
    }};

    ($v:ident, $name:ident, $Type:ty, $value:expr, postcard_legacy_ir_compile) => {{
        let _ = core::marker::PhantomData::<$Type>;
        bench!($v, $name, $value, postcard_legacy_ir_compile);
    }};
}
