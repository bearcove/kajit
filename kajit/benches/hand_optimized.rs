// Hand-optimized varint decoders to establish performance targets
// These represent what kajit COULD generate with better optimization passes

/// Hand-optimized u32 varint decoder - unrolled, minimal branches
/// Target: match or beat serde's performance
#[inline(never)]
fn hand_optimized_u32_unrolled(input: &[u8]) -> Result<u32, ()> {
    if input.is_empty() {
        return Err(());
    }

    let byte0 = unsafe { *input.get_unchecked(0) };
    let mut result = (byte0 & 0x7f) as u32;
    if byte0 & 0x80 == 0 {
        return Ok(result);
    }

    if input.len() < 2 {
        return Err(());
    }
    let byte1 = unsafe { *input.get_unchecked(1) };
    result |= ((byte1 & 0x7f) as u32) << 7;
    if byte1 & 0x80 == 0 {
        return Ok(result);
    }

    if input.len() < 3 {
        return Err(());
    }
    let byte2 = unsafe { *input.get_unchecked(2) };
    result |= ((byte2 & 0x7f) as u32) << 14;
    if byte2 & 0x80 == 0 {
        return Ok(result);
    }

    if input.len() < 4 {
        return Err(());
    }
    let byte3 = unsafe { *input.get_unchecked(3) };
    result |= ((byte3 & 0x7f) as u32) << 21;
    if byte3 & 0x80 == 0 {
        return Ok(result);
    }

    if input.len() < 5 {
        return Err(());
    }
    let byte4 = unsafe { *input.get_unchecked(4) };
    result |= ((byte4 & 0x7f) as u32) << 28;
    if byte4 & 0x80 != 0 {
        return Err(()); // invalid varint
    }

    Ok(result)
}

fn main() {
    use std::hint::black_box;

    let test_cases = vec![
        vec![0x00],                         // 0
        vec![0x01],                         // 1
        vec![0x7f],                         // 127
        vec![0x80, 0x01],                   // 128
        vec![0xff, 0xff, 0xff, 0xff, 0x0f], // u32::MAX
    ];

    let warmup = 100;
    let iters = 10000;

    // Warmup
    for _ in 0..warmup {
        for case in &test_cases {
            black_box(hand_optimized_u32_unrolled(case));
        }
    }

    // Measure
    let start = std::time::Instant::now();
    for _ in 0..iters {
        for case in &test_cases {
            black_box(hand_optimized_u32_unrolled(case));
        }
    }
    let elapsed = start.elapsed();

    let per_iter = elapsed / (iters * test_cases.len() as u32);
    eprintln!("hand_optimized_u32_unrolled: {:?} per decode", per_iter);
    eprintln!(
        "Total time for {} iterations: {:?}",
        iters * test_cases.len() as u32,
        elapsed
    );
}
