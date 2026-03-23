// Hand-optimized u32 varint decoder to establish performance baseline
// This shows what performance is achievable with optimal codegen

use std::hint::black_box;

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
    let test_cases = vec![
        vec![0x00],                         // 0
        vec![0x01],                         // 1
        vec![0x7f],                         // 127
        vec![0x80, 0x01],                   // 128
        vec![0xff, 0xff, 0xff, 0xff, 0x0f], // u32::MAX
    ];

    // Verify correctness first
    eprintln!("Verifying correctness...");
    assert_eq!(hand_optimized_u32_unrolled(&[0x00]).unwrap(), 0);
    assert_eq!(hand_optimized_u32_unrolled(&[0x01]).unwrap(), 1);
    assert_eq!(hand_optimized_u32_unrolled(&[0x7f]).unwrap(), 127);
    assert_eq!(hand_optimized_u32_unrolled(&[0x80, 0x01]).unwrap(), 128);
    assert_eq!(
        hand_optimized_u32_unrolled(&[0xff, 0xff, 0xff, 0xff, 0x0f]).unwrap(),
        u32::MAX
    );
    eprintln!("✓ All test cases pass\n");

    let warmup = 1000;
    let iters = 100_000;

    // Warmup
    for _ in 0..warmup {
        for case in &test_cases {
            black_box(hand_optimized_u32_unrolled(case).unwrap());
        }
    }

    // Measure
    let start = std::time::Instant::now();
    for _ in 0..iters {
        for case in &test_cases {
            black_box(hand_optimized_u32_unrolled(case).unwrap());
        }
    }
    let elapsed = start.elapsed();

    let total_decodes = iters * test_cases.len() as u32;
    let per_decode = elapsed.as_nanos() as f64 / total_decodes as f64;

    eprintln!("Hand-optimized u32 varint decoder:");
    eprintln!("  Total: {} decodes in {:?}", total_decodes, elapsed);
    eprintln!("  Per decode: {:.2} ns", per_decode);
    eprintln!("\nFor comparison:");
    eprintln!("  Current kajit: ~3.5 ns");
    eprintln!("  Current serde: ~0.7 ns");

    if per_decode < 1.0 {
        eprintln!("\n✓ Target achieved! This is the performance ceiling.");
    } else {
        eprintln!("\n→ This establishes the theoretical best case for kajit codegen.");
    }
}
