//! Benchmark for evaluate_offset_fft optimization
//!
//! This benchmark tests three implementations:
//! - `original`: poly.scale() + evaluate_fft() - creates intermediate polynomial
//! - `optimized`: evaluate_offset_fft() - scales directly into FFT buffer
//! - `with_buffer`: evaluate_offset_fft_with_buffer() - reuses buffer across iterations
//!
//! Two field options are available:
//! - `stark252` (default): Stark252PrimeField - 256-bit field, slower FFT
//! - `goldilocks`: Goldilocks64HybridField - 64-bit field, faster FFT
//!
//! Run with hyperfine:
//! ```sh
//! cargo build --release -p lambdaworks-math --example bench_evaluate_offset_fft
//!
//! # Stark252 (default) - FFT dominates, allocation savings negligible
//! hyperfine --warmup 3 \
//!   'target/release/examples/bench_evaluate_offset_fft original' \
//!   'target/release/examples/bench_evaluate_offset_fft optimized' \
//!   'target/release/examples/bench_evaluate_offset_fft with_buffer'
//!
//! # Goldilocks - faster FFT, allocation savings more visible
//! hyperfine --warmup 3 \
//!   'target/release/examples/bench_evaluate_offset_fft original goldilocks' \
//!   'target/release/examples/bench_evaluate_offset_fft optimized goldilocks' \
//!   'target/release/examples/bench_evaluate_offset_fft with_buffer goldilocks'
//! ```

use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        fields::u64_goldilocks_field::Goldilocks64Field,
        fields::u64_goldilocks_hybrid_field::Goldilocks64HybridField,
    },
    polynomial::Polynomial,
    unsigned_integer::element::UnsignedInteger,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;

const POLY_ORDER: u64 = 12; // 2^12 = 4096 coefficients
const ITERATIONS: usize = 1000;
const NUM_OFFSETS: usize = 10; // Simulate real-world loop with different offsets
const BLOWUP_FACTOR: usize = 1;

// ============================================================================
// Stark252 field types and helpers
// ============================================================================

type Stark252FE = FieldElement<Stark252PrimeField>;

fn rand_stark252_elements(order: u64, rng: &mut StdRng) -> Vec<Stark252FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        let rand_big = UnsignedInteger {
            limbs: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        };
        result.push(Stark252FE::new(rand_big));
    }
    result
}

fn rand_stark252_poly(order: u64, rng: &mut StdRng) -> Polynomial<Stark252FE> {
    Polynomial::new(&rand_stark252_elements(order, rng))
}

fn rand_stark252_offsets(count: usize, rng: &mut StdRng) -> Vec<Stark252FE> {
    (0..count)
        .map(|_| Stark252FE::from(rng.gen::<u64>()))
        .collect()
}

fn bench_stark252_original(poly: &Polynomial<Stark252FE>, offset: &Stark252FE) -> Vec<Stark252FE> {
    let scaled = poly.scale(offset);
    Polynomial::evaluate_fft::<Stark252PrimeField>(&scaled, BLOWUP_FACTOR, None).unwrap()
}

// ============================================================================
// Goldilocks Hybrid field types and helpers
// ============================================================================

type GoldilocksHybridFE = FieldElement<Goldilocks64HybridField>;

fn rand_goldilocks_hybrid_elements(order: u64, rng: &mut StdRng) -> Vec<GoldilocksHybridFE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        result.push(GoldilocksHybridFE::from(rng.gen::<u64>()));
    }
    result
}

fn rand_goldilocks_hybrid_poly(order: u64, rng: &mut StdRng) -> Polynomial<GoldilocksHybridFE> {
    Polynomial::new(&rand_goldilocks_hybrid_elements(order, rng))
}

fn rand_goldilocks_hybrid_offsets(count: usize, rng: &mut StdRng) -> Vec<GoldilocksHybridFE> {
    (0..count)
        .map(|_| GoldilocksHybridFE::from(rng.gen::<u64>()))
        .collect()
}

fn bench_goldilocks_hybrid_original(
    poly: &Polynomial<GoldilocksHybridFE>,
    offset: &GoldilocksHybridFE,
) -> Vec<GoldilocksHybridFE> {
    let scaled = poly.scale(offset);
    Polynomial::evaluate_fft::<Goldilocks64HybridField>(&scaled, BLOWUP_FACTOR, None).unwrap()
}

// ============================================================================
// Goldilocks Classic field types and helpers
// ============================================================================

type GoldilocksClassicFE = FieldElement<Goldilocks64Field>;

fn rand_goldilocks_classic_elements(order: u64, rng: &mut StdRng) -> Vec<GoldilocksClassicFE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        result.push(GoldilocksClassicFE::from(rng.gen::<u64>()));
    }
    result
}

fn rand_goldilocks_classic_poly(order: u64, rng: &mut StdRng) -> Polynomial<GoldilocksClassicFE> {
    Polynomial::new(&rand_goldilocks_classic_elements(order, rng))
}

fn rand_goldilocks_classic_offsets(count: usize, rng: &mut StdRng) -> Vec<GoldilocksClassicFE> {
    (0..count)
        .map(|_| GoldilocksClassicFE::from(rng.gen::<u64>()))
        .collect()
}

fn bench_goldilocks_classic_original(
    poly: &Polynomial<GoldilocksClassicFE>,
    offset: &GoldilocksClassicFE,
) -> Vec<GoldilocksClassicFE> {
    let scaled = poly.scale(offset);
    Polynomial::evaluate_fft::<Goldilocks64Field>(&scaled, BLOWUP_FACTOR, None).unwrap()
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("optimized");
    let field = args.get(2).map(|s| s.as_str()).unwrap_or("stark252");

    match field {
        "goldilocks" | "goldilocks-hybrid" => run_goldilocks_hybrid_benchmark(mode),
        "goldilocks-classic" => run_goldilocks_classic_benchmark(mode),
        "stark252" | _ => run_stark252_benchmark(mode),
    }
}

fn run_stark252_benchmark(mode: &str) {
    let mut rng = StdRng::seed_from_u64(42);
    let poly = rand_stark252_poly(POLY_ORDER, &mut rng);
    let offsets = rand_stark252_offsets(NUM_OFFSETS, &mut rng);

    match mode {
        "original" => {
            // Original approach: scale() + evaluate_fft() creates intermediate polynomial
            // This allocates: 1 scaled polynomial + 1 coeffs copy + 1 FFT result per call
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    let result = bench_stark252_original(black_box(&poly), black_box(offset));
                    black_box(result);
                }
            }
        }
        "optimized" => {
            // Optimized: scales directly into FFT buffer, no intermediate polynomial
            // This allocates: 1 FFT buffer per call (used as result)
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    let result = Polynomial::evaluate_offset_fft::<Stark252PrimeField>(
                        black_box(&poly),
                        BLOWUP_FACTOR,
                        None,
                        black_box(offset),
                    )
                    .unwrap();
                    black_box(result);
                }
            }
        }
        "with_buffer" => {
            // With reusable buffer: best for loops, zero allocations after warmup
            // The buffer is reused across all iterations
            // Note: buffer size must account for blowup_factor
            let len = poly.coeff_len().next_power_of_two() * BLOWUP_FACTOR;
            let mut buffer = Vec::with_capacity(len);
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    Polynomial::evaluate_offset_fft_with_buffer::<Stark252PrimeField>(
                        black_box(&poly),
                        BLOWUP_FACTOR,
                        None,
                        black_box(offset),
                        &mut buffer,
                    )
                    .unwrap();
                    black_box(&buffer);
                }
            }
        }
        _ => {
            eprintln!(
                "Usage: {} [original|optimized|with_buffer] [stark252|goldilocks|goldilocks-classic]",
                std::env::args().next().unwrap_or_default()
            );
            std::process::exit(1);
        }
    }
}

fn run_goldilocks_hybrid_benchmark(mode: &str) {
    let mut rng = StdRng::seed_from_u64(42);
    let poly = rand_goldilocks_hybrid_poly(POLY_ORDER, &mut rng);
    let offsets = rand_goldilocks_hybrid_offsets(NUM_OFFSETS, &mut rng);

    match mode {
        "original" => {
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    let result =
                        bench_goldilocks_hybrid_original(black_box(&poly), black_box(offset));
                    black_box(result);
                }
            }
        }
        "optimized" => {
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    let result = Polynomial::evaluate_offset_fft::<Goldilocks64HybridField>(
                        black_box(&poly),
                        BLOWUP_FACTOR,
                        None,
                        black_box(offset),
                    )
                    .unwrap();
                    black_box(result);
                }
            }
        }
        "with_buffer" => {
            let len = poly.coeff_len().next_power_of_two() * BLOWUP_FACTOR;
            let mut buffer = Vec::with_capacity(len);
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    Polynomial::evaluate_offset_fft_with_buffer::<Goldilocks64HybridField>(
                        black_box(&poly),
                        BLOWUP_FACTOR,
                        None,
                        black_box(offset),
                        &mut buffer,
                    )
                    .unwrap();
                    black_box(&buffer);
                }
            }
        }
        _ => {
            eprintln!(
                "Usage: {} [original|optimized|with_buffer] [stark252|goldilocks|goldilocks-classic]",
                std::env::args().next().unwrap_or_default()
            );
            std::process::exit(1);
        }
    }
}

fn run_goldilocks_classic_benchmark(mode: &str) {
    let mut rng = StdRng::seed_from_u64(42);
    let poly = rand_goldilocks_classic_poly(POLY_ORDER, &mut rng);
    let offsets = rand_goldilocks_classic_offsets(NUM_OFFSETS, &mut rng);

    match mode {
        "original" => {
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    let result =
                        bench_goldilocks_classic_original(black_box(&poly), black_box(offset));
                    black_box(result);
                }
            }
        }
        "optimized" => {
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    let result = Polynomial::evaluate_offset_fft::<Goldilocks64Field>(
                        black_box(&poly),
                        BLOWUP_FACTOR,
                        None,
                        black_box(offset),
                    )
                    .unwrap();
                    black_box(result);
                }
            }
        }
        "with_buffer" => {
            let len = poly.coeff_len().next_power_of_two() * BLOWUP_FACTOR;
            let mut buffer = Vec::with_capacity(len);
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    Polynomial::evaluate_offset_fft_with_buffer::<Goldilocks64Field>(
                        black_box(&poly),
                        BLOWUP_FACTOR,
                        None,
                        black_box(offset),
                        &mut buffer,
                    )
                    .unwrap();
                    black_box(&buffer);
                }
            }
        }
        _ => {
            eprintln!(
                "Usage: {} [original|optimized|with_buffer] [stark252|goldilocks|goldilocks-classic]",
                std::env::args().next().unwrap_or_default()
            );
            std::process::exit(1);
        }
    }
}
