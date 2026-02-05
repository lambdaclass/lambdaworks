//! Benchmark for evaluate_offset_fft optimization
//!
//! Run with hyperfine:
//! ```sh
//! cargo build --release -p lambdaworks-math --example bench_evaluate_offset_fft
//! hyperfine --warmup 3 \
//!   'target/release/examples/bench_evaluate_offset_fft original' \
//!   'target/release/examples/bench_evaluate_offset_fft optimized' \
//!   'target/release/examples/bench_evaluate_offset_fft with_buffer'
//! ```

use lambdaworks_math::{
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
    polynomial::Polynomial,
    unsigned_integer::element::UnsignedInteger,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;

type F = Stark252PrimeField;
type FE = FieldElement<F>;

const POLY_ORDER: u64 = 12; // 2^12 = 4096 coefficients
const ITERATIONS: usize = 1000;
const NUM_OFFSETS: usize = 10; // Simulate real-world loop with different offsets

fn rand_field_elements(order: u64, rng: &mut StdRng) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        let rand_big = UnsignedInteger {
            limbs: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        };
        result.push(FE::new(rand_big));
    }
    result
}

fn rand_poly(order: u64, rng: &mut StdRng) -> Polynomial<FE> {
    Polynomial::new(&rand_field_elements(order, rng))
}

fn rand_offsets(count: usize, rng: &mut StdRng) -> Vec<FE> {
    (0..count).map(|_| FE::from(rng.gen::<u64>())).collect()
}

/// Original implementation using poly.scale() - creates intermediate polynomial allocation
fn bench_original(poly: &Polynomial<FE>, offset: &FE) -> Vec<FE> {
    let scaled = poly.scale(offset);
    Polynomial::evaluate_fft::<F>(&scaled, 1, None).unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("optimized");

    let mut rng = StdRng::seed_from_u64(42);
    let poly = rand_poly(POLY_ORDER, &mut rng);
    let offsets = rand_offsets(NUM_OFFSETS, &mut rng);

    match mode {
        "original" => {
            // Original approach: scale() + evaluate_fft() creates intermediate polynomial
            // This allocates: 1 scaled polynomial + 1 coeffs copy + 1 FFT result per call
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    let result = bench_original(black_box(&poly), black_box(offset));
                    black_box(result);
                }
            }
        }
        "optimized" => {
            // Optimized: scales directly into FFT buffer, no intermediate polynomial
            // This allocates: 1 FFT buffer per call (used as result)
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    let result = Polynomial::evaluate_offset_fft::<F>(
                        black_box(&poly),
                        1,
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
            let len = poly.coeff_len().next_power_of_two();
            let mut buffer = Vec::with_capacity(len);
            for _ in 0..ITERATIONS {
                for offset in &offsets {
                    Polynomial::evaluate_offset_fft_with_buffer::<F>(
                        black_box(&poly),
                        1,
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
            eprintln!("Usage: {} [original|optimized|with_buffer]", args[0]);
            std::process::exit(1);
        }
    }
}
