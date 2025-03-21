use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use lambdaworks_math::field::fields::binary::field::{BinaryFieldError, TowerFieldElement};
use rand::Rng;

pub fn rand_element(num_level: usize) -> TowerFieldElement {
    let mut rng = rand::thread_rng();
    let value = rng.gen::<u128>();
    TowerFieldElement::new(value, num_level)
}

fn binary_add_bench(c: &mut Criterion, num_levels_a: usize, num_levels_b: usize) {
    let mut group = c.benchmark_group("Binary TowerField add");
    let a = rand_element(num_levels_a);
    let b = rand_element(num_levels_b);

    group.bench_with_input(
        BenchmarkId::from_parameter(format!("F{}×F{}", 1 << num_levels_a, 1 << num_levels_b)),
        &(num_levels_a, num_levels_b),
        |bencher, _params| {
            bencher.iter(|| {
                black_box(black_box(a) + black_box(b));
            });
        },
    );
    group.finish();
}

fn binary_mul_bench(c: &mut Criterion, num_levels_a: usize, num_levels_b: usize) {
    let mut group = c.benchmark_group("Binary TowerField mul");
    let a = rand_element(num_levels_a);
    let b = rand_element(num_levels_b);

    group.bench_with_input(
        BenchmarkId::from_parameter(format!("F{}×F{}", 1 << num_levels_a, 1 << num_levels_b)),
        &(num_levels_a, num_levels_b),
        |bencher, _params| {
            bencher.iter(|| {
                black_box(black_box(a) * black_box(b));
            });
        },
    );
    group.finish();
}

fn binary_pow_bench(c: &mut Criterion, num_levels: usize, exponent: u32) {
    if num_levels == 0 {
        return;
    }
    let mut group = c.benchmark_group("Binary TowerField pow");
    let a = rand_element(num_levels);

    group.bench_with_input(
        BenchmarkId::from_parameter(format!("F{} ^ {}", 1 << num_levels, exponent)),
        &(num_levels, exponent),
        |bencher, _params| {
            bencher.iter(|| {
                black_box(black_box(a).pow(exponent));
            });
        },
    );
    group.finish();
}

fn binary_inv_bench(c: &mut Criterion, num_levels: usize) {
    if num_levels == 0 {
        return;
    }

    let mut group = c.benchmark_group("Binary TowerField inv");

    // Generate non-zero element
    let mut rng = rand::thread_rng();
    let non_zero_val = rng.gen::<u128>() | 1;
    let a = TowerFieldElement::new(non_zero_val, num_levels);

    assert!(
        !a.is_zero(),
        "Failed to generate non-zero element for inversion benchmark"
    );

    group.bench_with_input(
        BenchmarkId::from_parameter(format!("F{}", 1 << num_levels)),
        &num_levels,
        |bencher, _params| {
            bencher.iter(|| match black_box(a).inv() {
                Ok(inv) => black_box(inv),
                Err(BinaryFieldError::InverseOfZero) => panic!("Attempted to invert zero element"),
            });
        },
    );
    group.finish();
}

// Main benchmarking function that runs all benches
pub fn binary_ops_benchmarks(c: &mut Criterion) {
    let levels = [0, 1, 2, 3, 4, 5, 6, 7];

    for &level in &levels {
        binary_add_bench(c, level, level);
        binary_mul_bench(c, level, level);
        binary_pow_bench(c, level, 5);
        binary_inv_bench(c, level);
    }

    // Benchmarks for operations between different levels
    binary_add_bench(c, 0, 1); // F₁ + F₂ (1-bit field + 2-bit field)
    binary_add_bench(c, 0, 4); // F₁ + F₁₆ (1-bit field + 16-bit field)
    binary_add_bench(c, 1, 4); // F₂ + F₁₆ (2-bit field + 16-bit field)
    binary_add_bench(c, 2, 6); // F₄ + F₆₄ (4-bit field + 64-bit field)
    binary_add_bench(c, 0, 7); // F₁ + F₁₂₈ (1-bit field + 128-bit field)
    binary_add_bench(c, 4, 7); // F₁₆ + F₁₂₈ (16-bit field + 128-bit field)
    binary_add_bench(c, 6, 7); // F₆₄ + F₁₂₈ (64-bit field + 128-bit field)

    binary_mul_bench(c, 0, 1); // F₁ × F₂ (1-bit field × 2-bit field)
    binary_mul_bench(c, 0, 4); // F₁ × F₁₆ (1-bit field × 16-bit field)
    binary_mul_bench(c, 1, 4); // F₂ × F₁₆ (2-bit field × 16-bit field)
    binary_mul_bench(c, 2, 6); // F₄ × F₆₄ (4-bit field × 64-bit field)
    binary_mul_bench(c, 0, 7); // F₁ × F₁₂₈ (1-bit field × 128-bit field)
    binary_mul_bench(c, 4, 7); // F₁₆ × F₁₂₈ (16-bit field × 128-bit field)
    binary_mul_bench(c, 6, 7); // F₆₄ × F₁₂₈ (64-bit field × 128-bit field)
}
