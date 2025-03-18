use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use lambdaworks_math::field::fields::binary::field::{BinaryFieldError, TowerFieldElement};
use rand::Rng;

pub type F = TowerFieldElement;

// Generate a random TowerFieldElement with specified level
pub fn rand_element(num_level: usize) -> F {
    let mut rng = rand::thread_rng();
    let safe_level = if num_level > 7 { 7 } else { num_level };
    let max_value = if safe_level == 7 {
        u128::MAX
    } else {
        (1u128 << (1 << safe_level)) - 1
    };
    F::new(rng.gen_range(0..=max_value), safe_level)
}

// // Generate a vector of random TowerFieldElement elements
// pub fn rand_vector(size: usize, num_level: usize) -> Vec<F> {
//     (0..size).map(|_| rand_element(num_level)).collect()
// }

// /// Generate pairs of random field elements for benchmarks
// pub fn rand_field_elements(num: usize, level: usize) -> Vec<(F, F)> {
//     (0..num)
//         .map(|_| (rand_element(level), rand_element(level)))
//         .collect()
// }

// Benchmark addition for specific levels
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

// Benchmark multiplication for specific levels
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

// Benchmark exponentiation for specific levels
fn binary_pow_bench(c: &mut Criterion, num_levels: usize, exponent: u32) {
    // Only makes sense for levels > 0
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

// Benchmark inversion for specific levels
fn binary_inv_bench(c: &mut Criterion, num_levels: usize) {
    // Only makes sense for levels > 0
    if num_levels == 0 {
        return;
    }

    let mut group = c.benchmark_group("Binary TowerField inv");

    // Generate a definitely non-zero element
    // For binary fields, any element with lowest bit set to 1 is guaranteed to be non-zero
    let mut rng = rand::thread_rng();
    let safe_level = if num_levels > 7 { 7 } else { num_levels };
    let max_value = if safe_level == 7 {
        u128::MAX
    } else {
        (1u128 << (1 << safe_level)) - 1
    };

    // Ensure at least the lowest bit is set to 1, guaranteeing a non-zero value
    let non_zero_val = rng.gen_range(0..=max_value) | 1;
    let a = F::new(non_zero_val, safe_level);

    // Double-check that the element is not zero before proceeding
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

// Main benchmarking function that runs all tests
pub fn binary_ops_benchmarks(c: &mut Criterion) {
    // Benchmarks for operations with same level
    let levels = [0, 1, 2, 3, 4, 5, 6, 7];

    for &level in &levels {
        binary_add_bench(c, level, level);
        binary_mul_bench(c, level, level);

        if level > 0 {
            binary_pow_bench(c, level, 5);
            binary_inv_bench(c, level);
        }
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
