//! Criterion benchmark for evaluate_offset_fft optimization (PR #1099)
//!
//! Compares three implementations:
//! - original: poly.scale() + evaluate_fft()
//! - optimized: evaluate_offset_fft()
//! - with_buffer: evaluate_offset_fft_with_buffer()

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
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

// ============================================================================
// Stark252
// ============================================================================

type Stark252FE = FieldElement<Stark252PrimeField>;

fn rand_stark252_poly(order: u64, rng: &mut StdRng) -> Polynomial<Stark252FE> {
    let mut coeffs = Vec::with_capacity(1 << order);
    for _ in 0..coeffs.capacity() {
        let rand_big = UnsignedInteger {
            limbs: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        };
        coeffs.push(Stark252FE::new(rand_big));
    }
    Polynomial::new(&coeffs)
}

// ============================================================================
// Goldilocks Hybrid
// ============================================================================

type GoldilocksHybridFE = FieldElement<Goldilocks64HybridField>;

fn rand_goldilocks_hybrid_poly(order: u64, rng: &mut StdRng) -> Polynomial<GoldilocksHybridFE> {
    let mut coeffs = Vec::with_capacity(1 << order);
    for _ in 0..coeffs.capacity() {
        coeffs.push(GoldilocksHybridFE::from(rng.gen::<u64>()));
    }
    Polynomial::new(&coeffs)
}

// ============================================================================
// Goldilocks Classic
// ============================================================================

type GoldilocksClassicFE = FieldElement<Goldilocks64Field>;

fn rand_goldilocks_classic_poly(order: u64, rng: &mut StdRng) -> Polynomial<GoldilocksClassicFE> {
    let mut coeffs = Vec::with_capacity(1 << order);
    for _ in 0..coeffs.capacity() {
        coeffs.push(GoldilocksClassicFE::from(rng.gen::<u64>()));
    }
    Polynomial::new(&coeffs)
}

// ============================================================================
// Benchmarks
// ============================================================================

fn stark252_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_offset_fft/Stark252");

    for order in [10u64, 12, 14] {
        let mut rng = StdRng::seed_from_u64(42);
        let poly = rand_stark252_poly(order, &mut rng);
        let offset = Stark252FE::from(12345u64);

        group.bench_with_input(BenchmarkId::new("original", 1 << order), &(), |b, _| {
            b.iter(|| {
                let scaled = poly.scale(&offset);
                Polynomial::evaluate_fft::<Stark252PrimeField>(&scaled, 1, None).unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("optimized", 1 << order), &(), |b, _| {
            b.iter(|| {
                Polynomial::evaluate_offset_fft::<Stark252PrimeField>(&poly, 1, None, &offset)
                    .unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("with_buffer", 1 << order), &(), |b, _| {
            let len = poly.coeff_len().next_power_of_two();
            let mut buffer = Vec::with_capacity(len);
            b.iter(|| {
                Polynomial::evaluate_offset_fft_with_buffer::<Stark252PrimeField>(
                    &poly,
                    1,
                    None,
                    &offset,
                    &mut buffer,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

fn goldilocks_hybrid_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_offset_fft/Goldilocks_Hybrid");

    for order in [10u64, 12, 14] {
        let mut rng = StdRng::seed_from_u64(42);
        let poly = rand_goldilocks_hybrid_poly(order, &mut rng);
        let offset = GoldilocksHybridFE::from(12345u64);

        group.bench_with_input(BenchmarkId::new("original", 1 << order), &(), |b, _| {
            b.iter(|| {
                let scaled = poly.scale(&offset);
                Polynomial::evaluate_fft::<Goldilocks64HybridField>(&scaled, 1, None).unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("optimized", 1 << order), &(), |b, _| {
            b.iter(|| {
                Polynomial::evaluate_offset_fft::<Goldilocks64HybridField>(&poly, 1, None, &offset)
                    .unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("with_buffer", 1 << order), &(), |b, _| {
            let len = poly.coeff_len().next_power_of_two();
            let mut buffer = Vec::with_capacity(len);
            b.iter(|| {
                Polynomial::evaluate_offset_fft_with_buffer::<Goldilocks64HybridField>(
                    &poly,
                    1,
                    None,
                    &offset,
                    &mut buffer,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

fn goldilocks_classic_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_offset_fft/Goldilocks_Classic");

    for order in [10u64, 12, 14] {
        let mut rng = StdRng::seed_from_u64(42);
        let poly = rand_goldilocks_classic_poly(order, &mut rng);
        let offset = GoldilocksClassicFE::from(12345u64);

        group.bench_with_input(BenchmarkId::new("original", 1 << order), &(), |b, _| {
            b.iter(|| {
                let scaled = poly.scale(&offset);
                Polynomial::evaluate_fft::<Goldilocks64Field>(&scaled, 1, None).unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("optimized", 1 << order), &(), |b, _| {
            b.iter(|| {
                Polynomial::evaluate_offset_fft::<Goldilocks64Field>(&poly, 1, None, &offset)
                    .unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("with_buffer", 1 << order), &(), |b, _| {
            let len = poly.coeff_len().next_power_of_two();
            let mut buffer = Vec::with_capacity(len);
            b.iter(|| {
                Polynomial::evaluate_offset_fft_with_buffer::<Goldilocks64Field>(
                    &poly,
                    1,
                    None,
                    &offset,
                    &mut buffer,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = evaluate_offset_fft;
    config = Criterion::default().sample_size(50);
    targets =
        goldilocks_hybrid_benchmarks,
        goldilocks_classic_benchmarks,
        stark252_benchmarks,
);

criterion_main!(evaluate_offset_fft);
