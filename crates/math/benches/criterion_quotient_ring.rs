use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::dilithium_prime::DilithiumField;
use lambdaworks_math::polynomial::quotient_ring::PolynomialRingElement;

type FE = FieldElement<DilithiumField>;
type R256 = PolynomialRingElement<DilithiumField, 256>;

const Q: u64 = 8380417;

/// Generate a deterministic full-degree (256-coeff) ring element.
fn make_ring_element(seed: u64) -> R256 {
    let coeffs: Vec<FE> = (0..256)
        .map(|i| FE::from((i * seed.wrapping_add(17).wrapping_mul(31 + i)) % Q))
        .collect();
    R256::new(&coeffs)
}

fn quotient_ring_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("quotient_ring_mul_256");

    let a = make_ring_element(7);
    let b = make_ring_element(13);

    group.bench_function("schoolbook", |bench| {
        bench.iter(|| black_box(&a).mul_schoolbook(black_box(&b)))
    });

    group.bench_function("ntt_standard", |bench| {
        bench.iter(|| black_box(&a).mul_ntt(black_box(&b)))
    });

    group.bench_function("negacyclic_ntt", |bench| {
        bench.iter(|| black_box(&a).mul_negacyclic_ntt(black_box(&b)))
    });

    group.finish();
}

criterion_group!(
    name = quotient_ring;
    config = Criterion::default();
    targets = quotient_ring_benchmarks
);
criterion_main!(quotient_ring);
