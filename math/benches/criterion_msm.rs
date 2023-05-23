use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::curve::BLS12381Curve, traits::IsEllipticCurve,
    },
    field::traits::IsField,
    msm::{naive, pippenger},
    unsigned_integer::element::UnsignedInteger,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

type F = <BLS12381Curve as IsEllipticCurve>::BaseField;
type FP = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
type UI = UnsignedInteger<6>;

pub fn generate_cs_and_points(msm_size: usize) -> (Vec<UI>, Vec<FP>) {
    // We use a seeded rng so the benchmarks are reproducible.
    let mut rng = StdRng::seed_from_u64(42);

    let g = BLS12381Curve::generator();

    let cs: Vec<_> = (0..msm_size)
        .map(|_| F::from_base_type(UI::from_limbs(rng.gen())))
        .collect();

    let points: Vec<_> = (0..msm_size)
        .map(|_| g.operate_with_self(F::from_base_type(UI::from_limbs(rng.gen()))))
        .collect();

    (cs, points)
}

pub fn msm_benchmarks_with_size(
    c: &mut Criterion,
    cs: &[UI],
    points: &[FP],
    window_sizes: &[usize],
) {
    assert_eq!(cs.len(), points.len());
    let msm_size = cs.len();

    let mut group = c.benchmark_group(format!("MSM benchmarks with size {msm_size}"));

    group.bench_function("Naive", |bench| {
        bench.iter(|| black_box(naive::msm(cs, points)));
    });

    for &window_size in window_sizes {
        group.bench_function(
            BenchmarkId::new("Sequential Pippenger", window_size),
            |bench| {
                bench.iter(|| black_box(pippenger::msm_with(cs, points, window_size)));
            },
        );

        group.bench_function(
            BenchmarkId::new("Parallel Pippenger", window_size),
            |bench| {
                bench.iter(|| black_box(pippenger::parallel_msm_with(cs, points, window_size)));
            },
        );
    }
}

pub fn run_benchmarks(c: &mut Criterion) {
    let exponents = 1..=10;
    let window_sizes = vec![1, 2, 4, 8, 12];

    for exp in exponents {
        let msm_size = 1 << exp;
        let (cs, points) = generate_cs_and_points(msm_size);

        msm_benchmarks_with_size(c, &cs, &points, &window_sizes);
    }
}

criterion_group!(msm, run_benchmarks);
criterion_main!(msm);
