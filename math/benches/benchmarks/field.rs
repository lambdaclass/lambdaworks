use const_random::const_random;
use criterion::{black_box, Criterion};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64FieldElement;

// Mersenne prime numbers
// https://www.math.utah.edu/~pa/math/mersenne.html
const PRIMES: [u64; 39] = [
    13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941,
    11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091, 756839, 859433, 1257787,
    1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457,
];

pub fn u64_benchmark(c: &mut Criterion) {
    const MODULUS: u64 = PRIMES[const_random!(usize) % PRIMES.len()];
    let mut group = c.benchmark_group("u64");

    group.bench_function("add", |bench| {
        let x: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        let y: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        bench.iter(|| black_box(x) + black_box(y));
    });

    group.bench_function("mul", |bench| {
        let x: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        let y: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        bench.iter(|| black_box(x) * black_box(y));
    });

    group.bench_function("pow", |bench| {
        let x: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        let y: u64 = 5;
        bench.iter(|| black_box(x).pow(black_box(y)));
    });

    group.bench_function("sub", |bench| {
        let x: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        let y: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        bench.iter(|| black_box(x) - black_box(y));
    });

    group.bench_function("inv", |bench| {
        let x: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        bench.iter(|| black_box(x).inv());
    });

    group.bench_function("div", |bench| {
        let x: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        let y: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        bench.iter(|| black_box(x) / black_box(y));
    });

    group.bench_function("eq", |bench| {
        let x: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        let y: U64FieldElement<MODULUS> = FieldElement::new(rand::random());
        bench.iter(|| black_box(x) == black_box(y));
    });
}
