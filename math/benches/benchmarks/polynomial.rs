use const_random::const_random;
use criterion::{black_box, Criterion};
use lambdaworks_math::field::fields::u64_prime_field::U64FieldElement;
use lambdaworks_math::polynomial::Polynomial;
use rand::Rng;

// Mersenne prime numbers
// https://www.math.utah.edu/~pa/math/mersenne.html
const PRIMES: [u64; 39] = [
    13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941,
    11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091, 756839, 859433, 1257787,
    1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457,
];

const MODULUS: u64 = PRIMES[const_random!(usize) % PRIMES.len()];
type FE = U64FieldElement<MODULUS>;

pub fn polynomial_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial");

    group.bench_function("evaluate", |bench| {
        let coeffs = gen_fe_vec();
        let poly = Polynomial::new(&coeffs);
        let x = FE::new(rand::random::<u64>());
        bench.iter(|| poly.evaluate(black_box(&x)));
    });

    group.bench_function("evaluate_slice", |bench| {
        let coeffs = gen_fe_vec();
        let poly = Polynomial::new(&coeffs);
        let inputs = gen_fe_vec();
        bench.iter(|| poly.evaluate_slice(black_box(&inputs)));
    });

    group.bench_function("add", |bench| {
        let x_poly = Polynomial::new(&gen_fe_vec());
        let y_poly = Polynomial::new(&gen_fe_vec());
        bench.iter(|| black_box(&x_poly) + black_box(&y_poly));
    });

    group.bench_function("neg", |bench| {
        let x_poly = Polynomial::new(&gen_fe_vec());
        bench.iter(|| -black_box(x_poly.clone()));
    });

    group.bench_function("sub", |bench| {
        let x_poly = Polynomial::new(&gen_fe_vec());
        let y_poly = Polynomial::new(&gen_fe_vec());
        bench.iter(|| black_box(x_poly.clone()) - black_box(y_poly.clone()));
    });

    group.bench_function("mul", |bench| {
        let x_poly = Polynomial::new(&gen_fe_vec());
        let y_poly = Polynomial::new(&gen_fe_vec());
        bench.iter(|| black_box(x_poly.clone()) + black_box(y_poly.clone()));
    });

    group.bench_function("div", |bench| {
        let x_poly = Polynomial::new(&gen_fe_vec());
        let y_poly = Polynomial::new(&gen_fe_vec());
        bench.iter(|| black_box(x_poly.clone()) + black_box(y_poly.clone()));
    });
}

fn gen_fe_vec() -> Vec<FE> {
    let mut rng = rand::thread_rng();
    let size: u64 = rng.gen_range(1..=256);
    let mut result = vec![];

    for _ in 0..size {
        result.push(FE::new(rng.gen_range(0..=u64::MAX)));
    }

    result
}
