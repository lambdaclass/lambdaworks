use const_random::const_random;
use criterion::{black_box, Criterion};
use lambdaworks_math::fft::fft_cooley_tukey::{fft, inverse_fft};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::test_fields::u64_test_field::U64TestField;
use rand::Rng;

// Mersenne prime numbers
// https://www.math.utah.edu/~pa/math/mersenne.html
const PRIMES: [u64; 39] = [
    13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941,
    11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091, 756839, 859433, 1257787,
    1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457,
];

const MODULUS: u64 = PRIMES[const_random!(usize) % PRIMES.len()];
type FE = FieldElement<U64TestField<MODULUS>>;

pub fn fft_benchmark(c: &mut Criterion) {
    c.bench_function("fft", |bench| {
        let mut rng = rand::thread_rng();
        let coeffs_size = 1 << rng.gen_range(1..8);
        let mut coeffs: Vec<FE> = vec![];

        for _ in 0..coeffs_size {
            coeffs.push(FE::new(rng.gen_range(1..=u64::MAX)));
        }

        bench.iter(|| fft(black_box(&coeffs)));
    });
}

pub fn inverse_fft_benchmark(c: &mut Criterion) {
    c.bench_function("inverse_fft", |bench| {
        let mut rng = rand::thread_rng();
        let coeffs_size = 1 << rng.gen_range(1..8);
        let mut coeffs: Vec<FE> = vec![];

        for _ in 0..coeffs_size {
            coeffs.push(FE::new(rng.gen_range(1..=u64::MAX)));
        }

        let evaluations = fft(&coeffs).unwrap();

        bench.iter(|| inverse_fft(black_box(&evaluations)));
    });
}
