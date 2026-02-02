use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lambdaworks_math::unsigned_integer::element::{UnsignedInteger, U256, U384};
use rand::random;

fn rand_u256_pairs(num: usize) -> Vec<(U256, U256)> {
    (0..num)
        .map(|_| {
            // Use smaller values to avoid overflow in wrapping mul
            let a = UnsignedInteger {
                limbs: [0, 0, random(), random()],
            };
            let b = UnsignedInteger {
                limbs: [0, 0, random(), random()],
            };
            (a, b)
        })
        .collect()
}

fn rand_u384_pairs(num: usize) -> Vec<(U384, U384)> {
    (0..num)
        .map(|_| {
            // Use smaller values to avoid overflow
            let a = UnsignedInteger {
                limbs: [0, 0, 0, random(), random(), random()],
            };
            let b = UnsignedInteger {
                limbs: [0, 0, 0, random(), random(), random()],
            };
            (a, b)
        })
        .collect()
}

fn unsigned_integer_mul_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("UnsignedInteger mul");

    // U256 multiplication
    for size in [100, 1000, 10000] {
        let pairs = rand_u256_pairs(size);
        group.bench_with_input(BenchmarkId::new("U256", size), &pairs, |b, pairs| {
            b.iter(|| {
                for (a, b) in pairs {
                    black_box(black_box(a) * black_box(b));
                }
            })
        });
    }

    // U384 multiplication
    for size in [100, 1000, 10000] {
        let pairs = rand_u384_pairs(size);
        group.bench_with_input(BenchmarkId::new("U384", size), &pairs, |b, pairs| {
            b.iter(|| {
                for (a, b) in pairs {
                    black_box(black_box(a) * black_box(b));
                }
            })
        });
    }

    group.finish();
}

fn unsigned_integer_mul_full_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("UnsignedInteger::mul (hi,lo)");

    // U256::mul returning (hi, lo) - no overflow possible
    for size in [100, 1000, 10000] {
        let pairs: Vec<(U256, U256)> = (0..size)
            .map(|_| {
                (
                    UnsignedInteger { limbs: random() },
                    UnsignedInteger { limbs: random() },
                )
            })
            .collect();
        group.bench_with_input(BenchmarkId::new("U256", size), &pairs, |b, pairs| {
            b.iter(|| {
                for (a, b) in pairs {
                    black_box(U256::mul(black_box(a), black_box(b)));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(
    unsigned_integer_benches,
    unsigned_integer_mul_benchmarks,
    unsigned_integer_mul_full_benchmarks,
);
criterion_main!(unsigned_integer_benches);
