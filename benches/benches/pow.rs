use ark_ff::Field;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use utils::generate_random_elements;

use crate::utils::to_lambdaworks_vec;

pub mod utils;

const BENCHMARK_NAME: &str = "pow";

pub fn criterion_benchmark(c: &mut Criterion) {
    let arkworks_vec = generate_random_elements(20000);

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9001);

    let mut v_ints = Vec::new();
    for _i in 0..10000 {
        v_ints.push(rng.gen::<u64>());
    }

    // arkworks-ff
    {
        c.bench_function(
            &format!("{} 10K elements | ark-ff - ef8f758", BENCHMARK_NAME),
            |b| {
                b.iter(|| {
                    let mut iter = arkworks_vec.iter();
                    let mut iter_ints = v_ints.iter();

                    for _i in 0..10000 {
                        let a = iter.next().unwrap();
                        let exp = iter_ints.next().unwrap();
                        black_box(black_box(&a).pow(black_box(&[*exp])));
                    }
                });
            },
        );
    }

    // lambdaworks-math
    {
        let lambdaworks_vec = to_lambdaworks_vec(&arkworks_vec);

        c.bench_function(
            &format!("{} 10K elements | lambdaworks", BENCHMARK_NAME,),
            |b| {
                b.iter(|| {
                    let mut iter = lambdaworks_vec.iter();
                    let mut iter_ints = v_ints.iter();

                    for _i in 0..10000 {
                        let a = iter.next().unwrap();
                        let exp = iter_ints.next().unwrap();
                        black_box(black_box(&a).pow(black_box(*exp)));
                    }
                });
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
