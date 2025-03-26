use ark_ff::Field;
use ark_std::UniformRand;
use ark_test_curves::starknet_fp::Fq as F;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use crate::utils::to_lambdaworks_vec;

pub mod utils;

const BENCHMARK_NAME: &str = "sqrt";

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = <rand_chacha::ChaCha20Rng as rand::SeedableRng>::seed_from_u64(9001);

    let mut arkworks_vec = Vec::new();
    for _i in 0..100 {
        let a = F::rand(&mut rng);
        let square = a * a;
        arkworks_vec.push(square);
    }

    // arkworks-ff
    {
        c.bench_function(
            &format!("{} 100 elements | ark-ff - ef8f758", BENCHMARK_NAME),
            |b| {
                b.iter(|| {
                    let mut iter = arkworks_vec.iter();

                    for _i in 0..100 {
                        let a = iter.next().unwrap();
                        black_box(black_box(a).sqrt());
                    }
                });
            },
        );
    }

    // lambdaworks-math
    {
        let lambdaworks_vec = to_lambdaworks_vec(&arkworks_vec);

        c.bench_function(
            &format!("{} 100 elements | lambdaworks", BENCHMARK_NAME,),
            |b| {
                b.iter(|| {
                    let mut iter = lambdaworks_vec.iter();

                    for _i in 0..100 {
                        let a = iter.next().unwrap();
                        black_box(black_box(a).sqrt());
                    }
                });
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
