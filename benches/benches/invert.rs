use ark_ff::Field;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use utils::generate_random_elements;

use crate::utils::to_lambdaworks_vec;

pub mod utils;

const BENCHMARK_NAME: &str = "invert";

pub fn criterion_benchmark(c: &mut Criterion) {
    let arkworks_vec = generate_random_elements(10000).to_vec();

    // arkworks-ff
    {
        c.bench_function(
            &format!("{} 10000 elements| ark-ff - ef8f758", BENCHMARK_NAME),
            |b| {
                b.iter_batched(
                    || arkworks_vec.clone(),
                    |mut v| {
                        for mut elem in v.iter_mut() {
                            black_box(black_box(&mut elem).inverse_in_place());
                        }
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    // lambdaworks-math
    {
        let lambdaworks_vec = to_lambdaworks_vec(&arkworks_vec);

        c.bench_function(
            &format!("{} 10000 elements | lambdaworks", BENCHMARK_NAME,),
            |b| {
                b.iter(|| {
                    for elem in lambdaworks_vec.iter() {
                        black_box(black_box(&elem).inv().unwrap());
                    }
                });
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
