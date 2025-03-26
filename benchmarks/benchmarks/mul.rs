use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::ops::Mul;
use utils::generate_random_elements;

use crate::utils::to_lambdaworks_vec;

pub mod utils;

const BENCHMARK_NAME: &str = "mul";

pub fn criterion_benchmark(c: &mut Criterion) {
    let arkworks_vec = generate_random_elements(20000);

    // arkworks-ff
    {
        c.bench_function(
            &format!(
                "{} 10K elements | ark-ff - commit: ef8f758 ",
                BENCHMARK_NAME
            ),
            |b| {
                b.iter(|| {
                    let mut iter = arkworks_vec.iter();

                    for _i in 0..10000 {
                        let a = iter.next().unwrap();
                        let b = iter.next().unwrap();
                        black_box(black_box(a).mul(black_box(b)));
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

                    for _i in 0..10000 {
                        let a = iter.next().unwrap();
                        let b = iter.next().unwrap();
                        black_box(black_box(&a).mul(black_box(b)));
                    }
                });
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
