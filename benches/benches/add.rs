use std::ops::Add;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use utils::generate_random_elements;

use crate::utils::to_lambdaworks_vec;

pub mod utils;

const BENCHMARK_NAME: &str = "add";

pub fn criterion_benchmark(c: &mut Criterion) {
    let arkworks_vec = generate_random_elements();

    // arkworks-ff
    {
        c.bench_function(
            &format!("{} 100000 elements | ark-ff - ef8f758", BENCHMARK_NAME),

            |b| {
                b.iter(|| {
                    let mut iter = arkworks_vec.iter();

                    for _i in 0..100000 {
                        let a = iter.next().unwrap();
                        let b = iter.next().unwrap();
                        black_box(black_box(&a).add(black_box(b)));
                    }
                });
            },
        );
    }

    // lambdaworks-math
    {
        let lambdaworks_vec = to_lambdaworks_vec(&arkworks_vec);

        c.bench_function(
            &format!("{} 100000 elements | lambdaworks", BENCHMARK_NAME,),
            |b| {
                println!("{}", lambdaworks_vec[1]);

                b.iter(|| {
                    let mut iter = lambdaworks_vec.iter();

                    for _i in 0..100000 {
                        let a = iter.next().unwrap();
                        let b = iter.next().unwrap();
                        black_box(black_box(&a).add(black_box(b)));
                    }
                });
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
