use std::{ops::Add, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use utils::generate_random_elements;

use crate::utils::to_lambdaworks_vec;

pub mod utils;

const BENCHMARK_NAME: &str = "add";

pub fn criterion_benchmark(c: &mut Criterion) {
    let arkworks_vec = generate_random_elements(2000000);
    let a = &arkworks_vec[0].clone();
    let d = &arkworks_vec[1].clone();

    // arkworks-ff
    {
        c.bench_function(
            &format!("{} 2 elements | ark-ff - ef8f758", BENCHMARK_NAME),

            |b| {
                b.iter(
                    || {
                        black_box(
                            black_box(&a).add(black_box(d))
                        );
                    }
                )
            },
        );
    }

    // lambdaworks-math
    {
        let lambdaworks_vec = to_lambdaworks_vec(&arkworks_vec);

        let a = &lambdaworks_vec[0].clone();
        let d = &lambdaworks_vec[1].clone();

        c.bench_function(
            &format!("{} 1M elements | lambdaworks", BENCHMARK_NAME,),
            |b| {
                b.iter(|| {
                    black_box(black_box(&a).add(black_box(d)));
                });
            },
        );
    }
}

criterion_group!{
    name = benches;
    config = Criterion::default()
        .significance_level(0.01)
        .measurement_time(Duration::from_secs(15))
        .sample_size(300);
    targets = criterion_benchmark
}
criterion_main!(benches);
