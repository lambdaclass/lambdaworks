use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use lambdaworks_math::{
    field::fields::{
        fft_friendly::stark_252_prime_field::{
            MontgomeryConfigStark252PrimeField, Stark252PrimeField,
        },
        montgomery_backed_prime_fields::IsModulus,
    },
    unsigned_integer::{element::U256, montgomery::MontgomeryAlgorithms},
};

mod util;

pub fn starkfield_ops_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stark FP operations");
    let (x, y) = util::get_field_elements();

    group.bench_with_input("add", &(x.clone(), y.clone()), |bench, (x, y)| {
        bench.iter(|| x + y);
    });

    group.bench_with_input("mul", &(x.clone(), y.clone()), |bench, (x, y)| {
        bench.iter(|| x * y);
    });

    group.bench_with_input("pow by 1", &x.clone(), |bench, x| {
        bench.iter(|| x.pow(1_u64));
    });

    // The non-boxed constants are intentional as they are
    // normally computed at compile time.
    group.bench_with_input("sos_square", &x.clone(), |bench, x| {
        bench.iter(|| {
            MontgomeryAlgorithms::sos_square(
                black_box(x.value()),
                &<MontgomeryConfigStark252PrimeField as IsModulus<U256>>::MODULUS,
                &Stark252PrimeField::MU,
            )
        });
    });

    group.bench_with_input("square", &x.clone(), |bench, x| {
        bench.iter(|| x.square());
    });

    group.bench_with_input("square with pow", &x.clone(), |bench, x| {
        bench.iter(|| x.pow(2_u64));
    });

    group.bench_with_input("square with mul", &x.clone(), |bench, x| {
        bench.iter(|| x * x);
    });

    group.bench_with_input("pow", &(x.clone(), 5u64), |bench, (x, y)| {
        bench.iter(|| x.pow(*y));
    });

    group.bench_with_input("sub", &(x.clone(), y.clone()), |bench, (x, y)| {
        bench.iter(|| x - y);
    });

    group.bench_with_input("inv", &x, |bench, x| {
        bench.iter(|| x.inv());
    });

    group.bench_with_input("div", &(x.clone(), y.clone()), |bench, (x, y)| {
        bench.iter(|| x / y);
    });

    group.bench_with_input("eq", &(x.clone(), y), |bench, (x, y)| {
        bench.iter(|| x == y);
    });

    group.bench_with_input("sqrt", &x, |bench, x| {
        bench.iter(|| x.sqrt());
    });

    group.bench_with_input("sqrt squared", &(&x * &x), |bench, x| {
        bench.iter(|| x.sqrt());
    });
}

criterion_group!(starkfp, starkfield_ops_benchmarks);
criterion_main!(starkfp);
