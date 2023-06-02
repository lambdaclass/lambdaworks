use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::{
            fft_friendly::stark_252_prime_field::{
                MontgomeryConfigStark252PrimeField, Stark252PrimeField,
            },
            montgomery_backed_prime_fields::IsModulus,
        },
    },
    unsigned_integer::{element::U256, montgomery::MontgomeryAlgorithms},
};

mod util;

pub fn starkfield_ops_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stark FP operations");

    let x = FieldElement::<Stark252PrimeField>::from_hex(
        "0x03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
    )
    .unwrap();
    let y = FieldElement::<Stark252PrimeField>::from_hex(
        "0x0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
    )
    .unwrap();

    group.bench_with_input("add", &(x.clone(), y.clone()), |bench, (x, y)| {
        bench.iter(|| x + y);
    });

    group.bench_with_input("mul", &(x.clone(), y.clone()), |bench, (x, y)| {
        bench.iter(|| x * y);
    });

    group.bench_with_input("sos_square", &x.clone(), |bench, x| {
        bench.iter(|| {
            MontgomeryAlgorithms::sos_square(
                &x.value(),
                &<MontgomeryConfigStark252PrimeField as IsModulus<U256>>::MODULUS,
                &Stark252PrimeField::MU,
            )
        });
    });

    group.bench_with_input("sos_square_black_box", &x.clone(), |bench, x| {
        bench.iter(|| {
            MontgomeryAlgorithms::sos_square(
                black_box(&x.value()),
                &<MontgomeryConfigStark252PrimeField as IsModulus<U256>>::MODULUS,
                &Stark252PrimeField::MU,
            )
        });
    });

    group.bench_with_input("noop", &x.clone(), |bench, x| {
        bench.iter(|| x.pow(1_u64));
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
