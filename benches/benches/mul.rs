use std::ops::Mul;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

const BENCHMARK_NAME: &str = "mul";

pub fn criterion_benchmark(c: &mut Criterion) {
    // arkworks-ff
    {
        use ark_ff::fields::PrimeField;
        use ark_test_curves::starknet_fp::Fq as F;

        let num_1 = F::from_be_bytes_mod_order(
            b"0x03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
        );
        let num_2 = F::from_be_bytes_mod_order(
            b"0x0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
        );

        c.bench_function(
            &format!(
                "{} | ark-ff - branch: faster-benchmarks-and-starknet-field",
                BENCHMARK_NAME
            ),
            |b| {
                b.iter(|| {
                    black_box(black_box(&num_1).mul(black_box(&num_2)));
                });
            },
        );
    }

    // lambdaworks-math
    {
        use lambdaworks_math::field::{
            element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        };

        let num_1 = FieldElement::<Stark252PrimeField>::from_hex(
            "03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
        )
        .unwrap();
        let num_2 = FieldElement::<Stark252PrimeField>::from_hex(
            "0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
        )
        .unwrap();

        c.bench_function(&format!("{} | lambdaworks", BENCHMARK_NAME,), |b| {
            b.iter(|| {
                black_box(black_box(&num_1).mul(black_box(&num_2)));
            });
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
