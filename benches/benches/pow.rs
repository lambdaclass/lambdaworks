use ark_ff::Field;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use rand::RngCore;

const BENCHMARK_NAME: &str = "pow";

pub fn criterion_benchmark(c: &mut Criterion) {
    // arkworks-ff
    {
        use ark_std::{test_rng, UniformRand};
        use ark_test_curves::starknet_fp::Fq as F;

        let mut rng = test_rng();

        let mut v = Vec::new();
        for _i in 0..10000 {
            let a = F::rand(&mut rng);
            v.push(a);
        }

        let mut rng = rand::thread_rng();

        let mut v_ints = Vec::new();
        for _i in 0..10000 {
            v_ints.push(rng.gen::<u64>());
        }

        c.bench_function(
            &format!(
                "{} | ark-ff - branch: faster-benchmarks-and-starknet-field",
                BENCHMARK_NAME
            ),
            |b| {
                b.iter(|| {
                    let mut iter = v.iter();
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
        use lambdaworks_math::field::{
            element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        };
        let mut v = Vec::new();
        let mut buf = [0u8; 32];
        for _i in 0..10000 {
            rand::thread_rng().fill_bytes(&mut buf[..]);

            let a = FieldElement::<Stark252PrimeField>::from_bytes_be(&buf).unwrap();

            v.push(a);
        }

        let mut rng = rand::thread_rng();

        let mut v_ints = Vec::new();
        for _i in 0..10000 {
            v_ints.push(rng.gen::<u64>());
        }

        c.bench_function(&format!("{} | lambdaworks", BENCHMARK_NAME,), |b| {
            b.iter(|| {
                let mut iter = v.iter();
                let mut iter_ints = v_ints.iter();

                for _i in 0..10000 {
                    let a = iter.next().unwrap();
                    let exp = iter_ints.next().unwrap();
                    black_box(black_box(&a).pow(black_box(*exp)));
                }
            });
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
