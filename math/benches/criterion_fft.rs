#![allow(dead_code)] // clippy has false positive in benchmarks
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use lambdaworks_math::field::traits::RootsConfig;
use utils::fft_functions;
use utils::stark252_utils;

mod utils;

const SIZE_ORDERS: [u64; 4] = [21, 22, 23, 24];

pub fn fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ordered FFT");

    for order in SIZE_ORDERS {
        group.throughput(criterion::Throughput::Elements(1 << order));

        let input_nat = stark252_utils::rand_field_elements(order);
        let twiddles_nat = stark252_utils::twiddles(order, RootsConfig::Natural);
        let mut input_bitrev = input_nat.clone();
        stark252_utils::bitrev_permute(&mut input_bitrev);
        let twiddles_bitrev = stark252_utils::twiddles(order, RootsConfig::BitReverse);

        group.bench_with_input(
            "Sequential from NR radix2",
            &(input_nat, twiddles_bitrev),
            |bench, (input, twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut input| {
                        fft_functions::ordered_fft_nr(&mut input, twiddles);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        group.bench_with_input(
            "Sequential from RN radix2",
            &(input_bitrev, twiddles_nat),
            |bench, (input, twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut input| {
                        fft_functions::ordered_fft_rn(&mut input, twiddles);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn twiddles_generation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT twiddles generation");
    const CONFIGS: [(&str, RootsConfig); 4] = [
        ("natural", RootsConfig::Natural),
        ("natural inversed", RootsConfig::NaturalInversed),
        ("bit-reversed", RootsConfig::BitReverse),
        ("bit-reversed inversed", RootsConfig::BitReverseInversed),
    ];

    for order in SIZE_ORDERS {
        group.throughput(criterion::Throughput::Elements(1 << (order - 1)));
        for (name, config) in CONFIGS {
            group.bench_with_input(name, &(order, config), |bench, (order, config)| {
                bench.iter_with_large_drop(|| {
                    fft_functions::twiddles_generation(*order, *config);
                });
            });
        }
    }

    group.finish();
}

fn bitrev_permutation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit-reverse permutation");

    for input in SIZE_ORDERS.map(stark252_utils::rand_field_elements) {
        group.throughput(criterion::Throughput::Elements(input.len() as u64));
        group.bench_with_input("Sequential", &input, |bench, input| {
            bench.iter_batched(
                || input.clone(),
                |mut input| {
                    stark252_utils::bitrev_permute(&mut input);
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn poly_evaluation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial evaluation");

    for poly in SIZE_ORDERS.map(stark252_utils::rand_poly) {
        group.throughput(criterion::Throughput::Elements(
            poly.coefficients().len() as u64
        ));
        group.bench_with_input("Sequential FFT", &poly, |bench, poly| {
            bench.iter_with_large_drop(|| {
                fft_functions::poly_evaluate_fft(poly);
            });
        });
    }

    group.finish();
}

fn poly_interpolation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial interpolation");

    for evals in SIZE_ORDERS.map(stark252_utils::rand_field_elements) {
        group.throughput(criterion::Throughput::Elements(evals.len() as u64));
        group.bench_with_input("Sequential FFT", &evals, |bench, evals| {
            bench.iter_with_large_drop(|| {
                fft_functions::poly_interpolate_fft(evals);
            });
        });
    }

    group.finish();
}

#[cfg(not(any(feature = "metal", feature = "cuda")))]
criterion_group!(
    name = seq_fft;
    config = Criterion::default().sample_size(10);
    targets =
        fft_benchmarks,
        twiddles_generation_benchmarks,
        bitrev_permutation_benchmarks,
        poly_evaluation_benchmarks,
        poly_interpolation_benchmarks,
);

#[cfg(any(feature = "metal", feature = "cuda"))]
criterion_group!(
    name = seq_fft;
    config = Criterion::default().sample_size(10);
    targets =
        fft_benchmarks,
        twiddles_generation_benchmarks,
        bitrev_permutation_benchmarks,
);

criterion_main!(seq_fft);
