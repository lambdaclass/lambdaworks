use criterion::{criterion_group, criterion_main, Criterion};
use lambdaworks_gpu::metal::abstractions::state::MetalState;
use lambdaworks_math::fft::gpu::metal::ops::gen_twiddles;
use lambdaworks_math::field::traits::RootsConfig;

use utils::stark252_utils::F;

mod utils;
use utils::metal_functions;
use utils::stark252_utils;

const SIZE_ORDERS: [u64; 4] = [21, 22, 23, 24];

fn fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ordered FFT");

    for (order, input) in SIZE_ORDERS
        .iter()
        .zip(SIZE_ORDERS.map(stark252_utils::rand_field_elements))
    {
        let metal_state = MetalState::new(None).unwrap();
        let twiddles = gen_twiddles::<F>(*order, RootsConfig::BitReverse, &metal_state).unwrap();

        group.throughput(criterion::Throughput::Elements(input.len() as u64));
        group.bench_with_input(
            "Parallel (Metal)",
            &(input, twiddles),
            |bench, (input, twiddles)| {
                bench.iter_with_large_drop(|| {
                    metal_functions::ordered_fft(input, twiddles);
                });
            },
        );
    }

    group.finish();
}

fn twiddles_generation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT twiddles generation");

    for order in SIZE_ORDERS {
        group.throughput(criterion::Throughput::Elements(1 << (order - 1)));
        group.bench_with_input("Parallel (Metal)", &order, |bench, order| {
            bench.iter_with_large_drop(|| {
                metal_functions::twiddles_generation(*order);
            });
        });
    }

    group.finish();
}

fn bitrev_permutation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit-reverse permutation");

    for input in SIZE_ORDERS.map(stark252_utils::rand_field_elements) {
        group.throughput(criterion::Throughput::Elements(input.len() as u64));
        group.bench_with_input("Parallel (Metal)", &input, |bench, input| {
            bench.iter_with_large_drop(|| {
                metal_functions::bitrev_permute(input);
            });
        });
    }

    group.finish();
}

fn poly_evaluation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial");

    for poly in SIZE_ORDERS.map(stark252_utils::rand_poly) {
        group.throughput(criterion::Throughput::Elements(
            poly.coefficients().len() as u64
        ));
        group.bench_with_input("evaluate_fft_metal", &poly, |bench, poly| {
            bench.iter_with_large_drop(|| {
                metal_functions::poly_evaluate_fft(poly);
            });
        });
    }

    group.finish();
}

fn poly_interpolation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial");

    for evals in SIZE_ORDERS.map(stark252_utils::rand_field_elements) {
        group.throughput(criterion::Throughput::Elements(evals.len() as u64));
        group.bench_with_input("interpolate_fft_metal", &evals, |bench, evals| {
            bench.iter_with_large_drop(|| {
                metal_functions::poly_interpolate_fft(evals);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = metal;
    config = Criterion::default().sample_size(10);
    targets =
        fft_benchmarks,
        twiddles_generation_benchmarks,
        bitrev_permutation_benchmarks,
        poly_evaluation_benchmarks,
        poly_interpolation_benchmarks,
);

criterion_main!(metal);
