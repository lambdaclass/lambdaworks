use criterion::{criterion_group, criterion_main, Criterion};
use lambdaworks_gpu::cuda::{abstractions::state::CudaState, fft::ops::gen_twiddles};
use lambdaworks_math::field::traits::RootsConfig;

use crate::util::F;

mod functions;
mod util;

const SIZE_ORDERS: [u64; 4] = [21, 22, 23, 24];

fn fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ordered FFT");

    for (order, input) in SIZE_ORDERS
        .iter()
        .zip(SIZE_ORDERS.map(util::rand_field_elements))
    {
        let cuda_state = CudaState::new(None).unwrap();
        let twiddles = gen_twiddles::<F>(*order, RootsConfig::BitReverse, &cuda_state).unwrap();

        group.throughput(criterion::Throughput::Elements(input.len() as u64));
        group.bench_with_input(
            "Parallel (Cuda)",
            &(input, twiddles),
            |bench, (input, twiddles)| {
                bench.iter_with_large_drop(|| {
                    functions::cuda::ordered_fft(input, twiddles);
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
        group.bench_with_input("Parallel (Cuda)", &order, |bench, order| {
            bench.iter_with_large_drop(|| {
                functions::cuda::twiddles_generation(*order);
            });
        });
    }

    group.finish();
}

fn bitrev_permutation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit-reverse permutation");

    for input in SIZE_ORDERS.map(util::rand_field_elements) {
        group.throughput(criterion::Throughput::Elements(input.len() as u64));
        group.bench_with_input("Parallel (Cuda)", &input, |bench, input| {
            bench.iter_with_large_drop(|| {
                functions::cuda::bitrev_permute(input);
            });
        });
    }

    group.finish();
}

fn poly_evaluation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial evaluation");

    for poly in SIZE_ORDERS.map(util::rand_poly) {
        group.throughput(criterion::Throughput::Elements(
            poly.coefficients().len() as u64
        ));
        group.bench_with_input("Parallel (CUDA)", &poly, |bench, poly| {
            bench.iter_with_large_drop(|| {
                functions::cuda::poly_evaluate_fft(poly);
            });
        });
    }

    group.finish();
}

fn poly_interpolation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial interpolation");

    for evals in SIZE_ORDERS.map(util::rand_field_elements) {
        group.throughput(criterion::Throughput::Elements(evals.len() as u64));
        group.bench_with_input("Parallel (CUDA)", &evals, |bench, evals| {
            bench.iter_with_large_drop(|| {
                functions::cuda::poly_interpolate_fft(evals);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = cuda;
    config = Criterion::default().sample_size(10);
    targets =
        fft_benchmarks,
        twiddles_generation_benchmarks,
        bitrev_permutation_benchmarks,
        poly_evaluation_benchmarks,
        poly_interpolation_benchmarks,
);

criterion_main!(cuda);
