use criterion::{criterion_group, criterion_main, Criterion};
use lambdaworks_math::{fft::roots_of_unity::get_twiddles, field::traits::RootsConfig};

mod functions;
mod stark252_utils;

const SIZE_ORDERS: [u64; 4] = [21, 22, 23, 24];

pub fn fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ordered FFT");

    for order in SIZE_ORDERS {
        group.throughput(criterion::Throughput::Elements(1 << order));

        let input = stark252_utils::rand_field_elements(order);
        let twiddles_bitrev = get_twiddles(order, RootsConfig::BitReverse).unwrap();
        let twiddles_nat = get_twiddles(order, RootsConfig::Natural).unwrap();

        group.bench_with_input(
            "Sequential from NR radix2",
            &(&input, twiddles_bitrev),
            |bench, (input, twiddles)| {
                bench.iter(|| {
                    functions::fft::ordered_fft_nr(input, twiddles);
                });
            },
        );
        group.bench_with_input(
            "Sequential from RN radix2",
            &(input, twiddles_nat),
            |bench, (input, twiddles)| {
                bench.iter(|| {
                    functions::fft::ordered_fft_rn(input, twiddles);
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
        group.bench_with_input("Sequential", &order, |bench, order| {
            bench.iter_with_large_drop(|| {
                functions::fft::twiddles_generation(*order);
            });
        });
    }

    group.finish();
}

fn bitrev_permutation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit-reverse permutation");

    for input in SIZE_ORDERS.map(stark252_utils::rand_field_elements) {
        group.throughput(criterion::Throughput::Elements(input.len() as u64));
        group.bench_with_input("Sequential", &input, |bench, input| {
            bench.iter_with_large_drop(|| {
                functions::fft::bitrev_permute(input);
            });
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
                functions::fft::poly_evaluate_fft(poly);
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
                functions::fft::poly_interpolate_fft(evals);
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
