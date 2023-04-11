use criterion::Criterion;
use lambdaworks_gpu::metal::{
    abstractions::state::MetalState,
    fft::{ops::*, polynomial::*},
};
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsTwoAdicField},
    field::{fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::RootsConfig},
    polynomial::Polynomial,
    unsigned_integer::element::UnsignedInteger,
};
use rand::random;

type F = Stark252PrimeField;
type FE = FieldElement<F>;
const INPUT_SET: [u64; 8] = [4, 5, 6, 7, 21, 22, 23, 24];

fn gen_coeffs(order: u64) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        let rand_big = UnsignedInteger { limbs: random() };

        result.push(FE::new(rand_big));
    }
    result
}

pub fn metal_fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT");
    group.sample_size(10); // too slow otherwise

    for order in INPUT_SET {
        let coeffs = gen_coeffs(order);
        group.throughput(criterion::Throughput::Elements(1 << order));

        // the objective is to bench ordered FFT, including twiddles generation and Metal setup
        group.bench_with_input(
            format!("Metal parallel NR radix2 FFT"),
            &coeffs,
            |bench, coeffs| {
                bench.iter(|| {
                    // TODO: autoreleaspool hurts perf. by 2-3%. Search for an alternative
                    objc::rc::autoreleasepool(|| {
                        let coeffs = coeffs.clone();
                        let metal_state = MetalState::new(None).unwrap();
                        let twiddles =
                            F::get_twiddles(order as u64, RootsConfig::BitReverse).unwrap();

                        fft(&coeffs, &twiddles, &metal_state).unwrap();
                    });
                });
            },
        );
    }

    group.finish();
}

pub fn metal_twiddles_gen_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT twiddles generation");
    group.sample_size(10); // too slow otherwise

    for order in INPUT_SET {
        group.throughput(criterion::Throughput::Elements(1 << (order - 1)));

        group.bench_with_input(
            format!("Metal parallel twiddles generation"),
            &order,
            |bench, order| {
                bench.iter(|| {
                    // TODO: autoreleaspool hurts perf. by 2-3%. Search for an alternative
                    objc::rc::autoreleasepool(|| {
                        let metal_state = MetalState::new(None).unwrap();
                        gen_twiddles::<F>(*order, RootsConfig::Natural, &metal_state).unwrap();
                    });
                });
            },
        );
    }

    group.finish();
}

pub fn metal_bitrev_permutation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit-reverse permutation");
    group.sample_size(10); // it becomes too slow with the default of 100

    for order in INPUT_SET {
        let coeffs = gen_coeffs(order);
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        group.bench_with_input(
            format!("Metal parallel bitrev permutation"),
            &coeffs,
            |bench, coeffs| {
                bench.iter(|| {
                    // TODO: autoreleaspool hurts perf. by 2-3%. Search for an alternative
                    objc::rc::autoreleasepool(|| {
                        let metal_state = MetalState::new(None).unwrap();
                        bitrev_permutation(coeffs, &metal_state).unwrap();
                    });
                });
            },
        );
    }

    group.finish();
}

pub fn metal_poly_interpolate_fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial interpolation");
    group.sample_size(10); // it becomes too slow with the default of 100

    for order in INPUT_SET {
        let evals = gen_coeffs(order);

        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        group.bench_with_input(
            format!("Metal FFT polynomial interpolation"),
            &evals,
            |bench, evals| {
                bench.iter(|| {
                    objc::rc::autoreleasepool(|| {
                        let metal_state = MetalState::new(None).unwrap();
                        Polynomial::interpolate_fft_metal(evals, &metal_state).unwrap();
                    });
                });
            },
        );
    }

    group.finish();
}
