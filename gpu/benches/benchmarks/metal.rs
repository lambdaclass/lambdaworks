use criterion::Criterion;
use lambdaworks_gpu::metal::{
    abstractions::state::MetalState,
    fft::{ops::*, polynomial::*},
};
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    field::traits::RootsConfig, polynomial::Polynomial, unsigned_integer::element::UnsignedInteger,
};
use rand::random;

type F = Stark252PrimeField;
type FE = FieldElement<F>;
const INPUT_SET: [u64; 4] = [21, 22, 23, 24];

fn rand_field_elements(order: u64) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        let rand_big = UnsignedInteger { limbs: random() };
        result.push(FE::new(rand_big));
    }
    result
}

pub fn metal_fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ordered FFT");
    group.sample_size(10); // too slow otherwise

    for order in INPUT_SET {
        group.throughput(criterion::Throughput::Elements(1 << order));

        let input = rand_field_elements(order);
        let metal_state = MetalState::new(None).unwrap();
        let twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverse, &metal_state).unwrap();

        group.bench_with_input(
            "Parallel (Metal)",
            &(input, twiddles),
            |bench, (input, twiddles)| {
                bench.iter(|| {
                    // TODO: autoreleaspool hurts perf. by 2-3%. Search for an alternative
                    objc::rc::autoreleasepool(|| {
                        let metal_state = MetalState::new(None).unwrap();
                        fft(input, twiddles, &metal_state).unwrap();
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

        group.bench_with_input("Parallel (Metal)", &order, |bench, order| {
            bench.iter(|| {
                // TODO: autoreleaspool hurts perf. by 2-3%. Search for an alternative
                objc::rc::autoreleasepool(|| {
                    let metal_state = MetalState::new(None).unwrap();
                    gen_twiddles::<F>(*order, RootsConfig::Natural, &metal_state).unwrap();
                });
            });
        });
    }

    group.finish();
}

pub fn metal_bitrev_permutation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit-reverse permutation");
    group.sample_size(10); // it becomes too slow with the default of 100

    for order in INPUT_SET {
        let coeffs = rand_field_elements(order);
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        group.bench_with_input("Parallel (Metal)", &coeffs, |bench, coeffs| {
            bench.iter(|| {
                // TODO: autoreleaspool hurts perf. by 2-3%. Search for an alternative
                objc::rc::autoreleasepool(|| {
                    let metal_state = MetalState::new(None).unwrap();
                    bitrev_permutation::<F, _>(coeffs, &metal_state).unwrap();
                });
            });
        });
    }

    group.finish();
}

pub fn metal_poly_interpolate_fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial interpolation");
    group.sample_size(10); // it becomes too slow with the default of 100

    for order in INPUT_SET {
        let evals = rand_field_elements(order);

        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        group.bench_with_input("Parallel FFT (Metal)", &evals, |bench, evals| {
            bench.iter(|| {
                // TODO: autoreleaspool hurts perf. by 2-3%. Search for an alternative
                objc::rc::autoreleasepool(|| {
                    interpolate_fft_metal(evals).unwrap();
                });
            });
        });
    }

    group.finish();
}

pub fn metal_poly_evaluate_fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial evaluation");
    group.sample_size(10); // too slow otherwise

    for order in INPUT_SET {
        let poly = Polynomial::new(&rand_field_elements(order));
        group.throughput(criterion::Throughput::Elements(1 << order));

        group.bench_with_input("Parallel FFT (Metal)", &poly, |bench, poly| {
            bench.iter(|| {
                // TODO: autoreleaspool hurts perf. by 2-3%. Search for an alternative
                objc::rc::autoreleasepool(|| {
                    evaluate_fft_metal(poly).unwrap();
                });
            });
        });
    }

    group.finish();
}
