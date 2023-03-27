use criterion::Criterion;
use lambdaworks_gpu::metal::{abstractions::state::MetalState, fft::ops::*};
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsTwoAdicField},
    field::{test_fields::u32_test_field::U32TestField, traits::RootsConfig},
};
use rand::random;

type F = U32TestField;
type FE = FieldElement<F>;

fn gen_coeffs(order: usize) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        result.push(FE::new(random()));
    }
    result
}

pub fn metal_fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT");

    for order in 21..=24 {
        let coeffs = gen_coeffs(order);
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        // the objective is to bench ordered FFT, including twiddles generation and Metal setup
        group.bench_with_input(
            format!("Parallel NR radix2 FFT for 2^{order} elements"),
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

pub fn metal_fft_twiddles_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT twiddles generation");
    group.sample_size(10); // it becomes too slow with the default of 100

    for order in 21..=24 {
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        group.bench_with_input(
            format!("Parallel twiddles generation for 2^({order}-1) elements"),
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

    for order in 21..=24 {
        let coeffs = gen_coeffs(order);
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        group.bench_with_input(
            format!("Parallel bitrev permutation for 2^{order} elements"),
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
