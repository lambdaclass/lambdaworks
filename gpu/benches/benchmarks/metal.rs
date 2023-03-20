use criterion::Criterion;
use lambdaworks_gpu::fft::fft_metal::*;
use lambdaworks_math::{
    fft::bit_reversing::in_place_bit_reverse_permute,
    field::{element::FieldElement, traits::IsTwoAdicField},
    field::{test_fields::u32_test_field::U32TestField, traits::RootsConfig},
};
use rand::random;

type F = U32TestField;
type FE = FieldElement<F>;

fn gen_coeffs(pow: usize) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << pow);
    for _ in 0..result.capacity() {
        result.push(FE::new(random()));
    }
    result
}

pub fn metal_fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("metal_fft");

    for order in 20..=24 {
        let coeffs = gen_coeffs(order);
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        // the objective is to bench ordered FFT, including twiddles generation and Metal setup
        group.bench_with_input(
            format!("parallel_nr_2radix_2^{order}_coeffs"),
            &coeffs,
            |bench, coeffs| {
                bench.iter(|| {
                    // TODO: autoreleaspool hurts perf. by 2-3%. Search for an alternative
                    objc::rc::autoreleasepool(|| {
                        let coeffs = coeffs.clone();
                        let twiddles =
                            F::get_twiddles(order as u64, RootsConfig::BitReverse).unwrap();
                        let fft_metal = FFTMetalState::new(None).unwrap();
                        let command_buff_encoder = fft_metal
                            .setup_fft("radix2_dit_butterfly", &twiddles)
                            .unwrap();

                        let mut result = fft_metal
                            .execute_fft(&coeffs, command_buff_encoder)
                            .unwrap();

                        in_place_bit_reverse_permute(&mut result);
                    });
                });
            },
        );
    }

    group.finish();
}

pub fn metal_fft_twiddles_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("metal_fft");
    group.sample_size(10); // it becomes too slow with the default of 100

    for order in 2..=4 {
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        // the objective is to bench ordered FFT, including twiddles generation and Metal setup
        group.bench_with_input(
            format!("parallel_twiddle_factors_2^({order}-1)_elems"),
            &order,
            |bench, order| {
                bench.iter(|| {
                    // TODO: autoreleaspool hurts perf. by 2-3%. Search for an alternative
                    objc::rc::autoreleasepool(|| {
                        let metal_state = FFTMetalState::new(None).unwrap();
                        let _gpu_twiddles = metal_state.gen_twiddles::<F>(*order).unwrap();
                    });
                });
            },
        );
    }

    group.finish();
}
