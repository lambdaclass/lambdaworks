use criterion::Criterion;
use lambdaworks_math::{
    fft::{bit_reversing::*, fft_iterative::*},
    field::{element::FieldElement, traits::IsTwoAdicField},
    field::{test_fields::u64_test_field::U64TestField, traits::RootsConfig},
};
use rand::random;

const MODULUS: u64 = 0xFFFFFFFF00000001;
type F = U64TestField<MODULUS>;
type FE = FieldElement<F>;

fn gen_coeffs(order: usize) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        result.push(FE::new(random()));
    }
    result
}

pub fn fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    group.sample_size(10); // it becomes too slow with the default

    for order in 21..=24 {
        let coeffs = gen_coeffs(order);
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        // the objective is to bench ordered FFT, including twiddles generation
        group.bench_with_input(
            format!("iterative_nr_2radix_2^{order}_coeffs"),
            &coeffs,
            |bench, coeffs| {
                bench.iter(|| {
                    let mut coeffs = coeffs.clone();
                    let twiddles = F::get_twiddles(order as u64, RootsConfig::BitReverse).unwrap();
                    in_place_nr_2radix_fft(&mut coeffs, &twiddles);
                });
            },
        );
        group.bench_with_input(
            format!("iterative_rn_2radix_2^{order}_coeffs"),
            &coeffs,
            |bench, coeffs| {
                bench.iter(|| {
                    let mut coeffs = coeffs.clone();
                    let twiddles = F::get_twiddles(order as u64, RootsConfig::Natural).unwrap();
                    in_place_bit_reverse_permute(&mut coeffs);
                    in_place_rn_2radix_fft(&mut coeffs, &twiddles);
                });
            },
        );
    }

    group.finish();
}

pub fn twiddles_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    group.sample_size(10); // it becomes too slow with the default

    for order in 21..=24 {
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        group.bench_with_input(
            format!("sequential_twiddle_gen_2^({order}-1)_elems"),
            &order,
            |bench, order| {
                bench.iter(|| {
                    F::get_twiddles(*order as u64, RootsConfig::Natural).unwrap();
                });
            },
        );
    }

    group.finish();
}

pub fn bitrev_permutation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");

    for order in 21..=24 {
        let coeffs = gen_coeffs(order);
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        group.bench_with_input(
            format!("sequential_bitrev_permutation_2^{order}_elems"),
            &coeffs,
            |bench, coeffs| {
                bench.iter(|| {
                    let mut coeffs = coeffs.clone();
                    in_place_bit_reverse_permute(&mut coeffs);
                });
            },
        );
    }

    group.finish();
}
