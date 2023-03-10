use criterion::Criterion;
use lambdaworks_math::{
    fft::fft_cooley_tukey::{fft, inverse_fft},
    field::test_fields::u64_test_field::U64TestField,
};
use lambdaworks_math::{
    fft::{bit_reversing::*, fft_iterative::*},
    field::{element::FieldElement, traits::IsTwoAdicField},
};
use rand::random;

const MODULUS: u64 = 0xFFFFFFFF00000001;
type F = U64TestField<MODULUS>;
type FE = FieldElement<F>;

fn gen_coeffs(pow: usize) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << pow);
    for _ in 0..result.capacity() {
        result.push(FE::new(random()));
    }
    result
}

pub fn fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    group.sample_size(10); // it becomes too slow with the default of 100

    for pow in 20..=24 {
        let coeffs = gen_coeffs(pow);
        let evals = fft(&coeffs).unwrap();
        group.throughput(criterion::Throughput::Elements(1 << pow)); // info for criterion

        // the objective is to bench ordered FFT, including twiddles generation
        group.bench_with_input(
            format!("iterative_nr_2radix_2^{pow}_coeffs"),
            &coeffs,
            |bench, coeffs| {
                bench.iter(|| {
                    let mut coeffs = coeffs.clone();
                    let root = F::get_root_of_unity(coeffs.len().trailing_zeros() as u64).unwrap();
                    let mut twiddles = (0..coeffs.len() as u64)
                        .map(|i| root.pow(i))
                        .collect::<Vec<FE>>();
                    in_place_bit_reverse_permute(&mut twiddles);
                    in_place_nr_2radix_fft(&mut coeffs[..], &twiddles[..]);
                });
            },
        );
        group.bench_with_input(
            format!("iterative_rn_2radix_2^{pow}_coeffs"),
            &coeffs,
            |bench, coeffs| {
                bench.iter(|| {
                    let mut coeffs = coeffs.clone();
                    let root = F::get_root_of_unity(coeffs.len().trailing_zeros() as u64).unwrap();
                    let twiddles = (0..coeffs.len() as u64)
                        .map(|i| root.pow(i))
                        .collect::<Vec<FE>>();
                    in_place_bit_reverse_permute(&mut coeffs);
                    in_place_rn_2radix_fft(&mut coeffs[..], &twiddles[..]);
                });
            },
        );

        // fft() and ifft() implicitly calculate a root and twiddles.
        group.bench_with_input(
            format!("recursive_forward_2^{pow}_coeffs"),
            &coeffs,
            |bench, coeffs| {
                bench.iter(|| fft(coeffs));
            },
        );
        group.bench_with_input(
            format!("recursive_inverse_2^{pow}_coeffs"),
            &evals,
            |bench, evals| {
                bench.iter(|| inverse_fft(evals));
            },
        );
    }

    group.finish();
}
