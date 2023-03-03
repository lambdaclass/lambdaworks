use super::util::u64_bench_field::*;
use criterion::{BatchSize, Criterion};
use lambdaworks_math::{
    fft::{bit_reversing::*, fft_iterative::*},
    field::{element::FieldElement, traits::IsTwoAdicField},
};
use rand::random;

const MODULUS: u64 = 0xFFFFFFFF00000001;
type F = U64BenchField<MODULUS>;
type FE = FieldElement<F>;

fn gen_coeffs(pow: usize) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << pow);
    for _ in 0..result.capacity() {
        result.push(FE::new(random()));
    }
    result
}

pub fn ntt_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt");
    group.sample_size(10); // it becomes too slow with the default of 100

    for pow in 20..=24 {
        let coeffs = gen_coeffs(pow);
        group.throughput(criterion::Throughput::Elements(1 << pow)); // info for criterion
        let root = F::get_root_of_unity(coeffs.len().trailing_zeros() as u64).unwrap();
        let twiddles = (0..coeffs.len() as u64)
            .map(|i| root.pow(i))
            .collect::<Vec<FE>>();

        group.bench_with_input::<String, _, (&Vec<FE>, &Vec<FE>)>(
            format!("nr_2radix_sequential_2^{}_coeffs", pow),
            &(&coeffs, &twiddles),
            |bench, &(coeffs, twiddles)| {
                bench.iter_batched(
                    || {
                        // input setup, doesn't get timed
                        let mut twiddles = twiddles.clone();
                        in_place_bit_reverse_permute(&mut twiddles);
                        (coeffs.clone(), twiddles) // clone needed because algos are in-place
                    },
                    |(mut coeffs, twiddles)| {
                        // this does get timed
                        in_place_nr_2radix_ntt(&mut coeffs[..], &twiddles[..]);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_with_input::<String, _, (&Vec<FE>, &Vec<FE>)>(
            format!("rn_2radix_sequential_2^{}_coeffs", pow),
            &(&coeffs, &twiddles),
            |bench, &(coeffs, twiddles)| {
                bench.iter_batched(
                    || {
                        let mut coeffs = coeffs.clone();
                        in_place_bit_reverse_permute(&mut coeffs);
                        (coeffs, twiddles)
                    },
                    |(mut coeffs, twiddles)| {
                        in_place_rn_2radix_ntt(&mut coeffs[..], &twiddles[..]);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}
