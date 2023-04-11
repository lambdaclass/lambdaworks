use criterion::Criterion;
use lambdaworks_math::{
    fft::{bit_reversing::*, fft_iterative::*},
    field::{element::FieldElement, traits::IsTwoAdicField},
    field::{fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::RootsConfig},
    polynomial::Polynomial,
    unsigned_integer::element::UnsignedInteger,
};
use rand::random;

type F = Stark252PrimeField;
type FE = FieldElement<F>;
const INPUT_SET: [u64; 6] = [4, 5, 6, 7, 21, 22];

fn gen_coeffs(order: u64) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        let rand_big = UnsignedInteger { limbs: random() };

        result.push(FE::new(rand_big));
    }
    result
}

pub fn fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ordered FFT");
    group.sample_size(10); // too slow otherwise

    for order in INPUT_SET {
        let coeffs = gen_coeffs(order);
        group.throughput(criterion::Throughput::Elements(1 << order));

        // the objective is to bench ordered FFT, including twiddles generation
        group.bench_with_input(
            format!("Sequential from NR radix2"),
            &coeffs,
            |bench, coeffs| {
                bench.iter(|| {
                    let mut coeffs = coeffs.clone();
                    let twiddles = F::get_twiddles(order as u64, RootsConfig::BitReverse).unwrap();
                    in_place_nr_2radix_fft(&mut coeffs, &twiddles);
                    in_place_bit_reverse_permute(&mut coeffs);
                });
            },
        );
        group.bench_with_input(
            format!("Sequential from RN radix2"),
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
    let mut group = c.benchmark_group("FFT twiddles generation");
    group.sample_size(10); // too slow otherwise

    for order in INPUT_SET {
        group.throughput(criterion::Throughput::Elements(1 << (order - 1)));

        group.bench_with_input(format!("Sequential"), &order, |bench, order| {
            bench.iter(|| {
                F::get_twiddles(*order as u64, RootsConfig::Natural).unwrap();
            });
        });
    }

    group.finish();
}

pub fn bitrev_permutation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit-reverse permutation");

    for order in INPUT_SET {
        let coeffs = gen_coeffs(order);
        group.throughput(criterion::Throughput::Elements(1 << order));

        group.bench_with_input(format!("Sequential"), &coeffs, |bench, coeffs| {
            bench.iter(|| {
                let mut coeffs = coeffs.clone();
                in_place_bit_reverse_permute(&mut coeffs);
            });
        });
    }

    group.finish();
}

pub fn poly_interpolate_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial interpolation");
    group.sample_size(10); // too slow otherwise

    for order in 4..=7 {
        // too slow for big inputs.
        let xs = gen_coeffs(order);
        let ys = gen_coeffs(order);

        group.throughput(criterion::Throughput::Elements(1 << order));

        group.bench_with_input(
            format!("Sequential lagrange"),
            &(xs, ys),
            |bench, (xs, ys)| {
                bench.iter(|| {
                    Polynomial::interpolate(xs, ys);
                });
            },
        );
    }

    group.finish();
}

pub fn poly_interpolate_fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial interpolation");
    group.sample_size(10); // too slow otherwise

    for order in INPUT_SET {
        let evals = gen_coeffs(order);

        group.throughput(criterion::Throughput::Elements(1 << order));

        group.bench_with_input(format!("Sequential FFT"), &evals, |bench, evals| {
            bench.iter(|| {
                Polynomial::interpolate_fft(evals).unwrap();
            });
        });
    }

    group.finish();
}
