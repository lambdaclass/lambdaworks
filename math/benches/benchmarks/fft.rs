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
const INPUT_SET: [u64; 2] = [21, 22];

fn rand_field_elements(order: u64) -> Vec<FE> {
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
        group.throughput(criterion::Throughput::Elements(1 << order));

        let input = rand_field_elements(order);
        let twiddles_bitrev = F::get_twiddles(order, RootsConfig::BitReverse).unwrap();
        let twiddles_nat = F::get_twiddles(order, RootsConfig::Natural).unwrap();

        // the objective is to bench ordered FFT, that's why a bitrev permutation is added.
        group.bench_with_input("Sequential from NR radix2", &input, |bench, input| {
            bench.iter(|| {
                let mut input = input.clone();
                in_place_nr_2radix_fft(&mut input, &twiddles_bitrev);
                in_place_bit_reverse_permute(&mut input);
            });
        });
        group.bench_with_input("Sequential from RN radix2", &input, |bench, input| {
            bench.iter(|| {
                let mut input = input.clone();
                in_place_bit_reverse_permute(&mut input);
                in_place_rn_2radix_fft(&mut input, &twiddles_nat);
            });
        });
    }

    group.finish();
}

pub fn twiddles_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT twiddles generation");
    group.sample_size(10); // too slow otherwise

    for order in INPUT_SET {
        group.throughput(criterion::Throughput::Elements(1 << (order - 1)));

        group.bench_with_input("Sequential", &order, |bench, order| {
            bench.iter(|| {
                F::get_twiddles(*order, RootsConfig::Natural).unwrap();
            });
        });
    }

    group.finish();
}

pub fn bitrev_permutation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit-reverse permutation");

    for order in INPUT_SET {
        let coeffs = rand_field_elements(order);
        group.throughput(criterion::Throughput::Elements(1 << order));

        group.bench_with_input("Sequential", &coeffs, |bench, coeffs| {
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
        let xs = rand_field_elements(order);
        let ys = rand_field_elements(order);

        group.throughput(criterion::Throughput::Elements(1 << order));

        group.bench_with_input("Sequential Lagrange", &(xs, ys), |bench, (xs, ys)| {
            bench.iter(|| {
                Polynomial::interpolate(xs, ys);
            });
        });
    }

    group.finish();
}

pub fn poly_interpolate_fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial interpolation");
    group.sample_size(10); // too slow otherwise

    for order in INPUT_SET {
        let evals = rand_field_elements(order);

        group.throughput(criterion::Throughput::Elements(1 << order));

        group.bench_with_input("Sequential FFT", &evals, |bench, evals| {
            bench.iter(|| {
                Polynomial::interpolate_fft(evals).unwrap();
            });
        });
    }

    group.finish();
}

pub fn poly_evaluate_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial evaluation");
    group.sample_size(10); // too slow otherwise

    for order in 4..=7 {
        // too slow for big inputs.
        let poly = Polynomial::new(&rand_field_elements(order));
        let input = rand_field_elements(order);

        group.throughput(criterion::Throughput::Elements(1 << order));

        group.bench_with_input(
            "Sequential Horner",
            &(poly, input),
            |bench, (poly, input)| {
                bench.iter(|| {
                    poly.evaluate_slice(input);
                });
            },
        );
    }

    group.finish();
}

pub fn poly_evaluate_fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial evaluation");
    group.sample_size(10); // too slow otherwise

    for order in INPUT_SET {
        // too slow for big inputs.
        let poly = Polynomial::new(&rand_field_elements(order));

        group.throughput(criterion::Throughput::Elements(1 << order));

        group.bench_with_input("Sequential FFT", &poly, |bench, poly| {
            bench.iter(|| {
                poly.evaluate_fft().unwrap();
            });
        });
    }

    group.finish();
}
