use criterion::Criterion;
use lambdaworks_math::{
    fft::{bit_reversing::*, fft_iterative::*},
    field::{element::FieldElement, traits::IsTwoAdicField},
    field::{test_fields::u64_test_field::U64TestField, traits::RootsConfig},
};
use rand::random;

type F = U64TestField;
type FE = FieldElement<F>;

fn gen_coeffs(order: usize) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        result.push(FE::new(random()));
    }
    result
}

pub fn fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT");
    group.sample_size(10); // it becomes too slow with the default

    for order in 21..=24 {
        let coeffs = gen_coeffs(order);
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        // the objective is to bench ordered FFT, including twiddles generation
        group.bench_with_input(
            format!("Sequential NR radix2 for 2^{order} elements"),
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
            format!("Sequential RN radix2 for 2^{order} elements"),
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
    group.sample_size(10); // it becomes too slow with the default

    for order in 21..=24 {
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        group.bench_with_input(
            format!("Sequential twiddles generation for 2^({order}-1) elements"),
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
    let mut group = c.benchmark_group("Bit-reverse permutation");

    for order in 21..=24 {
        let coeffs = gen_coeffs(order);
        group.throughput(criterion::Throughput::Elements(1 << order)); // info for criterion

        group.bench_with_input(
            format!("Sequential bitrev permutation for 2^{order} elements"),
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

pub fn poly_interpolate_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial interpolation");
    group.sample_size(10); // too slow otherwise

    for order in 4..=7 {
        // too slow for big inputs.
        let xs = gen_coeffs(order);
        let ys = gen_coeffs(order);

        group.throughput(criterion::Throughput::Elements(1 << order));

        group.bench_with_input(
            format!("Sequential lagrange polynomial interpolation"),
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

        group.bench_with_input(
            format!("Sequential FFT polynomial interpolation"),
            &evals,
            |bench, evals| {
                bench.iter(|| {
                    Polynomial::interpolate_fft(evals).unwrap();
                });
            },
        );
    }

    group.finish();
}
