use iai::black_box;
use lambdaworks_fft::roots_of_unity::get_twiddles;
use lambdaworks_math::field::traits::RootsConfig;

mod functions;
mod util;

const SIZE_ORDER: u64 = 4;

pub fn fft_benchmarks() {
    let input = util::rand_field_elements(SIZE_ORDER);
    let twiddles_bitrev = get_twiddles(SIZE_ORDER, RootsConfig::BitReverse).unwrap();
    let twiddles_nat = get_twiddles(SIZE_ORDER, RootsConfig::Natural).unwrap();

    functions::ordered_fft_nr(black_box(&input), black_box(&twiddles_bitrev));
    functions::ordered_fft_rn(black_box(&input), black_box(&twiddles_nat));
}

fn twiddles_generation_benchmarks() {
    functions::twiddles_generation(black_box(SIZE_ORDER));
}

fn bitrev_permutation_benchmarks() {
    let input = util::rand_field_elements(SIZE_ORDER);
    functions::bitrev_permute(black_box(&input));
}

fn poly_evaluation_benchmarks() {
    let poly = util::rand_poly(SIZE_ORDER);
    functions::poly_evaluate_fft(black_box(&poly));
}

fn poly_interpolation_benchmarks() {
    let evals = util::rand_field_elements(SIZE_ORDER);
    functions::poly_interpolate_fft(black_box(&evals));
}

iai::main!(
    fft_benchmarks,
    twiddles_generation_benchmarks,
    bitrev_permutation_benchmarks,
    poly_evaluation_benchmarks,
    poly_interpolation_benchmarks
);
