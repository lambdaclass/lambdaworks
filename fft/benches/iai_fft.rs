#![allow(dead_code)] // clippy has false positive in benchmarks
use iai_callgrind::black_box;
use lambdaworks_fft::roots_of_unity::get_twiddles;
use lambdaworks_math::field::traits::RootsConfig;

mod functions;
mod util;

const SIZE_ORDER: u64 = 10;

#[inline(never)]
fn seq_fft_benchmarks() {
    let input = util::rand_field_elements(SIZE_ORDER);
    let twiddles_bitrev = get_twiddles(SIZE_ORDER, RootsConfig::BitReverse).unwrap();
    let twiddles_nat = get_twiddles(SIZE_ORDER, RootsConfig::Natural).unwrap();

    functions::ordered_fft_nr(black_box(&input), black_box(&twiddles_bitrev));
    functions::ordered_fft_rn(black_box(&input), black_box(&twiddles_nat));
}

#[inline(never)]
fn seq_twiddles_generation_natural_benchmarks() {
    functions::twiddles_generation(black_box(SIZE_ORDER), RootsConfig::Natural);
}

#[inline(never)]
fn seq_twiddles_generation_natural_inversed_benchmarks() {
    functions::twiddles_generation(black_box(SIZE_ORDER), RootsConfig::NaturalInversed);
}

#[inline(never)]
fn seq_twiddles_generation_bitrev_benchmarks() {
    functions::twiddles_generation(black_box(SIZE_ORDER), RootsConfig::BitReverse);
}

#[inline(never)]
fn seq_twiddles_generation_bitrev_inversed_benchmarks() {
    functions::twiddles_generation(black_box(SIZE_ORDER), RootsConfig::BitReverseInversed);
}

#[inline(never)]
fn seq_bitrev_permutation_benchmarks() {
    let input = util::rand_field_elements(SIZE_ORDER);
    functions::bitrev_permute(black_box(&input));
}

#[inline(never)]
fn seq_poly_evaluation_benchmarks() {
    let poly = util::rand_poly(SIZE_ORDER);
    functions::poly_evaluate_fft(black_box(&poly));
}

#[inline(never)]
fn seq_poly_interpolation_benchmarks() {
    let evals = util::rand_field_elements(SIZE_ORDER);
    functions::poly_interpolate_fft(black_box(&evals));
}

#[cfg(not(any(feature = "metal", feature = "cuda")))]
iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = seq_fft_benchmarks,
    seq_twiddles_generation_natural_benchmarks,
    seq_twiddles_generation_natural_inversed_benchmarks,
    seq_twiddles_generation_bitrev_benchmarks,
    seq_twiddles_generation_bitrev_inversed_benchmarks,
    seq_bitrev_permutation_benchmarks,
    seq_poly_evaluation_benchmarks,
    seq_poly_interpolation_benchmarks
);

#[cfg(any(feature = "metal", feature = "cuda"))]
iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = seq_fft_benchmarks,
    seq_twiddles_generation_natural_benchmarks,
    seq_twiddles_generation_natural_inversed_benchmarks,
    seq_twiddles_generation_bitrev_benchmarks,
    seq_twiddles_generation_bitrev_inversed_benchmarks,
    seq_bitrev_permutation_benchmarks,
);
