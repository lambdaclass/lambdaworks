#![allow(dead_code)] // clippy has false positive in benchmarks
use iai_callgrind::black_box;
use lambdaworks_math::{fft::cpu::roots_of_unity::get_twiddles, field::traits::RootsConfig};

mod utils;
use utils::fft_functions;
use utils::stark252_utils;

const SIZE_ORDER: u64 = 10;

#[inline(never)]
fn seq_fft_benchmarks() {
    let input = stark252_utils::rand_field_elements(SIZE_ORDER);
    let twiddles_bitrev = get_twiddles(SIZE_ORDER, RootsConfig::BitReverse).unwrap();
    let twiddles_nat = get_twiddles(SIZE_ORDER, RootsConfig::Natural).unwrap();

    fft_functions::ordered_fft_nr(black_box(&input), black_box(&twiddles_bitrev));
    fft_functions::ordered_fft_rn(black_box(&input), black_box(&twiddles_nat));
}

#[inline(never)]
fn seq_twiddles_generation_benchmarks() {
    fft_functions::twiddles_generation(black_box(SIZE_ORDER));
}

#[inline(never)]
fn seq_bitrev_permutation_benchmarks() {
    let input = stark252_utils::rand_field_elements(SIZE_ORDER);
    fft_functions::bitrev_permute(black_box(&input));
}

#[inline(never)]
fn seq_poly_evaluation_benchmarks() {
    let poly = stark252_utils::rand_poly(SIZE_ORDER);
    fft_functions::poly_evaluate_fft(black_box(&poly));
}

#[inline(never)]
fn seq_poly_interpolation_benchmarks() {
    let evals = stark252_utils::rand_field_elements(SIZE_ORDER);
    fft_functions::poly_interpolate_fft(black_box(&evals));
}

#[cfg(not(any(feature = "metal", feature = "cuda")))]
iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = seq_fft_benchmarks,
    seq_twiddles_generation_benchmarks,
    seq_bitrev_permutation_benchmarks,
    seq_poly_evaluation_benchmarks,
    seq_poly_interpolation_benchmarks
);

#[cfg(any(feature = "metal", feature = "cuda"))]
iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = seq_fft_benchmarks,
    seq_twiddles_generation_benchmarks,
    seq_bitrev_permutation_benchmarks,
);
