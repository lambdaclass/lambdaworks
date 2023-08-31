#![allow(dead_code)] // clippy has false positive in benchmarks
use core::hint::black_box;
use lambdaworks_math::field::traits::RootsConfig;

mod utils;
use utils::fft_functions;
use utils::stark252_utils::{rand_field_elements, rand_poly, twiddles};

const SIZE_ORDER: u64 = 10;

#[inline(never)]
fn seq_fft_benchmarks_rn() {
    let mut input = rand_field_elements(SIZE_ORDER);
    let twiddles_nat = twiddles(SIZE_ORDER, RootsConfig::Natural);

    fft_functions::ordered_fft_rn(black_box(&mut input), black_box(&twiddles_nat));
}

#[inline(never)]
fn seq_fft_benchmarks_nr() {
    let mut input = rand_field_elements(SIZE_ORDER);
    let twiddles_bitrev = twiddles(SIZE_ORDER, RootsConfig::BitReverse);

    fft_functions::ordered_fft_nr(black_box(&mut input), black_box(&twiddles_bitrev));
}

#[inline(never)]
fn seq_twiddles_generation_natural_benchmarks() {
    fft_functions::twiddles_generation(black_box(SIZE_ORDER), black_box(RootsConfig::Natural));
}

#[inline(never)]
fn seq_twiddles_generation_natural_inversed_benchmarks() {
    fft_functions::twiddles_generation(
        black_box(SIZE_ORDER),
        black_box(RootsConfig::NaturalInversed),
    );
}

#[inline(never)]
fn seq_twiddles_generation_bitrev_benchmarks() {
    fft_functions::twiddles_generation(black_box(SIZE_ORDER), black_box(RootsConfig::BitReverse));
}

#[inline(never)]
fn seq_twiddles_generation_bitrev_inversed_benchmarks() {
    fft_functions::twiddles_generation(
        black_box(SIZE_ORDER),
        black_box(RootsConfig::BitReverseInversed),
    );
}

#[inline(never)]
fn seq_bitrev_permutation_benchmarks() {
    let mut input = rand_field_elements(SIZE_ORDER);
    fft_functions::bitrev_permute(black_box(&mut input));
}

#[inline(never)]
fn seq_poly_evaluation_benchmarks() {
    let poly = rand_poly(SIZE_ORDER);
    let _ = black_box(fft_functions::poly_evaluate_fft(black_box(&poly)));
}

#[inline(never)]
fn seq_poly_interpolation_benchmarks() {
    let evals = rand_field_elements(SIZE_ORDER);
    fft_functions::poly_interpolate_fft(black_box(&evals));
}

#[cfg(not(any(feature = "metal", feature = "cuda")))]
iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = seq_fft_benchmarks_nr,
    seq_fft_benchmarks_rn,
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
    functions = seq_fft_benchmarks_nr,
    seq_fft_benchmarks_rn,
    seq_twiddles_generation_natural_benchmarks,
    seq_twiddles_generation_natural_inversed_benchmarks,
    seq_twiddles_generation_bitrev_benchmarks,
    seq_twiddles_generation_bitrev_inversed_benchmarks,
    seq_bitrev_permutation_benchmarks,
);
