//#![allow(dead_code)] // clippy has false positive in benchmarks

use core::hint::black_box;

use lambdaworks_math::fft::{
    bit_reversing::in_place_bit_reverse_permute,
    fft_iterative::{in_place_nr_2radix_fft, in_place_rn_2radix_fft},
    polynomial::FFTPoly,
    roots_of_unity::get_twiddles,
};

use lambdaworks_math::{field::traits::RootsConfig, polynomial::Polynomial};

use super::fft_utils::{F, FE};

pub fn ordered_fft_nr(input: &mut [FE], twiddles: &[FE]) {
    in_place_nr_2radix_fft(input, twiddles);
}

pub fn ordered_fft_rn(input: &mut [FE], twiddles: &[FE]) {
    in_place_rn_2radix_fft(input, twiddles);
}

pub fn twiddles_generation(order: u64, config: RootsConfig) {
    get_twiddles::<F>(order, config).unwrap();
}

pub fn bitrev_permute(input: &mut [FE]) {
    in_place_bit_reverse_permute(input);
}

pub fn poly_evaluate_fft(poly: &Polynomial<FE>) -> Vec<FE> {
    poly.evaluate_fft(black_box(1), black_box(None)).unwrap()
}
pub fn poly_interpolate_fft(evals: &[FE]) {
    Polynomial::interpolate_fft(evals).unwrap();
}
