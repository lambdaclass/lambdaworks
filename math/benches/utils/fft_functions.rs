#![allow(dead_code)] // clippy has false positive in benchmarks
use lambdaworks_math::fft::{
    bit_reversing::in_place_bit_reverse_permute,
    fft_iterative::{in_place_nr_2radix_fft, in_place_rn_2radix_fft},
    polynomial::FFTPoly,
    roots_of_unity::get_twiddles,
};

use lambdaworks_math::{field::traits::RootsConfig, polynomial::Polynomial};

use super::stark252_utils::{F, FE};

pub fn ordered_fft_nr(input: &[FE], twiddles: &[FE]) {
    let mut input = input.to_vec();
    in_place_nr_2radix_fft(&mut input, twiddles);
    in_place_bit_reverse_permute(&mut input);
}

pub fn ordered_fft_rn(input: &[FE], twiddles: &[FE]) {
    let mut input = input.to_vec();
    in_place_bit_reverse_permute(&mut input);
    in_place_rn_2radix_fft(&mut input, twiddles);
}

pub fn twiddles_generation(order: u64) {
    get_twiddles::<F>(order, RootsConfig::Natural).unwrap();
}

pub fn bitrev_permute(input: &[FE]) {
    let mut input = input.to_vec();
    in_place_bit_reverse_permute(&mut input);
}

pub fn poly_evaluate_fft(poly: &Polynomial<FE>) {
    poly.evaluate_fft(1, None).unwrap();
}
pub fn poly_interpolate_fft(evals: &[FE]) {
    Polynomial::interpolate_fft(evals).unwrap();
}
