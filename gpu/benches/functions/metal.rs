use lambdaworks_fft::polynomial::FFTPoly;
use lambdaworks_gpu::metal::{abstractions::state::MetalState, fft::ops::*};
use lambdaworks_math::{field::traits::RootsConfig, polynomial::Polynomial};

// WARN: These should always be fields supported by Metal, else the last two benches will use CPU FFT.
use crate::util::{F, FE};

pub fn ordered_fft(input: &[FE], twiddles: &[FE]) {
    let metal_state = MetalState::new(None).unwrap();
    fft(input, twiddles, &metal_state).unwrap();
}

pub fn twiddles_generation(order: u64) {
    let metal_state = MetalState::new(None).unwrap();
    gen_twiddles::<F>(order, RootsConfig::Natural, &metal_state).unwrap();
}

pub fn bitrev_permute(input: &[FE]) {
    let metal_state = MetalState::new(None).unwrap();
    bitrev_permutation::<F, FE>(input, &metal_state).unwrap();
}

pub fn poly_evaluate_fft(poly: &Polynomial<FE>) {
    poly.evaluate_fft(1, None);
}
pub fn poly_interpolate_fft(evals: &[FE]) {
    Polynomial::interpolate_fft(evals).unwrap();
}
