use lambdaworks_fft::polynomial::FFTPoly;
use lambdaworks_gpu::cuda::{abstractions::state::CudaState, fft::ops::*};
use lambdaworks_math::{field::traits::RootsConfig, polynomial::Polynomial};

// WARN: These should always be fields supported by CUDA, else the last two benches will use CPU FFT.
use crate::util::{F, FE};

pub fn ordered_fft(input: &[FE], twiddles: &[FE]) {
    let cuda_state = CudaState::new(None).unwrap();
    fft(input, twiddles, &cuda_state).unwrap();
}

pub fn twiddles_generation(order: u64) {
    let cuda_state = CudaState::new(None).unwrap();
    gen_twiddles::<F>(order, RootsConfig::Natural, &cuda_state).unwrap();
}

pub fn bitrev_permute(input: &[FE]) {
    let cuda_state = CudaState::new(None).unwrap();
    bitrev_permutation::<F, FE>(input, &cuda_state).unwrap();
}

pub fn poly_evaluate_fft(poly: &Polynomial<FE>) {
    poly.evaluate_fft(1, None).unwrap();
}
pub fn poly_interpolate_fft(evals: &[FE]) {
    Polynomial::interpolate_fft(evals).unwrap();
}
