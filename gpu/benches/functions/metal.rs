use lambdaworks_gpu::metal::{
    abstractions::state::MetalState,
    fft::{ops::*, polynomial::*},
};
use lambdaworks_math::{field::traits::RootsConfig, polynomial::Polynomial};

use crate::util::{F, FE};

pub fn ordered_fft(input: &[FE], twiddles: &[FE]) {
    // TODO: autoreleasepool hurts perf. by 2-3%. Search for an alternative
    objc::rc::autoreleasepool(|| {
        let metal_state = MetalState::new(None).unwrap();
        fft(input, twiddles, &metal_state).unwrap();
    });
}

pub fn twiddles_generation(order: u64) {
    // TODO: autoreleasepool hurts perf. by 2-3%. Search for an alternative
    objc::rc::autoreleasepool(|| {
        let metal_state = MetalState::new(None).unwrap();
        gen_twiddles::<F>(order, RootsConfig::Natural, &metal_state).unwrap();
    });
}

pub fn bitrev_permute(input: &[FE]) {
    // TODO: autoreleasepool hurts perf. by 2-3%. Search for an alternative
    objc::rc::autoreleasepool(|| {
        let metal_state = MetalState::new(None).unwrap();
        bitrev_permutation(input, &metal_state).unwrap();
    });
}

pub fn poly_evaluate_fft(poly: &Polynomial<FE>) {
    // TODO: autoreleasepool hurts perf. by 2-3%. Search for an alternative
    objc::rc::autoreleasepool(|| {
        let metal_state = MetalState::new(None).unwrap();
        poly.evaluate_fft_metal(&metal_state).unwrap();
    });
}
pub fn poly_interpolate_fft(evals: &[FE]) {
    // TODO: autoreleasepool hurts perf. by 2-3%. Search for an alternative
    objc::rc::autoreleasepool(|| {
        let metal_state = MetalState::new(None).unwrap();
        Polynomial::interpolate_fft_metal(evals, &metal_state).unwrap();
    });
}
