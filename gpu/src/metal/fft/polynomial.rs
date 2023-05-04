use crate::metal::abstractions::{errors::MetalError, state::MetalState};
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },
    polynomial::Polynomial,
};

use super::ops::*;

pub fn evaluate_fft_metal<F>(input: &[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, MetalError>
where
    F: IsFFTField,
{
    let state = MetalState::new(None).unwrap();

    let order = input.len().trailing_zeros();
    let twiddles = gen_twiddles(order.into(), RootsConfig::BitReverse, &state)?;

    fft(input, &twiddles, &state)
}

/// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
/// factors. This is considered to be the inverse operation of [evaluate_fft_metal()].
pub fn interpolate_fft_metal<F>(
    fft_evals: &[FieldElement<F>],
) -> Result<Polynomial<FieldElement<F>>, MetalError>
where
    F: IsFFTField,
{
    let metal_state = MetalState::new(None).unwrap();

    // fft() can zero-pad the coeffs if there aren't 2^k of them (k being any integer).
    // TODO: twiddle factors need to be handled with too much care, the FFT API shouldn't accept
    // invalid twiddle factor collections. A better solution is needed.
    let order = fft_evals.len().next_power_of_two().trailing_zeros();

    let twiddles = gen_twiddles(order.into(), RootsConfig::BitReverseInversed, &metal_state)?;

    let coeffs = fft(fft_evals, &twiddles, &metal_state)?;

    let scale_factor = FieldElement::from(fft_evals.len() as u64).inv();
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}
