use crate::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },
    polynomial::Polynomial,
};
use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::MetalState};

use super::ops::*;

pub fn evaluate_fft_metal<F>(coeffs: &[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, MetalError>
where
    F: IsFFTField,
{
    let state = MetalState::new(None)?;

    let order = coeffs.len().trailing_zeros();
    let twiddles = gen_twiddles(order.into(), RootsConfig::BitReverse, &state)?;

    fft(coeffs, &twiddles, &state)
}

/// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
/// factors. This is considered to be the inverse operation of [evaluate_fft_metal()].
pub fn interpolate_fft_metal<F>(
    fft_evals: &[FieldElement<F>],
) -> Result<Polynomial<FieldElement<F>>, MetalError>
where
    F: IsFFTField,
{
    let metal_state = MetalState::new(None)?;

    let order = fft_evals.len().trailing_zeros();
    let twiddles = gen_twiddles(order.into(), RootsConfig::BitReverseInversed, &metal_state)?;

    let coeffs = fft(fft_evals, &twiddles, &metal_state)?;

    let scale_factor = FieldElement::from(fft_evals.len() as u64).inv().unwrap();
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}
