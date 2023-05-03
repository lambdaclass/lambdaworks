use crate::metal::abstractions::{errors::MetalError, state::MetalState};
use lambdaworks_math::{
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        traits::RootsConfig,
    },
    polynomial::Polynomial,
};

use super::ops::*;

pub fn evaluate_fft_metal(
    coeffs: &[FieldElement<Stark252PrimeField>],
) -> Result<Vec<FieldElement<Stark252PrimeField>>, MetalError>
where
    F: IsFFTField,
{
    let state = MetalState::new(None).unwrap();

    let order = coeffs.len().trailing_zeros();
    let twiddles = gen_twiddles(order.into(), RootsConfig::BitReverse, &state)?;

    fft(coeffs, &twiddles, &state)
}

/// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
/// factors. This is considered to be the inverse operation of [evaluate_fft_metal()].
pub fn interpolate_fft_metal(
    fft_evals: &[FieldElement<Stark252PrimeField>],
) -> Result<Polynomial<FieldElement<Stark252PrimeField>>, MetalError> {
    let metal_state = MetalState::new(None).unwrap();

    let order = fft_evals.len().trailing_zeros();
    let twiddles = gen_twiddles(order.into(), RootsConfig::BitReverseInversed, &metal_state)?;

    let coeffs = fft(fft_evals, &twiddles, &metal_state)?;

    let scale_factor = FieldElement::from(fft_evals.len() as u64).inv();
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}
