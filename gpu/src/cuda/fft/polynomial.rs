use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },
    polynomial::Polynomial,
};

use super::ops::fft;
use crate::cuda::abstractions::{errors::CudaError, state::CudaState};
use crate::cuda::fft::ops::gen_twiddles;

pub fn evaluate_fft_cuda<F>(coeffs: &[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, CudaError>
where
    F: IsFFTField,
    F::BaseType: Unpin,
{
    let state = CudaState::new()?;
    let order = log2(coeffs.len())?;
    let twiddles = gen_twiddles(order, RootsConfig::BitReverse, &state)?;

    fft(coeffs, &twiddles, &state)
}

/// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
/// factors. This is considered to be the inverse operation of [evaluate_fft_cuda()].
pub fn interpolate_fft_cuda<F>(
    fft_evals: &[FieldElement<F>],
) -> Result<Polynomial<FieldElement<F>>, CudaError>
where
    F: IsFFTField,
    F::BaseType: Unpin,
{
    let state = CudaState::new()?;

    // fft() can zero-pad the coeffs if there aren't 2^k of them (k being any integer).
    // TODO: twiddle factors need to be handled with too much care, the FFT API shouldn't accept
    // invalid twiddle factor collections. A better solution is needed.
    let order = log2(fft_evals.len())?;
    let twiddles = gen_twiddles(order, RootsConfig::BitReverseInversed, &state)?;

    let coeffs = fft(fft_evals, &twiddles, &state)?;

    let scale_factor = FieldElement::from(fft_evals.len() as u64).inv();
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}

// TODO: remove when fft works on non-multiple-of-two input length
fn log2(n: usize) -> Result<u64, CudaError> {
    if !n.is_power_of_two() {
        return Err(CudaError::InvalidOrder(n));
    }
    Ok(n.trailing_zeros() as u64)
}

#[cfg(test)]
mod tests {}
