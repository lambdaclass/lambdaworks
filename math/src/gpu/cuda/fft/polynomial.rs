use crate::{
    fft::errors::FFTError,
    field::{
        element::FieldElement,
        errors::FieldError,
        traits::{IsFFTField, RootsConfig},
    },
    polynomial::Polynomial,
};

use super::ops::{fft, reverse_index};
use lambdaworks_gpu::cuda::abstractions::{errors::CudaError, state::CudaState};

pub fn evaluate_fft_cuda<F>(coeffs: &[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, FFTError>
where
    F: IsFFTField,
    F::BaseType: Unpin,
{
    let state = CudaState::new()?;
    let order = log2(coeffs.len())?;
    let twiddles = get_twiddles(order, RootsConfig::BitReverse)?;

    fft(coeffs, &twiddles, &state)
}

/// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
/// factors. This is considered to be the inverse operation of [evaluate_fft_cuda()].
pub fn interpolate_fft_cuda<F>(
    fft_evals: &[FieldElement<F>],
) -> Result<Polynomial<FieldElement<F>>, FFTError>
where
    F: IsFFTField,
    F::BaseType: Unpin,
{
    let state = CudaState::new()?;

    // fft() can zero-pad the coeffs if there aren't 2^k of them (k being any integer).
    // TODO: twiddle factors need to be handled with too much care, the FFT API shouldn't accept
    // invalid twiddle factor collections. A better solution is needed.
    let order = log2(fft_evals.len())?;
    let twiddles = get_twiddles(order, RootsConfig::BitReverseInversed)?;

    let coeffs = fft(fft_evals, &twiddles, &state)?;

    let scale_factor = FieldElement::from(fft_evals.len() as u64).inv();
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}

// TODO: implement in CUDA
fn get_twiddles<F: IsFFTField>(
    order: u64,
    config: RootsConfig,
) -> Result<Vec<FieldElement<F>>, FieldError> {
    get_powers_of_primitive_root(order, (1 << order) / 2, config)
}

// TODO: remove after implementing in cuda
fn get_powers_of_primitive_root<F: IsFFTField>(
    n: u64,
    count: usize,
    config: RootsConfig,
) -> Result<Vec<FieldElement<F>>, FieldError> {
    let root = F::get_primitive_root_of_unity(n)?;

    let calc = |i| match config {
        RootsConfig::Natural => root.pow(i),
        RootsConfig::NaturalInversed => root.pow(i).inv(),
        RootsConfig::BitReverse => root.pow(reverse_index(&i, count as u64)),
        RootsConfig::BitReverseInversed => root.pow(reverse_index(&i, count as u64)).inv(),
    };

    let results = (0..count).map(calc);
    Ok(results.collect())
}

// TODO: remove when fft works on non-multiple-of-two input length
fn log2(n: usize) -> Result<u64, CudaError> {
    if !n.is_power_of_two() {
        return Err(CudaError::InvalidOrder(n));
    }
    Ok(n.trailing_zeros() as u64)
}
