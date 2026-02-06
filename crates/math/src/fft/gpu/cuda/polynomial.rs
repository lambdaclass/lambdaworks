use crate::{
    fft::{
        errors::FFTError,
        gpu::cuda::{
            ops::gen_twiddles,
            state::{CudaState, HasCudaExtFft},
        },
    },
    field::{
        element::FieldElement,
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        fields::u64_goldilocks_field::{
            Degree2GoldilocksExtensionField, Degree3GoldilocksExtensionField, Goldilocks64Field,
        },
        traits::{IsFFTField, IsField, RootsConfig},
    },
    polynomial::Polynomial,
};

use super::ops::{fft, fft_ext};
use lambdaworks_gpu::cuda::abstractions::errors::CudaError;
use std::cell::RefCell;

/// Trait for field types that support CUDA-accelerated FFT.
///
/// Each implementation encapsulates the correct CUDA kernel dispatch for a
/// specific field type. The generic `evaluate_fft`/`interpolate_fft` functions
/// in `fft/polynomial.rs` use this trait to delegate to the appropriate CUDA path.
pub(crate) trait HasCudaFft: IsField {
    fn cuda_evaluate_fft(
        coeffs: &[FieldElement<Self>],
    ) -> Result<Vec<FieldElement<Self>>, FFTError>;
    fn cuda_interpolate_fft(
        evals: &[FieldElement<Self>],
    ) -> Result<Polynomial<FieldElement<Self>>, FFTError>;
}

impl HasCudaFft for Stark252PrimeField {
    fn cuda_evaluate_fft(
        coeffs: &[FieldElement<Self>],
    ) -> Result<Vec<FieldElement<Self>>, FFTError> {
        evaluate_fft_cuda(coeffs).map_err(Into::into)
    }
    fn cuda_interpolate_fft(
        evals: &[FieldElement<Self>],
    ) -> Result<Polynomial<FieldElement<Self>>, FFTError> {
        interpolate_fft_cuda(evals)
    }
}

impl HasCudaFft for Goldilocks64Field {
    fn cuda_evaluate_fft(
        coeffs: &[FieldElement<Self>],
    ) -> Result<Vec<FieldElement<Self>>, FFTError> {
        evaluate_fft_cuda(coeffs).map_err(Into::into)
    }
    fn cuda_interpolate_fft(
        evals: &[FieldElement<Self>],
    ) -> Result<Polynomial<FieldElement<Self>>, FFTError> {
        interpolate_fft_cuda(evals)
    }
}

impl HasCudaFft for Degree2GoldilocksExtensionField {
    fn cuda_evaluate_fft(
        coeffs: &[FieldElement<Self>],
    ) -> Result<Vec<FieldElement<Self>>, FFTError> {
        evaluate_fft_ext_cuda::<Goldilocks64Field, Self>(coeffs).map_err(Into::into)
    }
    fn cuda_interpolate_fft(
        evals: &[FieldElement<Self>],
    ) -> Result<Polynomial<FieldElement<Self>>, FFTError> {
        interpolate_fft_ext_cuda::<Goldilocks64Field, Self>(evals)
    }
}

impl HasCudaFft for Degree3GoldilocksExtensionField {
    fn cuda_evaluate_fft(
        coeffs: &[FieldElement<Self>],
    ) -> Result<Vec<FieldElement<Self>>, FFTError> {
        evaluate_fft_ext_cuda::<Goldilocks64Field, Self>(coeffs).map_err(Into::into)
    }
    fn cuda_interpolate_fft(
        evals: &[FieldElement<Self>],
    ) -> Result<Polynomial<FieldElement<Self>>, FFTError> {
        interpolate_fft_ext_cuda::<Goldilocks64Field, Self>(evals)
    }
}

thread_local! {
    static CUDA_STATE: RefCell<Option<CudaState>> = const { RefCell::new(None) };
}

/// Returns a cached `CudaState`, initializing it on first use per thread.
/// This avoids re-creating the CUDA device and reloading PTX on every FFT call.
fn get_cuda_state() -> Result<CudaState, CudaError> {
    CUDA_STATE.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            *opt = Some(CudaState::new()?);
        }
        Ok(opt.as_ref().unwrap().clone())
    })
}

pub fn evaluate_fft_cuda<F>(coeffs: &[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, CudaError>
where
    F: IsFFTField,
    F::BaseType: Unpin,
{
    let state = get_cuda_state()?;
    let order = log2(coeffs.len())?;
    let twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverse, &state)?;

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
    let state = get_cuda_state()?;

    // fft() can zero-pad the coeffs if there aren't 2^k of them (k being any integer).
    // TODO: twiddle factors need to be handled with too much care, the FFT API shouldn't accept
    // invalid twiddle factor collections. A better solution is needed.
    let order = log2(fft_evals.len())?;
    let twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverseInversed, &state)?;

    let coeffs = fft(fft_evals, &twiddles, &state)?;

    let scale_factor = FieldElement::from(fft_evals.len() as u64)
        .inv()
        .map_err(|_| CudaError::FunctionError("Failed to invert scale factor".to_string()))?;
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}

/// Extension field evaluate: coefficients in E, twiddles generated in base field F.
pub(crate) fn evaluate_fft_ext_cuda<F, E>(
    coeffs: &[FieldElement<E>],
) -> Result<Vec<FieldElement<E>>, CudaError>
where
    F: IsFFTField,
    E: HasCudaExtFft,
{
    let state = get_cuda_state()?;
    let order = log2(coeffs.len())?;
    let twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverse, &state)?;

    fft_ext(coeffs, &twiddles, &state)
}

/// Extension field interpolate: evaluations in E, twiddles generated in base field F.
pub(crate) fn interpolate_fft_ext_cuda<F, E>(
    fft_evals: &[FieldElement<E>],
) -> Result<Polynomial<FieldElement<E>>, FFTError>
where
    F: IsFFTField,
    E: HasCudaExtFft,
{
    let state = get_cuda_state()?;
    let order = log2(fft_evals.len())?;
    let twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverseInversed, &state)?;

    let coeffs = fft_ext(fft_evals, &twiddles, &state)?;

    let scale_factor = FieldElement::from(fft_evals.len() as u64)
        .inv()
        .map_err(|_| CudaError::FunctionError("Failed to invert scale factor".to_string()))?;
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}

// TODO: remove when fft works on non-multiple-of-two input length
fn log2(n: usize) -> Result<u64, CudaError> {
    if !n.is_power_of_two() {
        return Err(CudaError::InvalidOrder(n));
    }
    Ok(n.trailing_zeros() as u64)
}
