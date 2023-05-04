use cudarc::driver::DriverError;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },
    polynomial::Polynomial,
};

use super::ops::{fft, get_twiddles, log2};

pub fn evaluate_fft_cuda<F>(
    poly: &Polynomial<FieldElement<F>>,
) -> Result<Vec<FieldElement<F>>, DriverError>
where
    F: IsFFTField,
    F::BaseType: Unpin,
{
    let order = log2(poly.coefficients.len()).unwrap();
    let twiddles = get_twiddles(order, RootsConfig::BitReverse).unwrap();

    fft(poly.coefficients(), &twiddles)
}

/// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
/// factors. This is considered to be the inverse operation of [evaluate_fft_cuda()].
pub fn interpolate_fft_cuda<F>(
    fft_evals: &[FieldElement<F>],
) -> Result<Polynomial<FieldElement<F>>, DriverError>
where
    F: IsFFTField,
    F::BaseType: Unpin,
{
    // fft() can zero-pad the coeffs if there aren't 2^k of them (k being any integer).
    // TODO: twiddle factors need to be handled with too much care, the FFT API shouldn't accept
    // invalid twiddle factor collections. A better solution is needed.
    let order = log2(fft_evals.len()).unwrap();
    let twiddles = get_twiddles(order, RootsConfig::BitReverseInversed).unwrap();

    let coeffs = fft(fft_evals, &twiddles)?;

    let scale_factor = FieldElement::from(fft_evals.len() as u64).inv();
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}

#[cfg(feature = "cuda")]
#[cfg(test)]
mod gpu_tests {}
