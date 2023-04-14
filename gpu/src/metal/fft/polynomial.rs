use crate::metal::abstractions::state::MetalState;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsTwoAdicField, RootsConfig},
    },
    polynomial::Polynomial,
};

use super::{errors::FFTMetalError, helpers::log2, ops::*};

pub trait MetalFFTPoly<F: IsTwoAdicField> {
    fn evaluate_fft_metal(&self) -> Result<Vec<FieldElement<F>>, FFTMetalError>;
    fn evaluate_offset_fft_metal(
        &self,
        offset: &FieldElement<F>,
        blowup_factor: usize,
    ) -> Result<Vec<FieldElement<F>>, FFTMetalError>;
    fn interpolate_fft_metal(
        fft_evals: &[FieldElement<F>],
    ) -> Result<Polynomial<FieldElement<F>>, FFTMetalError>;
}

impl<F: IsTwoAdicField> MetalFFTPoly<F> for Polynomial<FieldElement<F>> {
    /// Evaluates this polynomial using parallel FFT (so the function is evaluated using twiddle factors),
    /// in Metal.
    fn evaluate_fft_metal(&self) -> Result<Vec<FieldElement<F>>, FFTMetalError> {
        let metal_state = MetalState::new(None).unwrap();
        let order = log2(self.coefficients().len())?;
        let twiddles = gen_twiddles(order, RootsConfig::BitReverse, &metal_state)?;

        fft(self.coefficients(), &twiddles, &metal_state)
    }

    /// Evaluates this polynomial using parallel FFT in an extended domain by `blowup_factor` with an `offset`, in Metal.
    /// Usually used for Reed-Solomon encoding.
    fn evaluate_offset_fft_metal(
        &self,
        offset: &FieldElement<F>,
        blowup_factor: usize,
    ) -> Result<Vec<FieldElement<F>>, FFTMetalError> {
        let metal_state = MetalState::new(None).unwrap();
        let scaled = self.scale(offset);

        fft_with_blowup(scaled.coefficients(), blowup_factor, &metal_state)
    }

    /// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
    /// factors. This is considered to be the inverse operation of [Self::evaluate_fft()].
    fn interpolate_fft_metal(fft_evals: &[FieldElement<F>]) -> Result<Self, FFTMetalError> {
        let metal_state = MetalState::new(None).unwrap();
        let order = log2(fft_evals.len())?;
        let twiddles = gen_twiddles(order, RootsConfig::BitReverseInversed, &metal_state)?;

        let coeffs = fft(fft_evals, &twiddles, &metal_state)?;

        let scale_factor = FieldElement::from(fft_evals.len() as u64).inv();
        Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
    }
}
