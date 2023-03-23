use crate::metal::abstractions::state::MetalState;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsTwoAdicField},
    polynomial::Polynomial,
};

use super::{errors::FFTMetalError, helpers::log2, ops::*};

trait GpuPolyOperations<F: IsTwoAdicField> {
    fn evaluate_fft_metal(&self, state: MetalState) -> Result<Vec<FieldElement<F>>, FFTMetalError>;
    // fn evaluate_offset_fft_metal(
    //     &self,
    //     offset: &FieldElement<F>,
    //     blowup_factor: usize,
    //     state: MetalState
    // ) -> Result<Vec<FieldElement<F>>, FFTMetalError>;
    // fn interpolate_fft_metal(
    //     fft_evals: &[FieldElement<F>],
    // ) -> Result<Polynomial<FieldElement<F>>, FFTMetalError>;
}

impl<F: IsTwoAdicField> GpuPolyOperations<F> for Polynomial<FieldElement<F>> {
    /// Evaluates this polynomial using parallel FFT (so the function is evaluated using twiddle factors),
    /// in Metal.
    fn evaluate_fft_metal(&self, state: MetalState) -> Result<Vec<FieldElement<F>>, FFTMetalError> {
        let order = log2(self.coefficients().len()).map_err(FFTMetalError::FFT)?;
        let twiddles = gen_twiddles(order, &state)?;

        fft(self.coefficients(), &twiddles, state)
    }

    // /// Evaluates this polynomial using parallel FFT in an extended domain by `blowup_factor` with an `offset`, in Metal.
    // /// Usually used for Reed-Solomon encoding.
    // fn evaluate_offset_fft_metal(
    //     &self,
    //     offset: &FieldElement<F>,
    //     blowup_factor: usize,
    //     state: MetalState
    // ) -> Result<Vec<FieldElement<F>>, FFTMetalError> {
    //     let scaled = self.scale(offset);
    //     fft_with_blowup(scaled.coefficients(), blowup_factor, state)
    // }

    // /// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
    // /// factors. This is considered to be the inverse operation of [Self::evaluate_fft()].
    // fn interpolate_fft_metal(fft_evals: &[FieldElement<F>]) -> Result<Self, FFTMetalError> {
    //     let coeffs = inverse_fft(fft_evals)?;
    //     Ok(Polynomial::new(&coeffs))
    // }
}
