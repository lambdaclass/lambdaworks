use crate::metal::abstractions::state::MetalState;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsTwoAdicField, RootsConfig},
    },
    polynomial::Polynomial,
};

use super::{errors::FFTMetalError, helpers::log2, ops::*};

trait MetalFFTPoly<F: IsTwoAdicField> {
    fn evaluate_fft_metal(&self, state: &MetalState)
        -> Result<Vec<FieldElement<F>>, FFTMetalError>;
    // fn evaluate_offset_fft_metal(
    //     &self,
    //     offset: &FieldElement<F>,
    //     blowup_factor: usize,
    //     state: MetalState
    // ) -> Result<Vec<FieldElement<F>>, FFTMetalError>;
    fn interpolate_fft_metal(
        fft_evals: &[FieldElement<F>],
        state: &MetalState,
    ) -> Result<Polynomial<FieldElement<F>>, FFTMetalError>;
}

impl<F: IsTwoAdicField> MetalFFTPoly<F> for Polynomial<FieldElement<F>> {
    /// Evaluates this polynomial using parallel FFT (so the function is evaluated using twiddle factors),
    /// in Metal.
    fn evaluate_fft_metal(
        &self,
        state: &MetalState,
    ) -> Result<Vec<FieldElement<F>>, FFTMetalError> {
        let order = log2(self.coefficients().len()).map_err(FFTMetalError::FFT)?;
        let twiddles = gen_twiddles(order, RootsConfig::BitReverse, state)?;

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

    /// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
    /// factors. This is considered to be the inverse operation of [Self::evaluate_fft()].
    fn interpolate_fft_metal(
        fft_evals: &[FieldElement<F>],
        state: &MetalState,
    ) -> Result<Self, FFTMetalError> {
        let order = log2(fft_evals.len()).map_err(FFTMetalError::FFT)?;
        let twiddles = gen_twiddles(order, RootsConfig::BitReverseInversed, state)?;

        let coeffs = fft(fft_evals, &twiddles, state)?;

        let scale_factor = FieldElement::from(fft_evals.len() as u64).inv();
        Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        polynomial::Polynomial,
    };
    use proptest::{collection, prelude::*};

    use super::*;

    type F = Stark252PrimeField;
    type FE = FieldElement<F>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
        // also it can't exceed the test field's two-adicity.
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null polynomial", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size not power of two", |vec| vec.len().is_power_of_two())) -> Vec<FE> {
            vec
        }
    }
    prop_compose! {
        fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FE> {
            Polynomial::new(&coeffs)
        }
    }

    proptest! {
        #[test]
        fn test_metal_fft_poly_eval_matches_cpu(poly in poly(8)) {
            objc::rc::autoreleasepool(|| {
                let metal_state = MetalState::new(None).unwrap();

                let gpu_evals = poly.evaluate_fft_metal(&metal_state).unwrap();
                let cpu_evals = poly.evaluate_fft().unwrap();

                prop_assert_eq!(gpu_evals, cpu_evals);
                Ok(())
            }).unwrap();
        }
    }

    proptest! {
        #[test]
        fn test_metal_fft_poly_interpol_matches_cpu(evals in field_vec(8)) {
            objc::rc::autoreleasepool(|| {
                let metal_state = MetalState::new(None).unwrap();

                let gpu_evals = Polynomial::interpolate_fft_metal(&evals, &metal_state).unwrap();
                let cpu_evals = Polynomial::interpolate_fft(&evals).unwrap();

                prop_assert_eq!(gpu_evals, cpu_evals);
                Ok(())
            }).unwrap();
        }
    }
}
