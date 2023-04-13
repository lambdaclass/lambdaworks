use lambdaworks_math::{
    field::{element::FieldElement, traits::IsTwoAdicField},
    helpers,
    polynomial::Polynomial,
};

use crate::{
    abstractions::{fft, fft_with_blowup, inverse_fft},
    errors::FFTError,
};

pub trait FFTPoly<F: IsTwoAdicField> {
    fn evaluate_fft(&self) -> Result<Vec<FieldElement<F>>, FFTError>;
    fn evaluate_offset_fft(
        &self,
        offset: &FieldElement<F>,
        blowup_factor: usize,
    ) -> Result<Vec<FieldElement<F>>, FFTError>;
    fn interpolate_fft(
        fft_evals: &[FieldElement<F>],
    ) -> Result<Polynomial<FieldElement<F>>, FFTError>;
}

impl<F: IsTwoAdicField> FFTPoly<F> for Polynomial<FieldElement<F>> {
    /// Evaluates this polynomial using FFT (so the function is evaluated using twiddle factors).
    fn evaluate_fft(&self) -> Result<Vec<FieldElement<F>>, FFTError> {
        let num_coefficients = self.coefficients().len();
        let num_coeficcients_power_of_two = helpers::next_power_of_two(num_coefficients as u64);

        let mut padded_coefficients = self.coefficients().to_vec();
        padded_coefficients.resize(num_coeficcients_power_of_two as usize, FieldElement::zero());

        fft(padded_coefficients.as_slice())
    }

    /// Evaluates this polynomial in an extended domain by `blowup_factor` with an `offset`.
    /// Usually used for Reed-Solomon encoding.
    fn evaluate_offset_fft(
        &self,
        offset: &FieldElement<F>,
        blowup_factor: usize,
    ) -> Result<Vec<FieldElement<F>>, FFTError> {
        let scaled = self.scale(offset);
        fft_with_blowup(scaled.coefficients(), blowup_factor)
    }

    /// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
    /// factors. This is considered to be the inverse operation of [Self::evaluate_fft()].
    fn interpolate_fft(fft_evals: &[FieldElement<F>]) -> Result<Self, FFTError> {
        let coeffs = inverse_fft(fft_evals)?;
        Ok(Polynomial::new(&coeffs))
    }
}

pub fn compose_fft<F>(
    poly_1: &Polynomial<FieldElement<F>>,
    poly_2: &Polynomial<FieldElement<F>>,
) -> Polynomial<FieldElement<F>>
where
    F: IsTwoAdicField,
{
    let poly_2_evaluations = poly_2.evaluate_fft().unwrap();

    let values: Vec<_> = poly_2_evaluations
        .iter()
        .map(|value| poly_1.evaluate(value))
        .collect();

    Polynomial::interpolate_fft(values.as_slice()).unwrap()
}

// pub trait FFTPoly<F: IsTwoAdicField> {
//     fn evaluate_fft(&self) -> Result<Vec<FieldElement<F>>, FFTMetalError>;
//     fn evaluate_offset_fft(
//         &self,
//         offset: &FieldElement<F>,
//         blowup_factor: usize,
//     ) -> Result<Vec<FieldElement<F>>, FFTMetalError>;
//     fn interpolate_fft(
//         fft_evals: &[FieldElement<F>],
//     ) -> Result<Polynomial<FieldElement<F>>, FFTMetalError>;
// }

// impl<F: IsTwoAdicField> MetalFFTPoly<F> for Polynomial<FieldElement<F>> {
//     /// Evaluates this polynomial using parallel FFT (so the function is evaluated using twiddle factors),
//     /// in Metal.
//     fn evaluate_fft(&self) -> Result<Vec<FieldElement<F>>, FFTMetalError> {
//         let metal_state = MetalState::new(None).unwrap();
//         let order = log2(self.coefficients().len())?;
//         let twiddles = gen_twiddles(order, RootsConfig::BitReverse, &metal_state)?;

//         fft(self.coefficients(), &twiddles, &metal_state)
//     }

//     /// Evaluates this polynomial using parallel FFT in an extended domain by `blowup_factor` with an `offset`, in Metal.
//     /// Usually used for Reed-Solomon encoding.
//     fn evaluate_offset_fft(
//         &self,
//         offset: &FieldElement<F>,
//         blowup_factor: usize,
//     ) -> Result<Vec<FieldElement<F>>, FFTMetalError> {
//         let metal_state = MetalState::new(None).unwrap();
//         let scaled = self.scale(offset);
//         fft_with_blowup(scaled.coefficients(), blowup_factor, &metal_state)
//     }

//     /// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
//     /// factors. This is considered to be the inverse operation of [Self::evaluate_fft()].
//     fn interpolate_fft(fft_evals: &[FieldElement<F>]) -> Result<Self, FFTMetalError> {
//         let metal_state = MetalState::new(None).unwrap();
//         let order = log2(fft_evals.len())?;
//         let twiddles = gen_twiddles(order, RootsConfig::BitReverseInversed, &metal_state)?;

//         let coeffs = fft(fft_evals, &twiddles, &metal_state)?;

//         let scale_factor = FieldElement::from(fft_evals.len() as u64).inv();
//         Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
//     }
// }

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        field::{
            fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::RootsConfig,
        },
        polynomial::Polynomial,
    };
    use proptest::{collection, prelude::*};

    use crate::{helpers::log2, roots_of_unity::get_powers_of_primitive_root};

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
        fn offset()(num in any::<u64>(), factor in any::<u64>()) -> FE { FE::from(num).pow(factor) }
    }
    prop_compose! {
        fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FE> {
            Polynomial::new(&coeffs)
        }
    }

    // proptest! {
    //     #[test]
    //     fn test_metal_fft_poly_eval_matches_cpu(poly in poly(6)) {
    //         objc::rc::autoreleasepool(|| {
    //             let gpu_evals = poly.evaluate_fft().unwrap();
    //             let cpu_evals = poly.evaluate_fft().unwrap();

    //             prop_assert_eq!(gpu_evals, cpu_evals);
    //             Ok(())
    //         }).unwrap();
    //     }

    //     #[test]
    //     fn test_metal_fft_coset_poly_eval_matches_cpu(poly in poly(6), offset in offset(), blowup_factor in powers_of_two(4)) {
    //         objc::rc::autoreleasepool(|| {
    //             let gpu_evals = poly.evaluate_offset_fft(&offset, blowup_factor).unwrap();
    //             let cpu_evals = poly.evaluate_offset_fft(&offset, blowup_factor).unwrap();

    //             prop_assert_eq!(gpu_evals, cpu_evals);
    //             Ok(())
    //         }).unwrap();
    //     }

    //     #[test]
    //     fn test_metal_fft_poly_interpol_matches_cpu(evals in field_vec(6)) {
    //         objc::rc::autoreleasepool(|| {
    //             let gpu_evals = Polynomial::interpolate_fft(&evals).unwrap();
    //             let cpu_evals = Polynomial::interpolate_fft(&evals).unwrap();

    //             prop_assert_eq!(gpu_evals, cpu_evals);
    //             Ok(())
    //         }).unwrap();
    //     }
    // }

    proptest! {
        // Property-based test that ensures FFT eval. in the FFT friendly field gives same result as a naive polynomial evaluation.
        #[test]
        fn test_fft_evaluation_is_correct_in_u256_fft_friendly_field(poly in poly(8)) {
            let order = log2(poly.coefficients().len()).unwrap();
            let twiddles = get_powers_of_primitive_root(order, poly.coefficients.len(), RootsConfig::Natural).unwrap();

            let fft_eval = poly.evaluate_fft().unwrap();
            let naive_eval = poly.evaluate_slice(&twiddles);

            prop_assert_eq!(fft_eval, naive_eval);
        }
    }
}

#[cfg(test)]
mod fft_test {
    use lambdaworks_math::field::test_fields::u64_test_field::U64TestField;
    use lambdaworks_math::field::traits::IsField;
    use lambdaworks_math::field::traits::RootsConfig;
    use proptest::{collection, prelude::*};

    use crate::{
        helpers::log2,
        roots_of_unity::{get_powers_of_primitive_root, get_powers_of_primitive_root_coset},
    };

    use super::*;

    // FFT related tests
    type F = U64TestField;
    type FE = FieldElement<F>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
        // also it can't exceed the test field's two-adicity.
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }
    prop_compose! {
        fn offset()(num in 1..F::neg(&1)) -> FE { FE::from(num) }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size not power of two", |vec| vec.len().is_power_of_two())) -> Vec<FE> {
            vec
        }
    }
    prop_compose! {
        fn non_power_of_two_sized_field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size power of two", |vec| !vec.len().is_power_of_two())) -> Vec<FE> {
            vec
        }
    }
    prop_compose! {
        fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FE> {
            Polynomial::new(&coeffs)
        }
    }
    prop_compose! {
        fn poly_with_non_power_of_two_coeffs(max_exp: u8)(coeffs in non_power_of_two_sized_field_vec(max_exp)) -> Polynomial<FE> {
            Polynomial::new(&coeffs)
        }
    }
    proptest! {
        // Property-based test that ensures FFT eval. gives same result as a naive polynomial evaluation.
        #[test]
        fn test_fft_matches_naive_evaluation(poly in poly(8)) {
            let order = log2(poly.coefficients().len()).unwrap();
            let twiddles = get_powers_of_primitive_root(order, poly.coefficients.len(), RootsConfig::Natural).unwrap();

            let fft_eval = poly.evaluate_fft().unwrap();
            let naive_eval = poly.evaluate_slice(&twiddles);

            prop_assert_eq!(fft_eval, naive_eval);
        }

        // Property-based test that ensures FFT eval. with coset gives same result as a naive polynomial evaluation.
        #[test]
        fn test_fft_coset_matches_naive_evaluation(poly in poly(8), offset in offset(), blowup_factor in powers_of_two(4)) {
            let order = log2(poly.coefficients().len() * blowup_factor).unwrap();
            let twiddles = get_powers_of_primitive_root_coset(order, poly.coefficients.len() * blowup_factor, &offset).unwrap();

            let fft_eval = poly.evaluate_offset_fft(&offset, blowup_factor).unwrap();
            let naive_eval = poly.evaluate_slice(&twiddles);

            prop_assert_eq!(fft_eval, naive_eval);
        }

        // Property-based test that ensures FFT eval. using polynomials with a non-power-of-two amount of coefficients works.
        #[test]
        fn test_fft_non_power_of_two_poly(poly in poly_with_non_power_of_two_coeffs(8)) {
            let num_coefficients = poly.coefficients().len();
            let num_coeficcients_power_of_two = helpers::next_power_of_two(num_coefficients as u64) as usize;
            let order = log2(num_coeficcients_power_of_two).unwrap();
            let twiddles = get_powers_of_primitive_root(order, num_coeficcients_power_of_two, RootsConfig::Natural).unwrap();

            let fft_eval = poly.evaluate_fft().unwrap();
            let naive_eval = poly.evaluate_slice(&twiddles);

            prop_assert_eq!(fft_eval, naive_eval);
        }

        // Property-based test that ensures interpolation is the inverse operation of evaluation.
        #[test]
        fn test_fft_interpolate_is_inverse_of_evaluate(poly in poly(8)) {
            let eval = poly.evaluate_fft().unwrap();
            let new_poly = Polynomial::interpolate_fft(&eval).unwrap();

            prop_assert_eq!(poly, new_poly);
        }

        // Property-based test that ensures FFT won't work with a degree 0 polynomial.
        #[test]
        #[ignore] // TODO fix
        fn test_fft_constant_poly(elem in field_element()) {
            let poly = Polynomial::new(&[elem]);
            let result = poly.evaluate_fft();

            prop_assert!(matches!(result, Err(FFTError::RootOfUnityError(_, k)) if k == 0));
        }
    }

    #[test]
    fn composition_fft_works() {
        let p = Polynomial::new(&[FE::new(0), FE::new(2)]);
        let q = Polynomial::new(&[FE::new(0), FE::new(0), FE::new(0), FE::new(1)]);
        assert_eq!(
            compose_fft(&p, &q),
            Polynomial::new(&[FE::new(0), FE::new(0), FE::new(0), FE::new(2)])
        );
    }
}
