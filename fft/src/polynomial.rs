use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    helpers,
    polynomial::Polynomial,
};

use crate::{
    errors::FFTError,
    ops::{fft_with_blowup, inverse_fft},
};

pub trait FFTPoly<F: IsFFTField> {
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

impl<F: IsFFTField> FFTPoly<F> for Polynomial<FieldElement<F>> {
    /// Evaluates this polynomial using FFT (so the function is evaluated using twiddle factors).
    fn evaluate_fft(&self) -> Result<Vec<FieldElement<F>>, FFTError> {
        #[cfg(feature = "metal")]
        {
            if !F::field_name().is_empty() {
                Ok(lambdaworks_gpu::metal::fft::polynomial::evaluate_fft_metal(
                    self,
                )?)
            } else {
                evaluate_fft_cpu(self)
            }
        }

        #[cfg(feature = "cuda")]
        {
            // TODO: support multiple fields with CUDA
            if F::field_name() == "stark256" {
                Ok(lambdaworks_gpu::cuda::fft::polynomial::evaluate_fft_cuda(
                    self,
                )?)
            } else {
                evaluate_fft_cpu(self)
            }
        }

        #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
        {
            evaluate_fft_cpu(self)
        }
    }

    /// Evaluates this polynomial in an extended domain by `blowup_factor` with an `offset`.
    /// Usually used for Reed-Solomon encoding.
    fn evaluate_offset_fft(
        &self,
        offset: &FieldElement<F>,
        blowup_factor: usize,
    ) -> Result<Vec<FieldElement<F>>, FFTError> {
        #[cfg(feature = "metal")]
        {
            if !F::field_name().is_empty() {
                Ok(
                    lambdaworks_gpu::metal::fft::polynomial::evaluate_offset_fft_metal(
                        self,
                        offset,
                        blowup_factor,
                    )?,
                )
            } else {
                evaluate_offset_fft_cpu(self, offset, blowup_factor)
            }
        }

        #[cfg(not(feature = "metal"))]
        {
            evaluate_offset_fft_cpu(self, offset, blowup_factor)
        }
    }

    /// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
    /// factors. This is considered to be the inverse operation of [Self::evaluate_fft()].
    fn interpolate_fft(fft_evals: &[FieldElement<F>]) -> Result<Self, FFTError> {
        #[cfg(feature = "metal")]
        {
            if !F::field_name().is_empty() {
                Ok(lambdaworks_gpu::metal::fft::polynomial::interpolate_fft_metal(fft_evals)?)
            } else {
                interpolate_fft_cpu(fft_evals)
            }
        }

        #[cfg(feature = "cuda")]
        {
            if !F::field_name().is_empty() {
                Ok(lambdaworks_gpu::cuda::fft::polynomial::interpolate_fft_cuda(fft_evals)?)
            } else {
                interpolate_fft_cpu(fft_evals)
            }
        }

        #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
        {
            interpolate_fft_cpu(fft_evals)
        }
    }
}

pub fn evaluate_fft_cpu<F>(
    poly: &Polynomial<FieldElement<F>>,
) -> Result<Vec<FieldElement<F>>, FFTError>
where
    F: IsFFTField,
{
    let num_coefficients = poly.coefficients().len();
    let num_coeficcients_power_of_two = helpers::next_power_of_two(num_coefficients as u64);

    let mut padded_coefficients = poly.coefficients().to_vec();
    padded_coefficients.resize(num_coeficcients_power_of_two as usize, FieldElement::zero());
    crate::ops::fft(padded_coefficients.as_slice())
}

pub fn evaluate_offset_fft_cpu<F>(
    poly: &Polynomial<FieldElement<F>>,
    offset: &FieldElement<F>,
    blowup_factor: usize,
) -> Result<Vec<FieldElement<F>>, FFTError>
where
    F: IsFFTField,
{
    let scaled = poly.scale(offset);
    fft_with_blowup(scaled.coefficients(), blowup_factor)
}

pub fn interpolate_fft_cpu<F>(
    fft_evals: &[FieldElement<F>],
) -> Result<Polynomial<FieldElement<F>>, FFTError>
where
    F: IsFFTField,
{
    let coeffs = inverse_fft(fft_evals)?;
    Ok(Polynomial::new(&coeffs))
}

pub fn compose_fft<F>(
    poly_1: &Polynomial<FieldElement<F>>,
    poly_2: &Polynomial<FieldElement<F>>,
) -> Polynomial<FieldElement<F>>
where
    F: IsFFTField,
{
    let poly_2_evaluations = poly_2.evaluate_fft().unwrap();

    let values: Vec<_> = poly_2_evaluations
        .iter()
        .map(|value| poly_1.evaluate(value))
        .collect();

    Polynomial::interpolate_fft(values.as_slice()).unwrap()
}

/// Temporary hotfix for handling evaluations of the zero polynomial
/// in the STARK prover context. Checks if the the polynomial is the
/// zero polynomial, and if it is, returns its evaluations as a
/// vector of the zero field element of size domain_len * blowup_factor.
pub fn evaluate_offset_fft_with_len<F>(
    poly: &Polynomial<FieldElement<F>>,
    domain_len: usize,
    offset: &FieldElement<F>,
    blowup_factor: usize,
) -> Result<Vec<FieldElement<F>>, FFTError>
where
    F: IsFFTField,
{
    if poly == &Polynomial::zero() {
        return Ok(vec![FieldElement::<F>::zero(); domain_len * blowup_factor]);
    }

    poly.evaluate_offset_fft(offset, blowup_factor)
}

#[cfg(not(feature = "metal"))]
#[cfg(test)]
mod u64_field_tests {
    use lambdaworks_math::field::test_fields::u64_test_field::U64TestField;
    use lambdaworks_math::field::traits::IsField;
    use lambdaworks_math::field::traits::RootsConfig;
    use proptest::{collection, prelude::*};

    use crate::roots_of_unity::{get_powers_of_primitive_root, get_powers_of_primitive_root_coset};

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
            let order = poly.coefficients().len().trailing_zeros();
            let twiddles = get_powers_of_primitive_root(order.into(), poly.coefficients.len(), RootsConfig::Natural).unwrap();

            let fft_eval = poly.evaluate_fft().unwrap();
            let naive_eval = poly.evaluate_slice(&twiddles);

            prop_assert_eq!(fft_eval, naive_eval);
        }

        // Property-based test that ensures FFT eval. with coset gives same result as a naive polynomial evaluation.
        #[test]
        fn test_fft_coset_matches_naive_evaluation(poly in poly(8), offset in offset(), blowup_factor in powers_of_two(4)) {
            let order = (poly.coefficients().len() * blowup_factor).trailing_zeros();
            let twiddles = get_powers_of_primitive_root_coset(order.into(), poly.coefficients.len() * blowup_factor, &offset).unwrap();

            let fft_eval = poly.evaluate_offset_fft(&offset, blowup_factor).unwrap();
            let naive_eval = poly.evaluate_slice(&twiddles);

            prop_assert_eq!(fft_eval, naive_eval);
        }

        // Property-based test that ensures FFT eval. using polynomials with a non-power-of-two amount of coefficients works.
        #[test]
        fn test_fft_non_power_of_two(poly in poly(8)) {
            let order = poly.coefficients().len().trailing_zeros();
            let twiddles = get_powers_of_primitive_root(order.into(), poly.coefficients.len(), RootsConfig::Natural).unwrap();

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

#[cfg(test)]
mod u256_two_adic_prime_field_tests {
    use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
    use proptest::{
        collection, prelude::any, prop_assert_eq, prop_compose, proptest, strategy::Strategy,
    };

    use lambdaworks_math::{
        field::{element::FieldElement, traits::RootsConfig},
        polynomial::Polynomial,
    };

    use crate::{polynomial::FFTPoly, roots_of_unity::get_powers_of_primitive_root};

    type F = Stark252PrimeField;
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
        // Property-based test that ensures FFT eval. in the FFT friendly field gives same result as a naive polynomial evaluation.
        #[test]
        fn test_fft_evaluation_is_correct_in_u256_fft_friendly_field(poly in poly(8)) {
            let order = poly.coefficients().len().trailing_zeros();
            let twiddles = get_powers_of_primitive_root(order.into(), poly.coefficients.len(), RootsConfig::Natural).unwrap();

            let fft_eval = poly.evaluate_fft().unwrap();
            let naive_eval = poly.evaluate_slice(&twiddles);

            prop_assert_eq!(fft_eval, naive_eval);
        }
    }
}
