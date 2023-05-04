use crate::metal::abstractions::{errors::MetalError, state::MetalState};
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },
    polynomial::Polynomial,
};

use super::ops::*;

pub fn forward_fft_metal<F>(input: &[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, MetalError>
where
    F: IsFFTField,
{
    let state = MetalState::new(None).unwrap();

    let order = input.len().trailing_zeros();
    let twiddles = gen_twiddles(order.into(), RootsConfig::BitReverse, &state)?;

    fft(input, &twiddles, &state)
}

/// Evaluates this polynomial using parallel FFT in an extended domain by `blowup_factor` with an `offset`, in Metal.
/// Usually used for Reed-Solomon encoding.
pub fn evaluate_offset_fft_metal<F>(
    poly: &Polynomial<FieldElement<F>>,
    offset: &FieldElement<F>,
    blowup_factor: usize,
) -> Result<Vec<FieldElement<F>>, MetalError>
where
    F: IsFFTField,
{
    let metal_state = MetalState::new(None).unwrap();
    let scaled = poly.scale(offset);

    fft_with_blowup(scaled.coefficients(), blowup_factor, &metal_state)
}

/// Returns a new polynomial that interpolates `fft_evals`, which are evaluations using twiddle
/// factors. This is considered to be the inverse operation of [evaluate_fft_metal()].
pub fn interpolate_fft_metal<F>(
    fft_evals: &[FieldElement<F>],
) -> Result<Polynomial<FieldElement<F>>, MetalError>
where
    F: IsFFTField,
{
    let metal_state = MetalState::new(None).unwrap();

    // fft() can zero-pad the coeffs if there aren't 2^k of them (k being any integer).
    // TODO: twiddle factors need to be handled with too much care, the FFT API shouldn't accept
    // invalid twiddle factor collections. A better solution is needed.
    let order = fft_evals.len().next_power_of_two().trailing_zeros();

    let twiddles = gen_twiddles(order.into(), RootsConfig::BitReverseInversed, &metal_state)?;

    let coeffs = fft(fft_evals, &twiddles, &metal_state)?;

    let scale_factor = FieldElement::from(fft_evals.len() as u64).inv();
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}

#[cfg(feature = "metal")]
#[cfg(test)]
mod gpu_tests {
    use lambdaworks_fft::polynomial::{
        evaluate_offset_fft_cpu, forward_fft_cpu, interpolate_fft_cpu,
    };
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
        fn offset()(num in any::<u64>(), factor in any::<u64>()) -> FE { FE::from(num).pow(factor) }
    }
    prop_compose! {
        fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FE> {
            Polynomial::new(&coeffs)
        }
    }

    proptest! {
        #[test]
        fn test_metal_fft_poly_eval_matches_cpu(input in field_vec(6)) {
            objc::rc::autoreleasepool(|| {
                let gpu_evals = forward_fft_metal(&input).unwrap();
                let cpu_evals = forward_fft_cpu(&input).unwrap();

                prop_assert_eq!(gpu_evals, cpu_evals);
                Ok(())
            }).unwrap();
        }

        #[test]
        fn test_metal_fft_coset_poly_eval_matches_cpu(poly in poly(6), offset in offset(), blowup_factor in powers_of_two(4)) {
            objc::rc::autoreleasepool(|| {
                let gpu_evals = evaluate_offset_fft_metal(&poly, &offset, blowup_factor).unwrap();
                let cpu_evals = evaluate_offset_fft_cpu(&poly, &offset, blowup_factor).unwrap();

                prop_assert_eq!(gpu_evals, cpu_evals);
                Ok(())
            }).unwrap();
        }

        #[test]
        fn test_metal_fft_poly_interpol_matches_cpu(evals in field_vec(6)) {
            objc::rc::autoreleasepool(|| {
                let gpu_evals = interpolate_fft_metal(&evals).unwrap();
                let cpu_evals = interpolate_fft_cpu(&evals).unwrap();

                prop_assert_eq!(gpu_evals, cpu_evals);
                Ok(())
            }).unwrap();
        }
    }
}
