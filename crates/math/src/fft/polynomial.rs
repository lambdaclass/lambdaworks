use crate::fft::errors::FFTError;

use crate::field::errors::FieldError;
use crate::field::traits::{IsField, IsSubFieldOf};
use crate::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },
    polynomial::Polynomial,
};
use alloc::{vec, vec::Vec};

#[cfg(feature = "cuda")]
use crate::fft::gpu::cuda::polynomial::{evaluate_fft_cuda, interpolate_fft_cuda};

use super::cpu::{ops, roots_of_unity};

impl<E: IsField> Polynomial<FieldElement<E>> {
    /// Returns `N` evaluations of this polynomial using FFT over a domain in a subfield F of E (so the results
    /// are P(w^i), with w being a primitive root of unity).
    /// `N = max(self.coeff_len(), domain_size).next_power_of_two() * blowup_factor`.
    /// If `domain_size` is `None`, it defaults to 0.
    ///
    /// This uses degree-aware FFT optimization when the polynomial degree is much smaller
    /// than the domain size, achieving O(n log d) complexity instead of O(n log n).
    pub fn evaluate_fft<F: IsFFTField + IsSubFieldOf<E>>(
        poly: &Polynomial<FieldElement<E>>,
        blowup_factor: usize,
        domain_size: Option<usize>,
    ) -> Result<Vec<FieldElement<E>>, FFTError> {
        let domain_size = domain_size.unwrap_or(0);
        let len = core::cmp::max(poly.coeff_len(), domain_size).next_power_of_two() * blowup_factor;
        if len.trailing_zeros() as u64 > F::TWO_ADICITY {
            return Err(FFTError::DomainSizeError(len.trailing_zeros() as usize));
        }
        if poly.coefficients().is_empty() {
            return Ok(vec![FieldElement::zero(); len]);
        }

        // Capture the original coefficient count before padding for degree-aware optimization
        let original_coeff_count = poly.coeff_len();

        let mut coeffs = poly.coefficients().to_vec();
        coeffs.resize(len, FieldElement::zero());
        // padding with zeros will make FFT return more evaluations of the same polynomial.

        #[cfg(feature = "cuda")]
        {
            // TODO: support multiple fields with CUDA
            if F::field_name() == "stark256" {
                Ok(evaluate_fft_cuda(&coeffs)?)
            } else {
                evaluate_fft_cpu_degree_aware::<F, E>(&coeffs, original_coeff_count)
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            evaluate_fft_cpu_degree_aware::<F, E>(&coeffs, original_coeff_count)
        }
    }

    /// Returns `N` evaluations with an offset of this polynomial using FFT over a domain in a subfield F of E
    /// (so the results are P(w^i), with w being a primitive root of unity).
    /// `N = max(self.coeff_len(), domain_size).next_power_of_two() * blowup_factor`.
    /// If `domain_size` is `None`, it defaults to 0.
    pub fn evaluate_offset_fft<F: IsFFTField + IsSubFieldOf<E>>(
        poly: &Polynomial<FieldElement<E>>,
        blowup_factor: usize,
        domain_size: Option<usize>,
        offset: &FieldElement<F>,
    ) -> Result<Vec<FieldElement<E>>, FFTError> {
        let scaled = poly.scale(offset);
        Polynomial::evaluate_fft::<F>(&scaled, blowup_factor, domain_size)
    }

    /// Returns a new polynomial that interpolates `(w^i, fft_evals[i])`, with `w` being a
    /// Nth primitive root of unity in a subfield F of E, and `i in 0..N`, with `N = fft_evals.len()`.
    /// This is considered to be the inverse operation of [Self::evaluate_fft()].
    pub fn interpolate_fft<F: IsFFTField + IsSubFieldOf<E>>(
        fft_evals: &[FieldElement<E>],
    ) -> Result<Self, FFTError> {
        #[cfg(feature = "cuda")]
        {
            if !F::field_name().is_empty() {
                Ok(interpolate_fft_cuda(fft_evals)?)
            } else {
                interpolate_fft_cpu::<F, E>(fft_evals)
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            interpolate_fft_cpu::<F, E>(fft_evals)
        }
    }

    /// Returns a new polynomial that interpolates offset `(w^i, fft_evals[i])`, with `w` being a
    /// Nth primitive root of unity in a subfield F of E, and `i in 0..N`, with `N = fft_evals.len()`.
    /// This is considered to be the inverse operation of [Self::evaluate_offset_fft()].
    pub fn interpolate_offset_fft<F: IsFFTField + IsSubFieldOf<E>>(
        fft_evals: &[FieldElement<E>],
        offset: &FieldElement<F>,
    ) -> Result<Polynomial<FieldElement<E>>, FFTError> {
        let scaled = Polynomial::interpolate_fft::<F>(fft_evals)?;
        Ok(scaled.scale(&offset.inv().unwrap()))
    }

    /// Multiplies two polynomials using FFT.
    /// It's faster than naive multiplication when the degree of the polynomials is large enough (>=2**6).
    /// This works best with polynomials whose highest degree is equal to a power of 2 - 1.
    /// Will return an error if the degree of the resulting polynomial is greater than 2**63.
    ///
    /// This is an implementation of the fast division algorithm from
    /// [Gathen's book](https://www.cambridge.org/core/books/modern-computer-algebra/DB3563D4013401734851CF683D2F03F0)
    /// chapter 9
    pub fn fast_fft_multiplication<F: IsFFTField + IsSubFieldOf<E>>(
        &self,
        other: &Self,
    ) -> Result<Self, FFTError> {
        let domain_size = self.degree() + other.degree() + 1;
        let p = Polynomial::evaluate_fft::<F>(self, 1, Some(domain_size))?;
        let q = Polynomial::evaluate_fft::<F>(other, 1, Some(domain_size))?;
        let r = p.into_iter().zip(q).map(|(a, b)| a * b).collect::<Vec<_>>();

        Polynomial::interpolate_fft::<F>(&r)
    }

    /// Divides two polynomials with remainder.
    /// This is faster than the naive division if the degree of the divisor
    /// is greater than the degree of the dividend and both degrees are large enough.
    pub fn fast_division<F: IsSubFieldOf<E> + IsFFTField>(
        &self,
        divisor: &Self,
    ) -> Result<(Self, Self), FFTError> {
        let n = self.degree();
        let m = divisor.degree();
        if divisor.coefficients.is_empty()
            || divisor
                .coefficients
                .iter()
                .all(|c| c == &FieldElement::zero())
        {
            return Err(FieldError::DivisionByZero.into());
        }
        if n < m {
            return Ok((Self::zero(), self.clone()));
        }
        let d = n - m; // Degree of the quotient
        let a_rev = self.reverse(n);
        let b_rev = divisor.reverse(m);
        let inv_b_rev = b_rev.invert_polynomial_mod::<F>(d + 1)?;
        let q = a_rev
            .fast_fft_multiplication::<F>(&inv_b_rev)?
            .truncate(d + 1)
            .reverse(d);

        let r = self - q.fast_fft_multiplication::<F>(divisor)?;
        Ok((q, r))
    }

    /// Computes the inverse of polynomial P modulo x^k using Newton iteration.
    /// P must have an invertible constant term.
    pub fn invert_polynomial_mod<F: IsSubFieldOf<E> + IsFFTField>(
        &self,
        k: usize,
    ) -> Result<Self, FFTError> {
        if self.coefficients.is_empty()
            || self.coefficients.iter().all(|c| c == &FieldElement::zero())
        {
            return Err(FieldError::DivisionByZero.into());
        }
        let mut q = Self::new(&[self.coefficients[0].inv()?]);
        let mut current_precision = 1;

        let two = Self::new(&[FieldElement::<F>::one() + FieldElement::one()]);
        while current_precision < k {
            current_precision *= 2;
            let temp = self
                .fast_fft_multiplication::<F>(&q)?
                .truncate(current_precision);
            let correction = &two - temp;
            q = q
                .fast_fft_multiplication::<F>(&correction)?
                .truncate(current_precision);
        }

        // Final truncation to desired degree k
        Ok(q.truncate(k))
    }
}

pub fn compose_fft<F, E>(
    poly_1: &Polynomial<FieldElement<E>>,
    poly_2: &Polynomial<FieldElement<E>>,
) -> Polynomial<FieldElement<E>>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let poly_2_evaluations = Polynomial::evaluate_fft::<F>(poly_2, 1, None).unwrap();

    let values: Vec<_> = poly_2_evaluations
        .iter()
        .map(|value| poly_1.evaluate(value))
        .collect();

    Polynomial::interpolate_fft::<F>(values.as_slice()).unwrap()
}

pub fn evaluate_fft_cpu<F, E>(coeffs: &[FieldElement<E>]) -> Result<Vec<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let order = coeffs.len().trailing_zeros();
    let twiddles = roots_of_unity::get_twiddles::<F>(order.into(), RootsConfig::BitReverse)?;
    // Bit reverse order is needed for NR DIT FFT.
    ops::fft(coeffs, &twiddles)
}

/// FFT evaluation with degree-aware optimization for sparse polynomials.
///
/// When the polynomial has degree d << domain size n, this achieves O(n log d)
/// complexity instead of O(n log n).
///
/// Parameters:
/// - `coeffs`: Polynomial coefficients padded to domain_size with zeros
/// - `original_coeff_count`: Number of non-zero coefficients before padding
pub fn evaluate_fft_cpu_degree_aware<F, E>(
    coeffs: &[FieldElement<E>],
    original_coeff_count: usize,
) -> Result<Vec<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let order = coeffs.len().trailing_zeros();
    let twiddles = roots_of_unity::get_twiddles::<F>(order.into(), RootsConfig::BitReverse)?;
    // Bit reverse order is needed for NR DIT FFT.
    ops::fft_degree_aware(coeffs, &twiddles, original_coeff_count)
}

pub fn interpolate_fft_cpu<F, E>(
    fft_evals: &[FieldElement<E>],
) -> Result<Polynomial<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let order = fft_evals.len().trailing_zeros();
    let twiddles =
        roots_of_unity::get_twiddles::<F>(order.into(), RootsConfig::BitReverseInversed)?;

    let coeffs = ops::fft(fft_evals, &twiddles)?;

    let scale_factor = FieldElement::from(fft_evals.len() as u64).inv().unwrap();
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "cuda"))]
    use crate::field::traits::IsField;

    use crate::field::{
        test_fields::u64_test_field::{U64TestField, U64TestFieldExtension},
        traits::RootsConfig,
    };
    use proptest::{collection, prelude::*};

    use roots_of_unity::{get_powers_of_primitive_root, get_powers_of_primitive_root_coset};

    use super::*;

    fn gen_fft_and_naive_evaluation<F: IsFFTField>(
        poly: Polynomial<FieldElement<F>>,
    ) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
        let len = poly.coeff_len().next_power_of_two();
        let order = len.trailing_zeros();
        let twiddles =
            get_powers_of_primitive_root(order.into(), len, RootsConfig::Natural).unwrap();

        let fft_eval = Polynomial::evaluate_fft::<F>(&poly, 1, None).unwrap();
        let naive_eval = poly.evaluate_slice(&twiddles);

        (fft_eval, naive_eval)
    }

    fn gen_fft_coset_and_naive_evaluation<F: IsFFTField>(
        poly: Polynomial<FieldElement<F>>,
        offset: FieldElement<F>,
        blowup_factor: usize,
    ) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
        let len = poly.coeff_len().next_power_of_two();
        let order = (len * blowup_factor).trailing_zeros();
        let twiddles =
            get_powers_of_primitive_root_coset(order.into(), len * blowup_factor, &offset).unwrap();

        let fft_eval =
            Polynomial::evaluate_offset_fft::<F>(&poly, blowup_factor, None, &offset).unwrap();
        let naive_eval = poly.evaluate_slice(&twiddles);

        (fft_eval, naive_eval)
    }

    fn gen_fft_and_naive_interpolate<F: IsFFTField>(
        fft_evals: &[FieldElement<F>],
    ) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
        let order = fft_evals.len().trailing_zeros() as u64;
        let twiddles =
            get_powers_of_primitive_root(order, 1 << order, RootsConfig::Natural).unwrap();

        let naive_poly = Polynomial::interpolate(&twiddles, fft_evals).unwrap();
        let fft_poly = Polynomial::interpolate_fft::<F>(fft_evals).unwrap();

        (fft_poly, naive_poly)
    }

    fn gen_fft_and_naive_coset_interpolate<F: IsFFTField>(
        fft_evals: &[FieldElement<F>],
        offset: &FieldElement<F>,
    ) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
        let order = fft_evals.len().trailing_zeros() as u64;
        let twiddles = get_powers_of_primitive_root_coset(order, 1 << order, offset).unwrap();

        let naive_poly = Polynomial::interpolate(&twiddles, fft_evals).unwrap();
        let fft_poly = Polynomial::interpolate_offset_fft(fft_evals, offset).unwrap();

        (fft_poly, naive_poly)
    }

    fn gen_fft_interpolate_and_evaluate<F: IsFFTField>(
        poly: Polynomial<FieldElement<F>>,
    ) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
        let eval = Polynomial::evaluate_fft::<F>(&poly, 1, None).unwrap();
        let new_poly = Polynomial::interpolate_fft::<F>(&eval).unwrap();

        (poly, new_poly)
    }

    #[cfg(not(feature = "cuda"))]
    mod u64_field_tests {
        use super::*;
        use crate::field::test_fields::u64_test_field::U64TestField;

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
            fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<FE> {
                vec
            }
        }
        prop_compose! {
            fn non_empty_field_vec(max_exp: u8)(vec in collection::vec(field_element(), 1 << max_exp)) -> Vec<FE> {
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
            fn non_zero_poly(max_exp: u8)(coeffs in non_empty_field_vec(max_exp)) -> Polynomial<FE> {
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
                let (fft_eval, naive_eval) = gen_fft_and_naive_evaluation(poly);
                prop_assert_eq!(fft_eval, naive_eval);
            }

            // Property-based test that ensures FFT eval. with coset gives same result as a naive polynomial evaluation.
            #[test]
            fn test_fft_coset_matches_naive_evaluation(poly in poly(6), offset in offset(), blowup_factor in powers_of_two(4)) {
                let (fft_eval, naive_eval) = gen_fft_coset_and_naive_evaluation(poly, offset, blowup_factor);
                prop_assert_eq!(fft_eval, naive_eval);
            }

            // Property-based test that ensures FFT interpolation is the same as naive.
            #[test]
            fn test_fft_interpolate_matches_naive(fft_evals in field_vec(4)
                                                           .prop_filter("Avoid polynomials of size not power of two",
                                                                        |evals| evals.len().is_power_of_two())) {
                let (fft_poly, naive_poly) = gen_fft_and_naive_interpolate(&fft_evals);
                prop_assert_eq!(fft_poly, naive_poly);
            }

            // Property-based test that ensures FFT interpolation with an offset is the same as naive.
            #[test]
            fn test_fft_interpolate_coset_matches_naive(offset in offset(), fft_evals in field_vec(4)
                                                           .prop_filter("Avoid polynomials of size not power of two",
                                                                        |evals| evals.len().is_power_of_two())) {
                let (fft_poly, naive_poly) = gen_fft_and_naive_coset_interpolate(&fft_evals, &offset);
                prop_assert_eq!(fft_poly, naive_poly);
            }

            // Property-based test that ensures interpolation is the inverse operation of evaluation.
            #[test]
            fn test_fft_interpolate_is_inverse_of_evaluate(poly in poly(4)
                                                           .prop_filter("Avoid polynomials of size not power of two",
                                                                        |poly| poly.coeff_len().is_power_of_two())) {
                let (poly, new_poly) = gen_fft_interpolate_and_evaluate(poly);

                prop_assert_eq!(poly, new_poly);
            }

            #[test]
            fn test_fft_multiplication_works(poly in poly(7), other in poly(7)) {
                prop_assert_eq!(poly.fast_fft_multiplication::<F>(&other).unwrap(), poly * other);
            }

            #[test]
            fn test_fft_division_works(poly in non_zero_poly(7), other in non_zero_poly(7)) {
                prop_assert_eq!(poly.fast_division::<F>(&other).unwrap(), poly.long_division_with_remainder(&other));
            }

            #[test]
            fn test_invert_polynomial_mod_works(poly in non_zero_poly(7), k in powers_of_two(4)) {
                let inverted_poly = poly.invert_polynomial_mod::<F>(k).unwrap();
                prop_assert_eq!((poly * inverted_poly).truncate(k), Polynomial::new(&[FE::one()]));
            }
        }

        #[test]
        fn composition_fft_works() {
            let p = Polynomial::new(&[FE::new(0), FE::new(2)]);
            let q = Polynomial::new(&[FE::new(0), FE::new(0), FE::new(0), FE::new(1)]);
            assert_eq!(
                compose_fft::<F, F>(&p, &q),
                Polynomial::new(&[FE::new(0), FE::new(0), FE::new(0), FE::new(2)])
            );
        }
    }

    mod u256_field_tests {
        use super::*;
        use crate::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

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
            fn offset()(num in any::<u64>(), factor in any::<u64>()) -> FE { FE::from(num).pow(factor) }
        }
        prop_compose! {
            fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<FE> {
                vec
            }
        }
        prop_compose! {
            fn non_empty_field_vec(max_exp: u8)(vec in collection::vec(field_element(), 1 << max_exp)) -> Vec<FE> {
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
            fn non_zero_poly(max_exp: u8)(coeffs in non_empty_field_vec(max_exp)) -> Polynomial<FE> {
                Polynomial::new(&coeffs)
            }
        }
        prop_compose! {
            fn poly_with_non_power_of_two_coeffs(max_exp: u8)(coeffs in non_power_of_two_sized_field_vec(max_exp)) -> Polynomial<FE> {
                Polynomial::new(&coeffs)
            }
        }

        // FFT related tests
        type F = Stark252PrimeField;
        type FE = FieldElement<F>;

        proptest! {
            // Property-based test that ensures FFT eval. gives same result as a naive polynomial evaluation.
            #[test]
            fn test_fft_matches_naive_evaluation(poly in poly(8)) {
                let (fft_eval, naive_eval) = gen_fft_and_naive_evaluation(poly);
                prop_assert_eq!(fft_eval, naive_eval);
            }

            // Property-based test that ensures FFT eval. with coset gives same result as a naive polynomial evaluation.
            #[test]
            fn test_fft_coset_matches_naive_evaluation(poly in poly(4), offset in offset(), blowup_factor in powers_of_two(4)) {
                let (fft_eval, naive_eval) = gen_fft_coset_and_naive_evaluation(poly, offset, blowup_factor);
                prop_assert_eq!(fft_eval, naive_eval);
            }

            // Property-based test that ensures FFT interpolation is the same as naive..
            #[test]
            fn test_fft_interpolate_matches_naive(fft_evals in field_vec(4)
                                                           .prop_filter("Avoid polynomials of size not power of two",
                                                                        |evals| evals.len().is_power_of_two())) {
                let (fft_poly, naive_poly) = gen_fft_and_naive_interpolate(&fft_evals);
                prop_assert_eq!(fft_poly, naive_poly);
            }

            // Property-based test that ensures FFT interpolation with an offset is the same as naive.
            #[test]
            fn test_fft_interpolate_coset_matches_naive(offset in offset(), fft_evals in field_vec(4)
                                                           .prop_filter("Avoid polynomials of size not power of two",
                                                                        |evals| evals.len().is_power_of_two())) {
                let (fft_poly, naive_poly) = gen_fft_and_naive_coset_interpolate(&fft_evals, &offset);
                prop_assert_eq!(fft_poly, naive_poly);
            }

            // Property-based test that ensures interpolation is the inverse operation of evaluation.
            #[test]
            fn test_fft_interpolate_is_inverse_of_evaluate(
                poly in poly(4).prop_filter("Avoid non pows of two", |poly| poly.coeff_len().is_power_of_two())) {
                let (poly, new_poly) = gen_fft_interpolate_and_evaluate(poly);
                prop_assert_eq!(poly, new_poly);
            }

            #[test]
            fn test_fft_multiplication_works(poly in poly(7), other in poly(7)) {
                prop_assert_eq!(poly.fast_fft_multiplication::<F>(&other).unwrap(), poly * other);
            }

            #[test]
            fn test_fft_division_works(poly in poly(7), other in non_zero_poly(7)) {
                prop_assert_eq!(poly.fast_division::<F>(&other).unwrap(), poly.long_division_with_remainder(&other));
            }

            #[test]
            fn test_invert_polynomial_mod_works(poly in non_zero_poly(7), k in powers_of_two(4)) {
                let inverted_poly = poly.invert_polynomial_mod::<F>(k).unwrap();
                prop_assert_eq!((poly * inverted_poly).truncate(k), Polynomial::new(&[FE::one()]));
            }
        }
    }

    #[test]
    fn test_fft_with_values_in_field_extension_over_domain_in_prime_field() {
        type TF = U64TestField;
        type TL = U64TestFieldExtension;

        let a = FieldElement::<TL>::from(&[FieldElement::one(), FieldElement::one()]);
        let b = FieldElement::<TL>::from(&[-FieldElement::from(2), FieldElement::from(17)]);
        let c = FieldElement::<TL>::one();
        let poly = Polynomial::new(&[a, b, c]);

        let eval = Polynomial::evaluate_offset_fft::<TF>(&poly, 8, Some(4), &FieldElement::from(2))
            .unwrap();
        let new_poly =
            Polynomial::interpolate_offset_fft::<TF>(&eval, &FieldElement::from(2)).unwrap();
        assert_eq!(poly, new_poly);
    }
}
