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
use crate::fft::gpu::cuda::polynomial::HasCudaFft;
#[cfg(feature = "cuda")]
use crate::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
#[cfg(feature = "cuda")]
use crate::field::fields::u64_goldilocks_field::{
    Degree2GoldilocksExtensionField, Degree3GoldilocksExtensionField, Goldilocks64Field,
};

/// Attempts CUDA-accelerated FFT evaluation by dispatching to concrete type `T`.
///
/// Returns `None` if `E` is not the same type as `T` (detected via `type_name`).
/// Returns `Some(result)` if `E` matches `T` and the CUDA kernel was invoked.
///
/// # Safety rationale
/// The `unsafe` casts reinterpret `FieldElement<E>` as `FieldElement<T>`. This is sound
/// because:
/// - `type_name` comparison within the same binary guarantees `E` and `T` are the
///   same type (both calls originate from the same compiler invocation).
/// - Runtime `assert_eq!` on `size_of` and `align_of` provides defense-in-depth,
///   catching any hypothetical mismatch before the cast executes.
/// - When `E == T`, `FieldElement<E>` and `FieldElement<T>` are the identical type
///   with identical layout — the cast is a no-op.
#[cfg(feature = "cuda")]
fn try_cuda_evaluate_as<E: IsField, T: HasCudaFft>(
    coeffs: &[FieldElement<E>],
) -> Option<Result<Vec<FieldElement<E>>, FFTError>> {
    if core::any::type_name::<E>() != core::any::type_name::<T>() {
        return None;
    }
    // Defense-in-depth: verify layout compatibility at runtime.
    // These assertions are zero-cost when E == T (always true), since the
    // compiler can constant-fold them in monomorphized code.
    assert_eq!(
        core::mem::size_of::<FieldElement<E>>(),
        core::mem::size_of::<FieldElement<T>>(),
    );
    assert_eq!(
        core::mem::align_of::<FieldElement<E>>(),
        core::mem::align_of::<FieldElement<T>>(),
    );
    // SAFETY: E and T are the same type (verified above).
    let typed: &[FieldElement<T>] =
        unsafe { core::slice::from_raw_parts(coeffs.as_ptr().cast(), coeffs.len()) };
    Some(T::cuda_evaluate_fft(typed).map(|v| {
        // SAFETY: same type, converting Vec<FieldElement<T>> back to Vec<FieldElement<E>>.
        let mut v = core::mem::ManuallyDrop::new(v);
        unsafe {
            Vec::from_raw_parts(
                v.as_mut_ptr().cast::<FieldElement<E>>(),
                v.len(),
                v.capacity(),
            )
        }
    }))
}

/// Attempts CUDA-accelerated FFT interpolation by dispatching to concrete type `T`.
/// See [`try_cuda_evaluate_as`] for safety rationale.
#[cfg(feature = "cuda")]
fn try_cuda_interpolate_as<E: IsField, T: HasCudaFft>(
    evals: &[FieldElement<E>],
) -> Option<Result<Polynomial<FieldElement<E>>, FFTError>> {
    if core::any::type_name::<E>() != core::any::type_name::<T>() {
        return None;
    }
    assert_eq!(
        core::mem::size_of::<FieldElement<E>>(),
        core::mem::size_of::<FieldElement<T>>(),
    );
    assert_eq!(
        core::mem::align_of::<FieldElement<E>>(),
        core::mem::align_of::<FieldElement<T>>(),
    );
    // SAFETY: E and T are the same type (verified above).
    let typed: &[FieldElement<T>] =
        unsafe { core::slice::from_raw_parts(evals.as_ptr().cast(), evals.len()) };
    Some(T::cuda_interpolate_fft(typed).map(|poly| {
        let src = poly.coefficients();
        // SAFETY: same type, reinterpreting coefficients back.
        let reinterpreted: &[FieldElement<E>] =
            unsafe { core::slice::from_raw_parts(src.as_ptr().cast(), src.len()) };
        Polynomial::new(reinterpreted)
    }))
}

use super::cpu::{ops, roots_of_unity};

impl<E: IsField> Polynomial<FieldElement<E>> {
    /// Returns `N` evaluations of this polynomial using FFT over a domain in a subfield F of E (so the results
    /// are P(w^i), with w being a primitive root of unity).
    /// `N = max(self.coeff_len(), domain_size).next_power_of_two() * blowup_factor`.
    /// If `domain_size` is `None`, it defaults to 0.
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

        let mut coeffs = poly.coefficients().to_vec();
        coeffs.resize(len, FieldElement::zero());
        // padding with zeros will make FFT return more evaluations of the same polynomial.

        #[cfg(feature = "cuda")]
        {
            // Try CUDA dispatch for all known field types via HasCudaFft trait.
            // Each call checks type_name<E> == type_name<T> and returns None on mismatch.
            if let Some(r) = try_cuda_evaluate_as::<E, Stark252PrimeField>(&coeffs) {
                return r;
            }
            if let Some(r) = try_cuda_evaluate_as::<E, Goldilocks64Field>(&coeffs) {
                return r;
            }
            if let Some(r) = try_cuda_evaluate_as::<E, Degree2GoldilocksExtensionField>(&coeffs) {
                return r;
            }
            if let Some(r) = try_cuda_evaluate_as::<E, Degree3GoldilocksExtensionField>(&coeffs) {
                return r;
            }
            evaluate_fft_cpu::<F, E>(&coeffs)
        }

        #[cfg(not(feature = "cuda"))]
        {
            evaluate_fft_cpu::<F, E>(&coeffs)
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

    /// Like [`Self::evaluate_fft`], but returns evaluations in **bit-reversed order**.
    /// The NR DIT FFT naturally produces bit-reversed output; this variant skips
    /// the final permutation to avoid a redundant round-trip when the output is
    /// destined for a Merkle tree commitment.
    pub fn evaluate_fft_bitrev<F: IsFFTField + IsSubFieldOf<E>>(
        poly: &Self,
        blowup_factor: usize,
        domain_size: Option<usize>,
    ) -> Result<Vec<FieldElement<E>>, FFTError> {
        let domain_size = domain_size.unwrap_or_default();
        let len = core::cmp::max(poly.coeff_len(), domain_size).next_power_of_two() * blowup_factor;
        if len.trailing_zeros() as u64 > F::TWO_ADICITY {
            return Err(FFTError::DomainSizeError(len.trailing_zeros() as usize));
        }
        if poly.coefficients().is_empty() {
            return Ok(vec![FieldElement::zero(); len]);
        }

        let mut coeffs = poly.coefficients().to_vec();
        coeffs.resize(len, FieldElement::zero());

        evaluate_fft_bitrev_cpu::<F, E>(&coeffs)
    }

    /// Like [`Self::evaluate_offset_fft`], but returns evaluations in **bit-reversed order**.
    pub fn evaluate_offset_fft_bitrev<F: IsFFTField + IsSubFieldOf<E>>(
        poly: &Self,
        blowup_factor: usize,
        domain_size: Option<usize>,
        offset: &FieldElement<F>,
    ) -> Result<Vec<FieldElement<E>>, FFTError> {
        let scaled = poly.scale(offset);
        Self::evaluate_fft_bitrev::<F>(&scaled, blowup_factor, domain_size)
    }

    /// Returns a new polynomial that interpolates `(w^i, fft_evals[i])`, with `w` being a
    /// Nth primitive root of unity in a subfield F of E, and `i in 0..N`, with `N = fft_evals.len()`.
    /// This is considered to be the inverse operation of [Self::evaluate_fft()].
    pub fn interpolate_fft<F: IsFFTField + IsSubFieldOf<E>>(
        fft_evals: &[FieldElement<E>],
    ) -> Result<Self, FFTError> {
        #[cfg(feature = "cuda")]
        {
            // Try CUDA dispatch for all known field types via HasCudaFft trait.
            if let Some(r) = try_cuda_interpolate_as::<E, Stark252PrimeField>(fft_evals) {
                return r;
            }
            if let Some(r) = try_cuda_interpolate_as::<E, Goldilocks64Field>(fft_evals) {
                return r;
            }
            if let Some(r) =
                try_cuda_interpolate_as::<E, Degree2GoldilocksExtensionField>(fft_evals)
            {
                return r;
            }
            if let Some(r) =
                try_cuda_interpolate_as::<E, Degree3GoldilocksExtensionField>(fft_evals)
            {
                return r;
            }
            interpolate_fft_cpu::<F, E>(fft_evals)
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
        let offset_inv = offset.inv().map_err(|_| FFTError::InverseOfZero)?;
        Ok(scaled.scale(&offset_inv))
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
        if divisor.is_zero() {
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
        if self.is_zero() {
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
) -> Result<Polynomial<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let poly_2_evaluations = Polynomial::evaluate_fft::<F>(poly_2, 1, None)?;

    let values: Vec<_> = poly_2_evaluations
        .iter()
        .map(|value| poly_1.evaluate(value))
        .collect();

    Polynomial::interpolate_fft::<F>(values.as_slice())
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

/// Like [`evaluate_fft_cpu`], but returns evaluations in bit-reversed order.
pub fn evaluate_fft_bitrev_cpu<F, E>(
    coeffs: &[FieldElement<E>],
) -> Result<Vec<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let order = coeffs.len().trailing_zeros();
    let twiddles = roots_of_unity::get_twiddles::<F>(order.into(), RootsConfig::BitReverse)?;
    ops::fft_bitrev(coeffs, &twiddles)
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

    let scale_factor = FieldElement::from(fft_evals.len() as u64)
        .inv()
        .map_err(|_| FFTError::InverseOfZero)?;
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
            fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FE> {
                Polynomial::new(&coeffs)
            }
        }
        prop_compose! {
            fn non_zero_poly(max_exp: u8)(coeffs in non_empty_field_vec(max_exp)) -> Polynomial<FE> {
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
                // Both unwraps are safe: `other` is generated by non_zero_poly
                prop_assert_eq!(
                    poly.fast_division::<F>(&other).expect("divisor is non-zero"),
                    poly.long_division_with_remainder(&other).expect("divisor is non-zero")
                );
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
                compose_fft::<F, F>(&p, &q).unwrap(),
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
            fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FE> {
                Polynomial::new(&coeffs)
            }
        }
        prop_compose! {
            fn non_zero_poly(max_exp: u8)(coeffs in non_empty_field_vec(max_exp)) -> Polynomial<FE> {
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
                // Both unwraps are safe: `other` is generated by non_zero_poly
                prop_assert_eq!(
                    poly.fast_division::<F>(&other).expect("divisor is non-zero"),
                    poly.long_division_with_remainder(&other).expect("divisor is non-zero")
                );
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

    #[cfg(feature = "cuda")]
    mod goldilocks_ext_cuda_polynomial_tests {
        use super::*;
        use crate::field::fields::u64_goldilocks_field::{
            Degree2GoldilocksExtensionField, Degree3GoldilocksExtensionField, Goldilocks64Field,
        };

        type GF = Goldilocks64Field;
        type GFE = FieldElement<GF>;
        type Fp2 = Degree2GoldilocksExtensionField;
        type Fp2E = FieldElement<Fp2>;
        type Fp3 = Degree3GoldilocksExtensionField;
        type Fp3E = FieldElement<Fp3>;

        #[test]
        fn test_cuda_polynomial_evaluate_fft_fp2() {
            let coeffs: Vec<Fp2E> = (0..4u64)
                .map(|i| Fp2E::from(&[GFE::from(i + 1), GFE::from(i + 10)]))
                .collect();
            let poly = Polynomial::new(&coeffs);

            let eval = Polynomial::evaluate_fft::<GF>(&poly, 1, None).unwrap();

            // Verify by interpolating back
            let recovered = Polynomial::interpolate_fft::<GF>(&eval).unwrap();
            assert_eq!(poly, recovered);
        }

        #[test]
        fn test_cuda_polynomial_evaluate_fft_fp3() {
            let coeffs: Vec<Fp3E> = (0..4u64)
                .map(|i| Fp3E::from(&[GFE::from(i + 1), GFE::from(i + 10), GFE::from(i + 100)]))
                .collect();
            let poly = Polynomial::new(&coeffs);

            let eval = Polynomial::evaluate_fft::<GF>(&poly, 1, None).unwrap();

            // Verify by interpolating back
            let recovered = Polynomial::interpolate_fft::<GF>(&eval).unwrap();
            assert_eq!(poly, recovered);
        }

        #[test]
        fn test_cuda_polynomial_evaluate_fft_fp2_matches_cpu() {
            let coeffs: Vec<Fp2E> = (0..8u64)
                .map(|i| Fp2E::from(&[GFE::from(i * 3 + 1), GFE::from(i * 7 + 2)]))
                .collect();
            let poly = Polynomial::new(&coeffs);

            // CPU evaluation: get roots of unity from the base field and evaluate naively
            let len = poly.coeff_len().next_power_of_two();
            let order = len.trailing_zeros();
            let twiddles: Vec<GFE> = roots_of_unity::get_powers_of_primitive_root(
                order.into(),
                len,
                RootsConfig::Natural,
            )
            .unwrap();
            // Embed base field roots into extension field for naive evaluation
            let ext_twiddles: Vec<Fp2E> = twiddles
                .iter()
                .map(|t| Fp2E::from(&[t.clone(), GFE::zero()]))
                .collect();
            let naive_eval = poly.evaluate_slice(&ext_twiddles);

            // CUDA evaluation via Polynomial::evaluate_fft
            let cuda_eval = Polynomial::evaluate_fft::<GF>(&poly, 1, None).unwrap();

            assert_eq!(cuda_eval, naive_eval);
        }

        #[test]
        fn test_cuda_polynomial_evaluate_fft_fp3_matches_cpu() {
            let coeffs: Vec<Fp3E> = (0..8u64)
                .map(|i| {
                    Fp3E::from(&[
                        GFE::from(i * 3 + 1),
                        GFE::from(i * 7 + 2),
                        GFE::from(i * 11 + 3),
                    ])
                })
                .collect();
            let poly = Polynomial::new(&coeffs);

            // CPU evaluation: get roots of unity from the base field and evaluate naively
            let len = poly.coeff_len().next_power_of_two();
            let order = len.trailing_zeros();
            let twiddles: Vec<GFE> = roots_of_unity::get_powers_of_primitive_root(
                order.into(),
                len,
                RootsConfig::Natural,
            )
            .unwrap();
            // Embed base field roots into extension field for naive evaluation
            let ext_twiddles: Vec<Fp3E> = twiddles
                .iter()
                .map(|t| Fp3E::from(&[t.clone(), GFE::zero(), GFE::zero()]))
                .collect();
            let naive_eval = poly.evaluate_slice(&ext_twiddles);

            // CUDA evaluation via Polynomial::evaluate_fft
            let cuda_eval = Polynomial::evaluate_fft::<GF>(&poly, 1, None).unwrap();

            assert_eq!(cuda_eval, naive_eval);
        }

        #[test]
        fn test_cuda_fft_fp2_matches_cpu_fft() {
            let coeffs: Vec<Fp2E> = (0..16u64)
                .map(|i| Fp2E::from(&[GFE::from(i * 5 + 1), GFE::from(i * 13 + 7)]))
                .collect();

            // CPU FFT path directly
            let cpu_result = evaluate_fft_cpu::<GF, Fp2>(&coeffs).unwrap();

            // CUDA FFT path: goes through Polynomial::evaluate_fft → try_evaluate_fft_ext_cuda
            let poly = Polynomial::new(&coeffs);
            let cuda_result = Polynomial::evaluate_fft::<GF>(&poly, 1, None).unwrap();

            assert_eq!(cuda_result, cpu_result);
        }

        #[test]
        fn test_cuda_fft_fp3_matches_cpu_fft() {
            let coeffs: Vec<Fp3E> = (0..16u64)
                .map(|i| {
                    Fp3E::from(&[
                        GFE::from(i * 5 + 1),
                        GFE::from(i * 13 + 7),
                        GFE::from(i * 17 + 3),
                    ])
                })
                .collect();

            // CPU FFT path directly
            let cpu_result = evaluate_fft_cpu::<GF, Fp3>(&coeffs).unwrap();

            // CUDA FFT path: goes through Polynomial::evaluate_fft → try_evaluate_fft_ext_cuda
            let poly = Polynomial::new(&coeffs);
            let cuda_result = Polynomial::evaluate_fft::<GF>(&poly, 1, None).unwrap();

            assert_eq!(cuda_result, cpu_result);
        }

        #[test]
        fn test_cuda_polynomial_roundtrip_fp2_large() {
            let coeffs: Vec<Fp2E> = (0..256u64)
                .map(|i| Fp2E::from(&[GFE::from(i + 1), GFE::from(i * 2 + 1)]))
                .collect();
            let poly = Polynomial::new(&coeffs);

            let eval = Polynomial::evaluate_fft::<GF>(&poly, 1, None).unwrap();
            let recovered = Polynomial::interpolate_fft::<GF>(&eval).unwrap();
            assert_eq!(poly, recovered);
        }

        #[test]
        fn test_cuda_polynomial_roundtrip_fp3_large() {
            let coeffs: Vec<Fp3E> = (0..256u64)
                .map(|i| {
                    Fp3E::from(&[GFE::from(i + 1), GFE::from(i * 2 + 1), GFE::from(i * 3 + 1)])
                })
                .collect();
            let poly = Polynomial::new(&coeffs);

            let eval = Polynomial::evaluate_fft::<GF>(&poly, 1, None).unwrap();
            let recovered = Polynomial::interpolate_fft::<GF>(&eval).unwrap();
            assert_eq!(poly, recovered);
        }
    }
}
