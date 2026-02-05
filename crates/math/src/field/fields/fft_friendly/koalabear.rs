use crate::field::{
    fields::u32_montgomery_backend_prime_field::U32MontgomeryBackendPrimeField, traits::IsFFTField,
};

/// KoalaBear Prime Field p = 2^31 - 2^24 + 1 = 0x7f000001 = 2130706433
///
/// This is an FFT-friendly prime field with two-adicity 24.
/// p - 1 = 2^24 × 127, so the multiplicative group has a subgroup of order 2^24.
///
/// Reference: Plonky3's p3-koala-bear crate
pub type Koalabear31PrimeField = U32MontgomeryBackendPrimeField<2130706433>;

/// p = 2^31 - 2^24 + 1 = 2^24 * 127 + 1, then
/// there is a group in the field of order 2^24.
/// A two-adic primitive root of unity is 3^127 = 1791270792 because:
/// - (3^127)^(2^24) ≡ 1 mod 2130706433
/// - (3^127)^(2^23) ≡ -1 mod 2130706433 (primitive)
///
/// Reference: Derived from Plonky3's p3-koala-bear TWO_ADIC_GENERATOR = 3 (multiplicative generator)
#[cfg(not(feature = "cuda"))]
impl IsFFTField for Koalabear31PrimeField {
    const TWO_ADICITY: u64 = 24;

    const TWO_ADIC_PRIMITIVE_ROOT_OF_UNITY: Self::BaseType = 1791270792;

    fn field_name() -> &'static str {
        "koalabear31"
    }
}

// Comprehensive field axiom tests via macro
#[cfg(test)]
type KoalabearFE = crate::field::element::FieldElement<Koalabear31PrimeField>;

#[cfg(test)]
crate::impl_field_axiom_tests!(
    field: Koalabear31PrimeField,
    element: KoalabearFE,
);

// FFT field tests via macro (only when FFT is available)
#[cfg(all(test, not(feature = "cuda")))]
crate::impl_fft_field_tests!(
    field: Koalabear31PrimeField,
    element: KoalabearFE,
    two_adicity: 24,
);

#[cfg(test)]
mod tests {
    use super::*;
    mod koalabear_specific_tests {
        use super::*;
        #[cfg(feature = "alloc")]
        use crate::errors::CreationError;
        use crate::field::{element::FieldElement, errors::FieldError, traits::IsPrimeField};
        #[cfg(feature = "alloc")]
        use crate::traits::ByteConversion;
        type FE = FieldElement<Koalabear31PrimeField>;

        // KoalaBear modulus
        const ORDER: usize = 2130706433;

        #[test]
        fn two_plus_one_is_three() {
            let a = FE::from(2);
            let b = FE::one();
            let res = FE::from(3);

            assert_eq!(a + b, res)
        }

        #[test]
        fn one_minus_two_is_minus_one() {
            let a = FE::from(2);
            let b = FE::one();
            let res = FE::from((ORDER - 1) as u64);
            assert_eq!(b - a, res)
        }

        #[test]
        fn mul_by_zero_is_zero() {
            let a = FE::from(2);
            let b = FE::zero();
            assert_eq!(a * b, b)
        }

        #[test]
        fn neg_zero_is_zero() {
            let zero = FE::from(0);

            assert_eq!(-&zero, zero);
        }

        #[test]
        fn doubling() {
            assert_eq!(FE::from(2).double(), FE::from(2) + FE::from(2),);
        }

        #[test]
        fn order_is_0() {
            assert_eq!(FE::from((ORDER - 1) as u64) + FE::from(1), FE::from(0));
        }

        #[test]
        fn when_comparing_13_and_13_they_are_equal() {
            let a: FE = FE::from(13);
            let b: FE = FE::from(13);
            assert_eq!(a, b);
        }

        #[test]
        fn when_comparing_13_and_8_they_are_different() {
            let a: FE = FE::from(13);
            let b: FE = FE::from(8);
            assert_ne!(a, b);
        }

        #[test]
        fn mul_neutral_element() {
            let a: FE = FE::from(1);
            let b: FE = FE::from(2);
            assert_eq!(a * b, FE::from(2));
        }

        #[test]
        fn mul_2_3_is_6() {
            let a: FE = FE::from(2);
            let b: FE = FE::from(3);
            assert_eq!(a * b, FE::from(6));
        }

        #[test]
        fn mul_order_minus_1() {
            let a: FE = FE::from((ORDER - 1) as u64);
            let b: FE = FE::from((ORDER - 1) as u64);
            assert_eq!(a * b, FE::from(1));
        }

        #[test]
        fn inv_0_error() {
            let result = FE::from(0).inv();
            assert!(matches!(result, Err(FieldError::InvZeroError)))
        }

        #[test]
        fn inv_2_mul_2_is_1() {
            let a: FE = FE::from(2);
            assert_eq!(a * a.inv().expect("2 is invertible"), FE::from(1));
        }

        #[test]
        fn square_2_is_4() {
            assert_eq!(FE::from(2).square(), FE::from(4))
        }

        #[test]
        fn pow_2_3_is_8() {
            assert_eq!(FE::from(2).pow(3_u64), FE::from(8))
        }

        #[test]
        fn pow_p_minus_1() {
            assert_eq!(FE::from(2).pow(ORDER - 1), FE::from(1))
        }

        #[test]
        fn div_1() {
            assert_eq!(
                (FE::from(2) / FE::from(1)).expect("division by 1"),
                FE::from(2)
            )
        }

        #[test]
        fn div_4_2() {
            assert_eq!(
                (FE::from(4) / FE::from(2)).expect("division by 2"),
                FE::from(2)
            )
        }

        #[test]
        fn two_plus_its_additive_inv_is_0() {
            let two = FE::from(2);

            assert_eq!(two + (-&two), FE::from(0))
        }

        #[test]
        fn four_minus_three_is_1() {
            let four = FE::from(4);
            let three = FE::from(3);

            assert_eq!(four - three, FE::from(1))
        }

        #[test]
        fn zero_minus_1_is_order_minus_1() {
            let zero = FE::from(0);
            let one = FE::from(1);

            assert_eq!(zero - one, FE::from((ORDER - 1) as u64))
        }

        #[test]
        fn koalabear_uses_31_bits() {
            assert_eq!(Koalabear31PrimeField::field_bit_size(), 31);
        }

        #[test]
        fn test_two_adic_primitive_root_of_unity() {
            // Verify that 3^(2^24) = 1 mod p
            let root = FE::from(Koalabear31PrimeField::TWO_ADIC_PRIMITIVE_ROOT_OF_UNITY as u64);
            let order = 1u64 << Koalabear31PrimeField::TWO_ADICITY;
            assert_eq!(root.pow(order), FE::one());
        }

        #[test]
        fn test_two_adic_primitive_root_is_primitive() {
            // Verify that 3^(2^23) != 1 mod p (it's actually a primitive root)
            let root = FE::from(Koalabear31PrimeField::TWO_ADIC_PRIMITIVE_ROOT_OF_UNITY as u64);
            let half_order = 1u64 << (Koalabear31PrimeField::TWO_ADICITY - 1);
            assert_ne!(root.pow(half_order), FE::one());
        }

        #[test]
        #[cfg(feature = "alloc")]
        fn from_hex_bigger_than_u64_returns_error() {
            let x = FE::from_hex("5f103b0bd4397d4df560eb559f38353f80eeb6");
            assert!(matches!(x, Err(CreationError::InvalidHexString)))
        }

        #[test]
        #[cfg(feature = "alloc")]
        fn to_bytes_from_bytes_be_is_the_identity() {
            let x = FE::from_hex("5f103b").expect("valid hex");
            assert_eq!(FE::from_bytes_be(&x.to_bytes_be()).expect("valid bytes"), x);
        }

        #[test]
        #[cfg(feature = "alloc")]
        fn from_bytes_to_bytes_be_is_the_identity() {
            let bytes = [0, 0, 0, 1];
            assert_eq!(
                FE::from_bytes_be(&bytes)
                    .expect("valid bytes")
                    .to_bytes_be(),
                bytes
            );
        }

        #[test]
        #[cfg(feature = "alloc")]
        fn to_bytes_from_bytes_le_is_the_identity() {
            let x = FE::from_hex("5f103b").expect("valid hex");
            assert_eq!(FE::from_bytes_le(&x.to_bytes_le()).expect("valid bytes"), x);
        }

        #[test]
        #[cfg(feature = "alloc")]
        fn from_bytes_to_bytes_le_is_the_identity_4_bytes() {
            let bytes = [1, 0, 0, 0];
            assert_eq!(
                FE::from_bytes_le(&bytes)
                    .expect("valid bytes")
                    .to_bytes_le(),
                bytes
            );
        }

        #[test]
        #[cfg(feature = "alloc")]
        fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_le() {
            let element = FE::from_hex("0123456701234567").expect("valid hex");
            let bytes = element.to_bytes_le();
            let expected_bytes: [u8; 4] = ByteConversion::to_bytes_le(&element)
                .try_into()
                .expect("4 bytes");
            assert_eq!(bytes, expected_bytes);
        }

        #[test]
        #[cfg(feature = "alloc")]
        fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_be() {
            let element = FE::from_hex("0123456701234567").expect("valid hex");
            let bytes = element.to_bytes_be();
            let expected_bytes: [u8; 4] = ByteConversion::to_bytes_be(&element)
                .try_into()
                .expect("4 bytes");
            assert_eq!(bytes, expected_bytes);
        }

        #[test]
        #[cfg(feature = "alloc")]
        fn byte_serialization_and_deserialization_works_le() {
            let element = FE::from_hex("0x7654321076543210").expect("valid hex");
            let bytes = element.to_bytes_le();
            let from_bytes = FE::from_bytes_le(&bytes).expect("valid bytes");
            assert_eq!(element, from_bytes);
        }

        #[test]
        #[cfg(feature = "alloc")]
        fn byte_serialization_and_deserialization_works_be() {
            let element = FE::from_hex("7654321076543210").expect("valid hex");
            let bytes = element.to_bytes_be();
            let from_bytes = FE::from_bytes_be(&bytes).expect("valid bytes");
            assert_eq!(element, from_bytes);
        }
    }

    #[cfg(all(feature = "std", not(feature = "instruments")))]
    mod test_koalabear_31_fft {
        use super::*;
        #[cfg(not(feature = "cuda"))]
        use crate::fft::cpu::roots_of_unity::{
            get_powers_of_primitive_root, get_powers_of_primitive_root_coset,
        };
        use crate::field::element::FieldElement;
        #[cfg(not(feature = "cuda"))]
        use crate::field::traits::{IsFFTField, RootsConfig};
        use crate::polynomial::Polynomial;
        use proptest::{collection, prelude::*, std_facade::Vec};

        #[cfg(not(feature = "cuda"))]
        fn gen_fft_and_naive_evaluation<F: IsFFTField>(
            poly: Polynomial<FieldElement<F>>,
        ) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
            let len = poly.coeff_len().next_power_of_two();
            let order = len.trailing_zeros();
            let twiddles = get_powers_of_primitive_root(order.into(), len, RootsConfig::Natural)
                .expect("valid roots");

            let fft_eval = Polynomial::evaluate_fft::<F>(&poly, 1, None).expect("valid FFT");
            let naive_eval = poly.evaluate_slice(&twiddles);

            (fft_eval, naive_eval)
        }

        #[cfg(not(feature = "cuda"))]
        fn gen_fft_coset_and_naive_evaluation<F: IsFFTField>(
            poly: Polynomial<FieldElement<F>>,
            offset: FieldElement<F>,
            blowup_factor: usize,
        ) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
            let len = poly.coeff_len().next_power_of_two();
            let order = (len * blowup_factor).trailing_zeros();
            let twiddles =
                get_powers_of_primitive_root_coset(order.into(), len * blowup_factor, &offset)
                    .expect("valid coset roots");

            let fft_eval =
                Polynomial::evaluate_offset_fft::<F>(&poly, blowup_factor, None, &offset)
                    .expect("valid offset FFT");
            let naive_eval = poly.evaluate_slice(&twiddles);

            (fft_eval, naive_eval)
        }

        #[cfg(not(feature = "cuda"))]
        fn gen_fft_and_naive_interpolate<F: IsFFTField>(
            fft_evals: &[FieldElement<F>],
        ) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
            let order = fft_evals.len().trailing_zeros() as u64;
            let twiddles = get_powers_of_primitive_root(order, 1 << order, RootsConfig::Natural)
                .expect("valid roots");

            let naive_poly =
                Polynomial::interpolate(&twiddles, fft_evals).expect("valid interpolation");
            let fft_poly =
                Polynomial::interpolate_fft::<F>(fft_evals).expect("valid FFT interpolation");

            (fft_poly, naive_poly)
        }

        #[cfg(not(feature = "cuda"))]
        fn gen_fft_and_naive_coset_interpolate<F: IsFFTField>(
            fft_evals: &[FieldElement<F>],
            offset: &FieldElement<F>,
        ) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
            let order = fft_evals.len().trailing_zeros() as u64;
            let twiddles = get_powers_of_primitive_root_coset(order, 1 << order, offset)
                .expect("valid coset roots");

            let naive_poly =
                Polynomial::interpolate(&twiddles, fft_evals).expect("valid interpolation");
            let fft_poly = Polynomial::interpolate_offset_fft(fft_evals, offset)
                .expect("valid offset interpolation");

            (fft_poly, naive_poly)
        }

        #[cfg(not(feature = "cuda"))]
        fn gen_fft_interpolate_and_evaluate<F: IsFFTField>(
            poly: Polynomial<FieldElement<F>>,
        ) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
            let eval = Polynomial::evaluate_fft::<F>(&poly, 1, None).expect("valid FFT");
            let new_poly = Polynomial::interpolate_fft::<F>(&eval).expect("valid interpolation");

            (poly, new_poly)
        }

        prop_compose! {
            fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
            // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
            // also it can't exceed the test field's two-adicity.
        }
        prop_compose! {
            fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FieldElement<Koalabear31PrimeField> {
                FieldElement::<Koalabear31PrimeField>::from(num)
            }
        }
        prop_compose! {
            fn offset()(num in any::<u64>(), factor in any::<u64>()) -> FieldElement<Koalabear31PrimeField> { FieldElement::<Koalabear31PrimeField>::from(num).pow(factor) }
        }
        prop_compose! {
            fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<FieldElement<Koalabear31PrimeField>> {
                vec
            }
        }
        prop_compose! {
            fn non_power_of_two_sized_field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size power of two", |vec| !vec.len().is_power_of_two())) -> Vec<FieldElement<Koalabear31PrimeField>> {
                vec
            }
        }
        prop_compose! {
            fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FieldElement<Koalabear31PrimeField>> {
                Polynomial::new(&coeffs)
            }
        }
        prop_compose! {
            fn poly_with_non_power_of_two_coeffs(max_exp: u8)(coeffs in non_power_of_two_sized_field_vec(max_exp)) -> Polynomial<FieldElement<Koalabear31PrimeField>> {
                Polynomial::new(&coeffs)
            }
        }

        proptest! {
            // Property-based test that ensures FFT eval. gives same result as a naive polynomial evaluation.
            #[test]
            #[cfg(not(feature = "cuda"))]
            fn test_fft_matches_naive_evaluation(poly in poly(8)) {
                let (fft_eval, naive_eval) = gen_fft_and_naive_evaluation(poly);
                prop_assert_eq!(fft_eval, naive_eval);
            }

            // Property-based test that ensures FFT eval. with coset gives same result as a naive polynomial evaluation.
            #[test]
            #[cfg(not(feature = "cuda"))]
            fn test_fft_coset_matches_naive_evaluation(poly in poly(4), offset in offset(), blowup_factor in powers_of_two(4)) {
                let (fft_eval, naive_eval) = gen_fft_coset_and_naive_evaluation(poly, offset, blowup_factor);
                prop_assert_eq!(fft_eval, naive_eval);
            }

            // Property-based test that ensures FFT interpolation is the same as naive..
            #[test]
            #[cfg(not(feature = "cuda"))]
            fn test_fft_interpolate_matches_naive(fft_evals in field_vec(4)
                                                           .prop_filter("Avoid polynomials of size not power of two",
                                                                        |evals| evals.len().is_power_of_two())) {
                let (fft_poly, naive_poly) = gen_fft_and_naive_interpolate(&fft_evals);
                prop_assert_eq!(fft_poly, naive_poly);
            }

            // Property-based test that ensures FFT interpolation with an offset is the same as naive.
            #[test]
            #[cfg(not(feature = "cuda"))]
            fn test_fft_interpolate_coset_matches_naive(offset in offset(), fft_evals in field_vec(4)
                                                           .prop_filter("Avoid polynomials of size not power of two",
                                                                        |evals| evals.len().is_power_of_two())) {
                let (fft_poly, naive_poly) = gen_fft_and_naive_coset_interpolate(&fft_evals, &offset);
                prop_assert_eq!(fft_poly, naive_poly);
            }

            // Property-based test that ensures interpolation is the inverse operation of evaluation.
            #[test]
            #[cfg(not(feature = "cuda"))]
            fn test_fft_interpolate_is_inverse_of_evaluate(
                poly in poly(4).prop_filter("Avoid non pows of two", |poly| poly.coeff_len().is_power_of_two())) {
                let (poly, new_poly) = gen_fft_interpolate_and_evaluate(poly);
                prop_assert_eq!(poly, new_poly);
            }
        }
    }
}
