use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        traits::IsFFTField,
    },
    unsigned_integer::element::{UnsignedInteger, U64},
};

pub type U64MontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 1>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigBabybear31PrimeField;
impl IsModulus<U64> for MontgomeryConfigBabybear31PrimeField {
    //Babybear Prime p = 2^31 - 2^27 + 1 = 0x78000001
    const MODULUS: U64 = U64::from_u64(2013265921);
}

pub type Babybear31PrimeField =
    U64MontgomeryBackendPrimeField<MontgomeryConfigBabybear31PrimeField>;

//a two-adic primitive root of unity is 21^(2^24)
// 21^(2^24)=1 mod 2013265921
// 2^27(2^4-1)+1 where n=27 (two-adicity) and k=2^4+1
impl IsFFTField for Babybear31PrimeField {
    const TWO_ADICITY: u64 = 24;

    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType = UnsignedInteger { limbs: [21] };

    fn field_name() -> &'static str {
        "babybear31"
    }
}

impl FieldElement<Babybear31PrimeField> {
    pub fn to_bytes_le(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        limbs[0].to_le_bytes()
    }

    pub fn to_bytes_be(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        limbs[0].to_be_bytes()
    }
}

impl PartialOrd for FieldElement<Babybear31PrimeField> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.representative().partial_cmp(&other.representative())
    }
}

impl Ord for FieldElement<Babybear31PrimeField> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.representative().cmp(&other.representative())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(test)]
    mod test_babybear_31_bytes_ops {
        use super::*;
        use crate::{field::element::FieldElement, traits::ByteConversion};

        #[test]
        #[cfg(feature = "std")]
        fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_le() {
            let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked(
                "\
            0123456701234567\
        ",
            );
            let bytes = element.to_bytes_le();
            let expected_bytes: [u8; 8] = ByteConversion::to_bytes_le(&element).try_into().unwrap();
            assert_eq!(bytes, expected_bytes);
        }

        #[test]
        #[cfg(feature = "std")]
        fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_be() {
            let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked(
                "\
            0123456701234567\
        ",
            );
            let bytes = element.to_bytes_be();
            let expected_bytes: [u8; 8] = ByteConversion::to_bytes_be(&element).try_into().unwrap();
            assert_eq!(bytes, expected_bytes);
        }

        #[test]
        fn byte_serialization_and_deserialization_works_le() {
            let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked(
                "\
            7654321076543210\
        ",
            );
            let bytes = element.to_bytes_le();
            let from_bytes = FieldElement::<Babybear31PrimeField>::from_bytes_le(&bytes).unwrap();
            assert_eq!(element, from_bytes);
        }

        #[test]
        fn byte_serialization_and_deserialization_works_be() {
            let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked(
                "\
            7654321076543210\
        ",
            );
            let bytes = element.to_bytes_be();
            let from_bytes = FieldElement::<Babybear31PrimeField>::from_bytes_be(&bytes).unwrap();
            assert_eq!(element, from_bytes);
        }
    }

    mod test_babybear_31_primitive_root {
        use super::*;
        use crate::field::{element::FieldElement, traits::IsFFTField};

        #[test]
        fn test_two_adic_root_of_unity() {
            let root = FieldElement::<Babybear31PrimeField>::from(
                Babybear31PrimeField::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY.limbs[0],
            );
            let result = root.pow(u64::pow(2, 24));

            //checks that Babybear31PrimeField::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY is a root of unity
            assert_eq!(result, FieldElement::<Babybear31PrimeField>::one());
        }
    }

    mod test_babybear_31_fft {
        use super::*;
        use crate::fft::cpu::roots_of_unity::{
            get_powers_of_primitive_root, get_powers_of_primitive_root_coset,
        };
        use crate::fft::polynomial::FFTPoly;
        use crate::field::{
            element::FieldElement,
            traits::{IsFFTField, RootsConfig},
        };
        use crate::polynomial::Polynomial;
        use proptest::{collection, prelude::*, std_facade::Vec};

        fn gen_fft_and_naive_evaluation<F: IsFFTField>(
            poly: Polynomial<FieldElement<F>>,
        ) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
            let len = poly.coeff_len().next_power_of_two();
            let order = len.trailing_zeros();
            let twiddles =
                get_powers_of_primitive_root(order.into(), len, RootsConfig::Natural).unwrap();

            let fft_eval = poly.evaluate_fft(1, None).unwrap();
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
                get_powers_of_primitive_root_coset(order.into(), len * blowup_factor, &offset)
                    .unwrap();

            let fft_eval = poly
                .evaluate_offset_fft(blowup_factor, None, &offset)
                .unwrap();
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
            let fft_poly = Polynomial::interpolate_fft(fft_evals).unwrap();

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
            let eval = poly.evaluate_fft(1, None).unwrap();
            let new_poly = Polynomial::interpolate_fft(&eval).unwrap();

            (poly, new_poly)
        }

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

        // FFT related tests
        type F = Babybear31PrimeField;
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
        }
    }
}
