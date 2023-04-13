use crate::{
    field::{
        fields::montgomery_backed_prime_fields::{IsModulus, U256PrimeField},
        traits::IsTwoAdicField,
    },
    unsigned_integer::element::U256,
};

#[derive(Clone, Debug)]
pub struct MontgomeryConfigPlonkBLS12381ScalarField;
impl IsModulus<U256> for MontgomeryConfigPlonkBLS12381ScalarField {
    const MODULUS: U256 =
        U256::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
}

pub type BLS12381ScalarField = U256PrimeField<MontgomeryConfigPlonkBLS12381ScalarField>;

impl IsTwoAdicField for BLS12381ScalarField {
    const TWO_ADICITY: u64 = 32;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: U256 =
        U256::from("16a2a19edfe81f20d09b681922c813b4b63683508c2280b93829971f439f0d2b");
}

#[cfg(test)]
mod bls12381_two_adic_scalar_field_tests {
    use proptest::{
        collection, prelude::any, prop_assert_eq, prop_compose, proptest, strategy::Strategy,
    };

    use super::BLS12381ScalarField;
    use crate::{
        fft::helpers::log2,
        field::{
            element::FieldElement,
            traits::{IsTwoAdicField, RootsConfig},
        },
        polynomial::Polynomial,
    };

    type F = BLS12381ScalarField;
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
        fn test_fft_evaluation_is_correct_in_scalar_fft_friendly_field(poly in poly(8)) {
            let order = log2(poly.coefficients().len()).unwrap();
            let twiddles = F::get_powers_of_primitive_root(order, poly.coefficients.len(), RootsConfig::Natural).unwrap();

            let fft_eval = poly.evaluate_fft().unwrap();
            let naive_eval = poly.evaluate_slice(&twiddles);

            prop_assert_eq!(fft_eval, naive_eval);
        }
    }
}
