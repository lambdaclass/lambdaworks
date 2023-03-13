use crate::{
    field::{
        fields::montgomery_backed_prime_fields::{IsMontgomeryConfiguration, U256PrimeField},
        traits::IsTwoAdicField,
    },
    unsigned_integer::element::{UnsignedInteger, U256},
};

#[derive(Clone, Debug)]
pub struct U256MontgomeryConfigTwoAdic;
impl IsMontgomeryConfiguration<4> for U256MontgomeryConfigTwoAdic {
    const MODULUS: U256 =
        U256::from("800000000000011000000000000000000000000000000000000000000000001");
}

impl IsTwoAdicField for U256MontgomeryTwoAdicPrimeField {
    // FIXME this field should be removed in the future.
    const GENERATOR: U256 = U256::from_u64(0);
    const TWO_ADICITY: u64 = 48;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: U256 = UnsignedInteger {
        limbs: [
            219038664817244121,
            2879838607450979157,
            15244050560987562958,
            16338897044258952332,
        ],
    };
}

pub type U256MontgomeryTwoAdicPrimeField = U256PrimeField<U256MontgomeryConfigTwoAdic>;

#[cfg(test)]
mod u256_two_adic_prime_field_tests {
    use super::U256MontgomeryTwoAdicPrimeField;
    use crate::{
        fft::{errors::FFTError, operations::evaluate_poly},
        field::{element::FieldElement, traits::IsTwoAdicField},
        polynomial::Polynomial,
    };

    type F = U256MontgomeryTwoAdicPrimeField;
    type FE = FieldElement<F>;

    // FIXME this should be removed to
    fn log2(n: usize) -> Result<u64, FFTError> {
        if !n.is_power_of_two() {
            return Err(FFTError::InvalidOrder(
                "The order of polynomial + 1 should a be power of 2".to_string(),
            ));
        }
        Ok(n.trailing_zeros() as u64)
    }

    #[test]
    fn test() {
        let coeffs = vec![FE::one(); 4];
        let poly = Polynomial::new(&coeffs);
        let result = evaluate_poly(&poly).unwrap();

        let omega = F::get_root_of_unity(log2(poly.coefficients().len()).unwrap()).unwrap();
        let twiddles_iter = (0..poly.coefficients().len() as u64).map(|i| omega.pow(i));
        let expected: Vec<FE> = twiddles_iter.map(|x| poly.evaluate(&x)).collect();

        assert_eq!(result, expected);
    }
}
