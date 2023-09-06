use crate::field::traits::LegendreSymbol;

use super::{curve::BLS12381FieldElement, curve::BLS12381TwistCurveFieldElement};
use std::cmp::Ordering;

#[must_use]
pub fn select_sqrt_value_from_third_bit(
    sqrt_1: BLS12381FieldElement,
    sqrt_2: BLS12381FieldElement,
    third_bit: u8,
) -> BLS12381FieldElement {
    match (
        sqrt_1.representative().cmp(&sqrt_2.representative()),
        third_bit,
    ) {
        (Ordering::Greater, 0) => sqrt_2,
        (Ordering::Greater, _) | (Ordering::Less, 0) | (Ordering::Equal, _) => sqrt_1,
        (Ordering::Less, _) => sqrt_2,
    }
}

/// * `third_bit` - if 1, then the square root is the greater one, otherwise it is the smaller one.
#[must_use]
pub fn sqrt_qfe(
    input: &BLS12381TwistCurveFieldElement,
    third_bit: u8,
) -> Option<BLS12381TwistCurveFieldElement> {
    // Algorithm 8, https://eprint.iacr.org/2012/685.pdf
    if *input == BLS12381TwistCurveFieldElement::zero() {
        Some(BLS12381TwistCurveFieldElement::zero())
    } else {
        let a = input.value()[0].clone();
        let b = input.value()[1].clone();
        if b == BLS12381FieldElement::zero() {
            // second part is zero
            let (y_sqrt_1, y_sqrt_2) = a.sqrt()?;
            let y_aux = select_sqrt_value_from_third_bit(y_sqrt_1, y_sqrt_2, third_bit);

            Some(BLS12381TwistCurveFieldElement::new([
                y_aux,
                BLS12381FieldElement::zero(),
            ]))
        } else {
            // second part of the input field number is non-zero
            // instead of "sum" is: -beta
            let alpha = a.pow(2u64) + b.pow(2u64);
            let gamma = alpha.legendre_symbol();
            match gamma {
                LegendreSymbol::One => {
                    let two = BLS12381FieldElement::from(2u64);
                    let two_inv = two.inv().unwrap();
                    // calculate the square root of alpha
                    let (y_sqrt1, y_sqrt2) = alpha.sqrt()?;
                    let mut delta = (a.clone() + y_sqrt1) * two_inv.clone();

                    let legendre_delta = delta.legendre_symbol();
                    if legendre_delta == LegendreSymbol::MinusOne {
                        delta = (a + y_sqrt2) * two_inv;
                    };
                    let (x_sqrt_1, x_sqrt_2) = delta.sqrt()?;
                    let x_0 = select_sqrt_value_from_third_bit(x_sqrt_1, x_sqrt_2, third_bit);
                    let x_1 = b * (two * x_0.clone()).inv().unwrap();
                    Some(BLS12381TwistCurveFieldElement::new([x_0, x_1]))
                }
                LegendreSymbol::MinusOne => None,
                LegendreSymbol::Zero => {
                    unreachable!("The input is zero, but we already handled this case.")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::curve::BLS12381FieldElement;

    #[test]
    fn test_sqrt_qfe() {
        let c1 = BLS12381FieldElement::from_hex(
            "0x13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e",
        ).unwrap();
        let c0 = BLS12381FieldElement::from_hex(
        "0x024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8"
        ).unwrap();
        let qfe = super::BLS12381TwistCurveFieldElement::new([c0, c1]);

        let b1 = BLS12381FieldElement::from_hex("0x4").unwrap();
        let b0 = BLS12381FieldElement::from_hex("0x4").unwrap();
        let qfe_b = super::BLS12381TwistCurveFieldElement::new([b0, b1]);

        let cubic_value = qfe.pow(3_u64) + qfe_b;
        let root = super::sqrt_qfe(&cubic_value, 0).unwrap();

        let c0_expected = BLS12381FieldElement::from_hex("0x0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801").unwrap();
        let c1_expected = BLS12381FieldElement::from_hex("0x0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be").unwrap();
        let qfe_expected = super::BLS12381TwistCurveFieldElement::new([c0_expected, c1_expected]);

        let value_root = root.value();
        let value_qfe_expected = qfe_expected.value();

        assert_eq!(value_root[0].clone(), value_qfe_expected[0].clone());
        assert_eq!(value_root[1].clone(), value_qfe_expected[1].clone());
    }

    #[test]
    fn test_sqrt_qfe_2() {
        let c0 = BLS12381FieldElement::from_hex("0x02").unwrap();
        let c1 = BLS12381FieldElement::from_hex("0x00").unwrap();
        let qfe = super::BLS12381TwistCurveFieldElement::new([c0, c1]);

        let c0_expected = BLS12381FieldElement::from_hex("0x013a59858b6809fca4d9a3b6539246a70051a3c88899964a42bc9a69cf9acdd9dd387cfa9086b894185b9a46a402be73").unwrap();
        let c1_expected = BLS12381FieldElement::from_hex("0x02d27e0ec3356299a346a09ad7dc4ef68a483c3aed53f9139d2f929a3eecebf72082e5e58c6da24ee32e03040c406d4f").unwrap();
        let qfe_expected = super::BLS12381TwistCurveFieldElement::new([c0_expected, c1_expected]);

        let b1 = BLS12381FieldElement::from_hex("0x4").unwrap();
        let b0 = BLS12381FieldElement::from_hex("0x4").unwrap();
        let qfe_b = super::BLS12381TwistCurveFieldElement::new([b0, b1]);

        let root = super::sqrt_qfe(&(qfe.pow(3_u64) + qfe_b), 0).unwrap();

        let value_root = root.value();
        let value_qfe_expected = qfe_expected.value();

        assert_eq!(value_root[0].clone(), value_qfe_expected[0].clone());
        assert_eq!(value_root[1].clone(), value_qfe_expected[1].clone());
    }
}
