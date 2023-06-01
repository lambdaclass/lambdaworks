use super::{curve::BLS12381FieldElement, curve::BLS12381TwistCurveFieldElement};
use crate::field::element::LegendreSymbol;
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

            // instead of "sum" is -beta
            let alpha = a.pow(2u64) + b.pow(2u64);
            let gamma = alpha.legendre_symbol();
            match gamma {
                LegendreSymbol::One => {
                    let two = BLS12381FieldElement::from(2u64);
                    let two_inv = two.inv();
                    // calculate the square root of alpha
                    let (y_sqrt1, y_sqrt2) = alpha.sqrt()?;
                    let mut delta = (a.clone() + y_sqrt1) * two_inv.clone();

                    let legendre_delta = delta.legendre_symbol();
                    if legendre_delta == LegendreSymbol::MinusOne {
                        delta = (a + y_sqrt2) * two_inv;
                    };
                    let (x_sqrt_1, x_sqrt_2) = delta.sqrt()?;
                    let x_0 = select_sqrt_value_from_third_bit(x_sqrt_1, x_sqrt_2, third_bit);
                    let x_1 = b * (two * x_0.clone()).inv();
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
