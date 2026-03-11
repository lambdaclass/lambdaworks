use crate::field::traits::LegendreSymbol;

use super::curve::BLS12377FieldElement;
use super::field_extension::Degree2ExtensionField;
use crate::field::element::FieldElement;
use core::cmp::Ordering;

type BLS12377TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

#[must_use]
pub fn select_sqrt_value_from_third_bit(
    sqrt_1: BLS12377FieldElement,
    sqrt_2: BLS12377FieldElement,
    third_bit: u8,
) -> BLS12377FieldElement {
    match (sqrt_1.canonical().cmp(&sqrt_2.canonical()), third_bit) {
        (Ordering::Greater, 0) => sqrt_2,
        (Ordering::Greater, _) | (Ordering::Less, 0) | (Ordering::Equal, _) => sqrt_1,
        (Ordering::Less, _) => sqrt_2,
    }
}

/// Square root in Fp2 for BLS12-377.
/// * `third_bit` - if 1, then the square root is the greater one, otherwise it is the smaller one.
#[must_use]
pub fn sqrt_qfe(
    input: &BLS12377TwistCurveFieldElement,
    third_bit: u8,
) -> Option<BLS12377TwistCurveFieldElement> {
    // Algorithm 8, https://eprint.iacr.org/2012/685.pdf
    if *input == BLS12377TwistCurveFieldElement::zero() {
        Some(BLS12377TwistCurveFieldElement::zero())
    } else {
        let a = input.value()[0].clone();
        let b = input.value()[1].clone();
        if b == BLS12377FieldElement::zero() {
            let (y_sqrt_1, y_sqrt_2) = a.sqrt()?;
            let y_aux = select_sqrt_value_from_third_bit(y_sqrt_1, y_sqrt_2, third_bit);

            Some(BLS12377TwistCurveFieldElement::new([
                y_aux,
                BLS12377FieldElement::zero(),
            ]))
        } else {
            // alpha = a² + b² (since the non-residue is -5, norm = a² - (-5)*b² = a² + 5b²)
            // But for the algorithm from the paper, we use alpha = a² + b² when QNR = -1.
            // For BLS12-377, Fp2 = Fp[u]/(u² + 5), so norm(a + bu) = a² + 5b²
            let five = BLS12377FieldElement::from(5u64);
            let alpha = a.pow(2u64) + &five * b.pow(2u64);
            let gamma = alpha.legendre_symbol();
            match gamma {
                LegendreSymbol::One => {
                    let two = BLS12377FieldElement::from(2u64);
                    let two_inv = two.inv().unwrap();
                    let (y_sqrt1, y_sqrt2) = alpha.sqrt()?;
                    let mut delta = (a.clone() + y_sqrt1) * two_inv.clone();

                    let legendre_delta = delta.legendre_symbol();
                    if legendre_delta == LegendreSymbol::MinusOne {
                        delta = (a + y_sqrt2) * two_inv;
                    };
                    let (x_sqrt_1, x_sqrt_2) = delta.sqrt()?;
                    let x_0 = select_sqrt_value_from_third_bit(x_sqrt_1, x_sqrt_2, third_bit);
                    let x_1 = b * (two * x_0.clone()).inv().unwrap();
                    Some(BLS12377TwistCurveFieldElement::new([x_0, x_1]))
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
    use super::*;
    use crate::elliptic_curve::short_weierstrass::curves::bls12_377::twist::BLS12377TwistCurve;
    use crate::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;
    use crate::elliptic_curve::traits::IsEllipticCurve;

    #[test]
    fn test_sqrt_qfe_from_generator() {
        let g = BLS12377TwistCurve::generator();
        let x = g.x();
        let b = BLS12377TwistCurve::b();

        let y_squared = x.pow(3_u64) + b;
        let root = sqrt_qfe(&y_squared, 0).unwrap();
        assert_eq!(&root * &root, y_squared);
    }

    #[test]
    fn test_sqrt_qfe_real_only() {
        // Test with a pure real Fp2 element
        let val = BLS12377TwistCurveFieldElement::new([
            BLS12377FieldElement::from(4u64),
            BLS12377FieldElement::zero(),
        ]);
        let root = sqrt_qfe(&val, 0).unwrap();
        assert_eq!(&root * &root, val);
    }
}
