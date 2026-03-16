use crate::field::traits::LegendreSymbol;

use super::curve::BLS12377FieldElement;
use super::field_extension::Degree2ExtensionField;
use crate::field::element::FieldElement;

type BLS12377TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

/// Compute a deterministic square root in Fp2 for BLS12-377.
///
/// Returns the root whose base-field component (x_0) is the smaller of the two
/// possible values. The caller applies Fp2 lexicographic sign selection.
#[must_use]
pub fn sqrt_qfe(input: &BLS12377TwistCurveFieldElement) -> Option<BLS12377TwistCurveFieldElement> {
    // Algorithm 8, https://eprint.iacr.org/2012/685.pdf
    if *input == BLS12377TwistCurveFieldElement::zero() {
        Some(BLS12377TwistCurveFieldElement::zero())
    } else {
        let a = input.value()[0].clone();
        let b = input.value()[1].clone();
        if b == BLS12377FieldElement::zero() {
            let (y_sqrt_1, y_sqrt_2) = a.sqrt()?;
            let y_aux = if y_sqrt_1.canonical() <= y_sqrt_2.canonical() {
                y_sqrt_1
            } else {
                y_sqrt_2
            };

            Some(BLS12377TwistCurveFieldElement::new([
                y_aux,
                BLS12377FieldElement::zero(),
            ]))
        } else {
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
                    let x_0 = if x_sqrt_1.canonical() <= x_sqrt_2.canonical() {
                        x_sqrt_1
                    } else {
                        x_sqrt_2
                    };
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
        let root = sqrt_qfe(&y_squared).unwrap();
        assert_eq!(&root * &root, y_squared);
    }

    #[test]
    fn test_sqrt_qfe_real_only() {
        // Test with a pure real Fp2 element
        let val = BLS12377TwistCurveFieldElement::new([
            BLS12377FieldElement::from(4u64),
            BLS12377FieldElement::zero(),
        ]);
        let root = sqrt_qfe(&val).unwrap();
        assert_eq!(&root * &root, val);
    }
}
