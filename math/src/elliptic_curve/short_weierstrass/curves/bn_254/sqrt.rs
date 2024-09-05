use super::curve::{BN254FieldElement, BN254TwistCurveFieldElement};
use crate::field::traits::LegendreSymbol;
use core::cmp::Ordering;

#[must_use]
pub fn select_sqrt_value_from_third_bit(
    sqrt_1: BN254FieldElement,
    sqrt_2: BN254FieldElement,
    third_bit: u8,
) -> BN254FieldElement {
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
    input: &BN254TwistCurveFieldElement,
    third_bit: u8,
) -> Option<BN254TwistCurveFieldElement> {
    // Algorithm 8, https://eprint.iacr.org/2012/685.pdf
    if *input == BN254TwistCurveFieldElement::zero() {
        Some(BN254TwistCurveFieldElement::zero())
    } else {
        let a = input.value()[0].clone();
        let b = input.value()[1].clone();
        if b == BN254FieldElement::zero() {
            // second part is zero
            let (y_sqrt_1, y_sqrt_2) = a.sqrt()?;
            let y_aux = select_sqrt_value_from_third_bit(y_sqrt_1, y_sqrt_2, third_bit);

            Some(BN254TwistCurveFieldElement::new([
                y_aux,
                BN254FieldElement::zero(),
            ]))
        } else {
            // second part of the input field number is non-zero
            // instead of "sum" is: -beta
            let alpha = a.pow(2u64) + b.pow(2u64);
            let gamma = alpha.legendre_symbol();
            match gamma {
                LegendreSymbol::One => {
                    let two = BN254FieldElement::from(2u64);
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
                    Some(BN254TwistCurveFieldElement::new([x_0, x_1]))
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
    use super::super::curve::{BN254FieldElement, BN254TwistCurveFieldElement};
    use super::super::twist::BN254TwistCurve;
    use crate::cyclic_group::IsGroup;
    use crate::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;
    use crate::elliptic_curve::traits::IsEllipticCurve;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    /// We took the q1 point of the test two_pairs_of_points_match_1 from pairing.rs
    /// to get the values of x and y.
    fn test_sqrt_qfe() {
        // Coordinate x of q.
        let x = super::BN254TwistCurveFieldElement::new([
            BN254FieldElement::from_hex_unchecked(
                "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
            ),
            BN254FieldElement::from_hex_unchecked(
                "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
            ),
        ]);

        let qfe_b = BN254TwistCurve::b();
        // The equation of the twisted curve is y^2 = x^3 + 3 /(9+u)
        let y_square = x.pow(3_u64) + qfe_b;
        let y = super::sqrt_qfe(&y_square, 0).unwrap();

        // Coordinate y of q.
        let y_expected = super::BN254TwistCurveFieldElement::new([
            BN254FieldElement::from_hex_unchecked(
                "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
            ),
            BN254FieldElement::from_hex_unchecked(
                "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
            ),
        ]);

        let value_y = y.value();
        let value_y_expected = y_expected.value();

        assert_eq!(value_y[0].clone(), value_y_expected[0].clone());
        assert_eq!(value_y[1].clone(), value_y_expected[1].clone());
    }

    #[test]
    /// We took the q1 point of the test two_pairs_of_points_match_2 from pairing.rs
    fn test_sqrt_qfe_2() {
        let x = super::BN254TwistCurveFieldElement::new([
            BN254FieldElement::from_hex_unchecked(
                "3010c68cb50161b7d1d96bb71edfec9880171954e56871abf3d93cc94d745fa1",
            ),
            BN254FieldElement::from_hex_unchecked(
                "0476be093a6d2b4bbf907172049874af11e1b6267606e00804d3ff0037ec57fd",
            ),
        ]);

        let qfe_b = BN254TwistCurve::b();

        let y_square = x.pow(3_u64) + qfe_b;
        let y = super::sqrt_qfe(&y_square, 0).unwrap();

        let y_expected = super::BN254TwistCurveFieldElement::new([
            BN254FieldElement::from_hex_unchecked(
                "01b33461f39d9e887dbb100f170a2345dde3c07e256d1dfa2b657ba5cd030427",
            ),
            BN254FieldElement::from_hex_unchecked(
                "14c059d74e5b6c4ec14ae5864ebe23a71781d86c29fb8fb6cce94f70d3de7a21",
            ),
        ]);

        let value_y = y.value();
        let value_y_expected = y_expected.value();

        assert_eq!(value_y[0].clone(), value_y_expected[0].clone());
        assert_eq!(value_y[1].clone(), value_y_expected[1].clone());
    }

    #[test]
    fn test_sqrt_qfe_3() {
        let g = BN254TwistCurve::generator().to_affine();
        let y = &g.coordinates()[1];
        let y_square = &y.square();
        let y_result = super::sqrt_qfe(&y_square, 0).unwrap();

        assert_eq!(y_result, y.clone());
    }

    #[test]
    fn test_sqrt_qfe_4() {
        let g = BN254TwistCurve::generator()
            .operate_with_self(2 as u16)
            .to_affine();
        let y = &g.coordinates()[1];
        let y_square = &y.square();
        let y_result = super::sqrt_qfe(&y_square, 0).unwrap();

        assert_eq!(y_result, y.clone());
    }

    #[test]
    fn test_sqrt_qfe_5() {
        let a = BN254TwistCurveFieldElement::new([
            BN254FieldElement::from(3),
            BN254FieldElement::from(4),
        ]);
        let a_square = a.square();
        let a_result = super::sqrt_qfe(&a_square, 0).unwrap();

        assert_eq!(a_result, a);
    }
    #[test]
    fn test_sqrt_qfe_random() {
        let mut rng = StdRng::seed_from_u64(42);
        let a_val: u64 = rng.gen();
        let b_val: u64 = rng.gen();
        let a = BN254TwistCurveFieldElement::new([
            BN254FieldElement::from(a_val),
            BN254FieldElement::from(b_val),
        ]);
        let a_square = a.square();
        let a_result = super::sqrt_qfe(&a_square, 0).unwrap();

        assert_eq!(a_result, a);
    }
}
