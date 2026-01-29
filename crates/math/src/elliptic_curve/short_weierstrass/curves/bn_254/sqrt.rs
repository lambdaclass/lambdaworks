use super::curve::{BN254FieldElement, BN254TwistCurveFieldElement};
use crate::field::traits::LegendreSymbol;
use core::cmp::Ordering;

pub const TWO_INV: BN254FieldElement = BN254FieldElement::from_hex_unchecked(
    "183227397098D014DC2822DB40C0AC2ECBC0B548B438E5469E10460B6C3E7EA4",
);

/// Helper to compute x^(2^n) via repeated squaring
#[inline(always)]
fn square_repeated(x: &BN254FieldElement, n: usize) -> BN254FieldElement {
    let mut result = x.clone();
    for _ in 0..n {
        result = result.square();
    }
    result
}

/// Computes the inverse square root of `a` using an optimized addition chain.
/// Returns a^((p-3)/4) for BN254 prime p.
///
/// This is based on Constantine's addition chain for BN254.
/// Reference: https://github.com/mratsim/constantine
///
/// For p ≡ 3 (mod 4), we have:
/// - invsqrt(a) = a^((p-3)/4)
/// - sqrt(a) = invsqrt(a) * a = a^((p+1)/4)
#[must_use]
fn invsqrt_addchain(a: &BN254FieldElement) -> BN254FieldElement {
    // Addition chain for (p-3)/4 where p is the BN254 prime
    // Total: 301 operations (squarings + multiplications)

    let x10 = a.square();
    let x11 = &x10 * a;
    let x101 = &x10 * &x11;
    let x110 = &x101 * a;
    let x1000 = &x10 * &x110;
    let x1101 = &x101 * &x1000;
    let x10010 = &x101 * &x1101;
    let x10011 = &x10010 * a;
    let x10100 = &x10011 * a;
    let x10111 = &x11 * &x10100;
    let x11100 = &x101 * &x10111;
    let x100000 = &x1101 * &x10011;
    let x100011 = &x11 * &x100000;
    let x101011 = &x1000 * &x100011;
    let x101111 = &x10011 * &x11100;
    let x1000001 = &x10010 * &x101111;
    let x1010011 = &x10010 * &x1000001;
    let x1011011 = &x1000 * &x1010011;
    let x1100001 = &x110 * &x1011011;
    let x1110101 = &x10100 * &x1100001;
    let x10010001 = &x11100 * &x1110101;
    let x10010101 = &x100000 * &x1110101;
    let x10110101 = &x100000 * &x10010101;
    let x10111011 = &x110 * &x10110101;
    let x11000001 = &x110 * &x10111011;
    let x11000011 = &x10 * &x11000001;
    let x11010011 = &x10010 * &x11000001;
    let x11100001 = &x100000 * &x11000001;
    let x11100011 = &x10 * &x11100001;
    let x11100111 = &x110 * &x11100001; // 30 operations

    // 30 + 27 = 57 operations
    let mut r = square_repeated(&x11000001, 8);
    r = &r * &x10010001;
    r = square_repeated(&r, 10);
    r = &r * &x11100111;
    r = square_repeated(&r, 7);

    // 57 + 19 = 76 operations
    r = &r * &x10111;
    r = square_repeated(&r, 9);
    r = &r * &x10011;
    r = square_repeated(&r, 7);
    r = &r * &x1101;

    // 76 + 33 = 109 operations
    r = square_repeated(&r, 14);
    r = &r * &x1010011;
    r = square_repeated(&r, 9);
    r = &r * &x11100001;
    r = square_repeated(&r, 8);

    // 109 + 18 = 127 operations
    r = &r * &x1000001;
    r = square_repeated(&r, 10);
    r = &r * &x1011011;
    r = square_repeated(&r, 5);
    r = &r * &x1101;

    // 127 + 34 = 161 operations
    r = square_repeated(&r, 8);
    r = &r * &x11;
    r = square_repeated(&r, 12);
    r = &r * &x101011;
    r = square_repeated(&r, 12);

    // 161 + 25 = 186 operations
    r = &r * &x10111011;
    r = square_repeated(&r, 8);
    r = &r * &x101111;
    r = square_repeated(&r, 14);
    r = &r * &x10110101;

    // 186 + 28 = 214 operations
    r = square_repeated(&r, 9);
    r = &r * &x10010001;
    r = square_repeated(&r, 5);
    r = &r * &x1101;
    r = square_repeated(&r, 12);

    // 214 + 22 = 236 operations
    r = &r * &x11100011;
    r = square_repeated(&r, 8);
    r = &r * &x10010101;
    r = square_repeated(&r, 11);
    r = &r * &x11010011;

    // 236 + 32 = 268 operations
    r = square_repeated(&r, 7);
    r = &r * &x1100001;
    r = square_repeated(&r, 11);
    r = &r * &x100011;
    r = square_repeated(&r, 12);

    // 268 + 20 = 288 operations
    r = &r * &x1011011;
    r = square_repeated(&r, 9);
    r = &r * &x11000011;
    r = square_repeated(&r, 8);
    r = &r * &x11100111;

    // 288 + 13 = 301 operations
    r = square_repeated(&r, 7);
    r = &r * &x1110101;
    r = square_repeated(&r, 4);
    &r * a
}

/// Computes the square root of `a` in BN254 base field using an optimized addition chain.
///
/// For p ≡ 3 (mod 4), sqrt(a) = a^((p+1)/4).
/// This is computed as: sqrt(a) = invsqrt(a) * a where invsqrt(a) = a^((p-3)/4).
///
/// Returns `Some((sqrt, -sqrt))` if `a` is a quadratic residue, `None` otherwise.
#[must_use]
pub fn optimized_sqrt(a: &BN254FieldElement) -> Option<(BN254FieldElement, BN254FieldElement)> {
    if *a == BN254FieldElement::zero() {
        return Some((BN254FieldElement::zero(), BN254FieldElement::zero()));
    }

    // Compute sqrt using addition chain: sqrt(a) = invsqrt(a) * a
    let invsqrt = invsqrt_addchain(a);
    let sqrt = &invsqrt * a;

    // Verify that sqrt^2 == a (checks if a is a quadratic residue)
    if sqrt.square() == *a {
        let neg_sqrt = -&sqrt;
        Some((sqrt, neg_sqrt))
    } else {
        None
    }
}

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
            let alpha = a.square() + b.square();
            let gamma = alpha.legendre_symbol();
            match gamma {
                LegendreSymbol::One => {
                    let two = BN254FieldElement::from(2u64);
                    // calculate the square root of alpha
                    let (y_sqrt1, y_sqrt2) = alpha.sqrt()?;
                    let mut delta = (&a + y_sqrt1) * TWO_INV;

                    let legendre_delta = delta.legendre_symbol();
                    if legendre_delta == LegendreSymbol::MinusOne {
                        delta = (a + y_sqrt2) * TWO_INV;
                    };
                    let (x_sqrt_1, x_sqrt_2) = delta.sqrt()?;
                    let x_0 = select_sqrt_value_from_third_bit(x_sqrt_1, x_sqrt_2, third_bit);
                    let x_1 = b * (two * &x_0).inv().unwrap();
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
    use super::super::field_extension::BN254_PRIME_FIELD_ORDER;
    use super::super::twist::BN254TwistCurve;
    use crate::cyclic_group::IsGroup;
    use crate::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;
    use crate::elliptic_curve::traits::IsEllipticCurve;
    use crate::traits::ByteConversion;
    use crate::unsigned_integer::element::U256;
    use proptest::prelude::*;
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
        let y_square = x.square() * &x + qfe_b;
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
        let y_result = super::sqrt_qfe(y_square, 0).unwrap();

        assert_eq!(y_result, y.clone());
    }

    #[test]
    fn test_sqrt_qfe_4() {
        let g = BN254TwistCurve::generator()
            .operate_with_self(2_u16)
            .to_affine();
        let y = &g.coordinates()[1];
        let y_square = &y.square();
        let y_result = super::sqrt_qfe(y_square, 0).unwrap();

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

    // Tests for optimized optimized_sqrt (base field)

    #[test]
    fn test_optimized_sqrt_small_values() {
        // Test sqrt of 4 = 2
        let four = BN254FieldElement::from(4u64);
        let (sqrt1, sqrt2) = super::optimized_sqrt(&four).unwrap();
        assert!(sqrt1.square() == four);
        assert!(sqrt2.square() == four);

        // Test sqrt of 9 = 3
        let nine = BN254FieldElement::from(9u64);
        let (sqrt1, sqrt2) = super::optimized_sqrt(&nine).unwrap();
        assert!(sqrt1.square() == nine);
        assert!(sqrt2.square() == nine);
    }

    #[test]
    fn test_optimized_sqrt_zero() {
        let zero = BN254FieldElement::zero();
        let (sqrt1, sqrt2) = super::optimized_sqrt(&zero).unwrap();
        assert_eq!(sqrt1, BN254FieldElement::zero());
        assert_eq!(sqrt2, BN254FieldElement::zero());
    }

    #[test]
    fn test_optimized_sqrt_random_squares() {
        let mut rng = StdRng::seed_from_u64(12345);
        for _ in 0..10 {
            let val: u64 = rng.gen();
            let a = BN254FieldElement::from(val);
            let a_squared = a.square();
            let (sqrt1, sqrt2) = super::optimized_sqrt(&a_squared).unwrap();
            assert!(sqrt1.square() == a_squared);
            assert!(sqrt2.square() == a_squared);
            assert!(sqrt1 == a || sqrt2 == a);
        }
    }

    #[test]
    fn test_optimized_sqrt_matches_generic_sqrt() {
        // Verify our optimized sqrt matches the generic implementation
        let mut rng = StdRng::seed_from_u64(99999);
        for _ in 0..10 {
            let val: u64 = rng.gen();
            let a = BN254FieldElement::from(val);
            let a_squared = a.square();

            let (opt_sqrt1, opt_sqrt2) = super::optimized_sqrt(&a_squared).unwrap();
            let (gen_sqrt1, gen_sqrt2) = a_squared.sqrt().unwrap();

            // Both should produce valid square roots
            assert!(opt_sqrt1.square() == a_squared);
            assert!(gen_sqrt1.square() == a_squared);

            // The results should be the same (possibly with signs swapped)
            assert!(
                (opt_sqrt1 == gen_sqrt1 && opt_sqrt2 == gen_sqrt2)
                    || (opt_sqrt1 == gen_sqrt2 && opt_sqrt2 == gen_sqrt1)
            );
        }
    }

    #[test]
    fn test_optimized_sqrt_non_residue_returns_none() {
        // 3 is a quadratic non-residue for BN254
        let three = BN254FieldElement::from(3u64);
        assert!(super::optimized_sqrt(&three).is_none());
    }

    #[test]
    fn test_invsqrt_addchain_matches_pow_random() {
        let invsqrt_pow_exp = (BN254_PRIME_FIELD_ORDER - U256::from_u64(3)) >> 2;
        let mut rng = StdRng::seed_from_u64(424242);
        for _ in 0..32 {
            let bytes: [u8; 32] = rng.gen();
            let a = BN254FieldElement::from_bytes_be(&bytes).unwrap();
            let invsqrt_addchain = super::invsqrt_addchain(&a);
            let invsqrt_pow = a.pow(invsqrt_pow_exp);
            assert_eq!(invsqrt_addchain, invsqrt_pow);
        }
    }

    proptest! {
        #[test]
        fn prop_invsqrt_addchain_matches_pow(bytes in any::<[u8; 32]>()) {
            let invsqrt_pow_exp = (BN254_PRIME_FIELD_ORDER - U256::from_u64(3)) >> 2;
            let a = BN254FieldElement::from_bytes_be(&bytes).unwrap();
            let invsqrt_addchain = super::invsqrt_addchain(&a);
            let invsqrt_pow = a.pow(invsqrt_pow_exp);
            prop_assert_eq!(invsqrt_addchain, invsqrt_pow);
        }
    }
}
