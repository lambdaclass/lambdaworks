use super::{
    curve::BN254Curve,
    field_extension::{BN254PrimeField, Degree12ExtensionField, Degree2ExtensionField},
    twist::BN254TwistCurve,
};

use crate::elliptic_curve::traits::FromAffine;
use crate::{cyclic_group::IsGroup, elliptic_curve::traits::IsPairing, errors::PairingError};
use crate::{
    elliptic_curve::short_weierstrass::{
        curves::bn_254::field_extension::{Degree6ExtensionField, LevelTwoResidue},
        point::ShortWeierstrassProjectivePoint,
        traits::IsShortWeierstrass,
    },
    field::{element::FieldElement, extensions::cubic::HasCubicNonResidue},
    unsigned_integer::element::{UnsignedInteger, U256},
};
pub type BN254TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

type FpE = FieldElement<BN254PrimeField>;
type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp6E = FieldElement<Degree6ExtensionField>;
type Fp12E = FieldElement<Degree12ExtensionField>;

// Constants for the pairing
pub const X: u64 = 4965661367192848881;

/// @notice constant function for the coeffitients of the sextic twist of the BN256 curve.
/// @dev E': y' ** 2 = x' ** 3 + 3 / (9 + u)
/// @dev the curve E' is defined over Fp2 elements.
/// @dev See https://hackmd.io/@jpw/bn254#Twists for further details.
/// @return coefficients of the sextic twist of the BN256 curve
/// Constant from https://github.com/lambdaclass/zksync_era_precompiles/blob/4bdfebf831e21d58c5ba6945d4524763f1ef64d4/precompiles/EcPairing.yul
pub const TWISTED_CURVE_COEFFS: BN254TwistCurveFieldElement =
    BN254TwistCurveFieldElement::const_from_raw([
        FieldElement::from_hex_unchecked(
            "2514C6324384A86D26B7EDF049755260020B1B273633535D3BF938E377B802A8",
        ),
        FieldElement::from_hex_unchecked(
            "141b9ce4a688d4dd749d0dd22ac00aa65f0b37d93ce0d3e38e7ecccd1dcff67",
        ),
    ]);

// t = 6 * x^2
// NAF calculated in python.
pub const T: [i32; 128] = [
    0, -1, 0, 1, 0, 0, 1, 0, 1, 0, -1, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, -1,
    0, 0, -1, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1, 0, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1,
    0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1, 0, 0, 0, -1, 0, 0, -1, 0, -1, 0,
    0, 0, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 1, 0, 1, 0,
    -1, 0, 0, 0, -1, 0, 0, 1,
];

fn double_step(
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
) -> (Fp12E, ShortWeierstrassProjectivePoint<BN254TwistCurve>) {
    let two_inv = FpE::from(2).inv().unwrap();
    let t0 = q.x() * q.y();
    let t1 = &two_inv * t0;
    let t2 = q.y().square();
    let t3 = q.z().square();
    let mut t4 = t3.double();
    t4 = t4 + &t3;
    let mut t5 = TWISTED_CURVE_COEFFS;
    t5 = t4 * &t5;
    let mut t6 = t5.double();
    t6 = t6 + &t5;
    let mut t7 = &t2 + &t6;
    t7 = two_inv * t7;
    let mut t8 = q.y() + q.z();
    t8 = t8.square();
    let t9 = t3 + &t2;
    t8 = &t8 - t9;
    let t10 = &t5 - &t2;
    let t11 = q.x().square();
    let t12 = t5.square();
    let mut t13 = t12.double();
    t13 = t13 + t12;
    let l = Fp12E::new([
        Fp6E::new([t8.clone(), Fp2E::zero(), Fp2E::zero()]),
        Fp6E::new([t1.double() + &t1, t1.clone(), Fp2E::zero()]),
    ]);

    let x_r = (&t2 - t6) * t1;
    let y_r = t7.square() - t13;
    let z_r = t2 * t8;
    let r = ShortWeierstrassProjectivePoint::new([x_r, y_r, z_r]);

    (l, r)
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve,
        unsigned_integer::element::U384,
    };

    use super::*;

    #[test]
    // works
    fn double_step_doubles_point_correctly() {
        let q = BN254TwistCurve::generator();
        let r = double_step(&q).1;
        assert_eq!(r, q.operate_with_self(2usize));
    }
}
