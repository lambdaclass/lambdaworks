use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::element::U256;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

use super::field_extension::Degree2ExtensionField;

// X_0 : 10857046999023057135944570762232829481370756359578518086990519993285655852781
const GENERATOR_X_0: U256 =
    U256::from_hex_unchecked("1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed");
// X_1 : 11559732032986387107991004021392285783925812861821192530917403151452391805634
const GENERATOR_X_1: U256 =
    U256::from_hex_unchecked("198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2");
// Y_0 : 8495653923123431417604973247489272438418190587263600148770280649306958101930
const GENERATOR_Y_0: U256 =
    U256::from_hex_unchecked("12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa");
// Y_1 : 4082367875863433681332203403145435568316851327593401208105741076214120093531
const GENERATOR_Y_1: U256 =
    U256::from_hex_unchecked("90689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b");

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BN254TwistCurve;

impl IsEllipticCurve for BN254TwistCurve {
    type BaseField = Degree2ExtensionField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::new([
                FieldElement::new(GENERATOR_X_0),
                FieldElement::new(GENERATOR_X_1),
            ]),
            FieldElement::new([
                FieldElement::new(GENERATOR_Y_0),
                FieldElement::new(GENERATOR_Y_1),
            ]),
            FieldElement::one(),
        ])
    }
}

impl IsShortWeierstrass for BN254TwistCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::zero()
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::new([FieldElement::from(4), FieldElement::from(4)])
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bn_254::field_extension::{BN254PrimeField, Degree2ExtensionField},
                point::{Endianness, PointFormat, ShortWeierstrassProjectivePoint},
                traits::IsShortWeierstrass,
            },
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        unsigned_integer::element::U256,
    };

    use super::BN254TwistCurve;
    type Level0FE = FieldElement<BN254PrimeField>;
    type Level1FE = FieldElement<Degree2ExtensionField>;
}
