use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::element::U256;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

use super::field_extension::{Degree12ExtensionField, Degree2ExtensionField};

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
const BX_POINT: U256 =
    U256::from_hex_unchecked("0x2b149d40ceb8aaae81be18991be06ac3b5b4c5e559dbefa33267e6dc24a138e5");
const BY_POINT: U256 =
    U256::from_hex_unchecked("0x9713b03af0fed4cd2cafadeed8fdf4a74fa084e52d1852e4a2bd0685c315d2");
impl IsShortWeierstrass for BN254TwistCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::zero()
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::new([FieldElement::new(BX_POINT), FieldElement::new(BY_POINT)])
    }
}

impl ShortWeierstrassProjectivePoint<BN254TwistCurve> {
    /// This function is related to the map ψ: E_twist(𝔽p²) -> E(𝔽p¹²).
    /// Given an affine point G in E_twist(𝔽p²) returns x, y such that
    /// ψ(G) = (x', y', 1) with x' = x * x'' and y' = y * y''
    /// for some x'', y'' in 𝔽p².
    /// This is meant only to be used in the miller loop of the
    /// ate pairing before the final exponentiation.
    /// This is because elements in 𝔽p² raised to that
    /// power are 1 and so the final result of the ate pairing
    /// doens't depend on having this function output the exact
    /// values of x' and y'. And it is enough to work with x and y.
    pub fn to_fp12_unnormalized(&self) -> [FieldElement<Degree12ExtensionField>; 2] {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bn_254::field_extension::{BN254PrimeField, Degree2ExtensionField},
                traits::IsShortWeierstrass,
            },
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        unsigned_integer::element::{U256, U384},
    };

    use super::BN254TwistCurve;
    type Level0FE = FieldElement<BN254PrimeField>;
    type Level1FE = FieldElement<Degree2ExtensionField>;

    #[cfg(feature = "std")]
    use crate::elliptic_curve::short_weierstrass::point::{
        Endianness, PointFormat, ShortWeierstrassProjectivePoint,
    };

    #[test]
    fn create_generator() {
        let g = BN254TwistCurve::generator();
        let [x, y, _] = g.coordinates();
        assert_eq!(BN254TwistCurve::defining_equation(x, y), Level1FE::zero());
    }

    #[cfg(feature = "std")]
    #[test]
    fn serialize_deserialize_generator() {
        let g = BN254TwistCurve::generator();
        let bytes = g.serialize(PointFormat::Projective, Endianness::LittleEndian);

        let deserialized = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::deserialize(
            &bytes,
            PointFormat::Projective,
            Endianness::LittleEndian,
        )
        .unwrap();

        assert_eq!(deserialized, g);
    }

    #[test]
    fn add_points() {
        let px = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "2b4580f90a0e9fa8b93997b04e615edc9402fe4fe3d0ba28eb999ce0f81d0dc",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "24b5da2f6e95abba5b57be0fc16add2b2a4ee52450079095b5b64742d13bfa6b",
            )),
        ]);
        let py = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "12eefcc585ca0b8a84927abb81b801c79c639ef587493b3dc1b6487f0d79e3cb",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "1952b0717cbd43394858dce7095bb5756f53a03cca162bf498eaef57c10d384c",
            )),
        ]);

        // Similarly for point Q
        let qx = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "279b8ee0886dcf8e1bd526c09adf9447e02b545a12ebd1e6bfba393a05a37f0d",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "27cad5af482e60b7dc08f30c10fd632cf4de741279fa8464f8cd7f52a2b565e0",
            )),
        ]);
        let qy = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "a110620524dd503d81a0467c03bcb94f2d015e18fbc3e48bec76c700ad9d28a",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "14864350e4f2bdef7b39914359838a6bd53bda430dae1adf6ea541ad28a6de81",
            )),
        ]);
        let expectedx = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "279b8ee0886dcf8e1bd526c09adf9447e02b545a12ebd1e6bfba393a05a37f0d",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "248dc631199b98f427ae38c65de8fc03ad5fe320d8aaf56e98e842a89447e43e",
            )),
        ]);
        let expectedy = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "a110620524dd503d81a0467c03bcb94f2d015e18fbc3e48bec76c700ad9d28a",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "14864350e4f2bdef7b39914359838a6bd53bda430dae1adf6ea541ad28a6de81",
            )),
        ]);
        let p = BN254TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BN254TwistCurve::create_point_from_affine(qx, qy).unwrap();
        let expected = BN254TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();
        assert_eq!(p.operate_with(&q), expected);
    }
}
