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
        FieldElement::new([
            FieldElement::from_hex_unchecked(
                "2b149d40ceb8aaae81be18991be06ac3b5b4c5e559dbefa33267e6dc24a138e5",
            ),
            FieldElement::from_hex_unchecked(
                "9713b03af0fed4cd2cafadeed8fdf4a74fa084e52d1852e4a2bd0685c315d2",
            ),
        ])
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
        unsigned_integer::element::U256,
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
                "8ae7459fe0d23419ec54b150574b77b1d0aa0785ce98d43365898a1d9168a2a",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "235ec75b0bbcca3f1bab9f3aa4c65d52ccb479cb398b54bd4b0f7e3a24454b44",
            )),
        ]);
        let py = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "214ebea9c718706be05072da305b74c1585f9e75dbf99c7859bf2292e07c1691",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "2efdafd49b6e2d718b4e3b3d78939a6463f5f84f4343ec6b1161971dd38af12f",
            )),
        ]);
        // Similarly for point Q
        let qx = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "2b87b85159311f97b20b4c0e27eed978cf94984b06203f853c43192de1579324",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "27b8bc83db0109e29df764a1379a0b9ba3dab41db33aa9be4aef88d4dd9cb275",
            )),
        ]);
        let qy = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "18e358db9be18771bb6d8ba89a7be0d521782f10af8398e981b1dd252d114bed",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "8aa3f8a241032d3832b0f52403eb4ea852e23ec4c1e6d39b08de5ac36a2d43b",
            )),
        ]);
        let expectedx = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "27e1bb6cb3f893ef4af84ff82bd36b0c0832e3c5d4649da024b41bfecdc74233",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "b04a4feada4eba73191184c5f39f98e7319dc888a2b258697511a2035723656",
            )),
        ]);
        let expectedy = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked(
                "a5490e3b00bc8e434f9a1ba734b05c27c525889bf117bb4d293f5aa54b238c5",
            )),
            Level0FE::new(U256::from_hex_unchecked(
                "136ce0ba382e5d37c3e05eff8365e0e6857eefa150096af33bdbdf327649c0eb",
            )),
        ]);
        let p = BN254TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BN254TwistCurve::create_point_from_affine(qx, qy).unwrap();
        //let expected = BN254TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();
        let expected = BN254TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();
        assert_eq!(p.operate_with(&q), expected);
    }
}
