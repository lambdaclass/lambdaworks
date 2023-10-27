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

impl IsShortWeierstrass for BN254TwistCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::zero()
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::new([FieldElement::from(4), FieldElement::from(4)])
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
            Level0FE::new(U256::from_hex_unchecked("b7b3ccb18ad732d1a8b3e40484b7363f75cef89af6f8db06c35f88148b9706619e23a79d7ec4da022e8f99e4aad1e1a")),
            Level0FE::new(U256::from_hex_unchecked("0"))  // Assuming imaginary part is 0
        ]);
        let py = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked("83c7351a041b93078edcd824cfbf0986238a5537abf31c39bda0100f15080982abb5bae91cabb6e2ae057912c174f75")),
            Level0FE::new(U256::from_hex_unchecked("0"))  // Assuming imaginary part is 0
        ]);

        // Similarly for point Q
        let qx = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked("848d4c38b6cabfba64b43091135755cffcd6da65f6eb9a9529ce6ffd359f180642aa9393e66db97f7a8542c1d8e27dc")),
            Level0FE::new(U256::from_hex_unchecked("0"))  // Assuming imaginary part is 0
        ]);
        let qy = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked("ca390d05f986dba142f6f9b6d5467118bce40e10b4b2ffe86cc80ebbd1f1ccbe2a539d6fb6f570030691c3c62d84eca")),
            Level0FE::new(U256::from_hex_unchecked("0"))  // Assuming imaginary part is 0
        ]);
        let expectedx = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked("189abdeda1abc26e67f36216d2270212eff2161639eca25d9625229a476b204e80dda02a82cff695f6ea71d033741ae5")),
            Level0FE::new(U256::from_hex_unchecked("0")),
        ]);
        let expectedy = Level1FE::new([
            Level0FE::new(U256::from_hex_unchecked("18a35df258be38d31c58019d5f1a0bd68f399bc961e58e16afb74eaf0c3585ffd8cca6bffa32f4ee776a12adb24d4359")),
            Level0FE::new(U256::from_hex_unchecked("0")),
        ]);
        let p = BN254TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BN254TwistCurve::create_point_from_affine(qx, qy).unwrap();
        let expected = BN254TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();
        assert_eq!(p.operate_with(&q), expected);
    }
}
