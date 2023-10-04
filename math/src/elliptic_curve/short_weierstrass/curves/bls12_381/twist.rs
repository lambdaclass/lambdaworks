use super::curve::SEED;
use super::field_extension::{Degree12ExtensionField, Degree2ExtensionField};
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::traits::IsField;
use crate::unsigned_integer::element::U384;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

const GENERATOR_X_0: U384 = U384::from_hex_unchecked("024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8");
const GENERATOR_X_1: U384 = U384::from_hex_unchecked("13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e");
const GENERATOR_Y_0: U384 = U384::from_hex_unchecked("0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801");
const GENERATOR_Y_1: U384 = U384::from_hex_unchecked("0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be");

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12381TwistCurve;

impl IsEllipticCurve for BLS12381TwistCurve {
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

impl IsShortWeierstrass for BLS12381TwistCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::zero()
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::new([FieldElement::from(4), FieldElement::from(4)])
    }
}

impl BLS12381TwistCurve {
    pub fn is_on_curve(point: &<Self as IsEllipticCurve>::PointRepresentation) -> bool {
        if point.z() == &FieldElement::zero() {
            true
        } else {
            let point = point.to_affine();
            Self::defining_equation(&point.x(), &point.y()) == FieldElement::zero()
        }
    }
}

impl ShortWeierstrassProjectivePoint<BLS12381TwistCurve> {
    /// Returns the "Untwist-Frobenius-Twist" endomorphism
    pub fn psi(&self) -> Self {
        // Coefficients borrowed from https://github.com/Consensys/gnark-crypto/blob/55489ac07ca1a88bf0b830a29625fcb0d8879a48/ecc/bls12-381/bls12-381.go#L132C1-L135C138
        let psi_coeff_x = FieldElement::<<BLS12381TwistCurve as IsEllipticCurve>::BaseField>::new([
            FieldElement::zero(),
            FieldElement::new(U384::from_hex_unchecked("1A0111EA397FE699EC02408663D4DE85AA0D857D89759AD4897D29650FB85F9B409427EB4F49FFFD8BFD00000000AAAD"))
        ]);

        let psi_coeff_y = FieldElement::<<BLS12381TwistCurve as IsEllipticCurve>::BaseField>::new([
            FieldElement::new(U384::from_hex_unchecked("135203E60180A68EE2E9C448D77A2CD91C3DEDD930B1CF60EF396489F61EB45E304466CF3E67FA0AF1EE7B04121BDEA2")),
            FieldElement::new(U384::from_hex_unchecked("6AF0E0437FF400B6831E36D6BD17FFE48395DABC2D3435E77F76E17009241C5EE67992F72EC05F4C81084FBEDE3CC09"))
        ]);

        let frob_map_x = Degree2ExtensionField::frobenius_map(self.x().value());
        let frob_map_x_with_psi_coeff_x =
            Degree2ExtensionField::mul(&frob_map_x, psi_coeff_x.value());

        let frob_map_y = Degree2ExtensionField::frobenius_map(self.y().value());
        let frob_map_y_with_psi_coeff_y =
            Degree2ExtensionField::mul(&frob_map_y, psi_coeff_y.value());

        let frob_map_z = Degree2ExtensionField::frobenius_map(self.z().value());

        <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation::new([
            FieldElement::new(frob_map_x_with_psi_coeff_x),
            FieldElement::new(frob_map_y_with_psi_coeff_y),
            FieldElement::new(frob_map_z),
        ])
    }

    /// Returns true if the point is in the prime subgroup `G_2` of order `r`.
    /// Makes use of endomorphism for efficient point multiplication.
    pub fn is_in_subgroup(&self) -> bool {
        let seed_times_p = self.operate_with_self(SEED);
        let psi_plus_seed_times_p = self.psi().operate_with(&seed_times_p);

        BLS12381TwistCurve::is_on_curve(&psi_plus_seed_times_p)
    }

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
        if self.is_neutral_element() {
            [FieldElement::zero(), FieldElement::one()]
        } else {
            let [qx, qy, _] = self.coordinates();

            let result_x = FieldElement::new([
                FieldElement::new([FieldElement::zero(), qx.clone(), FieldElement::zero()]),
                FieldElement::zero(),
            ]);

            let result_y = FieldElement::new([
                FieldElement::zero(),
                FieldElement::new([FieldElement::zero(), qy.clone(), FieldElement::zero()]),
            ]);

            [result_x, result_y]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BLS12381TwistCurve;
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bls12_381::field_extension::{BLS12381PrimeField, Degree2ExtensionField},
                point::{Endianness, PointFormat, ShortWeierstrassProjectivePoint},
                traits::IsShortWeierstrass,
            },
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        unsigned_integer::element::U384,
    };

    type Level0FE = FieldElement<BLS12381PrimeField>;
    type Level1FE = FieldElement<Degree2ExtensionField>;

    #[test]
    fn create_generator() {
        let g = BLS12381TwistCurve::generator();
        let [x, y, _] = g.coordinates();
        assert_eq!(
            BLS12381TwistCurve::defining_equation(x, y),
            Level1FE::zero()
        );
    }

    #[test]
    fn serialize_deserialize_generator() {
        let g = BLS12381TwistCurve::generator();
        let bytes = g.serialize(PointFormat::Projective, Endianness::LittleEndian);

        let deserialized = ShortWeierstrassProjectivePoint::<BLS12381TwistCurve>::deserialize(
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
            Level0FE::new(U384::from_hex_unchecked("97798b4a61ac301bbee71e36b5174e2f4adfe3e1729bdae1fcc9965ae84181be373aa80414823eed694f1270014012d")),
            Level0FE::new(U384::from_hex_unchecked("c9852cc6e61868966249aec153b50b29b3c22409f4c7880fd13121981c103c8ef84d9ea29b552431360e82cf69219fa"))
        ]);
        let py = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("16cb3a60f3fa52c8273aceeb94c4c7303e8074aa9eedec7355bbb1e8cceedd4ec1497f573f62822140377b8e339619ed")),
            Level0FE::new(U384::from_hex_unchecked("1cd919b08afe06bebe9adf6223a55868a6fd8b77efc5c67b60fff39be36e9b44b7f10db16827c83b43ad2dad1947778"))
        ]);
        let qx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("b6bce994c23f6505131a5f6d4ce4ba30f5dab726780bef00517585cab02e17f4d015b26eeaff376dc236af26c0210f1")),
            Level0FE::new(U384::from_hex_unchecked("163163e71fdd96a84b6a24d3e7cd9d7c0f06961e6fe8b7ec9b27bef1dbef5cbaf557563f725229fc79814a294c0b8511"))
        ]);
        let qy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("1c6afffac96cd457b4ac797e5cef6951c83bb328737f57df44ba0cc513d499f736816877a6cf87f1359e79d10151e14")),
            Level0FE::new(U384::from_hex_unchecked("79e40e569c20182726c148ca72a6e862d03317a2cf75cd19c2be36e29e03da70acbefbfa7a4c4e1c088bf94ae6ba6ce"))
        ]);
        let expectedx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("63f209cd306e632cc91089bd6b3bb02a6679fd02931a6e2292976589426dfdff9366829d5f45d982413e8b9514e8965")),
            Level0FE::new(U384::from_hex_unchecked("11aae43845fcb3e633217c2851889cddb939a3d2ddf00a64e4e0a723c362dff2caabc640a1095ac5be4075d4f7edf17f"))
        ]);
        let expectedy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("83e21ca01826bca9221373faf03132b80128c24760c639b44bd7e0b6c11537ef239c01d31a25a58c7f67fb16df0234b")),
            Level0FE::new(U384::from_hex_unchecked("f45243fd699bba6c6ca644ad8070f7812e4987fb2c91f64139a293958ed373814ef7317c11c3496cd93b88871f5d2c7"))
        ]);
        let p = BLS12381TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BLS12381TwistCurve::create_point_from_affine(qx, qy).unwrap();
        let expected = BLS12381TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();
        assert_eq!(p.operate_with(&q), expected);
    }

    #[test]
    fn check_generator_g2_in_subgroup() {
        let gen = BLS12381TwistCurve::generator();
        assert!(gen.is_in_subgroup());
    }

    #[test]
    fn check_arbitrary_g2_point_in_subgroup() {
        let arb_point = BLS12381TwistCurve::generator().operate_with_self(420_u32);
        assert!(arb_point.is_in_subgroup());
    }
}
