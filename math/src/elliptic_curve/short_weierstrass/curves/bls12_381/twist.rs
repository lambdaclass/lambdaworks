use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::u64::element::U384;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

use super::field_extension::{Degree12ExtensionField, Degree2ExtensionField};

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

impl ShortWeierstrassProjectivePoint<BLS12381TwistCurve> {
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
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bls12_381::field_extension::{BLS12381PrimeField, Degree2ExtensionField},
                traits::IsShortWeierstrass,
            },
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        unsigned_integer::u64::element::U384,
    };

    use super::BLS12381TwistCurve;
    type Level0FE = FieldElement<BLS12381PrimeField>;
    type Level1FE = FieldElement<Degree2ExtensionField>;

    #[cfg(feature = "alloc")]
    use crate::elliptic_curve::short_weierstrass::point::{
        Endianness, PointFormat, ShortWeierstrassProjectivePoint,
    };

    #[test]
    fn create_generator() {
        let g = BLS12381TwistCurve::generator();
        let [x, y, _] = g.coordinates();
        assert_eq!(
            BLS12381TwistCurve::defining_equation(x, y),
            Level1FE::zero()
        );
    }

    #[cfg(feature = "alloc")]
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
    // Numbers checked in SAGE
    fn add_points2() {
        let px = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x1414a51107b5ca989957046a1126425d371f5124215e294770f67fbf14dd92bbf1c9c2dbf35441769fa88427c17f0bb5")),
            Level0FE::new(U384::from_hex_unchecked("0x6224c8c8d6ecb882197551c68a25340be33975948d7da7568f6e00131307dc3688d320ad3c3c7cb95625082a47908f2"))
        ]);
        let py = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0xa69bb992a48dabcc49ab3fa1508bbc1acae14a9af09db39290b303de518806cec0067486adb6044f936d4bd2e5a151")),
            Level0FE::new(U384::from_hex_unchecked("0x98d34282ed5a2e265e455af63c66f7b5dd1557296f775463bcea891a14a801baa172e923055c4bb0fd5343e86294f41"))
        ]);

        let qx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0xae980b8c7483736e1904a1643e7a46f9980e52a5e65f0c5f7d195b30efd7173adea992c49a9073572d64ba67470e406")),
            Level0FE::new(U384::from_hex_unchecked("0x57d195c5f11d93558b52a74be27ae07f82f908ce35fabe58ce212c6d0bcef4a9f25e31fe92b2a49ea3fbc5d6c8cde99"))
        ]);
        let qy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x160c8799bfe8b80732e9ccb33ad35d1d23b6d01d8170ad088118a3cfe2a97dba2cf06ac3ce202fe039b105082ea48c22")),
            Level0FE::new(U384::from_hex_unchecked("0xc1e4cf5b2ca3deb2ec4f95b0ca0dbe79b0fc119f16f525e1d00054f009bcf2e2bde26f820b163e488bdc248adee1bfe"))
        ]);

        let expectedx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0xc9ef648884ce132f76f51a3fc536cc33e93c3a687c518abe1a0ddd4389df68c28214505a4b4a11a62d5b251badc7f9")),
            Level0FE::new(U384::from_hex_unchecked("0x6a43f39c279791e3ae0d36c3c24bee770593b80f44f931d70bbdcda17f9ed53b682dba192aa3c92b18d25e5d49b7d04"))
        ]);
        let expectedy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x128764b35cb5dbfe604968c5e3734ca80e16ee940f966c260096c29dd15aa6c6de3b1fc68a085403b8bdc012bf3b5b30")),
            Level0FE::new(U384::from_hex_unchecked("0x13471bd588bda43ce76f52dba32298bb46cfcf97a5f4484486e4394f6e38f7bd807ba62216d57ed8fd9df5f608c55ef1"))
        ]);

        let p = BLS12381TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BLS12381TwistCurve::create_point_from_affine(qx, qy).unwrap();
        let expected = BLS12381TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();
        assert_eq!(p.operate_with(&q), expected);
    }

    #[test]
    // Numbers checked in SAGE
    fn operate_with_self_test() {
        let px = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x1414a51107b5ca989957046a1126425d371f5124215e294770f67fbf14dd92bbf1c9c2dbf35441769fa88427c17f0bb5")),
            Level0FE::new(U384::from_hex_unchecked("0x6224c8c8d6ecb882197551c68a25340be33975948d7da7568f6e00131307dc3688d320ad3c3c7cb95625082a47908f2"))
        ]);

        let py = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0xa69bb992a48dabcc49ab3fa1508bbc1acae14a9af09db39290b303de518806cec0067486adb6044f936d4bd2e5a151")),
            Level0FE::new(U384::from_hex_unchecked("0x98d34282ed5a2e265e455af63c66f7b5dd1557296f775463bcea891a14a801baa172e923055c4bb0fd5343e86294f41"))
        ]);

        let qx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x16ba99ac9a28190dd74b8988e7a833f60e398472c363c2254c7db3138aff3a0858fb23e6cd2ca814a021b6b3b983f14a")),
            Level0FE::new(U384::from_hex_unchecked("0xe1356660c4a00b7ba5021f81949bd96680df9fa464a70d257c7b1bcae0e28ec15d84ddcef2ca2e4e8531f50177685dd"))
        ]);

        let qy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x9883c1c7d10c32d584f1cf5f0a7c0742f9b283144290afd6871abcb585e434516cefd2b159d99d75771f5658f0af628")),
            Level0FE::new(U384::from_hex_unchecked("0x13e5df65d734c9decf24356dacfcf9c4a317e5d21a7d1ada728f59e46ddfb137214bab47e8629a8016b6e508cafe141a"))
        ]);

        let scalar = U384::from_hex_unchecked(
            "0x1752428b56412bc55b5c6aca6e1811d1b5d810afd55169d8cffeae326bc8d6ea",
        );

        let p = BLS12381TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BLS12381TwistCurve::create_point_from_affine(qx, qy).unwrap();

        assert_eq!(p.operate_with_self(scalar), q);
    }
}
