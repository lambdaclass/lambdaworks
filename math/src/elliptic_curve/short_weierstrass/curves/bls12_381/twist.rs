use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::element::U384;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

use super::field_extension::LevelOneField;

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12381TwistCurve;

const C0X: U384 = U384::from("024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8");
const C1X: U384 = U384::from("13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e");
const C0Y: U384 = U384::from("0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801");
const C1Y: U384 = U384::from("0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be");

impl IsEllipticCurve for BLS12381TwistCurve {
    type BaseField = LevelOneField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::new([FieldElement::new(C0X),FieldElement::new(C1X)]),
            FieldElement::new([FieldElement::new(C0Y),FieldElement::new(C1Y)]),
            FieldElement::one()
        ])
    }
}

impl IsShortWeierstrass for BLS12381TwistCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::zero()
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::new([
            FieldElement::from(4),
            FieldElement::from(4)
        ])
    }
}

#[cfg(test)]
mod tests {
    use crate::{elliptic_curve::{traits::IsEllipticCurve, short_weierstrass::{traits::IsShortWeierstrass, curves::bls12_381::field_extension::LevelOneField}}, field::element::FieldElement, unsigned_integer::element::U384, cyclic_group::IsGroup};

    use super::BLS12381TwistCurve;
    type F = LevelOneField;

    #[test]
    fn create_generator() {
        let x = BLS12381TwistCurve::generator();
        assert_eq!(BLS12381TwistCurve::defining_equation(x.coordinates()), FieldElement::<F>::zero());
    }

    #[test]
    fn add_points() {
        let px = FieldElement::<F>::new([
            FieldElement::new(U384::from("97798b4a61ac301bbee71e36b5174e2f4adfe3e1729bdae1fcc9965ae84181be373aa80414823eed694f1270014012d")),
            FieldElement::new(U384::from("c9852cc6e61868966249aec153b50b29b3c22409f4c7880fd13121981c103c8ef84d9ea29b552431360e82cf69219fa"))
        ]);
        let py = FieldElement::<F>::new([
            FieldElement::new(U384::from("16cb3a60f3fa52c8273aceeb94c4c7303e8074aa9eedec7355bbb1e8cceedd4ec1497f573f62822140377b8e339619ed")),
            FieldElement::new(U384::from("1cd919b08afe06bebe9adf6223a55868a6fd8b77efc5c67b60fff39be36e9b44b7f10db16827c83b43ad2dad1947778"))
        ]);
        let qx = FieldElement::<F>::new([
            FieldElement::new(U384::from("b6bce994c23f6505131a5f6d4ce4ba30f5dab726780bef00517585cab02e17f4d015b26eeaff376dc236af26c0210f1")),
            FieldElement::new(U384::from("163163e71fdd96a84b6a24d3e7cd9d7c0f06961e6fe8b7ec9b27bef1dbef5cbaf557563f725229fc79814a294c0b8511"))
        ]);
        let qy = FieldElement::<F>::new([
            FieldElement::new(U384::from("1c6afffac96cd457b4ac797e5cef6951c83bb328737f57df44ba0cc513d499f736816877a6cf87f1359e79d10151e14")),
            FieldElement::new(U384::from("79e40e569c20182726c148ca72a6e862d03317a2cf75cd19c2be36e29e03da70acbefbfa7a4c4e1c088bf94ae6ba6ce"))
        ]);
        let expectedx = FieldElement::<F>::new([
            FieldElement::new(U384::from("63f209cd306e632cc91089bd6b3bb02a6679fd02931a6e2292976589426dfdff9366829d5f45d982413e8b9514e8965")),
            FieldElement::new(U384::from("11aae43845fcb3e633217c2851889cddb939a3d2ddf00a64e4e0a723c362dff2caabc640a1095ac5be4075d4f7edf17f"))
        ]);
        let expectedy = FieldElement::<F>::new([
            FieldElement::new(U384::from("83e21ca01826bca9221373faf03132b80128c24760c639b44bd7e0b6c11537ef239c01d31a25a58c7f67fb16df0234b")),
            FieldElement::new(U384::from("f45243fd699bba6c6ca644ad8070f7812e4987fb2c91f64139a293958ed373814ef7317c11c3496cd93b88871f5d2c7"))
        ]);
        let p = BLS12381TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BLS12381TwistCurve::create_point_from_affine(qx, qy).unwrap();
        let expected = BLS12381TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();
        assert_eq!(p.operate_with(&q), expected);
    }
}
