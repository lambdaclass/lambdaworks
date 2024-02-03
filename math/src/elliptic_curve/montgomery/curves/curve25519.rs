use crate::{
    elliptic_curve::{
        montgomery::{point::MontgomeryProjectivePoint, traits::IsMontgomery},
        traits::IsEllipticCurve,
    },
    field::{element::FieldElement, fields::curve25519_field::Curve25519PrimeField},
    unsigned_integer::element::U256,
};

#[derive(Debug, Clone)]
pub struct Curve25519;

pub const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

impl FieldElement<Curve25519PrimeField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U256::from(a_hex))
    }
}

impl IsEllipticCurve for Curve25519 {
    type BaseField = Curve25519PrimeField;
    type PointRepresentation = MontgomeryProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new_base("0x09"),
            FieldElement::<Self::BaseField>::new_base(
                "0x20ae19a1b8a086b4e01edd2c7748d14c923d4d7e6d7c61b229e9c5a27eced3d9",
            ),
            FieldElement::one(),
        ])
    }
}

impl IsMontgomery for Curve25519 {
    fn a() -> FieldElement<Self::BaseField> {
        // a = 0x76d06
        FieldElement::from(486662)
    }

    fn b() -> FieldElement<Self::BaseField> {
        // b = 0x1
        FieldElement::one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement,
    };

    #[allow(clippy::upper_case_acronyms)]
    type FE = FieldElement<Curve25519PrimeField>;

    fn point_1() -> MontgomeryProjectivePoint<Curve25519> {
        let x = FE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        Curve25519::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> MontgomeryProjectivePoint<Curve25519> {
        let x = FE::new_base("32bcce7e71eb50384918e0c9809f73bde357027c6bf15092dd849aa0eac274d43af4c68a65fb2cda381734af5eecd5c");
        let y = FE::new_base("11e48467b19458aabe7c8a42dc4b67d7390fdf1e150534caadddc7e6f729d8890b68a5ea6885a21b555186452b954d88");
        Curve25519::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn adding_five_times_point_1_works() {
        let point_1 = point_1();
        let point_1_times_5 = point_1_times_5();
        assert_eq!(point_1.operate_with_self(5_u16), point_1_times_5);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(*p.x(), FE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5"));
        assert_eq!(*p.y(), FE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0"));
        assert_eq!(*p.z(), FE::new_base("1"));
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            Curve25519::create_point_from_affine(FE::from(0), FE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = Curve25519::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    #[ignore]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = Curve25519::generator();
        let g2 = g.operate_with_self(2_u64);

        todo!()
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = Curve25519::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }
}
