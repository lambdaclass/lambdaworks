use crate::{
    elliptic_curve::{
        edwards::{point::EdwardsProjectivePoint, traits::IsEdwards},
        traits::IsEllipticCurve,
    },
    field::{element::FieldElement, fields::p448_goldilocks_prime_field::P448GoldilocksPrimeField},
};

#[derive(Debug, Clone)]
pub struct Ed448Goldilocks;

impl IsEllipticCurve for Ed448Goldilocks {
    type BaseField = P448GoldilocksPrimeField;
    type PointRepresentation = EdwardsProjectivePoint<Self>;

    /// Taken from https://www.rfc-editor.org/rfc/rfc7748#page-6
    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::from_hex("4f1970c66bed0ded221d15a622bf36da9e146570470f1767ea6de324a3d3a46412ae1af72ab66511433b80e18b00938e2626a82bc70cc05e").unwrap(),
            FieldElement::<Self::BaseField>::from_hex("693f46716eb6bc248876203756c9c7624bea73736ca3984087789c1e05a0c2d73ad3ff1ce67c39c4fdbd132c4ed7c8ad9808795bf230fa14").unwrap(),
            FieldElement::one(),
        ])
    }
}

impl IsEdwards for Ed448Goldilocks {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::one()
    }

    fn d() -> FieldElement<Self::BaseField> {
        -FieldElement::from(39081)
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
    type FE = FieldElement<P448GoldilocksPrimeField>;

    fn generator_times_5() -> EdwardsProjectivePoint<Ed448Goldilocks> {
        let x = FE::from_hex("7a9f9335a48dcb0e2ba7601eedb50def80cbcf728562ada756d761e8958812808bc0d57a920c3c96f07b2d8cefc6f950d0a99d1092030034").unwrap();
        let y = FE::from_hex("adfd751a2517edd3b9109ce4fd580ade260ca1823ab18fced86551f7b698017127d7a4ee59d2b33c58405512881f225443b4731472f435eb").unwrap();
        Ed448Goldilocks::create_point_from_affine(x, y).unwrap()
    }

    fn point_1() -> EdwardsProjectivePoint<Ed448Goldilocks> {
        let x = FE::from_hex("c591e0987244569fbb80a8edda5f9c5b30d9e7862acbb19ac0f24b766fe29c5e5782efc0e7a169f0c55c5524f8a9f9333ca985ec56404926").unwrap();
        let y = FE::from_hex("f969573bf05f19ac5824718d7483d3b83a3ece5847e25487ae2a176290ad2b1cb9c7f4dd55f6ca6c50209556d7fc16e0683c3177356ac9bc").unwrap();
        Ed448Goldilocks::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_7() -> EdwardsProjectivePoint<Ed448Goldilocks> {
        let x = FE::from_hex("63c9cd1d79f027458015c2013fc819dd0f46f71c21a11fee0c32998acd17bac5b06d0f2f1e1539cfc33223a6e989b2b119dae9bbb16c3743").unwrap();
        let y = FE::from_hex("654de66ab8be9fbeec6e72798a0ba2bb39c1888b99104de6cb0acf4516ea5e018bd292a1855f0fea673a5d8e8724d1b19ca52817db624f06").unwrap();
        Ed448Goldilocks::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn generator_satisfies_defining_equation() {
        let g = Ed448Goldilocks::generator().to_affine();
        assert_eq!(Ed448Goldilocks::defining_equation(g.x(), g.y()), FE::zero());
    }

    #[test]
    fn adding_generator_five_times_works() {
        let g_times_5 = Ed448Goldilocks::generator().operate_with_self(5_u16);
        assert_eq!(g_times_5, generator_times_5());
    }

    #[test]
    fn adding_point_1_seven_times_works() {
        let point_1 = point_1();
        let point_1_times_7 = point_1_times_7();
        assert_eq!(point_1.operate_with_self(7_u16), point_1_times_7);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(*p.x(), FE::from_hex("c591e0987244569fbb80a8edda5f9c5b30d9e7862acbb19ac0f24b766fe29c5e5782efc0e7a169f0c55c5524f8a9f9333ca985ec56404926").unwrap());
        assert_eq!(*p.y(), FE::from_hex("f969573bf05f19ac5824718d7483d3b83a3ece5847e25487ae2a176290ad2b1cb9c7f4dd55f6ca6c50209556d7fc16e0683c3177356ac9bc").unwrap());
        assert_eq!(*p.z(), FE::one());
    }

    #[test]
    fn create_invalid_points_panics() {
        assert_eq!(
            Ed448Goldilocks::create_point_from_affine(FE::from(1), FE::from(1)).unwrap_err(),
            EllipticCurveError::InvalidPoint
        )
    }
}
