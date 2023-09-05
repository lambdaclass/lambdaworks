pub use super::field::FqField;
use crate::elliptic_curve::edwards::point::EdwardsProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::{elliptic_curve::edwards::traits::IsEdwards, field::element::FieldElement};

pub type BaseBandersnatchFieldElement = FqField;

#[derive(Clone, Debug)]
pub struct BandersnatchCurve;

impl IsEllipticCurve for BandersnatchCurve {
    type BaseField = BaseBandersnatchFieldElement;
    type PointRepresentation = EdwardsProjectivePoint<Self>;

    // Values are from https://github.com/arkworks-rs/curves/blob/5a41d7f27a703a7ea9c48512a4148443ec6c747e/ed_on_bls12_381_bandersnatch/src/curves/mod.rs#L120
    // Converted to Hex
    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new_base(
                "29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18",
            ),
            FieldElement::<Self::BaseField>::new_base(
                "2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166",
            ),
            FieldElement::one(),
        ])
    }
}

impl IsEdwards for BandersnatchCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base(
            "73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFEFFFFFFFC",
        )
    }

    fn d() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base(
            "6389C12633C267CBC66E3BF86BE3B6D8CB66677177E54F92B369F2F5188D58E7",
        )
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement, unsigned_integer::element::U256,
    };

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BaseBandersnatchFieldElement>;

    fn point_1() -> EdwardsProjectivePoint<BandersnatchCurve> {
        let x = FEE::new_base("29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18");
        let y = FEE::new_base("2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166");

        BandersnatchCurve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn test_scalar_mul() {
        let g = BandersnatchCurve::generator();
        let result1 = g.operate_with_self(5u16);

        assert_eq!(
            result1.x().clone(),
            FEE::new_base("68CBECE0B8FB55450410CBC058928A567EED293D168FAEF44BFDE25F943AABE0")
        );

        let scalar =
            U256::from_hex("1CFB69D4CA675F520CCE760202687600FF8F87007419047174FD06B52876E7E6")
                .unwrap();
        let result2 = g.operate_with_self(scalar);

        assert_eq!(
            result2.x().clone(),
            FEE::new_base("68CBECE0B8FB55450410CBC058928A567EED293D168FAEF44BFDE25F943AABE0")
        );
    }

    #[test]
    fn test_create_valid_point_works() {
        let p = BandersnatchCurve::generator();

        assert_eq!(p, p.clone());
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(
            *p.x(),
            FEE::new_base("29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18")
        );
        assert_eq!(
            *p.y(),
            FEE::new_base("2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166")
        );
        assert_eq!(*p.z(), FEE::new_base("1"));
    }

    #[test]
    fn create_invalid_points_panics() {
        assert_eq!(
            BandersnatchCurve::create_point_from_affine(FEE::from(1), FEE::from(1)).unwrap_err(),
            EllipticCurveError::InvalidPoint
        )
    }

    #[test]
    fn equality_works() {
        let g = BandersnatchCurve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = BandersnatchCurve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }
}
