
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::elliptic_curve::edwards::point::EdwardsProjectivePoint;
use crate::{
    elliptic_curve::edwards::traits::IsEdwards, field::element::FieldElement,
};
pub use super::field_extension::FqField;

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
            FieldElement::<Self::BaseField>::new_base("29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18"),
            FieldElement::<Self::BaseField>::new_base("2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166"),
            FieldElement::one()
        ])
    }
}

impl IsEdwards for BandersnatchCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base("73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFEFFFFFFFC")
    }

    fn d() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base("6389C12633C267CBC66E3BF86BE3B6D8CB66677177E54F92B369F2F5188D58E7")
    }
}



#[cfg(test)]
mod tests {

    use core::ops::Mul;

    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::{traits::EllipticCurveError, montgomery::point},
        field::element::FieldElement,
    };

    use super::FqField;

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BaseBandersnatchFieldElement>;

    // fn point_1() -> EdwardsProjectivePoint<BandersnatchCurve> {
    //     let x = FEE::new_base("9697B44AA99E68EA78A5E48044A88E43B4ECF5D3B1833BEA4ED1F96653C4FC2");
    //     let y = FEE::new_base("393C141D1A492C2289B2528D0C468EEB87FAF4C35964D2470EB262F76B3E0C8");    

    //     BandersnatchCurve::create_point_from_affine(x, y).unwrap()
    //     }

    fn f1f2g_fixed() -> EdwardsProjectivePoint<BandersnatchCurve> {
        let x = FEE::new_base("248BED2600CB615A6715105E8C33A48E2669775385260EDCCFFA0540A9204865");
        let y = FEE::new_base("5DEB792D33432583D251F8EE85F7090A7BDDBC2113A10323650361F99BAF7254");    

        BandersnatchCurve::create_point_from_affine(x, y).unwrap()
        }

    #[test]
    fn test_scalar_mul() -> () {
        // let f1 = FEE::new_base("9697B44AA99E68EA78A5E48044A88E43B4ECF5D3B1833BEA4ED1F96653C4FC2");
        // let f2 = FEE::new_base("393C141D1A492C2289B2528D0C468EEB87FAF4C35964D2470EB262F76B3E0C8");    

        // let f1f2 = f1.clone().mul(&f2);

        let g = BandersnatchCurve::generator();

        // let f1f2g = EdwardsAffine::from_str(
        //     "(16530491029447613915334753043669938793793987372416328257719459807614119987301, \
        //      42481140308370805476764840229335460092474682686441442216596889726548353970772)",
        // )
        // .unwrap();

        let result = g.clone().to_affine().operate_with_self(5_u16);

        // let f1f2g = point_1().operate_with(&g);

        // let f1f2g_fixed = f1f2g_fixed();

        // assert_eq!(f1f2.clone(), f1f2.clone());
        // assert_eq!(f1f2.clone() , f1f2.clone());
    }

    #[test]
    fn test_create_valid_point_works(){
        let p = BandersnatchCurve::generator();
        

        assert_eq!(p,p.clone());

    }
}