
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
            FieldElement::<Self::BaseField>::new_base("41805FAE2224FAC314FF0D6A07713EB490D7DE3F01A4C6ECF10E502BD002599D"),
            FieldElement::<Self::BaseField>::new_base("3CC5E0409B7814DEB8D217956F2F64DC73906EF5FFC9AC291FEC2C6C42DCAC7A"),
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
        field::element::FieldElement, unsigned_integer::element::U256,
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
        let x = FEE::new_base("488D26D8ECCA1C6F5BE328547612C4A2E629CA6B6509FAA1E59DF5F180271664");
        let y = FEE::new_base("019E2A29248E9C7AE0383F7F3BF6A50D703415A980BD40D67C2D5C3BE46BE633");    

        BandersnatchCurve::create_point_from_affine(x, y).unwrap()
        }

    #[test]
    fn test_scalar_mul() -> () {
        // let f1 = FEE::new_base("9697B44AA99E68EA78A5E48044A88E43B4ECF5D3B1833BEA4ED1F96653C4FC2");
        // let f2 = FEE::new_base("393C141D1A492C2289B2528D0C468EEB87FAF4C35964D2470EB262F76B3E0C8");    

        // let f1f2 = f1.clone().mul(&f2);

        let g = BandersnatchCurve::generator();

        let result = g.operate_with_self(5u16);

        assert_eq!(result.x(), f1f2g_fixed().x());

        // let f1f2g = EdwardsAffine::from_str(
        //     "(16530491029447613915334753043669938793793987372416328257719459807614119987301, \
        //      42481140308370805476764840229335460092474682686441442216596889726548353970772)",
        // )
        // .unwrap();
        // let f1 = U256::from_hex("9697B44AA99E68EA78A5E48044A88E43B4ECF5D3B1833BEA4ED1F96653C4FC2").unwrap();
        // let f2 = U256::from_hex("393C141D1A492C2289B2528D0C468EEB87FAF4C35964D2470EB262F76B3E0C8").unwrap();    


        // let resultf1 = g.clone().to_affine().operate_with_self(f1);

        // let resultf1f2 = resultf1.clone().operate_with_self(f2);

        // let f1f2g = point_1().operate_with(&g);

        // let f1f2g_fixed = f1f2g_fixed();
        // assert_eq!(f1f2g_fixed().x(), resultf1f2.x());
        // assert_eq!(f1f2g_fixed().y(), resultf1f2.y());
        // assert_eq!(f1f2g_fixed().z(), resultf1f2.z());

        // assert_eq!(f1f2.clone(), f1f2.clone());
        // assert_eq!(f1f2.clone() , f1f2.clone());
    }

    #[test]
    fn test_create_valid_point_works(){
        let p = BandersnatchCurve::generator();
        

        assert_eq!(p,p.clone());

    }
}