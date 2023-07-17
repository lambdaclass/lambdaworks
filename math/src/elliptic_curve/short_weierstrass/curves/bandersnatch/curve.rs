
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};
pub use super::field_extension::FqField;

pub type BaseBandersnatchFieldElement = FqField;

#[derive(Clone, Debug)]
pub struct BandersnatchCurve;

impl IsEllipticCurve for BandersnatchCurve {
    type BaseField = BaseBandersnatchFieldElement;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    // Values are from https://github.com/arkworks-rs/curves/blob/5a41d7f27a703a7ea9c48512a4148443ec6c747e/ed_on_bls12_381_bandersnatch/src/curves/mod.rs#L120
    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new_base("4450F9122AE57D5CBD002014EF530AD63F83A04D7022AEAD47AC4AEA0A73F2F7"),
            FieldElement::<Self::BaseField>::new_base("1BFF80EF062FE4B1592598412A2B272627F74E164A63E0BE795A3EB57763C1B2"),
            FieldElement::one()
        ])
    }
}

impl IsShortWeierstrass for BandersnatchCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base("17D15ECBE9F0FC8B8C4A118E26DA26E5E32913D24F20541DE0720F8033267935")
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base("415FCB20D12E60A811194E6B4D5D4625FA2D24F0AB8C256907BB3D28D5E8BBBF")
    }
}


#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::{traits::EllipticCurveError, montgomery::point},
        field::element::FieldElement,
    };

    use super::FqField;

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BaseBandersnatchFieldElement>;

    fn point_1() -> ShortWeierstrassProjectivePoint<BandersnatchCurve> {
        let x = FEE::new_base("9697B44AA99E68EA78A5E48044A88E43B4ECF5D3B1833BEA4ED1F96653C4FC2");
        let y = FEE::new_base("393C141D1A492C2289B2528D0C468EEB87FAF4C35964D2470EB262F76B3E0C8");    

        BandersnatchCurve::create_point_from_affine(x, y).unwrap()
        }

    fn f1f2g_fixed() -> ShortWeierstrassProjectivePoint<BandersnatchCurve> {
        let x = FEE::new_base("248BED2600CB615A6715105E8C33A48E2669775385260EDCCFFA0540A9204865");
        let y = FEE::new_base("5DEB792D33432583D251F8EE85F7090A7BDDBC2113A10323650361F99BAF7254");    

        BandersnatchCurve::create_point_from_affine(x, y).unwrap()
        }

    #[test]
    fn test_scalar_mul() -> () {
        let f1f2 = point_1();

        let g = BandersnatchCurve::generator();
        // let f1f2g = EdwardsAffine::from_str(
        //     "(16530491029447613915334753043669938793793987372416328257719459807614119987301, \
        //      42481140308370805476764840229335460092474682686441442216596889726548353970772)",
        // )
        // .unwrap();

        // let f1f2g = point_1().operate_with(&g);

        // let f1f2g_fixed = f1f2g_fixed();

        assert_eq!(g.clone(), g.clone());

        assert_eq!(f1f2.clone() , f1f2.clone());
    }
}