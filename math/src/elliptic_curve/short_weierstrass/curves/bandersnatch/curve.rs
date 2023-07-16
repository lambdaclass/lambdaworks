
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
            FieldElement::<Self::BaseField>::new_base("30900340493481298850216505686589334086208278925799850409469406976849338430199"),
            FieldElement::<Self::BaseField>::new_base("12663882780877899054958035777720958383845500985908634476792678820121468453298"),
            FieldElement::one()
        ])
    }
}

impl IsShortWeierstrass for BandersnatchCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base("10773120815616481058602537765553212789256758185246796157495669123169359657269")
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base("29569587568322301171008055308580903175558631321415017492731745847794083609535")
    }
}

// #[cfg(test)]
// mod tests {

//     use super::*;
//     use crate::{
//         cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
//         field::element::FieldElement,
//     };

//     use super::FqField;

//     #[allow(clippy::upper_case_acronyms)]
//     type FEE = FieldElement<FqField>;

//     fn point_1() -> ShortWeierstrassProjectivePoint<BandersnatchCurve> {
//         let x = FEE::new_base("30900340493481298850216505686589334086208278925799850409469406976849338430199");
//         let y = FEE::new_base("12663882780877899054958035777720958383845500985908634476792678820121468453298");    
        
//         BandersnatchCurve::create_point_from_affine(x, y).unwrap()
//         }
    
//     fn point_1_times_5() -> ShortWeierstrassProjectivePoint<BandersnatchCurve> {
//         let x = FEE::new_base("30900340493481298850216505686589334086208278925799850409469406976849338430199");
//         let y = FEE::new_base("12663882780877899054958035777720958383845500985908634476792678820121468453298");    
        
//         BandersnatchCurve::create_point_from_affine(x, y).unwrap().mul(&5)
//         }
// }