use crate::{
    elliptic_curve::{
        short_weierstrass::{point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass},
        traits::IsEllipticCurve,
    },
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
};

#[derive(Clone, Debug)]
pub struct StarkCurve;

impl IsEllipticCurve for StarkCurve {
    type BaseField = Stark252PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "1EF15C18599971B7BECED415A40F0C7DEACFD9B0D1819E03D723D8BC943CFCA",
            ),
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "5668060AA49730B7BE4801DF46EC62DE53ECD11ABE43A32873000C36E8DC1F",
            ),
            FieldElement::one(),
        ])
    }
}

impl IsShortWeierstrass for StarkCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::from_hex_unchecked("1")
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::from_hex_unchecked(
            "6F21413EFBE40DE150E596D72F7A8C5609AD26C15C915C1F4CDFCB99CEE9E89",
        )
    }
}
