use crate::{
    elliptic_curve::{
        point::ProjectivePoint,
        short_weierstrass::{point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass},
        traits::IsEllipticCurve,
    },
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
};

#[derive(Clone, Debug)]
pub struct StarkCurve;

impl StarkCurve {
    pub const fn from_affine_hex_string(
        x_hex: &str,
        y_hex: &str,
    ) -> ShortWeierstrassProjectivePoint<Self> {
        ShortWeierstrassProjectivePoint(ProjectivePoint::new([
            FieldElement::<Stark252PrimeField>::from_hex_unchecked(x_hex),
            FieldElement::<Stark252PrimeField>::from_hex_unchecked(y_hex),
            FieldElement::<Stark252PrimeField>::from_hex_unchecked("1"),
        ]))
    }
}

impl IsEllipticCurve for StarkCurve {
    type BaseField = Stark252PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - The generator point is mathematically verified to be a valid point on the curve.
        // - `unwrap()` is safe because the provided coordinates satisfy the curve equation.
        let point = Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "1EF15C18599971B7BECED415A40F0C7DEACFD9B0D1819E03D723D8BC943CFCA",
            ),
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "5668060AA49730B7BE4801DF46EC62DE53ECD11ABE43A32873000C36E8DC1F",
            ),
            FieldElement::one(),
        ]);
        point.unwrap()
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
