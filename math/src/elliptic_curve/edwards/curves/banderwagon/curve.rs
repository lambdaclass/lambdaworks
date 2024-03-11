pub use super::field::FqField;
use crate::elliptic_curve::edwards::point::EdwardsProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::{elliptic_curve::edwards::traits::IsEdwards, field::element::FieldElement};

pub type BaseBanderwagonFieldElement = FqField;

#[derive(Clone, Debug)]

pub struct BanderwagonCurve;

impl IsEllipticCurve for BanderwagonCurve {
    type BaseField = BaseBanderwagonFieldElement;
    type PointRepresentation = EdwardsProjectivePoint<Self>;

    // Values are from https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/edwards/curves/bandersnatch/curve.rs
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

impl IsEdwards for BanderwagonCurve {
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

impl BanderwagonCurve {
    fn serialize(&self) -> String {
        let y_sign = if self.y >= FieldElement::<Self::BaseField>::zero() { 1 } else { -1 };
        let result = self.x * y_sign;
        format!("{}", result)
    }

    fn deserialize(input: &str) -> Option<Self> {
        let xk = FieldElement::<Self::BaseField>::from_str(input).ok()?;
        let one = FieldElement::<Self::BaseField>::one();
        let a = Self::a();
        let d = Self::d();

        let numerator = one - a * xk.pow(2);
        let denominator = one - d * xk.pow(2);

        if numerator.is_square().is_none() || denominator.is_square().is_none() {
            return None;
        }

        let yk_square = numerator / denominator;
        let yk = yk_square.sqrt()?;

        if yk.signum() != 1 {
            return None;
        }

        Some(Self { x: xk, y: yk })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_and_deserialization_large_numbers() {
        let curve = BanderwagonCurve {
            x: FieldElement::<Self::BaseField>::from(1234567890),
            y: FieldElement::<Self::BaseField>::from(9876543210),
        };

        let serialized = curve.serialize();
        let deserialized = BanderwagonCurve::deserialize(&serialized);

        assert!(deserialized.is_some());
        let deserialized = deserialized.unwrap();

        assert_eq!(curve.x, deserialized.x);
        assert_eq!(curve.y.signum(), deserialized.y.signum());
    }
}
