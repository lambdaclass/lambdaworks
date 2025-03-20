use crate::field::{
    element::FieldElement, extensions::quadratic::*,
    fields::fft_friendly::babybear::Babybear31PrimeField,
};

/// Quadratic field extension of Babybear
pub type QuadraticBabybearField =
    QuadraticExtensionField<Babybear31PrimeField, Babybear31PrimeField>;

impl HasQuadraticNonResidue<Babybear31PrimeField> for Babybear31PrimeField {
    fn residue() -> FieldElement<Babybear31PrimeField> {
        -FieldElement::one()
    }
}

/// Field element type for the quadratic extension of Babybear
pub type QuadraticBabybearFieldElement =
    QuadraticExtensionFieldElement<Babybear31PrimeField, Babybear31PrimeField>;

#[cfg(test)]
mod tests {
    use super::*;

    type FE = FieldElement<Babybear31PrimeField>;
    type Fee = QuadraticBabybearFieldElement;

    #[test]
    fn test_add_quadratic() {
        let a = Fee::new([FE::from(0), FE::from(3)]);
        let b = Fee::new([-FE::from(2), FE::from(8)]);
        let expected_result = Fee::new([FE::from(0) - FE::from(2), FE::from(3) + FE::from(8)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_sub_quadratic() {
        let a = Fee::new([FE::from(0), FE::from(3)]);
        let b = Fee::new([-FE::from(2), FE::from(8)]);
        let expected_result = Fee::new([FE::from(0) + FE::from(2), FE::from(3) - FE::from(8)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_mul_quadratic() {
        let a = Fee::new([FE::from(12), FE::from(5)]);
        let b = Fee::new([-FE::from(4), FE::from(2)]);
        let expected_result = Fee::new([
            FE::from(12) * (-FE::from(4))
                + FE::from(5) * FE::from(2) * Babybear31PrimeField::residue(),
            FE::from(12) * FE::from(2) + FE::from(5) * (-FE::from(4)),
        ]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_inv_quadratic() {
        let a = Fee::new([FE::from(12), FE::from(5)]);
        let inv_norm = (FE::from(12).pow(2_u64)
            - Babybear31PrimeField::residue() * FE::from(5).pow(2_u64))
        .inv()
        .unwrap();
        let expected_result = Fee::new([FE::from(12) * &inv_norm, -&FE::from(5) * inv_norm]);
        assert_eq!(a.inv().unwrap(), expected_result);
    }

    #[test]
    fn test_conjugate_quadratic() {
        let a = Fee::new([FE::from(12), FE::from(5)]);
        let expected_result = Fee::new([FE::from(12), -FE::from(5)]);
        assert_eq!(a.conjugate(), expected_result);
    }
}
