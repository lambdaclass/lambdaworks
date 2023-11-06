use crate::field::{
    element::FieldElement,
    extensions::quadratic::*,
    fields::fft_friendly::{
        quadratic_babybear::QuadraticBabybearField,
        quadratic_babybear::QuadraticBabybearFieldElement,
    },
};

/// Quartic field extension of Babybear
pub type QuarticBabybearField = QuadraticExtensionField<QuadraticBabybearField>;

/// Field element type for the quartic extension of Babybear
pub type QuarticBabybearFieldElement = QuadraticExtensionFieldElement<QuadraticBabybearField>;

impl HasQuadraticNonResidue for QuadraticBabybearField {
    type BaseField = QuadraticBabybearField;

    fn residue() -> QuadraticBabybearFieldElement {
        -FieldElement::one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::fields::fft_friendly::babybear::Babybear31PrimeField;

    type FE = FieldElement<Babybear31PrimeField>;
    type Fee = QuadraticBabybearFieldElement;

    #[test]
    fn test_add_quartic() {
        let a = Fee::new([FE::from(0), FE::from(3)]);
        let b = Fee::new([-FE::from(2), FE::from(8)]);
        let c = Fee::new([FE::from(3), FE::from(5)]);
        let d = Fee::new([-FE::from(2), FE::from(8)]);

        let x = QuarticBabybearFieldElement::new([a.clone(), b.clone()]);
        let y = QuarticBabybearFieldElement::new([c.clone(), d.clone()]);
        let expected_result = QuarticBabybearFieldElement::new([a + c, b + d]);
        assert_eq!(x + y, expected_result);
    }

    #[test]
    fn test_sub_quartic() {
        let a = Fee::new([FE::from(0), FE::from(3)]);
        let b = Fee::new([-FE::from(2), FE::from(8)]);
        let c = Fee::new([FE::from(3), FE::from(5)]);
        let d = Fee::new([-FE::from(2), FE::from(8)]);

        let x = QuarticBabybearFieldElement::new([a.clone(), b.clone()]);
        let y = QuarticBabybearFieldElement::new([c.clone(), d.clone()]);

        let expected_result = QuarticBabybearFieldElement::new([a - c, b - d]);
        assert_eq!(x - y, expected_result);
    }

    #[test]
    fn test_mul_quartic() {
        let a = Fee::new([FE::from(0), FE::from(3)]);
        let b = Fee::new([-FE::from(2), FE::from(8)]);
        let c = Fee::new([FE::from(3), FE::from(5)]);
        let d = Fee::new([-FE::from(2), FE::from(8)]);

        let x = QuarticBabybearFieldElement::new([a.clone(), b.clone()]);
        let y = QuarticBabybearFieldElement::new([c.clone(), d.clone()]);

        let expected_result = QuarticBabybearFieldElement::new([
            a.clone() * c.clone() + b.clone() * d.clone() * QuadraticBabybearField::residue(),
            a.clone() * d.clone() + b.clone() * c.clone(),
        ]);
        assert_eq!(x * y, expected_result);
    }

    #[test]
    fn test_inv_quartic() {
        let a = Fee::new([FE::from(0), FE::from(3)]);
        let b = Fee::new([-FE::from(2), FE::from(8)]);

        let x = QuarticBabybearFieldElement::new([a.clone(), b.clone()]);
        let inv_norm = (a.pow(2_u64) - QuadraticBabybearField::residue() * b.pow(2_u64))
            .inv()
            .unwrap();
        let expected_result = QuarticBabybearFieldElement::new([a * &inv_norm, -b * inv_norm]);
        assert_eq!(x.inv().unwrap(), expected_result);
    }

    #[test]
    fn test_div_quartic() {
        let a = Fee::new([FE::from(0), FE::from(3)]);
        let b = Fee::new([-FE::from(2), FE::from(8)]);
        let c = Fee::new([FE::from(3), FE::from(5)]);
        let d = Fee::new([-FE::from(2), FE::from(8)]);

        let x = QuarticBabybearFieldElement::new([a.clone(), b.clone()]);
        let y = QuarticBabybearFieldElement::new([c.clone(), d.clone()]);

        let expected_result = &x * y.inv().unwrap();
        assert_eq!(x / y, expected_result);
    }

    #[test]
    fn test_conjugate_quartic() {
        let a = Fee::new([FE::from(0), FE::from(3)]);
        let b = Fee::new([-FE::from(2), FE::from(8)]);

        let x = QuarticBabybearFieldElement::new([a.clone(), b.clone()]);
        let expected_result = QuarticBabybearFieldElement::new([a, -b]);
        assert_eq!(x.conjugate(), expected_result);
    }
}
