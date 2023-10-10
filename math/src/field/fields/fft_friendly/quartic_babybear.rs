use Babybear31PrimeField;
use QuadraticExtensionField;

pub type QuadraticBabybearField = QuadraticExtensionField<Babybear31PrimeField>;

pub type QuadraticBabybearFieldElement = QuadraticExtensionFieldElement<QuadraticBabybearField>;

impl HasQuadraticNonResidue for QuadraticBabybearField {
    type BaseType = Babybear31PrimeField;

    fn residue() -> FieldElement<Babybear31PrimeField> {
        -FieldElement::one()
    }
}

#[cfg(test)]
mod tests {

}