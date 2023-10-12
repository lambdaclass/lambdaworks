use lambdaworks_math::fft::polynomial::FFTPoly;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField},
    },
    polynomial::Polynomial,
};

#[derive(Clone, Debug)]
pub struct ConstraintEvaluations<F: IsField> {
    // Accumulation of the evaluation of the constraints
    pub evaluations_acc: Vec<FieldElement<F>>,
}

impl<F: IsField> ConstraintEvaluations<F> {
    pub fn new(evaluations_acc: Vec<FieldElement<F>>) -> Self {
        ConstraintEvaluations { evaluations_acc }
    }

    pub fn compute_composition_poly(&self, offset: &FieldElement<F>) -> Polynomial<FieldElement<F>>
    where
        F: IsFFTField,
        Polynomial<FieldElement<F>>: FFTPoly<F>,
    {
        Polynomial::interpolate_offset_fft(&self.evaluations_acc, offset).unwrap()
    }
}
