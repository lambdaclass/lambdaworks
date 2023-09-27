use lambdaworks_math::fft::polynomial::FFTPoly;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField},
    },
    polynomial::Polynomial,
};

#[derive(Clone, Debug)]
pub struct ConstraintEvaluationTable<F: IsField> {
    // Accumulation of the evaluation of the constraints
    pub evaluations_acc: Vec<FieldElement<F>>,
    pub trace_length: usize,
}

impl<F: IsField> ConstraintEvaluationTable<F> {
    pub fn new(_n_cols: usize, domain: &[FieldElement<F>]) -> Self {
        let evaluations_acc = Vec::with_capacity(domain.len());

        ConstraintEvaluationTable {
            evaluations_acc,
            trace_length: domain.len(),
        }
    }

    pub fn compute_composition_poly(&self, offset: &FieldElement<F>) -> Polynomial<FieldElement<F>>
    where
        F: IsFFTField,
        Polynomial<FieldElement<F>>: FFTPoly<F>,
    {
        Polynomial::interpolate_offset_fft(&self.evaluations_acc, offset).unwrap()
    }
}
