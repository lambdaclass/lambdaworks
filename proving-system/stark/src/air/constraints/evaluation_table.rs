use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

pub struct ConstraintEvaluationTable<F: IsField> {
    // Inner vectors are rows
    pub evaluations: Vec<Vec<FieldElement<F>>>,
    // divisors: Vec<Polynomial<FieldElement<F>>>,
    trace_length: usize,
}

impl<F: IsField> ConstraintEvaluationTable<F> {
    // pub fn new(n_cols: usize, domain: &[FE], divisors: &[Polynomial<FieldElement<F>>]) -> Self {
    pub fn new(n_cols: usize, domain: &[FieldElement<F>]) -> Self {
        let col = Vec::with_capacity(domain.len());

        let evaluations = vec![col; n_cols];

        ConstraintEvaluationTable {
            evaluations,
            // divisors: divisors.to_vec(),
            trace_length: domain.len(),
        }
    }

    pub fn compute_composition_poly(
        &self,
        lde_coset: &[FieldElement<F>],
    ) -> Polynomial<FieldElement<F>> {
        let merged_evals: Vec<FieldElement<F>> = self
            .evaluations
            .iter()
            .map(|row| row.iter().fold(FieldElement::zero(), |acc, d| acc + d))
            .collect();

        Polynomial::interpolate(lde_coset, &merged_evals)
    }
}
