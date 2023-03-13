use crate::FE;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

pub struct ConstraintEvaluationTable<F: IsField> {
    pub evaluations: Vec<Vec<FieldElement<F>>>,
    divisors: Vec<Polynomial<FieldElement<F>>>,
    trace_length: usize,
}

impl<F: IsField> ConstraintEvaluationTable {
    pub fn new(n_cols: usize, domain: &[FE], divisors: &Vec<Polynomial<FieldElement<F>>>) -> Self {
        let col = Vec::with_capacity(domain.len());

        let evaluations = vec![col; n_cols];

        ConstraintEvaluationTable {
            evaluations,
            divisors: divisors.clone(),
            trace_length: domain.len(),
        }
    }
}
