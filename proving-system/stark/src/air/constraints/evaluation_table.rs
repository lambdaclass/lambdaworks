use crate::FE;
use lambdaworks_math::polynomial::Polynomial;

pub struct ConstraintEvaluationTable {
    evaluations: Vec<Vec<FE>>,
    divisors: Vec<Polynomial<FE>>,
    trace_length: usize,
}

impl ConstraintEvaluationTable {
    fn new(n_cols: usize, domain: &[FE], divisors: Vec<Polynomial<FE>>) -> Self {
        let col = Vec::with_capacity(domain.len());

        let evaluations = vec![col; n_cols];

        ConstraintEvaluationTable {
            evaluations,
            divisors,
            trace_length: domain.len(),
        }
    }
}
