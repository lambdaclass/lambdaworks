use crate::{
    air::{trace::TraceTable, Air},
    FE,
};

use super::{boundary::BoundaryConstraints, evaluation_table::ConstraintEvaluationTable};

pub struct ConstraintEvaluator<'a, A: Air> {
    air: &'a A,
    boundary_constraints: BoundaryConstraints<FE>,
    primitive_root: FE,
}

impl<'a, A: Air> ConstraintEvaluator<'a, A> {
    fn new() -> Self {
        todo!()
    }

    fn evaluate(&self, lde_trace: TraceTable, lde_domain: Vec<FE>) -> ConstraintEvaluationTable {
        // Get all divisors in a vector
        // The first divisors appearing in the vector will be transition ones
        // and the last the one from the boundary constraints.
        let divisors = self.air.transition_divisors();
        divisors.push(
            self.boundary_constraints
                .compute_zerofier(&self.primitive_root),
        );

        todo!()
    }
}
