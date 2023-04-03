use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

#[derive(Debug)]
/// Represents a boundary constraint that must hold in an execution
/// trace:
///   * col: The column of the trace where the constraint must hold
///   * step: The step (or row) of the trace where the constraint must hold
///   * value: The value the constraint must have in that column and step
pub struct BoundaryConstraint<F: IsField> {
    col: usize,
    step: usize,
    value: FieldElement<F>,
}

impl<F: IsField> BoundaryConstraint<F> {
    pub fn new(col: usize, step: usize, value: FieldElement<F>) -> Self {
        Self { col, step, value }
    }

    /// Used for creating boundary constraints for a trace with only one column
    pub fn new_simple(step: usize, value: FieldElement<F>) -> Self {
        Self {
            col: 0,
            step,
            value,
        }
    }
}

/// Data structure that stores all the boundary constraints that must
/// hold for the execution trace
#[derive(Default, Debug)]
pub struct BoundaryConstraints<F: IsField> {
    constraints: Vec<BoundaryConstraint<F>>,
}

impl<F: IsField> BoundaryConstraints<F> {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            constraints: Vec::<BoundaryConstraint<F>>::new(),
        }
    }

    /// To instantiate from a vector of BoundaryConstraint elements
    pub fn from_constraints(constraints: Vec<BoundaryConstraint<F>>) -> Self {
        Self { constraints }
    }

    /// Returns all the steps where boundary conditions exist for the given column
    pub fn steps(&self, col: usize) -> Vec<usize> {
        self.constraints
            .iter()
            .filter(|v| v.col == col)
            .map(|c| c.step)
            .collect()
    }

    /// Given the primitive root of some domain, returns the domain values corresponding
    /// to the steps where the boundary conditions hold. This is useful when interpolating
    /// the boundary conditions, since we must know the x values
    pub fn generate_roots_of_unity(
        &self,
        primitive_root: &FieldElement<F>,
        count_cols_trace: usize,
    ) -> Vec<Vec<FieldElement<F>>> {
        let mut ret = Vec::new();

        for i in 0..count_cols_trace {
            ret.push(
                self.steps(i)
                    .into_iter()
                    .map(|s| primitive_root.pow(s))
                    .collect(),
            );
        }
        ret
    }

    /// For every trace column, give all the values the trace must be equal to in
    /// the steps where the boundary constraints hold
    pub fn values(&self, n_trace_columns: usize) -> Vec<Vec<FieldElement<F>>> {
        (0..n_trace_columns)
            .map(|i| {
                self.constraints
                    .iter()
                    .filter(|c| c.col == i)
                    .map(|c| c.value.clone())
                    .collect()
            })
            .collect()
    }

    /// Computes the zerofier of the boundary quotient. The result is the
    /// multiplication of each binomial that evaluates to zero in the domain
    /// values where the boundary constraints must hold.
    ///
    /// Example: If there are boundary conditions in the third and fifth steps,
    /// then the zerofier will be (x - w^3) * (x - w^5)
    pub fn compute_zerofier(
        &self,
        primitive_root: &FieldElement<F>,
        col: usize,
    ) -> Polynomial<FieldElement<F>> {
        let mut zerofier = Polynomial::new_monomial(FieldElement::<F>::one(), 0);
        for step in self.steps(col).into_iter() {
            let binomial = Polynomial::new(&[-primitive_root.pow(step), FieldElement::<F>::one()]);
            // TODO: Implement the MulAssign trait for Polynomials?
            zerofier = zerofier * binomial;
        }

        zerofier
    }
}

#[cfg(test)]
mod test {
    use lambdaworks_math::field::traits::IsTwoAdicField;

    use crate::PrimeField;

    use super::*;

    #[test]
    fn zerofier_is_the_correct_one() {
        let one = FieldElement::<PrimeField>::one();

        // Fibonacci constraints:
        //   * a0 = 1
        //   * a1 = 1
        //   * a7 = 32
        let a0 = BoundaryConstraint::new_simple(0, one.clone());
        let a1 = BoundaryConstraint::new_simple(1, one.clone());
        let result = BoundaryConstraint::new_simple(7, FieldElement::<PrimeField>::from(32));

        let constraints = BoundaryConstraints::from_constraints(vec![a0, a1, result]);

        let primitive_root = PrimeField::get_primitive_root_of_unity(3).unwrap();

        // P_0(x) = (x - 1)
        let a0_zerofier = Polynomial::new(&[-one.clone(), one.clone()]);
        // P_1(x) = (x - w^1)
        let a1_zerofier = Polynomial::new(&[-primitive_root.pow(1u32), one.clone()]);
        // P_res(x) = (x - w^7)
        let res_zerofier = Polynomial::new(&[-primitive_root.pow(7u32), one]);

        let expected_zerofier = a0_zerofier * a1_zerofier * res_zerofier;

        let zerofier = constraints.compute_zerofier(&primitive_root, 0);

        assert_eq!(expected_zerofier, zerofier);
    }
}
