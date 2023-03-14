use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

/// Represents a boundary constraint that must hold in an execution
/// trace:
///   * col: The column of the trace where the constraint must hold
///   * step: The step (or row) of the trace where the constraint must hold
///   * value: The value the constraint must have in that column and step
pub(crate) struct BoundaryConstraint<FE> {
    col: usize,
    step: usize,
    value: FE,
}

impl<F: IsField> BoundaryConstraint<FieldElement<F>> {
    #[allow(dead_code)]
    pub(crate) fn new(col: usize, step: usize, value: FieldElement<F>) -> Self {
        Self { col, step, value }
    }

    /// Used for creating boundary constraints for a trace with only one column
    pub(crate) fn new_simple(step: usize, value: FieldElement<F>) -> Self {
        Self {
            col: 0,
            step,
            value,
        }
    }
}

/// Data structure that stores all the boundary constraints that must
/// hold for the execution trace
pub(crate) struct BoundaryConstraints<FE> {
    constraints: Vec<BoundaryConstraint<FE>>,
}

impl<F: IsField> BoundaryConstraints<FieldElement<F>> {
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self {
            constraints: Vec::<BoundaryConstraint<FieldElement<F>>>::new(),
        }
    }

    /// To instantiate from a vector of BoundaryConstraint elements
    pub(crate) fn from_constraints(constraints: Vec<BoundaryConstraint<FieldElement<F>>>) -> Self {
        Self { constraints }
    }

    /// Returns all the steps where boundary conditions exist
    pub(crate) fn steps(&self) -> Vec<usize> {
        self.constraints.iter().map(|c| c.step).collect()
    }

    /// Given the primitive root of some domain, returns the domain values corresponding
    /// to the steps where the boundary conditions hold. This is useful when interpolating
    /// the boundary conditions, since we must know the x values
    pub(crate) fn generate_roots_of_unity(
        &self,
        primitive_root: &FieldElement<F>,
    ) -> Vec<FieldElement<F>> {
        self.steps()
            .into_iter()
            .map(|s| primitive_root.pow(s))
            .collect()
    }

    /// Given a trace column, gives all the values the trace must be equal to where
    /// the boundary constraints hold
    pub(crate) fn values(&self, col: usize) -> Vec<FieldElement<F>> {
        self.constraints
            .iter()
            .filter(|c| c.col == col)
            .map(|c| c.value.clone())
            .collect()
    }

    /// Computes the zerofier of the boundary quotient. The result is the
    /// multiplication of each binomial that evaluates to zero in the domain
    /// values where the boundary constraints must hold.
    ///
    /// Example: If there are boundary conditions in the third and fifth steps,
    /// then the zerofier will be (x - w^3) * (x - w^5)
    pub(crate) fn compute_zerofier(
        &self,
        primitive_root: &FieldElement<F>,
    ) -> Polynomial<FieldElement<F>> {
        let mut zerofier = Polynomial::new_monomial(FieldElement::one(), 0);
        for step in self.steps().into_iter() {
            let binomial = Polynomial::new(&[-primitive_root.pow(step), FieldElement::one()]);
            // TODO: Implement the MulAssign trait for Polynomials?
            zerofier = zerofier * binomial;
        }

        zerofier
    }
}

#[cfg(test)]
mod test {
    use lambdaworks_math::{field::traits::IsTwoAdicField, unsigned_integer::element::U256};

    use crate::{PrimeField, FE};

    use super::*;

    #[test]
    fn zerofier_is_the_correct_one() {
        let a0 = BoundaryConstraint::new_simple(0, FE::new(U256::from("1")));
        let a1 = BoundaryConstraint::new_simple(1, FE::new(U256::from("1")));
        let result = BoundaryConstraint::new_simple(7, FE::new(U256::from("32")));

        let constraints = BoundaryConstraints::from_constraints(vec![a0, a1, result]);

        let primitive_root = PrimeField::get_primitive_root_of_unity(3).unwrap();

        let a0_zerofier = Polynomial::new(&[-FE::one(), FE::one()]);
        let a1_zerofier = Polynomial::new(&[-primitive_root.pow(1u32), FE::one()]);
        let res_zerofier = Polynomial::new(&[-primitive_root.pow(7u32), FE::one()]);

        let expected_zerofier = a0_zerofier * a1_zerofier * res_zerofier;

        let zerofier = constraints.compute_zerofier(&primitive_root);

        assert_eq!(expected_zerofier, zerofier);
    }
}
