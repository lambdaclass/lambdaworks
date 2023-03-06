use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

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

    pub(crate) fn new_simple(step: usize, value: FieldElement<F>) -> Self {
        Self {
            col: 0,
            step,
            value,
        }
    }
}

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

    pub(crate) fn from_constraints(constraints: Vec<BoundaryConstraint<FieldElement<F>>>) -> Self {
        Self { constraints }
    }

    pub(crate) fn steps(&self) -> Vec<usize> {
        self.constraints.iter().map(|c| c.step).collect()
    }

    pub(crate) fn get_boundary_roots_of_unity(
        &self,
        primitive_root: &FieldElement<F>,
    ) -> Vec<FieldElement<F>> {
        self.steps()
            .into_iter()
            .map(|s| primitive_root.pow(s))
            .collect()
    }

    pub(crate) fn values(&self, col: usize) -> Vec<FieldElement<F>> {
        self.constraints
            .iter()
            .filter(|c| c.col == col)
            .map(|c| c.value.clone())
            .collect()
    }

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
    use lambdaworks_math::unsigned_integer::element::U384;

    use crate::{generate_primitive_root, FE};

    use super::*;

    #[test]
    fn zerofier_is_the_correct_one() {
        let a0 = BoundaryConstraint::new_simple(0, FE::new(U384::from("1")));
        let a1 = BoundaryConstraint::new_simple(1, FE::new(U384::from("1")));
        let result = BoundaryConstraint::new_simple(7, FE::new(U384::from("32")));

        let constraints = BoundaryConstraints::from_constraints(vec![a0, a1, result]);

        let primitive_root = generate_primitive_root(8);

        let a0_zerofier = Polynomial::new(&[-FE::one(), FE::one()]);
        let a1_zerofier = Polynomial::new(&[-primitive_root.pow(1u32), FE::one()]);
        let res_zerofier = Polynomial::new(&[-primitive_root.pow(7u32), FE::one()]);

        let expected_zerofier = a0_zerofier * a1_zerofier * res_zerofier;

        let zerofier = constraints.compute_zerofier(&primitive_root);

        assert_eq!(expected_zerofier, zerofier);
    }
}
