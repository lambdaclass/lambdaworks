use lambdaworks_math::field::{element::FieldElement, traits::IsField};

#[allow(dead_code)]
pub(crate) struct BoundaryConstraint<FE> {
    col: usize,
    step: usize,
    value: FE,
}

#[allow(dead_code)]
impl<F: IsField> BoundaryConstraint<FieldElement<F>> {
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

#[allow(dead_code)]
pub(crate) struct BoundaryConstraints<FE> {
    constraints: Vec<BoundaryConstraint<FE>>,
}

#[allow(dead_code)]
impl<F: IsField> BoundaryConstraints<FieldElement<F>> {
    pub(crate) fn new() -> Self {
        Self {
            constraints: Vec::<BoundaryConstraint<FieldElement<F>>>::new(),
        }
    }

    pub(crate) fn from_constraints(constraints: Vec<BoundaryConstraint<FieldElement<F>>>) -> Self {
        Self { constraints }
    }

    pub(crate) fn get_steps(&self) -> Vec<usize> {
        let mut steps = Vec::with_capacity(self.constraints.len());
        for constraint in self.constraints.iter() {
            steps.push(constraint.step);
        }

        steps
    }

    pub(crate) fn get_boundary_poly_domain(
        &self,
        trace_domain: &[FieldElement<F>],
    ) -> Vec<FieldElement<F>> {
        let mut domain = Vec::with_capacity(self.constraints.len());
        for step in self.get_steps().into_iter() {
            // TODO: Handle Option from get()
            let domain_point = trace_domain.get(step).unwrap();
            domain.push(domain_point.clone());
        }

        domain
    }

    pub(crate) fn get_values(&self, col: usize) -> Vec<FieldElement<F>> {
        let mut values = Vec::with_capacity(self.constraints.len());
        for constraint in self.constraints.iter().filter(|c| c.col == col) {
            values.push(constraint.value.clone());
        }

        values
    }
}
