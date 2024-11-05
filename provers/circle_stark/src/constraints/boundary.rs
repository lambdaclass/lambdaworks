use itertools::Itertools;
use lambdaworks_math::{
    circle::point::CirclePoint,
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field, traits::IsField},
    polynomial::Polynomial,
};

#[derive(Debug)]
/// Represents a boundary constraint that must hold in an execution
/// trace:
///   * col: The column of the trace where the constraint must hold
///   * step: The step (or row) of the trace where the constraint must hold
///   * value: The value the constraint must have in that column and step
pub struct BoundaryConstraint {
    pub col: usize,
    pub step: usize,
    pub value: FieldElement<Mersenne31Field>,
}

impl BoundaryConstraint {
    pub fn new(col: usize, step: usize, value: FieldElement<Mersenne31Field>) -> Self {
        Self { col, step, value }
    }

    /// Used for creating boundary constraints for a trace with only one column
    pub fn new_simple(step: usize, value: FieldElement<Mersenne31Field>) -> Self {
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
pub struct BoundaryConstraints {
    pub constraints: Vec<BoundaryConstraint>,
}

impl BoundaryConstraints {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            constraints: Vec::<BoundaryConstraint>::new(),
        }
    }

    /// To instantiate from a vector of BoundaryConstraint elements
    pub fn from_constraints(constraints: Vec<BoundaryConstraint>) -> Self {
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

    /// Return all the steps where boundary constraints hold.
    pub fn steps_for_boundary(&self) -> Vec<usize> {
        self.constraints
            .iter()
            .unique_by(|elem| elem.step)
            .map(|v| v.step)
            .collect()
    }

    /// Return all the columns where boundary constraints hold.
    pub fn cols_for_boundary(&self) -> Vec<usize> {
        self.constraints
            .iter()
            .unique_by(|elem| elem.col)
            .map(|v| v.col)
            .collect()
    }

    /// Given the group generator of some domain, returns for each column the domain values corresponding
    /// to the steps where the boundary conditions hold. This is useful when interpolating
    /// the boundary conditions, since we must know the x values
    pub fn generate_roots_of_unity(
        &self,
        group_generator: &CirclePoint<Mersenne31Field>,
        cols_trace: &[usize],
    ) -> Vec<Vec<CirclePoint<Mersenne31Field>>> {
        cols_trace
            .iter()
            .map(|i| {
                self.steps(*i)
                    .into_iter()
                    .map(|s| group_generator * (s as u128))
                    .collect::<Vec<CirclePoint<Mersenne31Field>>>()
            })
            .collect::<Vec<Vec<CirclePoint<Mersenne31Field>>>>()
    }

    /// For every trace column, give all the values the trace must be equal to in
    /// the steps where the boundary constraints hold
    pub fn values(&self, cols_trace: &[usize]) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
        cols_trace
            .iter()
            .map(|i| {
                self.constraints
                    .iter()
                    .filter(|c| c.col == *i)
                    .map(|c| c.value.clone())
                    .collect()
            })
            .collect()
    }

    /// Evaluate the zerofier of the boundary constraints for a column. The result is the
    /// multiplication of each zerofier that evaluates to zero in the domain
    /// values where the boundary constraints must hold.
    ///
    /// Example: If there are boundary conditions in the third and fifth steps,
    /// then the zerofier will be f(x, y) = ( ((x, y) + p3.conjugate()).x - 1 ) * ( ((x, y) + p5.conjugate()).x - 1 )
    /// (eval_point + vanish_point.conjugate()).x - FieldElement::<Mersenne31Field>::one()
    /// TODO: Optimize this function so we don't need to look up and indexes in the coset vector and clone its value.
    pub fn evaluate_zerofier(
        &self,
        trace_coset: &Vec<CirclePoint<Mersenne31Field>>,
        col: usize,
        eval_point: &CirclePoint<Mersenne31Field>,
    ) -> FieldElement<Mersenne31Field> {
        self.steps(col).into_iter().fold(
            FieldElement::<Mersenne31Field>::one(),
            |zerofier, step| {
                let vanish_point = trace_coset[step].clone();
                let evaluation = (eval_point + vanish_point.conjugate()).x
                    - FieldElement::<Mersenne31Field>::one();
                // TODO: Implement the MulAssign trait for Polynomials?
                zerofier * evaluation
            },
        )
    }
}

#[cfg(test)]
mod test {
    use lambdaworks_math::{
        circle::cosets::Coset,
        field::{
            fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::IsFFTField,
        },
    };
    type PrimeField = Stark252PrimeField;

    use super::*;

    #[test]
    fn zerofier_is_the_correct_one() {
        let one = FieldElement::<Mersenne31Field>::one();

        // Fibonacci constraints:
        //   * a0 = 1
        //   * a1 = 1
        //   * a7 = 32
        let a0 = BoundaryConstraint::new_simple(0, one);
        let a1 = BoundaryConstraint::new_simple(1, one);
        let result = BoundaryConstraint::new_simple(7, FieldElement::<Mersenne31Field>::from(32));

        let trace_coset = Coset::get_coset_points(&Coset::new_standard(3));
        let eval_point = CirclePoint::<Mersenne31Field>::GENERATOR * 2;
        let a0_zerofier = (&eval_point + &trace_coset[0].clone().conjugate()).x - one;
        let a1_zerofier = (&eval_point + &trace_coset[1].clone().conjugate()).x - one;
        let res_zerofier = (&eval_point + &trace_coset[7].clone().conjugate()).x - one;
        let expected_zerofier = a0_zerofier * a1_zerofier * res_zerofier;

        let constraints = BoundaryConstraints::from_constraints(vec![a0, a1, result]);
        let zerofier = constraints.evaluate_zerofier(&trace_coset, 0, &eval_point);

        assert_eq!(expected_zerofier, zerofier);
    }
}
