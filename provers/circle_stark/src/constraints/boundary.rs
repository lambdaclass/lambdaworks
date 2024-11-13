use itertools::Itertools;
use lambdaworks_math::{
    circle::point::CirclePoint,
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field, traits::IsField},
    polynomial::Polynomial,
};

use crate::trace::LDETraceTable;

use super::{evaluator::line, utils::interpolant};

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

    /// For every column, it returns for each pair of boundary constraints in that column, the corresponding evaluation
    /// in all the `eval_points`.
    // We assume that there are an even number of boundary contrainsts in the column `col`.
    // TODO: If two columns have a pair of boundary constraints that hold in the same rows, we can optimize
    // this function so that we don't calculate the same line evaluations for the two of them. Maybe a hashmap?
    pub fn evaluate_zerofiers(
        &self,
        trace_coset: &Vec<CirclePoint<Mersenne31Field>>,
        eval_points: &Vec<CirclePoint<Mersenne31Field>>,
    ) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
        let mut zerofiers_evaluations = Vec::new();
        for col in self.cols_for_boundary() {
            self.constraints
                .iter()
                .filter(|constraint| constraint.col == col)
                .chunks(2)
                .into_iter()
                .map(|chunk| {
                    let chunk: Vec<_> = chunk.collect();
                    let first_constraint = &chunk[0];
                    let second_constraint = &chunk[1];
                    let first_vanish_point = &trace_coset[first_constraint.step];
                    let second_vanish_point = &trace_coset[second_constraint.step];

                    let mut boundary_evaluations: Vec<FieldElement<Mersenne31Field>> = eval_points
                        .iter()
                        .map(|eval_point| {
                            line(eval_point, &first_vanish_point, &second_vanish_point)
                        })
                        .collect();
                    FieldElement::inplace_batch_inverse(&mut boundary_evaluations).unwrap();
                    zerofiers_evaluations.push(boundary_evaluations);
                })
                .collect() // TODO: We don't use this collect.
        }
        zerofiers_evaluations
    }

    /// For every columnm, and every constrain, returns the evaluation on the entire lde domain of the constrain function F(x) - I(x).
    // Note this is the numerator of each pair of constrains for the composition polynomial.
    pub fn evaluate_poly_constraints(
        &self,
        trace_coset: &Vec<CirclePoint<Mersenne31Field>>,
        eval_points: &Vec<CirclePoint<Mersenne31Field>>,
        lde_trace: &LDETraceTable,
    ) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
        let mut poly_evaluations = Vec::new();
        for col in self.cols_for_boundary() {
            self.constraints
                .iter()
                .filter(|constraint| constraint.col == col)
                .chunks(2)
                .into_iter()
                .map(|chunk| {
                    let chunk: Vec<_> = chunk.collect();
                    let first_constraint = &chunk[0];
                    let second_constraint = &chunk[1];
                    let first_vanish_point = &trace_coset[first_constraint.step];
                    let first_value = first_constraint.value;
                    let second_vanish_point = &trace_coset[second_constraint.step];
                    let second_value = second_constraint.value;
                    let boundary_evaluations = eval_points
                        .iter()
                        .zip(&lde_trace.table.columns()[col])
                        .map(|(eval_point, lde_eval)| {
                            lde_eval
                                - interpolant(
                                    &first_vanish_point,
                                    &second_vanish_point,
                                    first_value,
                                    second_value,
                                    eval_point,
                                )
                        })
                        .collect::<Vec<FieldElement<Mersenne31Field>>>();
                    poly_evaluations.push(boundary_evaluations);
                })
                .collect() // TODO: We are not using this collect.
        }
        poly_evaluations
    }

    // /// For every columnm, and every constrain, returns the evaluation on the entire lde domain of the constrain function F(x) - I(x).
    // // Note this is the numerator of each pair of constrains for the composition polynomial.
    // pub fn evaluate_poly_constraints(
    //     &self,
    //     trace_coset: &Vec<CirclePoint<Mersenne31Field>>,
    //     eval_points: &Vec<CirclePoint<Mersenne31Field>>,
    //     lde_trace: &LDETraceTable,
    // ) -> Vec<Vec<Vec<FieldElement<Mersenne31Field>>>> {
    //     self.cols_for_boundary()
    //         .iter()
    //         .map(|col| {
    //             self.constraints
    //                 .iter()
    //                 .filter(|constraint| constraint.col == *col)
    //                 .chunks(2)
    //                 .into_iter()
    //                 .map(|chunk| {
    //                     let chunk: Vec<_> = chunk.collect();
    //                     let first_constraint = &chunk[0];
    //                     let second_constraint = &chunk[1];
    //                     let first_vanish_point = &trace_coset[first_constraint.step];
    //                     let first_value = first_constraint.value;
    //                     let second_vanish_point = &trace_coset[second_constraint.step];
    //                     let second_value = second_constraint.value;
    //                     eval_points
    //                         .iter()
    //                         .zip(&lde_trace.table.columns()[*col])
    //                         .map(|(eval_point, lde_eval)| {
    //                             lde_eval
    //                                 - interpolant(
    //                                     &first_vanish_point,
    //                                     &second_vanish_point,
    //                                     first_value,
    //                                     second_value,
    //                                     eval_point,
    //                                 )
    //                         })
    //                         .collect::<Vec<FieldElement<Mersenne31Field>>>()
    //                 })
    //                 .collect()
    //         })
    //         .collect()
    // }
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
    fn simple_fibonacci_boundary_zerofiers() {
        let one = FieldElement::<Mersenne31Field>::one();

        // Fibonacci constraints:
        // a0 = 1
        // a1 = 1
        let a0 = BoundaryConstraint::new_simple(0, one);
        let a1 = BoundaryConstraint::new_simple(1, one);

        let trace_coset = Coset::get_coset_points(&Coset::new_standard(3));
        let eval_points = Coset::get_coset_points(&Coset::new_standard(4));

        let expected_zerofier: Vec<Vec<FieldElement<Mersenne31Field>>> = vec![eval_points
            .iter()
            .map(|point| line(point, &trace_coset[0], &trace_coset[1]))
            .collect()];

        let constraints = BoundaryConstraints::from_constraints(vec![a0, a1]);
        let zerofier = constraints.evaluate_zerofiers(&trace_coset, &eval_points);

        assert_eq!(expected_zerofier, zerofier);
    }
}
