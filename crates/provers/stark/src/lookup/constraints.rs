use lambdaworks_math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

use crate::{
    constraints::transition::TransitionConstraint, table::TableView,
    traits::TransitionEvaluationContext,
};

use super::types::{
    BusInteraction, LinearTerm, Multiplicity, LOGUP_CHALLENGE_ALPHA, LOGUP_CHALLENGE_Z,
};

// =============================================================================
// Lookup Term Constraint (degree 2)
// =============================================================================

/// Constraint for each term column.
///
/// Verifies: `term[i] * fingerprint[i] - sign * multiplicity[i] = 0`
///
/// This is degree 2 because it multiplies the aux column (`term`) by
/// the fingerprint (which is linear in main trace values).
pub(crate) struct LookupTermConstraint {
    interaction: BusInteraction,
    term_column_idx: usize,
    constraint_idx: usize,
}

impl LookupTermConstraint {
    pub fn new(interaction: BusInteraction, term_column_idx: usize, constraint_idx: usize) -> Self {
        Self {
            interaction,
            term_column_idx,
            constraint_idx,
        }
    }
}

impl<F, E> TransitionConstraint<F, E> for LookupTermConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        fn evaluate_term_constraint<A: IsSubFieldOf<B>, B: IsField>(
            step: &TableView<A, B>,
            term_column_idx: usize,
            interaction: &BusInteraction,
            rap_challenges: &[FieldElement<B>],
        ) -> FieldElement<B> {
            let term = step.get_aux_evaluation_element(0, term_column_idx);

            let z = &rap_challenges[LOGUP_CHALLENGE_Z];
            let alpha = &rap_challenges[LOGUP_CHALLENGE_ALPHA];

            // Compute multiplicity
            let multiplicity: FieldElement<A> =
                compute_multiplicity(step, &interaction.multiplicity);

            // Bus elements: [bus_id, ...values...]
            let mut bus_elements: Vec<FieldElement<B>> =
                vec![FieldElement::from(interaction.bus_id)];

            bus_elements.extend(interaction.values.iter().map(|bv| {
                let combined: FieldElement<A> =
                    bv.combine_from(|col| step.get_main_evaluation_element(0, col).clone());
                combined.to_extension()
            }));

            // Compute fingerprint = z - (v₀·α⁰ + v₁·α¹ + v₂·α² + ...)
            // Uses iterative multiplication instead of pow() to avoid
            // recomputing powers on every row of the LDE domain.
            let mut linear_combination = FieldElement::<B>::zero();
            let mut alpha_power = FieldElement::<B>::one();
            for v in &bus_elements {
                linear_combination += v * &alpha_power;
                alpha_power = &alpha_power * alpha;
            }
            let fingerprint = z - linear_combination;

            // Sign: +1 for senders, -1 for receivers
            let sign = if interaction.is_sender {
                FieldElement::<B>::one()
            } else {
                -FieldElement::<B>::one()
            };

            // Constraint: term * fingerprint - sign * multiplicity = 0
            term * &fingerprint - multiplicity * sign
        }

        let res = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                rap_challenges,
                ..
            } => evaluate_term_constraint(
                frame.get_evaluation_step(0),
                self.term_column_idx,
                &self.interaction,
                rap_challenges,
            ),
            TransitionEvaluationContext::Verifier {
                frame,
                rap_challenges,
                ..
            } => evaluate_term_constraint(
                frame.get_evaluation_step(0),
                self.term_column_idx,
                &self.interaction,
                rap_challenges,
            ),
        };

        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
            *eval = res;
        }
    }
}

// =============================================================================
// Lookup Accumulated Constraint (degree 1)
// =============================================================================

/// Constraint for the accumulated column.
///
/// Verifies: `acc[i+1] - acc[i] - Σ terms[i+1] = 0`
pub(crate) struct LookupAccumulatedConstraint {
    constraint_idx: usize,
    num_term_columns: usize,
    acc_column_idx: usize,
}

impl LookupAccumulatedConstraint {
    pub fn new(constraint_idx: usize, num_term_columns: usize) -> Self {
        Self {
            constraint_idx,
            num_term_columns,
            acc_column_idx: num_term_columns,
        }
    }
}

impl<F, E> TransitionConstraint<F, E> for LookupAccumulatedConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        fn evaluate_accumulated_constraint<A: IsSubFieldOf<B>, B: IsField>(
            first_step: &TableView<A, B>,
            second_step: &TableView<A, B>,
            acc_column_idx: usize,
            num_term_columns: usize,
        ) -> FieldElement<B> {
            let acc_curr = first_step.get_aux_evaluation_element(0, acc_column_idx);
            let acc_next = second_step.get_aux_evaluation_element(0, acc_column_idx);

            let terms_sum: FieldElement<B> = (0..num_term_columns)
                .map(|i| second_step.get_aux_evaluation_element(0, i).clone())
                .sum();

            acc_next - acc_curr - terms_sum
        }

        let res = match evaluation_context {
            TransitionEvaluationContext::Prover { frame, .. } => evaluate_accumulated_constraint(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                self.acc_column_idx,
                self.num_term_columns,
            ),
            TransitionEvaluationContext::Verifier { frame, .. } => evaluate_accumulated_constraint(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                self.acc_column_idx,
                self.num_term_columns,
            ),
        };

        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
            *eval = res;
        }
    }
}

// =============================================================================
// Helper: compute multiplicity from a TableView
// =============================================================================

fn compute_multiplicity<A: IsSubFieldOf<B>, B: IsField>(
    step: &TableView<A, B>,
    multiplicity: &Multiplicity,
) -> FieldElement<A> {
    match multiplicity {
        Multiplicity::One => FieldElement::<A>::one(),
        Multiplicity::Column(col) => step.get_main_evaluation_element(0, *col).clone(),
        Multiplicity::Sum(col_a, col_b) => {
            step.get_main_evaluation_element(0, *col_a)
                + step.get_main_evaluation_element(0, *col_b)
        }
        Multiplicity::Negated(col) => {
            FieldElement::<A>::one() - step.get_main_evaluation_element(0, *col)
        }
        Multiplicity::Linear(terms) => {
            let mut result = FieldElement::<A>::zero();
            for term in terms {
                match term {
                    LinearTerm::Column {
                        coefficient,
                        column,
                    } => {
                        let coeff = if *coefficient >= 0 {
                            FieldElement::<A>::from(*coefficient as u64)
                        } else {
                            -FieldElement::<A>::from((-*coefficient) as u64)
                        };
                        result += step.get_main_evaluation_element(0, *column) * coeff;
                    }
                    LinearTerm::ColumnUnsigned {
                        coefficient,
                        column,
                    } => {
                        let coeff = FieldElement::<A>::from(*coefficient);
                        result += step.get_main_evaluation_element(0, *column) * coeff;
                    }
                    LinearTerm::Constant(value) => {
                        if *value >= 0 {
                            result += FieldElement::<A>::from(*value as u64);
                        } else {
                            result = result - FieldElement::<A>::from((-*value) as u64);
                        }
                    }
                    LinearTerm::ConstantUnsigned(value) => {
                        result += FieldElement::<A>::from(*value);
                    }
                }
            }
            result
        }
    }
}
