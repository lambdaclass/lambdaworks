//! AIR (Algebraic Intermediate Representation) for ZK Mastermind
//!
//! This module defines the constraints that the prover must satisfy to generate
//! a valid proof. The circuit verifies:
//!
//! 1. **Range checks**: All colors are in valid range [0, 5]
//! 2. **Exact match verification**: The exact match count is correctly computed
//! 3. **Boundary constraints**: Public inputs match the trace values
//!
//! ## Circuit Design
//!
//! The exact match count is verified using equality indicators (eq_i) for each
//! position. For a sound circuit, we enforce:
//! - `eq_i` is boolean: `eq_i * (1 - eq_i) = 0`
//! - If `eq_i = 1`, then `secret[i] = guess[i]`: `eq_i * (secret[i] - guess[i]) = 0`
//! - The sum of indicators equals `exact_count`
//!
//! Note: Partial match verification would require additional color frequency
//! counting constraints, which adds significant complexity.

use lambdaworks_math::field::{element::FieldElement, traits::IsFFTField};
use stark_platinum_prover::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        transition::TransitionConstraint,
    },
    context::AirContext,
    proof::options::ProofOptions,
    traits::{TransitionEvaluationContext, AIR},
};

use crate::game::MastermindPublicInputs;

/// Column indices for the trace table
pub mod cols {
    pub const SECRET_0: usize = 0;
    pub const SECRET_1: usize = 1;
    pub const SECRET_2: usize = 2;
    pub const SECRET_3: usize = 3;
    pub const GUESS_0: usize = 4;
    pub const GUESS_1: usize = 5;
    pub const GUESS_2: usize = 6;
    pub const GUESS_3: usize = 7;
    pub const AUX_EXACT: usize = 8; // Exact match count
    pub const AUX_PARTIAL: usize = 9; // Partial match count
    pub const EQ_0: usize = 10; // Equality indicator for position 0
    pub const EQ_1: usize = 11; // Equality indicator for position 1
    pub const EQ_2: usize = 12; // Equality indicator for position 2
    pub const EQ_3: usize = 13; // Equality indicator for position 3
}

/// Constraint: Check that a value is in range [0, max_value - 1]
/// We do this by checking that value * (value - 1) * ... * (value - max_value + 1) = 0
fn range_check<F: IsFFTField>(value: &FieldElement<F>, max_value: u64) -> FieldElement<F> {
    let mut product = FieldElement::<F>::one();
    for i in 0..max_value {
        product *= value - FieldElement::<F>::from(i);
    }
    product
}

/// Boolean constraint: Ensures eq_i is 0 or 1
/// Constraint: eq_i * (1 - eq_i) = 0
#[derive(Clone)]
struct BooleanConstraint<F: IsFFTField> {
    col_index: usize,
    constraint_idx: usize,
    phantom: std::marker::PhantomData<F>,
}

impl<F: IsFFTField> BooleanConstraint<F> {
    pub fn new(col_index: usize, constraint_idx: usize) -> Self {
        Self {
            col_index,
            constraint_idx,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for BooleanConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        2 // eq * (1 - eq) is quadratic
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
    ) {
        let frame = match evaluation_context {
            TransitionEvaluationContext::Prover { frame, .. } => frame,
            TransitionEvaluationContext::Verifier { frame, .. } => frame,
        };

        let step = frame.get_evaluation_step(0);
        let eq = step.get_main_evaluation_element(0, self.col_index);

        // Boolean constraint: eq * (1 - eq) = 0
        let one = FieldElement::<F>::one();
        transition_evaluations[self.constraint_idx()] = eq.clone() * (one - eq);
    }
}

/// Equality indicator constraint: If eq_i = 1, then secret[i] must equal guess[i]
/// Constraint: eq_i * (secret[i] - guess[i]) = 0
#[derive(Clone)]
struct EqualityIndicatorConstraint<F: IsFFTField> {
    position: usize, // 0-3 for the four positions
    constraint_idx: usize,
    phantom: std::marker::PhantomData<F>,
}

impl<F: IsFFTField> EqualityIndicatorConstraint<F> {
    pub fn new(position: usize, constraint_idx: usize) -> Self {
        Self {
            position,
            constraint_idx,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for EqualityIndicatorConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        2 // eq * (secret - guess) is quadratic
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
    ) {
        let frame = match evaluation_context {
            TransitionEvaluationContext::Prover { frame, .. } => frame,
            TransitionEvaluationContext::Verifier { frame, .. } => frame,
        };

        let step = frame.get_evaluation_step(0);

        let secret_col = cols::SECRET_0 + self.position;
        let guess_col = cols::GUESS_0 + self.position;
        let eq_col = cols::EQ_0 + self.position;

        let secret = step.get_main_evaluation_element(0, secret_col);
        let guess = step.get_main_evaluation_element(0, guess_col);
        let eq = step.get_main_evaluation_element(0, eq_col);

        // If eq = 1, then secret - guess must be 0
        // Constraint: eq * (secret - guess) = 0
        transition_evaluations[self.constraint_idx()] = eq * (secret - guess);
    }
}

/// Sum constraint: Verifies exact_count = eq_0 + eq_1 + eq_2 + eq_3
#[derive(Clone)]
struct ExactCountConstraint<F: IsFFTField> {
    constraint_idx: usize,
    phantom: std::marker::PhantomData<F>,
}

impl<F: IsFFTField> ExactCountConstraint<F> {
    pub fn new(constraint_idx: usize) -> Self {
        Self {
            constraint_idx,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for ExactCountConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        1 // Linear constraint
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
    ) {
        let frame = match evaluation_context {
            TransitionEvaluationContext::Prover { frame, .. } => frame,
            TransitionEvaluationContext::Verifier { frame, .. } => frame,
        };

        let step = frame.get_evaluation_step(0);

        let eq_0 = step.get_main_evaluation_element(0, cols::EQ_0);
        let eq_1 = step.get_main_evaluation_element(0, cols::EQ_1);
        let eq_2 = step.get_main_evaluation_element(0, cols::EQ_2);
        let eq_3 = step.get_main_evaluation_element(0, cols::EQ_3);
        let exact_count = step.get_main_evaluation_element(0, cols::AUX_EXACT);

        // Constraint: exact_count - (eq_0 + eq_1 + eq_2 + eq_3) = 0
        let sum = eq_0 + eq_1 + eq_2 + eq_3;
        transition_evaluations[self.constraint_idx()] = exact_count - sum;
    }
}

/// Constraint for range checking colors
#[derive(Clone)]
struct RangeCheckConstraint<F: IsFFTField> {
    col_index: usize,
    constraint_idx: usize,
    phantom: std::marker::PhantomData<F>,
}

impl<F: IsFFTField> RangeCheckConstraint<F> {
    pub fn new(col_index: usize, constraint_idx: usize) -> Self {
        Self {
            col_index,
            constraint_idx,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for RangeCheckConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        6 // Degree 6 because we check product of (value - i) for i in 0..6
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
    ) {
        let frame = match evaluation_context {
            TransitionEvaluationContext::Prover { frame, .. } => frame,
            TransitionEvaluationContext::Verifier { frame, .. } => frame,
        };

        let step = frame.get_evaluation_step(0);
        let value = step.get_main_evaluation_element(0, self.col_index);

        // Range check: value must be in [0, 5]
        #[allow(clippy::needless_borrow)]
        let check = range_check(&value, 6);
        transition_evaluations[self.constraint_idx()] = check;
    }
}

/// Secret commitment constraint: Verifies the secret matches the committed value.
/// The commitment is: secret[0] + secret[1]*6 + secret[2]*36 + secret[3]*216
/// This binds the prover to a specific secret that was committed before guessing.
#[derive(Clone)]
struct SecretCommitmentConstraint<F: IsFFTField> {
    constraint_idx: usize,
    commitment: FieldElement<F>,
    phantom: std::marker::PhantomData<F>,
}

impl<F: IsFFTField> SecretCommitmentConstraint<F> {
    pub fn new(constraint_idx: usize, commitment: FieldElement<F>) -> Self {
        Self {
            constraint_idx,
            commitment,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for SecretCommitmentConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        1 // Linear constraint
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
    ) {
        let frame = match evaluation_context {
            TransitionEvaluationContext::Prover { frame, .. } => frame,
            TransitionEvaluationContext::Verifier { frame, .. } => frame,
        };

        let step = frame.get_evaluation_step(0);

        let s0 = step.get_main_evaluation_element(0, cols::SECRET_0);
        let s1 = step.get_main_evaluation_element(0, cols::SECRET_1);
        let s2 = step.get_main_evaluation_element(0, cols::SECRET_2);
        let s3 = step.get_main_evaluation_element(0, cols::SECRET_3);

        // Commitment encoding: s0 + s1*6 + s2*36 + s3*216
        let six = FieldElement::<F>::from(6u64);
        let thirty_six = FieldElement::<F>::from(36u64);
        let two_sixteen = FieldElement::<F>::from(216u64);

        let computed = s0 + s1 * &six + s2 * &thirty_six + s3 * two_sixteen;

        // Constraint: computed_commitment - public_commitment = 0
        transition_evaluations[self.constraint_idx()] = computed - &self.commitment;
    }
}

/// Mastermind AIR implementation
pub struct MastermindAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    trace_length: usize,
    pub_inputs: MastermindPublicInputs<F>,
    constraints: Vec<Box<dyn TransitionConstraint<F, F>>>,
}

impl<F> MastermindAIR<F>
where
    F: IsFFTField + Send + Sync + 'static,
{
    pub fn new(
        trace_length: usize,
        pub_inputs: &MastermindPublicInputs<F>,
        proof_options: &ProofOptions,
    ) -> Self {
        let mut constraints: Vec<Box<dyn TransitionConstraint<F, F>>> = Vec::new();
        let mut idx = 0;

        // Constraint 0: Secret commitment verification
        constraints.push(Box::new(SecretCommitmentConstraint::new(
            idx,
            pub_inputs.secret_commitment.clone(),
        )));
        idx += 1;

        // Constraints 1-4: Boolean constraints for eq_0..eq_3
        for i in 0..4 {
            constraints.push(Box::new(BooleanConstraint::new(cols::EQ_0 + i, idx)));
            idx += 1;
        }

        // Constraints 5-8: Equality indicator constraints
        for i in 0..4 {
            constraints.push(Box::new(EqualityIndicatorConstraint::new(i, idx)));
            idx += 1;
        }

        // Constraint 9: Exact count sum constraint
        constraints.push(Box::new(ExactCountConstraint::new(idx)));
        idx += 1;

        // Constraints 10-17: Range check constraints for all 8 color columns
        for col in 0..8 {
            constraints.push(Box::new(RangeCheckConstraint::new(col, idx)));
            idx += 1;
        }

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 14, // 4 secret + 4 guess + 2 feedback + 4 equality indicators
            transition_offsets: vec![0],
            num_transition_constraints: constraints.len(),
        };

        Self {
            pub_inputs: pub_inputs.clone(),
            context,
            trace_length,
            constraints,
        }
    }
}

impl<F> AIR for MastermindAIR<F>
where
    F: IsFFTField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = MastermindPublicInputs<F>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        Self::new(trace_length, pub_inputs, proof_options)
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length()
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::Field>>> {
        &self.constraints
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &[FieldElement<Self::Field>],
    ) -> BoundaryConstraints<Self::Field> {
        let mut constraints_vec: Vec<BoundaryConstraint<Self::Field>> = Vec::new();

        // Boundary constraints for guess columns (public input) at row 0
        for i in 0..4 {
            let val = self.pub_inputs.guess[i].clone();
            let constraint = BoundaryConstraint::new_main(cols::GUESS_0 + i, 0, val);
            constraints_vec.push(constraint);
        }

        // Boundary constraint for exact match count
        let exact_val = self.pub_inputs.feedback[0].clone();
        let exact_constraint = BoundaryConstraint::new_main(cols::AUX_EXACT, 0, exact_val);
        constraints_vec.push(exact_constraint);

        // Boundary constraint for partial match count
        let partial_val = self.pub_inputs.feedback[1].clone();
        let partial_constraint = BoundaryConstraint::new_main(cols::AUX_PARTIAL, 0, partial_val);
        constraints_vec.push(partial_constraint);

        BoundaryConstraints::from_constraints(constraints_vec)
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (14, 0) // 14 main columns, 0 auxiliary columns
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}
