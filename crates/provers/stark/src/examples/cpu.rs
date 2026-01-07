use crate::{
    constraints::{boundary::BoundaryConstraints, transition::TransitionConstraint},
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{TransitionEvaluationContext, AIR},
    Felt252,
};
use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
    },
};
type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;

pub struct CPUAir {
    context: AirContext,
    trace_length: usize,
    pub_inputs: Vec<F>,
    transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
}

struct LogUpAddConstraint {}

impl LogUpAddConstraint {
    pub fn new() -> Self {
        Self {}
    }
}

impl TransitionConstraint<F, E> for LogUpAddConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        0
    }

    fn end_exemptions(&self) -> usize {
        1
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _periodic_values,
                rap_challenges,
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                // Auxiliary frame elements
                let s0 = first_step.get_aux_evaluation_element(0, 0);
                let s1 = second_step.get_aux_evaluation_element(0, 0);

                // Challenges
                let z = &rap_challenges[0];
                let alpha = &rap_challenges[1];

                // Main frame elements
                let flag = second_step.get_main_evaluation_element(0, 0);
                let a = second_step.get_main_evaluation_element(0, 2);
                let b = second_step.get_main_evaluation_element(0, 3);
                let c = second_step.get_main_evaluation_element(0, 4);

                let fingerprint = -(a + b * alpha + c * alpha.square()) + z;

                // We are using the following LogUp equation:
                // s1 = s0 + flag / fingerprint
                // 0 =  s0 * fingerprint + flag - s1 * fingerprint
                // Since constraints must be expressed without division, we multiply each term by sorted_term * unsorted_term:
                let res = flag + s0 * &fingerprint - s1 * fingerprint;

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
                    *eval = res;
                }
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges,
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                // Auxiliary frame elements
                let s0 = first_step.get_aux_evaluation_element(0, 0);
                let s1 = second_step.get_aux_evaluation_element(0, 0);

                // Challenges
                let z = &rap_challenges[0];
                let alpha = &rap_challenges[1];

                // Main frame elements
                let flag = second_step.get_main_evaluation_element(0, 0);
                let a = second_step.get_main_evaluation_element(0, 2);
                let b = second_step.get_main_evaluation_element(0, 3);
                let c = second_step.get_main_evaluation_element(0, 4);

                let fingerprint = -(a + b * alpha + c * alpha.square()) + z;

                // We are using the following LogUp equation:
                // s1 = s0 + flag / fingerprint
                // Since constraints must be expressed without division, we multiply each term by sorted_term * unsorted_term:
                let res = s0 * &fingerprint + flag - s1 * fingerprint;

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
                    *eval = res;
                }
            }
        }
    }
}

struct LogUpMulConstraint {}

impl LogUpMulConstraint {
    pub fn new() -> Self {
        Self {}
    }
}

impl TransitionConstraint<F, E> for LogUpMulConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        1
    }

    fn end_exemptions(&self) -> usize {
        1
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _periodic_values,
                rap_challenges,
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                // Auxiliary frame elements
                let s0 = first_step.get_aux_evaluation_element(0, 1);
                let s1 = second_step.get_aux_evaluation_element(0, 1);

                // Challenges
                let z = &rap_challenges[0];
                let alpha = &rap_challenges[1];

                // Main frame elements
                let flag = second_step.get_main_evaluation_element(0, 1);
                let a = second_step.get_main_evaluation_element(0, 2);
                let b = second_step.get_main_evaluation_element(0, 3);
                let c = second_step.get_main_evaluation_element(0, 4);

                let fingerprint = -(a + b * alpha + c * alpha.square()) + z;

                // We are using the following LogUp equation:
                // s1 = s0 + flag / fingerprint
                // Since constraints must be expressed without division, we multiply each term by sorted_term * unsorted_term:
                let res = flag + s0 * &fingerprint - s1 * fingerprint;

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
                    *eval = res;
                }
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges,
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                // Auxiliary frame elements
                let s0 = first_step.get_aux_evaluation_element(0, 1);
                let s1 = second_step.get_aux_evaluation_element(0, 1);

                // Challenges
                let z = &rap_challenges[0];
                let alpha = &rap_challenges[1];

                // Main frame elements
                let flag = second_step.get_main_evaluation_element(0, 1);
                let a = second_step.get_main_evaluation_element(0, 2);
                let b = second_step.get_main_evaluation_element(0, 3);
                let c = second_step.get_main_evaluation_element(0, 4);

                let fingerprint = -(a + b * alpha + c * alpha.square()) + z;

                // We are using the following LogUp equation:
                // s1 = s0 + flag / fingerprint
                // Since constraints must be expressed without division, we multiply each term by sorted_term * unsorted_term:
                let res = s0 * &fingerprint + flag - s1 * fingerprint;

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
                    *eval = res;
                }
            }
        }
    }
}

impl AIR for CPUAir {
    type Field = F;
    type FieldExtension = E;
    type PublicInputs = Vec<F>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        let transition_constraints: Vec<
            Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>,
        > = vec![
            Box::new(LogUpAddConstraint::new()),
            Box::new(LogUpMulConstraint::new()),
        ];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 8,
            transition_offsets: vec![0, 1],
            num_transition_constraints: transition_constraints.len(),
        };

        Self {
            context,
            trace_length,
            pub_inputs: pub_inputs.clone(),
            transition_constraints,
        }
    }

    fn build_auxiliary_trace(
        &self,
        trace: &mut TraceTable<Self::Field, Self::FieldExtension>,
        challenges: &[FieldElement<E>],
    ) {
        // Main table
        let main_segment_cols = trace.columns_main();
        let add_flag = &main_segment_cols[0];
        let mul_flag = &main_segment_cols[1];
        let a = &main_segment_cols[2];
        let b = &main_segment_cols[3];
        let c = &main_segment_cols[4];

        // Challenges
        let z = &challenges[0];
        let alpha = &challenges[1];

        let trace_len = trace.num_rows();
        let mut aux_add_col = Vec::new();
        let mut aux_mul_col = Vec::new();
        let mut aux_total_col = Vec::new();

        // s_0 = flag / fingerprint
        // s1 = s0 + flag / fingerprint
        // fingerprint = z - (a + b * alpha + c * alpha^2)
        let fingerprint_inv =
            (-(a[0].clone() + b[0].clone() * alpha + c[0].clone() * alpha.square()) + z)
                .inv()
                .unwrap();
        aux_add_col.push(add_flag[0].clone() * fingerprint_inv.clone());
        aux_mul_col.push(mul_flag[0].clone() * fingerprint_inv.clone());
        aux_total_col.push(aux_add_col[0].clone() + aux_mul_col[0].clone());

        // Apply the same equation given in the permutation transition contraint to the rest of the trace.
        // s_{i+1} = s_i + flag_{i+1}/fingerprint
        for i in 0..trace_len - 1 {
            let fingerprint_inv = (-(a[i + 1].clone()
                + b[i + 1].clone() * alpha
                + c[i + 1].clone() * alpha.square())
                + z)
                .inv()
                .unwrap();
            aux_add_col.push(&aux_add_col[i] + &add_flag[i + 1] * fingerprint_inv.clone());
            aux_mul_col.push(&aux_mul_col[i] + &mul_flag[i + 1] * fingerprint_inv.clone());
            aux_total_col.push(&aux_total_col[i] + &aux_add_col[i + 1] + &aux_mul_col[i + 1]);
        }

        for i in 0..trace_len {
            trace.set_aux(i, 0, aux_add_col[i].clone());
            trace.set_aux(i, 1, aux_mul_col[i].clone());
            trace.set_aux(i, 2, aux_total_col[i].clone());
        }
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut dyn IsStarkTranscript<Self::FieldExtension, Self::Field>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        vec![
            transcript.sample_field_element(),
            transcript.sample_field_element(),
        ]
    }

    fn trace_layout(&self) -> (usize, usize) {
        (5, 3)
    }

    fn boundary_constraints(
        &self,
        rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::FieldExtension> {
        BoundaryConstraints::from_constraints(vec![])
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> {
        &self.transition_constraints
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    // The prover use this function to define the number of parts of the composition polynomial.
    // The number of parts will be: composition_poly_degree_bound() / trace_length().
    // Since we have a transition constraint of degree 3, we need the bound to be two times the trace length.
    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length() * 2
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}
