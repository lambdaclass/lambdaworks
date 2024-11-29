use std::{iter::once, marker::PhantomData};

use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        transition::TransitionConstraint,
    },
    context::AirContext,
    frame::Frame,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::AIR,
};
use itertools::Itertools;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::traits::IsPrimeField;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    traits::ByteConversion,
};

/// Transition Constraint that ensures the continuity of the sorted address column of a memory.
#[derive(Clone)]
struct ContinuityConstraint<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> ContinuityConstraint<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for ContinuityConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        0
    }

    fn end_exemptions(&self) -> usize {
        // NOTE: We are assuming that the trace has as length a power of 2.
        1
    }

    fn evaluate(
        &self,
        frame: &Frame<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
        _periodic_values: &[FieldElement<F>],
        _rap_challenges: &[FieldElement<F>],
    ) {
        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);

        let a_sorted_0 = first_step.get_main_evaluation_element(0, 2);
        let a_sorted_1 = second_step.get_main_evaluation_element(0, 2);
        // (a'_{i+1} - a'_i)(a'_{i+1} - a'_i - 1) = 0 where a' is the sorted address
        let res = (a_sorted_1 - a_sorted_0) * (a_sorted_1 - a_sorted_0 - FieldElement::<F>::one());

        // The eval always exists, except if the constraint idx were incorrectly defined.
        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
            *eval = res;
        }
    }
}
/// Transition constraint that ensures that same addresses have same values, making the sorted memory read-only.
#[derive(Clone)]
struct SingleValueConstraint<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> SingleValueConstraint<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for SingleValueConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        1
    }

    fn end_exemptions(&self) -> usize {
        // NOTE: We are assuming that the trace has as length a power of 2.
        1
    }

    fn evaluate(
        &self,
        frame: &Frame<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
        _periodic_values: &[FieldElement<F>],
        _rap_challenges: &[FieldElement<F>],
    ) {
        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);

        let a_sorted_0 = first_step.get_main_evaluation_element(0, 2);
        let a_sorted_1 = second_step.get_main_evaluation_element(0, 2);
        let v_sorted_0 = first_step.get_main_evaluation_element(0, 3);
        let v_sorted_1 = second_step.get_main_evaluation_element(0, 3);
        // (v'_{i+1} - v'_i) * (a'_{i+1} - a'_i - 1) = 0
        let res = (v_sorted_1 - v_sorted_0) * (a_sorted_1 - a_sorted_0 - FieldElement::<F>::one());

        // The eval always exists, except if the constraint idx were incorrectly defined.
        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
            *eval = res;
        }
    }
}
/// Transition constraint that ensures that the sorted columns are a permutation of the original ones.
/// We are using the LogUp construction described in:
/// <https://0xpolygonmiden.github.io/miden-vm/design/lookups/logup.html>
#[derive(Clone)]
struct PermutationConstraint<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> PermutationConstraint<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for PermutationConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        3
    }

    fn constraint_idx(&self) -> usize {
        2
    }

    fn end_exemptions(&self) -> usize {
        1
    }

    fn evaluate(
        &self,
        frame: &Frame<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
        _periodic_values: &[FieldElement<F>],
        rap_challenges: &[FieldElement<F>],
    ) {
        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);

        // Auxiliary constraints
        let s0 = first_step.get_aux_evaluation_element(0, 0);
        let s1 = second_step.get_aux_evaluation_element(0, 0);
        let z = &rap_challenges[0];
        let alpha = &rap_challenges[1];
        let a1 = second_step.get_main_evaluation_element(0, 0);
        let v1 = second_step.get_main_evaluation_element(0, 1);
        let a_sorted_1 = second_step.get_main_evaluation_element(0, 2);
        let v_sorted_1 = second_step.get_main_evaluation_element(0, 3);
        let m = second_step.get_main_evaluation_element(0, 4);

        let unsorted_term = z - (a1 + alpha * v1);
        let sorted_term = z - (a_sorted_1 + alpha * v_sorted_1);

        // We are using the following LogUp equation:
        // s1 = s0 + m / sorted_term - 1/unsorted_term.
        // Since constraints must be expressed without division, we multiply each term by sorted_term * unsorted_term:
        let res = s0 * &unsorted_term * &sorted_term + m * &unsorted_term
            - &sorted_term
            - s1 * unsorted_term * sorted_term;

        // The eval always exists, except if the constraint idx were incorrectly defined.
        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
            *eval = res;
        }
    }
}

/// AIR for a continuous read-only memory.
pub struct LogReadOnlyRAP<F>
where
    F: IsFFTField,
{
    context: AirContext,
    trace_length: usize,
    pub_inputs: LogReadOnlyPublicInputs<F>,
    transition_constraints: Vec<Box<dyn TransitionConstraint<F, F>>>,
}

#[derive(Clone, Debug)]
pub struct LogReadOnlyPublicInputs<F>
where
    F: IsFFTField,
{
    pub a0: FieldElement<F>,
    pub v0: FieldElement<F>,
    pub a_sorted_0: FieldElement<F>,
    pub v_sorted_0: FieldElement<F>,
    // The multiplicity of (a_sorted_0, v_sorted_0)
    pub m0: FieldElement<F>,
}

impl<F> AIR for LogReadOnlyRAP<F>
where
    F: IsFFTField + Send + Sync + 'static,
    FieldElement<F>: ByteConversion,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = LogReadOnlyPublicInputs<F>;

    const STEP_SIZE: usize = 1;

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        let transition_constraints: Vec<
            Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>,
        > = vec![
            Box::new(ContinuityConstraint::new()),
            Box::new(SingleValueConstraint::new()),
            Box::new(PermutationConstraint::new()),
        ];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 6,
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
        challenges: &[FieldElement<F>],
    ) {
        let main_segment_cols = trace.columns_main();
        let a = &main_segment_cols[0];
        let v = &main_segment_cols[1];
        let a_sorted = &main_segment_cols[2];
        let v_sorted = &main_segment_cols[3];
        let m = &main_segment_cols[4];
        let z = &challenges[0];
        let alpha = &challenges[1];

        let trace_len = trace.num_rows();
        let mut aux_col = Vec::new();

        let unsorted_term = (z - (&a[0] + alpha * &v[0])).inv().unwrap();
        let sorted_term = (z - (&a_sorted[0] + alpha * &v_sorted[0])).inv().unwrap();
        aux_col.push(&m[0] * sorted_term - unsorted_term);

        // Apply the same equation given in the permutation case to the rest of the trace
        for i in 0..trace_len - 1 {
            let unsorted_term = (z - (&a[i + 1] + alpha * &v[i + 1])).inv().unwrap();
            let sorted_term = (z - (&a_sorted[i + 1] + alpha * &v_sorted[i + 1]))
                .inv()
                .unwrap();
            aux_col.push(&aux_col[i] + &m[i + 1] * sorted_term - unsorted_term);
        }

        for (i, aux_elem) in aux_col.iter().enumerate().take(trace.num_rows()) {
            trace.set_aux(i, 0, aux_elem.clone())
        }
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut impl IsTranscript<Self::Field>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        vec![
            transcript.sample_field_element(),
            transcript.sample_field_element(),
        ]
    }

    fn trace_layout(&self) -> (usize, usize) {
        (5, 1)
    }

    fn boundary_constraints(
        &self,
        rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::FieldExtension> {
        let a0 = &self.pub_inputs.a0;
        let v0 = &self.pub_inputs.v0;
        let a_sorted_0 = &self.pub_inputs.a_sorted_0;
        let v_sorted_0 = &self.pub_inputs.v_sorted_0;
        let m0 = &self.pub_inputs.m0;
        let z = &rap_challenges[0];
        let alpha = &rap_challenges[1];

        // Main boundary constraints
        let c1 = BoundaryConstraint::new_main(0, 0, a0.clone());
        let c2 = BoundaryConstraint::new_main(1, 0, v0.clone());
        let c3 = BoundaryConstraint::new_main(2, 0, a_sorted_0.clone());
        let c4 = BoundaryConstraint::new_main(3, 0, v_sorted_0.clone());
        let c5 = BoundaryConstraint::new_main(4, 0, m0.clone());

        // Auxiliary boundary constraints
        let unsorted_term = (z - (a0 + alpha * v0)).inv().unwrap();
        let sorted_term = (z - (a_sorted_0 + alpha * v_sorted_0)).inv().unwrap();
        let p0_value = m0 * sorted_term - unsorted_term;

        let c_aux1 = BoundaryConstraint::new_aux(0, 0, p0_value);
        let c_aux2 = BoundaryConstraint::new_aux(
            0,
            self.trace_length - 1,
            FieldElement::<Self::FieldExtension>::zero(),
        );

        BoundaryConstraints::from_constraints(vec![c1, c2, c3, c4, c5, c_aux1, c_aux2])
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

    fn compute_transition_verifier(
        &self,
        frame: &Frame<Self::FieldExtension, Self::FieldExtension>,
        periodic_values: &[FieldElement<Self::FieldExtension>],
        rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> Vec<FieldElement<Self::Field>> {
        self.compute_transition_prover(frame, periodic_values, rap_challenges)
    }
}

/// Return a trace table with an auxiliary column full of zeros (that will be completed by the air) and
/// the following five main columns: The original addresses and values, the sorted addresses and values without
/// repetition and the multiplicities that tell
pub fn read_only_logup_trace<F: IsFFTField + IsPrimeField>(
    addresses: Vec<FieldElement<F>>,
    values: Vec<FieldElement<F>>,
) -> TraceTable<F, F> {
    let mut address_value_pairs: Vec<_> = addresses.iter().zip(values.iter()).collect();
    address_value_pairs.sort_by_key(|(addr, _)| addr.representative());

    let mut multiplicities = Vec::new();
    let mut sorted_addresses = Vec::new();
    let mut sorted_values = Vec::new();

    for (key, group) in &address_value_pairs.into_iter().group_by(|&(a, v)| (a, v)) {
        let group_vec: Vec<_> = group.collect();
        multiplicities.push(FieldElement::<F>::from(group_vec.len() as u64));
        sorted_addresses.push(key.0.clone());
        sorted_values.push(key.1.clone());
    }

    // We resize the sorted addresses and values with the last value of each one so they have the
    // same number of rows as the original addresses and values. However, their multiplicity should be zero.
    sorted_addresses.resize(addresses.len(), sorted_addresses.last().unwrap().clone());
    sorted_values.resize(addresses.len(), sorted_values.last().unwrap().clone());
    multiplicities.resize(addresses.len(), FieldElement::<F>::zero());

    let main_columns = vec![
        addresses.clone(),
        values.clone(),
        sorted_addresses,
        sorted_values,
        multiplicities,
    ];

    // create a vector with zeros of the same length as the main columns
    let zero_vec = vec![FieldElement::<F>::zero(); main_columns[0].len()];
    TraceTable::from_columns(main_columns, vec![zero_vec], 1)
}

#[cfg(test)]
mod test {
    use super::*;
    use lambdaworks_math::field::fields::{
        fft_friendly::stark_252_prime_field::Stark252PrimeField, u64_prime_field::FE17,
    };

    #[test]
    fn tes_logup_trace_construction() {
        let address_col = vec![
            FE17::from(3),
            FE17::from(7),
            FE17::from(2),
            FE17::from(8),
            FE17::from(4),
            FE17::from(5),
            FE17::from(1),
            FE17::from(6),
        ];
        let value_col = vec![
            FE17::from(30),
            FE17::from(70),
            FE17::from(20),
            FE17::from(80),
            FE17::from(40),
            FE17::from(50),
            FE17::from(10),
            FE17::from(60),
        ];

        let logup_trace = read_only_logup_trace(address_col, value_col);

        let expected_sorted_addresses = vec![
            FE17::from(1),
            FE17::from(2),
            FE17::from(3),
            FE17::from(4),
            FE17::from(5),
            FE17::from(6),
            FE17::from(7),
            FE17::from(8),
        ];
        let expected_sorted_values = vec![
            FE17::from(10),
            FE17::from(20),
            FE17::from(30),
            FE17::from(40),
            FE17::from(50),
            FE17::from(60),
            FE17::from(70),
            FE17::from(80),
        ];
        let expected_multiplicities = vec![
            FE17::one(),
            FE17::one(),
            FE17::one(),
            FE17::one(),
            FE17::one(),
            FE17::one(),
            FE17::one(),
            FE17::one(),
        ];
        assert_eq!(logup_trace.columns_main()[2], expected_sorted_addresses);
        assert_eq!(logup_trace.columns_main()[3], expected_sorted_values);
        assert_eq!(logup_trace.columns_main()[4], expected_multiplicities);
    }

    #[test]
    fn test_logup_trace_construction_2() {
        let address_col = vec![
            FieldElement::<Stark252PrimeField>::from(3), // a0
            FieldElement::<Stark252PrimeField>::from(2), // a1
            FieldElement::<Stark252PrimeField>::from(2), // a2
            FieldElement::<Stark252PrimeField>::from(3), // a3
            FieldElement::<Stark252PrimeField>::from(4), // a4
            FieldElement::<Stark252PrimeField>::from(5), // a5
            FieldElement::<Stark252PrimeField>::from(1), // a6
            FieldElement::<Stark252PrimeField>::from(3), // a7
        ];
        let value_col = vec![
            FieldElement::<Stark252PrimeField>::from(30), // v0
            FieldElement::<Stark252PrimeField>::from(20), // v1
            FieldElement::<Stark252PrimeField>::from(20), // v2
            FieldElement::<Stark252PrimeField>::from(30), // v3
            FieldElement::<Stark252PrimeField>::from(40), // v4
            FieldElement::<Stark252PrimeField>::from(50), // v5
            FieldElement::<Stark252PrimeField>::from(10), // v6
            FieldElement::<Stark252PrimeField>::from(30), // v7
        ];

        let sorted_address_col = vec![
            FieldElement::<Stark252PrimeField>::from(1), // a0
            FieldElement::<Stark252PrimeField>::from(2), // a1
            FieldElement::<Stark252PrimeField>::from(3), // a2
            FieldElement::<Stark252PrimeField>::from(4), // a3
            FieldElement::<Stark252PrimeField>::from(5), // a4
            FieldElement::<Stark252PrimeField>::from(5), // a5
            FieldElement::<Stark252PrimeField>::from(5), // a6
            FieldElement::<Stark252PrimeField>::from(5), // a7
        ];
        let sorted_value_col = vec![
            FieldElement::<Stark252PrimeField>::from(10), // v0
            FieldElement::<Stark252PrimeField>::from(20), // v1
            FieldElement::<Stark252PrimeField>::from(30), // v2
            FieldElement::<Stark252PrimeField>::from(40), // v3
            FieldElement::<Stark252PrimeField>::from(50), // v4
            FieldElement::<Stark252PrimeField>::from(50), // v5
            FieldElement::<Stark252PrimeField>::from(50), // v6
            FieldElement::<Stark252PrimeField>::from(50), // v7
        ];

        let multiplicity_col = vec![
            FieldElement::<Stark252PrimeField>::from(1), // v0
            FieldElement::<Stark252PrimeField>::from(2), // v1
            FieldElement::<Stark252PrimeField>::from(3), // v2
            FieldElement::<Stark252PrimeField>::from(1), // v3
            FieldElement::<Stark252PrimeField>::from(1), // v4
            FieldElement::<Stark252PrimeField>::from(0), // v5
            FieldElement::<Stark252PrimeField>::from(0), // v6
            FieldElement::<Stark252PrimeField>::from(0), // v7
        ];
        let logup_trace = read_only_logup_trace(address_col, value_col);

        assert_eq!(logup_trace.columns_main()[2], sorted_address_col);
        assert_eq!(logup_trace.columns_main()[3], sorted_value_col);
        assert_eq!(logup_trace.columns_main()[4], multiplicity_col);
    }
}
