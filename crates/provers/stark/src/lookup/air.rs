use std::marker::PhantomData;
use std::sync::RwLock;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use lambdaworks_math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        transition::TransitionConstraint,
    },
    context::AirContext,
    proof::options::ProofOptions,
    prover::ProvingError,
    trace::TraceTable,
    traits::AIR,
};

use super::{
    constraints::{LookupAccumulatedConstraint, LookupTermConstraint},
    trace_builder::{build_accumulated_column, build_logup_term_column},
    types::{BoundaryConstraintBuilder, BusInteraction},
};

/// AIR with LogUp lookup argument support.
///
/// This struct wraps the LogUp protocol machinery, auto-generating:
/// - Auxiliary trace columns (term columns + accumulated column)
/// - Transition constraints for term verification and accumulation
/// - Boundary constraints for the accumulated column
///
/// Users provide their own transition constraints and bus interactions;
/// `AirWithLogUp` handles the rest.
pub struct AirWithLogUp<F, E, B, PI>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    B: BoundaryConstraintBuilder<F, E, PI>,
    PI: Send + Sync,
{
    context: AirContext,
    step_size: usize,
    trace_layout: (usize, usize),
    transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    interactions: Vec<BusInteraction>,
    trace_length: usize,
    pub_inputs: PI,
    /// Stores the computed initial term values (row 0) after `build_auxiliary_trace()`.
    initial_terms: RwLock<Vec<FieldElement<E>>>,
    /// Stores the computed final accumulated value (last row) after `build_auxiliary_trace()`.
    acc_final_value: RwLock<FieldElement<E>>,
    _phantom: PhantomData<(F, B)>,
}

impl<F, E, B, PI> AirWithLogUp<F, E, B, PI>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    B: BoundaryConstraintBuilder<F, E, PI>,
    PI: Send + Sync,
{
    /// Creates an AirWithLogUp with LogUp-specific transition constraints.
    ///
    /// Auxiliary column layout (when interactions are present):
    /// - Columns 0..N-1: Term columns (one per interaction)
    /// - Column N: Accumulated column (running sum of all terms)
    ///
    /// Total aux columns = N + 1
    pub fn new(
        trace_length: usize,
        pub_inputs: PI,
        num_main_columns: usize,
        interactions: Vec<BusInteraction>,
        proof_options: &ProofOptions,
        step_size: usize,
        mut transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    ) -> Self {
        let num_interactions = interactions.len();

        // Add a term constraint for each interaction
        for (i, interaction) in interactions.iter().enumerate() {
            let constraint =
                LookupTermConstraint::new(interaction.clone(), i, transition_constraints.len());
            transition_constraints.push(Box::new(constraint));
        }

        // Add the accumulated constraint
        if num_interactions > 0 {
            let accumulated_constraint =
                LookupAccumulatedConstraint::new(transition_constraints.len(), num_interactions);
            transition_constraints.push(Box::new(accumulated_constraint));
        }

        let num_aux_columns = if num_interactions > 0 {
            num_interactions + 1
        } else {
            0
        };
        let trace_layout = (num_main_columns, num_aux_columns);

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: trace_layout.0 + trace_layout.1,
            transition_offsets: vec![0, 1],
            num_transition_constraints: transition_constraints.len(),
        };

        Self {
            context,
            step_size,
            trace_layout,
            transition_constraints,
            interactions,
            trace_length,
            pub_inputs,
            initial_terms: RwLock::new(Vec::new()),
            acc_final_value: RwLock::new(FieldElement::<E>::zero()),
            _phantom: PhantomData,
        }
    }
}

impl<F, E, B, PI> AIR for AirWithLogUp<F, E, B, PI>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    B: BoundaryConstraintBuilder<F, E, PI>,
    PI: Send + Sync,
{
    type Field = F;
    type FieldExtension = E;
    type PublicInputs = PI;

    fn step_size(&self) -> usize {
        self.step_size
    }

    fn new(
        _trace_length: usize,
        _pub_inputs: &Self::PublicInputs,
        _proof_options: &ProofOptions,
    ) -> Self
    where
        Self: Sized,
    {
        unreachable!("AirWithLogUp should only be created via AirWithLogUp::new()")
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut dyn IsStarkTranscript<Self::FieldExtension, Self::Field>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        vec![
            transcript.sample_field_element(), // z
            transcript.sample_field_element(), // alpha
        ]
    }

    fn build_auxiliary_trace(
        &self,
        trace: &mut TraceTable<Self::Field, Self::FieldExtension>,
        challenges: &[FieldElement<Self::FieldExtension>],
    ) -> Result<(), ProvingError> {
        let num_interactions = self.interactions.len();
        if num_interactions == 0 {
            return Ok(());
        }

        // Allocate aux table if dimensions don't match
        let (_, num_aux_columns) = self.trace_layout;
        if num_aux_columns > 0 && trace.num_aux_columns != num_aux_columns {
            trace.allocate_aux_table(num_aux_columns);
        }

        // Build term columns
        let main_segment_cols = trace.columns_main();
        for (i, interaction) in self.interactions.iter().enumerate() {
            build_logup_term_column(i, interaction, &main_segment_cols, trace, challenges)?;
        }

        // Store initial term values (row 0) for boundary constraints.
        let terms: Vec<FieldElement<E>> = (0..num_interactions)
            .map(|i| trace.get_aux(0, i).clone())
            .collect();
        *self.initial_terms.write().unwrap() = terms;

        // Build accumulated column
        let acc_col_idx = num_interactions;
        build_accumulated_column(acc_col_idx, num_interactions, trace);

        // Store final accumulated value for boundary constraint.
        let final_acc = trace.get_aux(trace.num_rows() - 1, acc_col_idx).clone();
        *self.acc_final_value.write().unwrap() = final_acc;

        Ok(())
    }

    fn boundary_constraints(
        &self,
        rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::FieldExtension> {
        let mut constraints = vec![];

        // LogUp boundary constraints:
        // 1. Pin each term column at row 0: term[k](0) = initial_terms[k]
        // 2. Pin acc[0] = Σ initial_terms
        // 3. Pin acc[N-1] = final_accumulated
        //
        // Pinning row-0 terms prevents the prover from injecting offsets
        // into acc[0]. For multi-table systems the caller verifies
        // Σ final_accumulated across all tables equals 0.
        if !self.interactions.is_empty() {
            let acc_col_idx = self.interactions.len();
            let initial_terms = self.initial_terms.read().unwrap();

            // Boundary constraint per term column at row 0.
            // Iterate over acc_col_idx (= num_interactions) rather than
            // initial_terms.len() so that missing entries default to zero.
            for i in 0..acc_col_idx {
                let expected = initial_terms
                    .get(i)
                    .cloned()
                    .unwrap_or_else(FieldElement::<Self::FieldExtension>::zero);
                constraints.push(BoundaryConstraint::new_aux(i, 0, expected));
            }

            // acc[0] = Σ initial_terms
            let initial_acc: FieldElement<Self::FieldExtension> =
                initial_terms.iter().cloned().sum();
            constraints.push(BoundaryConstraint::new_aux(acc_col_idx, 0, initial_acc));

            // acc[N-1] = final_accumulated
            let final_value = self.acc_final_value.read().unwrap().clone();
            constraints.push(BoundaryConstraint::new_aux(
                acc_col_idx,
                self.trace_length - 1,
                final_value,
            ));
        }

        // User-defined boundary constraints
        constraints.extend(B::boundary_constraints(&self.pub_inputs, rap_challenges));

        BoundaryConstraints::from_constraints(constraints)
    }

    fn trace_layout(&self) -> (usize, usize) {
        self.trace_layout
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length * 2
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> {
        &self.transition_constraints
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        proof::options::ProofOptions,
        prover::{IsStarkProver, Prover},
        verifier::{IsStarkVerifier, Verifier},
    };
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::{
        element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field,
    };

    use super::super::types::{BusValue, Multiplicity, NullBoundaryConstraintBuilder};

    // Use Goldilocks as both base and extension field (F = E).
    // This works because F: IsSubFieldOf<F> has a blanket impl.
    type F = Goldilocks64Field;
    type E = Goldilocks64Field;
    type FE = FieldElement<F>;

    #[test]
    fn test_bus_interaction_creation() {
        let sender = BusInteraction::sender(
            0u64,
            Multiplicity::One,
            vec![BusValue::column(0), BusValue::column(1)],
        );
        assert!(sender.is_sender);
        assert_eq!(sender.bus_id, 0);
        // bus_id + 2 values = 3 bus elements
        assert_eq!(sender.num_bus_elements(), 3);

        let receiver =
            BusInteraction::receiver(1u64, Multiplicity::Column(2), vec![BusValue::column(0)]);
        assert!(!receiver.is_sender);
        assert_eq!(receiver.bus_id, 1);
        assert_eq!(receiver.num_bus_elements(), 2);
    }

    #[test]
    fn test_bus_value_combine() {
        let val = BusValue::column(0);
        let result: FieldElement<F> = val.combine_from(|_col| FE::from(42u64));
        assert_eq!(result, FE::from(42u64));

        let val = BusValue::linear(vec![
            super::super::types::LinearTerm::Column {
                coefficient: 3,
                column: 0,
            },
            super::super::types::LinearTerm::Constant(7),
        ]);
        // 3 * 10 + 7 = 37
        let result: FieldElement<F> = val.combine_from(|_col| FE::from(10u64));
        assert_eq!(result, FE::from(37u64));
    }

    #[test]
    fn test_multiplicity_variants() {
        let m = Multiplicity::One;
        assert!(matches!(m, Multiplicity::One));

        let m = Multiplicity::Column(3);
        assert!(matches!(m, Multiplicity::Column(3)));

        let m = Multiplicity::Sum(1, 2);
        assert!(matches!(m, Multiplicity::Sum(1, 2)));

        let m = Multiplicity::Negated(0);
        assert!(matches!(m, Multiplicity::Negated(0)));
    }

    /// Simple permutation test: 8 rows, one sender and one receiver with the same
    /// values but in different order. The bus should balance (accumulated = 0 at end).
    #[test]
    fn test_logup_simple_permutation() {
        let trace_length = 8usize;
        let num_main_columns = 2;

        let sender_values: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let receiver_values: Vec<FE> = (1..=8).rev().map(|i| FE::from(i as u64)).collect();

        let main_columns = vec![sender_values, receiver_values];

        let aux_zero = vec![FieldElement::<E>::zero(); trace_length];
        let trace_aux = vec![aux_zero.clone(), aux_zero.clone(), aux_zero];
        let mut trace = TraceTable::from_columns(main_columns, trace_aux, 1);

        let interactions = vec![
            BusInteraction::sender(0u64, Multiplicity::One, vec![BusValue::column(0)]),
            BusInteraction::receiver(0u64, Multiplicity::One, vec![BusValue::column(1)]),
        ];

        let proof_options = ProofOptions::default_test_options();

        let air = AirWithLogUp::<F, E, NullBoundaryConstraintBuilder, ()>::new(
            trace_length,
            (),
            num_main_columns,
            interactions,
            &proof_options,
            1,
            vec![],
        );

        let proof = Prover::prove(&air, &mut trace, &mut DefaultTranscript::<E>::new(&[]))
            .expect("proof generation failed");

        assert!(Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<E>::new(&[]),
        ));
    }

    /// Test with multiplicities > 1: a lookup table with values [10, 20, 30, 40]
    /// and lookup requests that repeat some values.
    #[test]
    fn test_logup_with_multiplicities() {
        let trace_length = 8usize;
        let num_main_columns = 3;

        let table_values: Vec<FE> = vec![
            FE::from(10u64),
            FE::from(20u64),
            FE::from(30u64),
            FE::from(40u64),
            FE::from(40u64),
            FE::from(40u64),
            FE::from(40u64),
            FE::from(40u64),
        ];
        let multiplicities: Vec<FE> = vec![
            FE::from(2u64),
            FE::from(1u64),
            FE::from(3u64),
            FE::from(2u64),
            FE::from(0u64),
            FE::from(0u64),
            FE::from(0u64),
            FE::from(0u64),
        ];
        let lookup_values: Vec<FE> = vec![
            FE::from(10u64),
            FE::from(10u64),
            FE::from(20u64),
            FE::from(30u64),
            FE::from(30u64),
            FE::from(30u64),
            FE::from(40u64),
            FE::from(40u64),
        ];

        let main_columns = vec![table_values, multiplicities, lookup_values];

        let aux_zero = vec![FieldElement::<E>::zero(); trace_length];
        let trace_aux = vec![aux_zero.clone(), aux_zero.clone(), aux_zero];
        let mut trace = TraceTable::from_columns(main_columns, trace_aux, 1);

        let interactions = vec![
            BusInteraction::sender(0u64, Multiplicity::Column(1), vec![BusValue::column(0)]),
            BusInteraction::receiver(0u64, Multiplicity::One, vec![BusValue::column(2)]),
        ];

        let proof_options = ProofOptions::default_test_options();

        let air = AirWithLogUp::<F, E, NullBoundaryConstraintBuilder, ()>::new(
            trace_length,
            (),
            num_main_columns,
            interactions,
            &proof_options,
            1,
            vec![],
        );

        let proof = Prover::prove(&air, &mut trace, &mut DefaultTranscript::<E>::new(&[]))
            .expect("proof generation failed");

        assert!(Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<E>::new(&[]),
        ));
    }

    /// Verify that for a balanced bus, the accumulated column's final value
    /// makes the proof succeed.
    #[test]
    fn test_logup_bus_balance() {
        let trace_length = 8usize;
        let num_main_columns = 2;

        let values: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();

        let main_columns = vec![values.clone(), values];

        let aux_zero = vec![FieldElement::<E>::zero(); trace_length];
        let trace_aux = vec![aux_zero.clone(), aux_zero.clone(), aux_zero];
        let mut trace = TraceTable::from_columns(main_columns, trace_aux, 1);

        let interactions = vec![
            BusInteraction::sender(0u64, Multiplicity::One, vec![BusValue::column(0)]),
            BusInteraction::receiver(0u64, Multiplicity::One, vec![BusValue::column(1)]),
        ];

        let proof_options = ProofOptions::default_test_options();

        let air = AirWithLogUp::<F, E, NullBoundaryConstraintBuilder, ()>::new(
            trace_length,
            (),
            num_main_columns,
            interactions,
            &proof_options,
            1,
            vec![],
        );

        // Build the aux trace manually to check the accumulated column sums to 0
        let challenges = vec![
            FieldElement::<E>::from(123u64),
            FieldElement::<E>::from(456u64),
        ];

        let mut check_trace = trace.clone();
        air.build_auxiliary_trace(&mut check_trace, &challenges)
            .expect("aux trace build failed");

        // The accumulated column should end at 0 for a balanced bus where
        // sender and receiver have identical values with multiplicity 1
        let last_row = trace_length - 1;
        let acc_col_idx = 2; // 2 interactions -> acc at index 2
        let final_acc = check_trace.get_aux(last_row, acc_col_idx);
        assert_eq!(*final_acc, FieldElement::<E>::zero());

        // Full prove/verify
        let proof = Prover::prove(&air, &mut trace, &mut DefaultTranscript::<E>::new(&[]))
            .expect("proof generation failed");

        assert!(Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<E>::new(&[]),
        ));
    }

    /// Cross-table bus test: Table A only sends, Table B only receives.
    /// Neither table balances individually (acc[N-1] != 0), but
    /// final_acc_A + final_acc_B = 0 (global bus balance).
    #[test]
    fn test_logup_cross_table_bus_balance() {
        let trace_length = 8usize;
        let challenges = vec![FE::from(123u64), FE::from(456u64)];

        let values: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();

        // --- Table A: sender only ---
        let main_a = vec![values.clone()];
        let aux_zero = vec![FieldElement::<E>::zero(); trace_length];
        let trace_aux_a = vec![aux_zero.clone(), aux_zero.clone()]; // 1 term + 1 acc
        let mut trace_a = TraceTable::from_columns(main_a, trace_aux_a, 1);

        let interactions_a = vec![BusInteraction::sender(
            0u64,
            Multiplicity::One,
            vec![BusValue::column(0)],
        )];

        let proof_options = ProofOptions::default_test_options();

        let air_a = AirWithLogUp::<F, E, NullBoundaryConstraintBuilder, ()>::new(
            trace_length,
            (),
            1,
            interactions_a,
            &proof_options,
            1,
            vec![],
        );

        air_a
            .build_auxiliary_trace(&mut trace_a, &challenges)
            .unwrap();

        let final_acc_a = *trace_a.get_aux(trace_length - 1, 1); // acc column at idx 1
                                                                 // Sender-only table should NOT balance individually
        assert_ne!(final_acc_a, FieldElement::<E>::zero());

        // --- Table B: receiver only (same values) ---
        let main_b = vec![values];
        let trace_aux_b = vec![aux_zero.clone(), aux_zero];
        let mut trace_b = TraceTable::from_columns(main_b, trace_aux_b, 1);

        let interactions_b = vec![BusInteraction::receiver(
            0u64,
            Multiplicity::One,
            vec![BusValue::column(0)],
        )];

        let air_b = AirWithLogUp::<F, E, NullBoundaryConstraintBuilder, ()>::new(
            trace_length,
            (),
            1,
            interactions_b,
            &proof_options,
            1,
            vec![],
        );

        air_b
            .build_auxiliary_trace(&mut trace_b, &challenges)
            .unwrap();

        let final_acc_b = *trace_b.get_aux(trace_length - 1, 1);
        // Receiver-only table should NOT balance individually
        assert_ne!(final_acc_b, FieldElement::<E>::zero());

        // --- Cross-table check: Σ final_accumulated = 0 ---
        assert_eq!(
            final_acc_a + final_acc_b,
            FieldElement::<E>::zero(),
            "Cross-table bus balance failed: sender + receiver should sum to zero"
        );
    }
}
