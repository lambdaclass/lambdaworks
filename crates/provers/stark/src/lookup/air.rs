use std::marker::PhantomData;

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

        // Build accumulated column
        let acc_col_idx = num_interactions;
        build_accumulated_column(acc_col_idx, num_interactions, trace);

        Ok(())
    }

    fn boundary_constraints(
        &self,
        rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::FieldExtension> {
        let mut constraints = vec![];

        // LogUp boundary constraint: accumulated column must equal zero at last row.
        // This enforces bus balance (total senders = total receivers).
        // The accumulated transition constraint has end_exemptions = 0, so the
        // wrap-around at the last row gives: acc[0] = acc[N-1] + Σ terms[0],
        // which pins acc[0] when combined with this boundary constraint.
        if !self.interactions.is_empty() {
            let acc_col_idx = self.interactions.len();
            constraints.push(BoundaryConstraint::new_aux(
                acc_col_idx,
                self.trace_length - 1,
                FieldElement::<Self::FieldExtension>::zero(),
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
}
