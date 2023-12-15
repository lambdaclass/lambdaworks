use itertools::Itertools;
use lambdaworks_math::{
    fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset,
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

use crate::{
    constraints::transition::{TransitionConstraint, TransitionZerofiersIter},
    transcript::IsStarkTranscript,
};

use super::{
    constraints::boundary::BoundaryConstraints, context::AirContext, frame::Frame,
    proof::options::ProofOptions, trace::TraceTable,
};

/// AIR is a representation of the Constraints
pub trait AIR {
    type Field: IsFFTField;
    type RAPChallenges;
    type PublicInputs;
    type TransitionConstraints: TransitionConstraint<Self::Field>;

    const STEP_SIZE: usize;

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self;

    fn build_auxiliary_trace(
        &self,
        main_trace: &TraceTable<Self::Field>,
        rap_challenges: &Self::RAPChallenges,
    ) -> TraceTable<Self::Field>;

    fn build_rap_challenges(
        &self,
        transcript: &mut impl IsStarkTranscript<Self::Field>,
    ) -> Self::RAPChallenges;

    fn number_auxiliary_rap_columns(&self) -> usize;

    fn composition_poly_degree_bound(&self) -> usize;

    fn compute_transition(
        &self,
        frame: &Frame<Self::Field>,
        periodic_values: &[FieldElement<Self::Field>],
        rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>>;

    fn boundary_constraints(
        &self,
        rap_challenges: &Self::RAPChallenges,
    ) -> BoundaryConstraints<Self::Field>;

    fn transition_exemptions(&self) -> Vec<Polynomial<FieldElement<Self::Field>>> {
        let trace_length = self.trace_length();
        let roots_of_unity_order = trace_length.trailing_zeros();
        let roots_of_unity = get_powers_of_primitive_root_coset(
            roots_of_unity_order as u64,
            self.trace_length(),
            &FieldElement::<Self::Field>::one(),
        )
        .unwrap();
        let root_of_unity_len = roots_of_unity.len();

        let x = Polynomial::new_monomial(FieldElement::one(), 1);

        self.context()
            .transition_exemptions
            .iter()
            .unique_by(|elem| *elem)
            .filter(|v| *v > &0_usize)
            .map(|cant_take| {
                roots_of_unity
                    .iter()
                    .take(root_of_unity_len)
                    .rev()
                    .take(*cant_take)
                    .fold(
                        Polynomial::new_monomial(FieldElement::one(), 0),
                        |acc, root| acc * (&x - root),
                    )
            })
            .collect()
    }

    fn context(&self) -> &AirContext;

    fn trace_length(&self) -> usize;

    fn options(&self) -> &ProofOptions {
        &self.context().proof_options
    }

    fn blowup_factor(&self) -> u8 {
        self.options().blowup_factor
    }

    fn coset_offset(&self) -> FieldElement<Self::Field> {
        FieldElement::from(self.options().coset_offset)
    }

    fn trace_primitive_root(&self) -> FieldElement<Self::Field> {
        let trace_length = self.trace_length();
        let root_of_unity_order = u64::from(trace_length.trailing_zeros());

        Self::Field::get_primitive_root_of_unity(root_of_unity_order).unwrap()
    }

    fn num_transition_constraints(&self) -> usize {
        self.context().num_transition_constraints
    }

    fn pub_inputs(&self) -> &Self::PublicInputs;

    fn transition_exemptions_verifier(
        &self,
        root: &FieldElement<Self::Field>,
    ) -> Vec<Polynomial<FieldElement<Self::Field>>> {
        let x = Polynomial::new_monomial(FieldElement::one(), 1);

        let max = self
            .context()
            .transition_exemptions
            .iter()
            .max()
            .expect("has maximum");
        (1..=*max)
            .map(|index| {
                (1..=index).fold(
                    Polynomial::new_monomial(FieldElement::one(), 0),
                    |acc, k| acc * (&x - root.pow(k)),
                )
            })
            .collect()
    }

    fn get_periodic_column_values(&self) -> Vec<Vec<FieldElement<Self::Field>>> {
        vec![]
    }

    fn get_periodic_column_polynomials(&self) -> Vec<Polynomial<FieldElement<Self::Field>>> {
        let mut result = Vec::new();
        for periodic_column in self.get_periodic_column_values() {
            let values: Vec<_> = periodic_column
                .iter()
                .cycle()
                .take(self.trace_length())
                .cloned()
                .collect();
            let poly = Polynomial::interpolate_fft::<Self::Field>(&values).unwrap();
            result.push(poly);
        }
        result
    }

    // NOTE: Remember to index constraints correctly!!!!
    // fn transition_constraints<T: TransitionConstraint<Self::Field>>(&self) -> Vec<T>;
    fn transition_constraints(&self) -> Vec<Self::TransitionConstraints>;

    fn transition_zerofier_evaluations<'a>(&'a self) -> TransitionZerofiersIter<'a, Self::Field> {
        let trace_length = self.trace_length();
        let blowup_factor = usize::from(self.blowup_factor());
        let offset = self.coset_offset();
        let trace_primitive_root = self.trace_primitive_root();

        let evals: Vec<_> = self
            .transition_constraints()
            .iter()
            .map(|c| {
                c.zerofier_evaluations(blowup_factor, &offset, trace_length, &trace_primitive_root)
            })
            .collect();

        TransitionZerofiersIter::new(evals)
    }
}
