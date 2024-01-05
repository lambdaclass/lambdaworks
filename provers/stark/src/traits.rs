use itertools::Itertools;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField},
    },
    polynomial::Polynomial,
};

use crate::{
    constraints::transition::{TransitionConstraint, TransitionZerofiersIter},
    domain::Domain,
    transcript::IsStarkTranscript,
};

use super::{
    constraints::boundary::BoundaryConstraints, context::AirContext, frame::Frame,
    proof::options::ProofOptions, trace::TraceTable,
};

/// AIR is a representation of the Constraints
pub trait AIR {
    type Field: IsFFTField;
    type PublicInputs;

    const STEP_SIZE: usize;

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self;

    fn build_auxiliary_trace(
        &self,
        _main_trace: &TraceTable<Self::Field>,
        _rap_challenges: &[FieldElement<Self::Field>],
    ) -> TraceTable<Self::Field> {
        TraceTable::empty()
    }

    fn build_rap_challenges(
        &self,
        _transcript: &mut impl IsStarkTranscript<Self::Field>,
    ) -> Vec<FieldElement<Self::Field>> {
        Vec::new()
    }

    fn number_auxiliary_rap_columns(&self) -> usize {
        0
    }

    fn composition_poly_degree_bound(&self) -> usize;

    fn compute_transition(
        &self,
        frame: &Frame<Self::Field>,
        periodic_values: &[FieldElement<Self::Field>],
        rap_challenges: &[FieldElement<Self::Field>],
    ) -> Vec<FieldElement<Self::Field>> {
        let mut evaluations =
            vec![FieldElement::<Self::Field>::zero(); self.num_transition_constraints()];
        self.transition_constraints()
            .iter()
            .for_each(|c| c.evaluate(frame, &mut evaluations, periodic_values, rap_challenges));

        evaluations
    }

    fn boundary_constraints(
        &self,
        rap_challenges: &[FieldElement<Self::Field>],
    ) -> BoundaryConstraints<Self::Field>;

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
    // fn transition_constraints<T: TransitionConstraint<Self::Field>>(&self) -> Vec<Box<dyn T>>;
    fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraint<Self::Field>>>;

    fn transition_zerofier_evaluations(
        &self,
        domain: &Domain<Self::Field>,
    ) -> TransitionZerofiersIter<Self::Field> {
        let evals: Vec<_> = self
            .transition_constraints()
            .iter()
            .map(|c| c.zerofier_evaluations_on_extended_domain(domain))
            .collect();

        TransitionZerofiersIter::new(evals)
    }
}
