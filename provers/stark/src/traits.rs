use itertools::Itertools;
use lambdaworks_math::{
    fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField},
    },
    polynomial::univariate::UnivariatePolynomial,
};

use crate::transcript::IsStarkTranscript;

use super::{
    constraints::boundary::BoundaryConstraints, context::AirContext, frame::Frame,
    proof::options::ProofOptions, trace::TraceTable,
};

/// AIR is a representation of the Constraints
pub trait AIR: Clone {
    type Field: IsFFTField;
    type RAPChallenges;
    type PublicInputs;

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
        rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>>;

    fn boundary_constraints(
        &self,
        rap_challenges: &Self::RAPChallenges,
    ) -> BoundaryConstraints<Self::Field>
    where
        <<Self as AIR>::Field as IsField>::BaseType: Send + Sync;

    fn transition_exemptions(&self) -> Vec<UnivariatePolynomial<Self::Field>>
    where
        <<Self as AIR>::Field as IsField>::BaseType: Send + Sync,
    {
        let trace_length = self.trace_length();
        let roots_of_unity_order = trace_length.trailing_zeros();
        let roots_of_unity = get_powers_of_primitive_root_coset(
            roots_of_unity_order as u64,
            self.trace_length(),
            &FieldElement::<Self::Field>::one(),
        )
        .unwrap();
        let root_of_unity_len = roots_of_unity.len();

        let x = UnivariatePolynomial::new_monomial(FieldElement::one(), 1);

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
                        UnivariatePolynomial::new_monomial(FieldElement::one(), 0),
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

    fn num_transition_constraints(&self) -> usize {
        self.context().num_transition_constraints
    }

    fn pub_inputs(&self) -> &Self::PublicInputs;

    fn transition_exemptions_verifier(
        &self,
        root: &FieldElement<Self::Field>,
    ) -> Vec<UnivariatePolynomial<Self::Field>>
    where
        <<Self as AIR>::Field as IsField>::BaseType: Send + Sync,
    {
        let x = UnivariatePolynomial::new_monomial(FieldElement::one(), 1);

        let max = self
            .context()
            .transition_exemptions
            .iter()
            .max()
            .expect("has maximum");
        (1..=*max)
            .map(|index| {
                (1..=index).fold(
                    UnivariatePolynomial::new_monomial(FieldElement::one(), 0),
                    |acc, k| acc * (&x - root.pow(k)),
                )
            })
            .collect()
    }
}
