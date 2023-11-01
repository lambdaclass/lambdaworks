use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use winterfell::{Air, EvaluationFrame, FieldExtension, ProofOptions, TraceInfo};

use crate::{
    constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
    traits::AIR,
};

#[derive(Clone)]
pub struct Adapter<A>
where
    A: Air<BaseField = FieldElement<Stark252PrimeField>>,
    A::PublicInputs: Clone,
{
    winterfell_air: A,
    public_inputs: A::PublicInputs,
    air_context: crate::context::AirContext,
    composition_poly_degree_bound: usize,
}

impl<A> AIR for Adapter<A>
where
    A: Air<BaseField = FieldElement<Stark252PrimeField>> + Clone,
    A::PublicInputs: Clone,
{
    type Field = Stark252PrimeField;
    type RAPChallenges = (); // RAP Challenges not supported?
    type PublicInputs = A::PublicInputs;

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        lambda_proof_options: &crate::proof::options::ProofOptions,
    ) -> Self {
        let winter_trace_info = TraceInfo::new(2, trace_length);
        let lambda_context = crate::context::AirContext {
            proof_options: lambda_proof_options.clone(),
            transition_degrees: vec![1, 1],
            transition_exemptions: vec![1, 1],
            transition_offsets: vec![0, 1],
            num_transition_constraints: 2,
            trace_columns: 2,
            num_transition_exemptions: 2,
        };

        let winter_proof_options = ProofOptions::new(
            lambda_proof_options.fri_number_of_queries,
            lambda_proof_options.blowup_factor as usize,
            lambda_proof_options.grinding_factor as u32,
            FieldExtension::None,
            2,
            0, // TODO: Check
        );

        Self {
            winterfell_air: A::new(winter_trace_info, pub_inputs.clone(), winter_proof_options),
            public_inputs: pub_inputs.clone(),
            air_context: lambda_context,
            composition_poly_degree_bound: 16,
        }
    }

    fn build_auxiliary_trace(
        &self,
        _main_trace: &crate::trace::TraceTable<Self::Field>,
        _rap_challenges: &Self::RAPChallenges,
    ) -> crate::trace::TraceTable<Self::Field> {
        // Not supported
        crate::trace::TraceTable::empty()
    }

    fn build_rap_challenges(
        &self,
        _transcript: &mut impl crate::transcript::IsStarkTranscript<Self::Field>,
    ) -> Self::RAPChallenges {
        // Not supported
        
    }

    fn number_auxiliary_rap_columns(&self) -> usize {
        // Not supported
        0
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.composition_poly_degree_bound
    }

    fn compute_transition(
        &self,
        frame: &crate::frame::Frame<Self::Field>,
        _rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>> {
        let frame = EvaluationFrame::from_rows(
            frame.get_row(0).to_vec(),
            frame.get_row(1).to_vec(),
        );
        let mut result = vec![FieldElement::zero(); self.num_transition_constraints()];
        self.winterfell_air
            .evaluate_transition::<FieldElement<Stark252PrimeField>>(
                &frame,
                &[],
                &mut result,
            ); // Periodic values not supported
        result
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &Self::RAPChallenges,
    ) -> crate::constraints::boundary::BoundaryConstraints<Self::Field> {
        let mut result = Vec::new();
        for assertion in self.winterfell_air.get_assertions() {
            assert!(assertion.is_single());
            result.push(BoundaryConstraint::new(
                assertion.column(),
                assertion.first_step(),
                assertion.values()[0],
            ));
        }
        BoundaryConstraints::from_constraints(result)
    }

    fn context(&self) -> &crate::context::AirContext {
        &self.air_context
    }

    fn trace_length(&self) -> usize {
        self.winterfell_air.context().trace_len()
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.public_inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::TraceTable as LambdaTraceTable;
    use crate::{
        proof::options::ProofOptions,
        prover::{IsStarkProver, Prover},
        transcript::StoneProverTranscript,
        verifier::{IsStarkVerifier, Verifier},
        winterfell_adapter::example_air::{build_trace, FibAir},
    };

    #[test]
    fn test_1() {
        let lambda_proof_options = ProofOptions::default_test_options();
        let trace = build_trace(16);
        let pub_inputs = trace.get_column(1)[7];

        let mut columns = Vec::new();
        for i in 0..trace.width() {
            columns.push(trace.get_column(i).to_owned());
        }

        let lambda_trace = LambdaTraceTable::from_columns(&columns);

        let proof = Prover::prove::<Adapter<FibAir>>(
            &lambda_trace,
            &pub_inputs,
            &lambda_proof_options,
            StoneProverTranscript::new(&[]),
        )
        .unwrap();
        assert!(Verifier::verify::<Adapter<FibAir>>(
            &proof,
            &pub_inputs,
            &lambda_proof_options,
            StoneProverTranscript::new(&[]),
        ));
    }
}
