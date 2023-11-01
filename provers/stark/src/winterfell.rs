use core::{
    marker::PhantomData,
    mem,
    ops::{DivAssign, MulAssign, Shl, Shr, ShrAssign, SubAssign},
    slice,
};
use lambdaworks_math::{
    field::{
        element::FieldElement as LambdaFieldElement,
        fields::{
            fft_friendly::stark_252_prime_field::Stark252PrimeField,
            montgomery_backed_prime_fields::MontgomeryBackendPrimeField,
        },
        traits::{IsFFTField, IsField},
    },
    traits::ByteConversion,
    unsigned_integer::element::U256,
};
use winterfell::{
    crypto::{DefaultRandomCoin, ElementHasher},
    math::ExtensibleField,
    matrix::ColMatrix,
    Air, AirContext, Assertion, AuxTraceRandElements, ConstraintCompositionCoefficients,
    DefaultConstraintEvaluator, DefaultTraceLde, EvaluationFrame, ProofOptions, Prover,
    StarkDomain, Trace, TraceInfo, TracePolyTable, TraceTable, TransitionConstraintDegree, FieldExtension,
};
use winterfell::{
    math::{ExtensionOf, FieldElement as IsWinterFieldElement, StarkField},
    Deserializable, Serializable,
};

use crate::{traits::AIR, constraints::boundary::{BoundaryConstraint, BoundaryConstraints}, examples::fibonacci_2_cols_shifted::PublicInputs};

// WINTERFELL FIBONACCI AIR
// ================================================================================================

const TRACE_WIDTH: usize = 2;

pub fn are_equal<E: IsWinterFieldElement>(a: E, b: E) -> E {
    a - b
}

#[derive(Clone)]
pub struct FibAir {
    context: AirContext<LambdaFieldElement<Stark252PrimeField>>,
    result: LambdaFieldElement<Stark252PrimeField>,
}

impl Air for FibAir {
    type BaseField = LambdaFieldElement<Stark252PrimeField>;
    type PublicInputs = LambdaFieldElement<Stark252PrimeField>;

    // CONSTRUCTOR
    // --------------------------------------------------------------------------------------------
    fn new(trace_info: TraceInfo, pub_inputs: Self::BaseField, options: ProofOptions) -> Self {
        let degrees = vec![
            TransitionConstraintDegree::new(1),
            TransitionConstraintDegree::new(1),
        ];
        assert_eq!(TRACE_WIDTH, trace_info.width());
        FibAir {
            context: AirContext::new(trace_info, degrees, 3, options),
            result: pub_inputs,
        }
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }

    fn evaluate_transition<E: IsWinterFieldElement + From<Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();
        // expected state width is 2 field elements
        debug_assert_eq!(TRACE_WIDTH, current.len());
        debug_assert_eq!(TRACE_WIDTH, next.len());

        // constraints of Fibonacci sequence (2 terms per step):
        // s_{0, i+1} = s_{0, i} + s_{1, i}
        // s_{1, i+1} = s_{1, i} + s_{0, i+1}
        result[0] = are_equal(next[0], current[0] + current[1]);
        result[1] = are_equal(next[1], current[1] + next[0]);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        // a valid Fibonacci sequence should start with two ones and terminate with
        // the expected result
        let last_step = self.trace_length() - 1;
        vec![
            Assertion::single(0, 0, Self::BaseField::ONE),
            Assertion::single(1, 0, Self::BaseField::ONE),
            Assertion::single(1, last_step, self.result),
        ]
    }
}

// -------------------------
// Lambdaworks AIR

#[derive(Clone)]
pub struct Adapter<A>
where
    A: Air<BaseField=LambdaFieldElement<Stark252PrimeField>>,
    A::PublicInputs: Clone
{
    winterfell_air: A,
    public_inputs: A::PublicInputs,
    air_context: crate::context::AirContext,
    composition_poly_degree_bound: usize
}

impl<A> AIR for Adapter<A>
where
    A: Air<BaseField=LambdaFieldElement<Stark252PrimeField>> + Clone,
    A::PublicInputs: Clone
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
            num_transition_exemptions: 2
        };

        let winter_proof_options = ProofOptions::new(
            lambda_proof_options.fri_number_of_queries as usize,
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
            composition_poly_degree_bound: 16
        }
    }

    fn build_auxiliary_trace(
        &self,
        main_trace: &crate::trace::TraceTable<Self::Field>,
        rap_challenges: &Self::RAPChallenges,
    ) -> crate::trace::TraceTable<Self::Field> {
        // Not supported
        crate::trace::TraceTable::empty()
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut impl crate::transcript::IsStarkTranscript<Self::Field>,
    ) -> Self::RAPChallenges {
        // Not supported
        ()
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
        rap_challenges: &Self::RAPChallenges,
    ) -> Vec<LambdaFieldElement<Self::Field>> {
        let frame = EvaluationFrame::from_rows(
            frame.get_row(0).clone().to_vec(),
            frame.get_row(1).clone().to_vec(),
        );
        let mut result = vec![LambdaFieldElement::zero(); self.num_transition_constraints()];
        self.winterfell_air.evaluate_transition::<LambdaFieldElement<Stark252PrimeField>>(&frame, &[], & mut result); // Periodic values not supported
        result
    }

    fn boundary_constraints(
        &self,
        rap_challenges: &Self::RAPChallenges,
    ) -> crate::constraints::boundary::BoundaryConstraints<Self::Field> {
        let mut result = Vec::new();
        for assertion in self.winterfell_air.get_assertions() {
            assert!(assertion.is_single());
            result.push(BoundaryConstraint::new(assertion.column(), assertion.first_step(), assertion.values()[0]));
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
    use crate::{proof::options::ProofOptions, prover::{Prover, IsStarkProver}, transcript::StoneProverTranscript, verifier::{Verifier, IsStarkVerifier}};
    use crate::trace::TraceTable as LambdaTraceTable;

    pub fn build_proof_options(use_extension_field: bool) -> winterfell::ProofOptions {
        use winterfell::{FieldExtension, ProofOptions};
    
        let extension = if use_extension_field {
            FieldExtension::Quadratic
        } else {
            FieldExtension::None
        };
        ProofOptions::new(28, 8, 0, extension, 4, 7)
    }

    pub fn build_trace(
        sequence_length: usize,
    ) -> TraceTable<LambdaFieldElement<Stark252PrimeField>> {
        assert!(
            sequence_length.is_power_of_two(),
            "sequence length must be a power of 2"
        );

        let mut trace = TraceTable::new(TRACE_WIDTH, sequence_length / 2);
        trace.fill(
            |state| {
                state[0] = LambdaFieldElement::one();
                state[1] = LambdaFieldElement::one();
            },
            |_, state| {
                state[0] += state[1];
                state[1] += state[0];
            },
        );
        
        trace
    }

    #[test]
    fn test_1() {
        //let trace = simple_fibonacci::fibonacci_trace([Felt252::from(1u64), Felt252::from(1u64)], 8);

        //let proof_options = build_proof_options(false);
        let lambda_proof_options = ProofOptions::default_test_options();
        //let fib_prover = FibProver::new(proof_options);
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
