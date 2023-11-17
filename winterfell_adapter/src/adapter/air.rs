use crate::utils::{matrix_winter2lambda, vec_lambda2winter, vec_winter2lambda};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField};
use lambdaworks_math::traits::ByteConversion;
use miden_core::Felt;
use stark_platinum_prover::{
    constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
    traits::AIR,
};
use std::marker::PhantomData;
use winter_air::{Air, AuxTraceRandElements, EvaluationFrame, FieldExtension, ProofOptions};
use winter_math::{FieldElement as IsWinterfellFieldElement, StarkField};
use winter_prover::{ColMatrix, Trace, TraceTable};

use super::public_inputs::AirAdapterPublicInputs;

pub trait FromColumns<A> {
    fn from_cols(columns: Vec<Vec<A>>) -> Self;
}

impl FromColumns<Felt> for TraceTable<Felt> {
    fn from_cols(columns: Vec<Vec<Felt>>) -> Self {
        TraceTable::init(columns)
    }
}

#[derive(Clone)]
pub struct AirAdapter<A, T, FE>
where
    FE: IsWinterfellFieldElement + StarkField + ByteConversion + Unpin + IsFFTField,
    A: Air<BaseField = FE>,
    A::PublicInputs: Clone,
    T: Trace<BaseField = FE> + Clone, //+ FromColumns<FE>,
{
    winterfell_air: A,
    public_inputs: AirAdapterPublicInputs<A, T, FE>,
    air_context: stark_platinum_prover::context::AirContext,
    phantom: PhantomData<T>,
}

impl<A, T, FE> AirAdapter<A, T, FE>
where
    FE: IsWinterfellFieldElement
        + StarkField
        + ByteConversion
        + Unpin
        + IsFFTField
        + IsField<BaseType = FE>,
    A: Air<BaseField = FE> + Clone,
    A::PublicInputs: Clone,
    T: Trace<BaseField = FE> + Clone, //+ FromColumns<FE>,
{
    pub fn convert_winterfell_trace_table(
        trace: ColMatrix<FE>,
    ) -> stark_platinum_prover::trace::TraceTable<FE> {
        let mut columns = Vec::new();
        for i in 0..trace.num_cols() {
            columns.push(trace.get_column(i).to_owned());
        }

        stark_platinum_prover::trace::TraceTable::from_columns(matrix_winter2lambda(&columns), 1)
    }
}

impl<A, T, FE> AIR for AirAdapter<A, T, FE>
where
    FE: IsWinterfellFieldElement
        + StarkField
        + ByteConversion
        + Unpin
        + IsFFTField
        + IsField<BaseType = FE>,
    A: Air<BaseField = FE> + Clone,
    A::PublicInputs: Clone,
    T: Trace<BaseField = FE> + Clone, //+ FromColumns<FE>,
{
    type Field = FE;
    type RAPChallenges = Vec<FE>;
    type PublicInputs = AirAdapterPublicInputs<A, T, FE>;
    const STEP_SIZE: usize = 1;

    fn new(
        _trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        lambda_proof_options: &stark_platinum_prover::proof::options::ProofOptions,
    ) -> Self {
        let winter_proof_options = ProofOptions::new(
            lambda_proof_options.fri_number_of_queries,
            lambda_proof_options.blowup_factor as usize,
            lambda_proof_options.grinding_factor as u32,
            FieldExtension::None,
            2,
            0,
        );

        let winterfell_air = A::new(
            pub_inputs.trace_info.clone(),
            pub_inputs.winterfell_public_inputs.clone(),
            winter_proof_options,
        );
        let winterfell_context = winterfell_air.context();

        let lambda_context = stark_platinum_prover::context::AirContext {
            proof_options: lambda_proof_options.clone(),
            transition_degrees: pub_inputs.transition_degrees.to_owned(),
            transition_exemptions: pub_inputs.transition_exemptions.to_owned(),
            transition_offsets: pub_inputs.transition_offsets.to_owned(),
            num_transition_constraints: winterfell_context.num_transition_constraints(),
            trace_columns: pub_inputs.trace_info.width(),
            num_transition_exemptions: pub_inputs.num_transition_exemptions,
        };

        Self {
            winterfell_air,
            public_inputs: pub_inputs.clone(),
            air_context: lambda_context,
            phantom: PhantomData,
        }
    }

    fn build_auxiliary_trace(
        &self,
        _main_trace: &stark_platinum_prover::trace::TraceTable<Self::Field>,
        rap_challenges: &Self::RAPChallenges,
    ) -> stark_platinum_prover::trace::TraceTable<Self::Field> {
        // We support at most a one-stage RAP. This covers most use cases.
        //if let Some(winter_trace) = T::from_cols(matrix_lambda2winter(&main_trace.columns()))
        //    .build_aux_segment(&[], rap_challenges)
        if let Some(winter_trace) = self
            .pub_inputs()
            .clone()
            .trace
            .build_aux_segment(&[], rap_challenges)
        {
            let mut columns = Vec::new();
            for i in 0..winter_trace.num_cols() {
                columns.push(winter_trace.get_column(i).to_owned());
            }
            stark_platinum_prover::trace::TraceTable::from_columns(
                matrix_winter2lambda(&columns),
                1,
            )
        } else {
            stark_platinum_prover::trace::TraceTable::<FE>::empty()
        }
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut impl stark_platinum_prover::transcript::IsStarkTranscript<Self::Field>,
    ) -> Self::RAPChallenges {
        let trace_layout = self.winterfell_air.trace_layout();
        let num_segments = trace_layout.num_aux_segments();

        if num_segments == 1 {
            let mut result = Vec::new();
            for _ in 0..trace_layout.get_aux_segment_rand_elements(0) {
                result.push(transcript.sample_field_element());
            }
            vec_lambda2winter(&result)
        } else if num_segments == 0 {
            Vec::new()
        } else {
            panic!("The winterfell adapter does not support AIR's with more than one auxiliary segment");
        }
    }

    fn number_auxiliary_rap_columns(&self) -> usize {
        self.winterfell_air.trace_layout().aux_trace_width()
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.winterfell_air
            .context()
            .num_constraint_composition_columns()
            * self.trace_length()
    }

    fn compute_transition(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Self::Field>,
        periodic_values: &[FieldElement<Self::Field>],
        rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>> {
        let num_aux_columns = self.number_auxiliary_rap_columns();
        let num_main_columns = self.context().trace_columns - num_aux_columns;

        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);

        let main_frame = EvaluationFrame::from_rows(
            vec_lambda2winter(&first_step.get_row(0)[..num_main_columns]),
            vec_lambda2winter(&second_step.get_row(0)[..num_main_columns]),
        );

        let periodic_values = vec_lambda2winter(periodic_values);

        let mut main_result = vec![
            FieldElement::zero();
            self.winterfell_air
                .context()
                .num_main_transition_constraints()
        ];

        let mut main_result_winter = vec_lambda2winter(&main_result);
        self.winterfell_air.evaluate_transition::<FE>(
            &main_frame,
            &periodic_values,
            &mut main_result_winter,
        ); // Periodic values not supported

        main_result = vec_winter2lambda(&main_result_winter);

        if self.winterfell_air.trace_layout().num_aux_segments() == 1 {
            let mut rand_elements = AuxTraceRandElements::new();
            rand_elements.add_segment_elements(rap_challenges.clone());

            let first_step = frame.get_evaluation_step(0);
            let second_step = frame.get_evaluation_step(1);

            let aux_frame = EvaluationFrame::from_rows(
                vec_lambda2winter(&first_step.get_row(0)[num_main_columns..]),
                vec_lambda2winter(&second_step.get_row(0)[num_main_columns..]),
            );

            let mut aux_result = vec![
                FieldElement::zero();
                self.winterfell_air
                    .context()
                    .num_aux_transition_constraints()
            ];
            let mut winter_aux_result = vec_lambda2winter(&aux_result);
            self.winterfell_air.evaluate_aux_transition(
                &main_frame,
                &aux_frame,
                &periodic_values,
                &rand_elements,
                &mut winter_aux_result,
            );
            aux_result = vec_winter2lambda(&winter_aux_result);
            main_result.extend_from_slice(&aux_result);
        }
        main_result
    }

    fn boundary_constraints(
        &self,
        rap_challenges: &Self::RAPChallenges,
    ) -> stark_platinum_prover::constraints::boundary::BoundaryConstraints<FE> {
        let num_aux_columns = self.number_auxiliary_rap_columns();
        let num_main_columns = self.context().trace_columns - num_aux_columns;

        let mut result = Vec::new();
        for assertion in self.winterfell_air.get_assertions() {
            assert!(assertion.is_single());
            result.push(BoundaryConstraint::new(
                assertion.column(),
                assertion.first_step(),
                FieldElement::<FE>::const_from_raw(assertion.values()[0]),
            ));
        }

        let mut rand_elements = AuxTraceRandElements::new();
        rand_elements.add_segment_elements(rap_challenges.clone());

        for assertion in self.winterfell_air.get_aux_assertions(&rand_elements) {
            assert!(assertion.is_single());
            result.push(BoundaryConstraint::new(
                assertion.column() + num_main_columns,
                assertion.first_step(),
                FieldElement::<FE>::const_from_raw(assertion.values()[0]),
            ));
        }

        BoundaryConstraints::from_constraints(result)
    }

    fn context(&self) -> &stark_platinum_prover::context::AirContext {
        &self.air_context
    }

    fn trace_length(&self) -> usize {
        self.winterfell_air.context().trace_len()
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.public_inputs
    }

    fn get_periodic_column_values(&self) -> Vec<Vec<FieldElement<Self::Field>>> {
        matrix_winter2lambda(&self.winterfell_air.get_periodic_column_values())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::cubic::{self, Cubic};
    use crate::examples::fibonacci_2_terms::{self, FibAir2Terms};
    use crate::examples::fibonacci_rap::{self, FibonacciRAP, RapTraceTable};
    use crate::utils::matrix_lambda2winter;
    use miden_air::{ProcessorAir, ProvingOptions, PublicInputs};
    use miden_assembly::Assembler;
    use miden_core::{Felt, StackInputs};
    use miden_processor::DefaultHost;
    use miden_processor::{self as processor};
    use processor::ExecutionTrace;
    use stark_platinum_prover::prover::MidenProver;
    use stark_platinum_prover::transcript::MidenProverTranscript;
    use stark_platinum_prover::verifier::MidenVerifier;
    use stark_platinum_prover::{
        proof::options::ProofOptions, prover::IsStarkProver, verifier::IsStarkVerifier,
    };
    use winter_air::{TraceInfo, TraceLayout};

    #[test]
    fn prove_and_verify_miden_readme_example() {
        let mut lambda_proof_options = ProofOptions::default_test_options();
        lambda_proof_options.blowup_factor = 32;
        let assembler = Assembler::default();

        let program = assembler.compile("begin push.3 push.5 add end").unwrap();

        let winter_trace = processor::execute(
            &program,
            StackInputs::default(),
            DefaultHost::default(),
            *ProvingOptions::default().execution_options(),
        )
        .unwrap();
        let program_info = winter_trace.program_info().clone();
        let stack_outputs = winter_trace.stack_outputs().clone();

        let pub_inputs = PublicInputs::new(program_info, StackInputs::default(), stack_outputs);

        let pub_inputs = AirAdapterPublicInputs {
            winterfell_public_inputs: pub_inputs,
            transition_degrees: vec![0; 182], // Not used, but still has to have 182 things because of zip's.
            transition_exemptions: vec![2; 182],
            transition_offsets: vec![0, 1],
            trace: winter_trace.clone(),
            trace_info: winter_trace.get_info(),
            num_transition_exemptions: 1,
        };

        let trace =
            AirAdapter::<FibAir2Terms, ExecutionTrace, Felt>::convert_winterfell_trace_table(
                winter_trace.main_segment().clone(),
            );

        let proof = MidenProver::prove::<AirAdapter<ProcessorAir, ExecutionTrace, Felt>>(
            &trace,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        )
        .unwrap();

        assert!(MidenVerifier::verify::<
            AirAdapter<ProcessorAir, ExecutionTrace, Felt>,
        >(
            &proof,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        ));
    }

    fn compute_fibonacci(n: usize) -> Felt {
        let mut t0 = Felt::ZERO;
        let mut t1 = Felt::ONE;

        for _ in 0..n {
            t1 = t0 + t1;
            core::mem::swap(&mut t0, &mut t1);
        }
        t0
    }

    #[test]
    fn prove_and_verify_miden_fibonacci() {
        let fibonacci_number = 16;
        let program = format!(
            "begin
                repeat.{}
                    swap dup.1 add
                end
            end",
            fibonacci_number - 1
        );
        let program = Assembler::default().compile(&program).unwrap();
        let expected_result = vec![compute_fibonacci(fibonacci_number).as_int()];
        let stack_inputs = StackInputs::try_from_values([0, 1]).unwrap();

        let mut lambda_proof_options = ProofOptions::default_test_options();
        lambda_proof_options.blowup_factor = 8;

        let winter_trace = processor::execute(
            &program,
            stack_inputs.clone(),
            DefaultHost::default(),
            *ProvingOptions::default().execution_options(),
        )
        .unwrap();
        let program_info = winter_trace.program_info().clone();
        let stack_outputs = winter_trace.stack_outputs().clone();

        let pub_inputs = PublicInputs::new(program_info, stack_inputs, stack_outputs.clone());

        assert_eq!(
            expected_result,
            stack_outputs.clone().stack_truncated(1),
            "Program result was computed incorrectly"
        );

        let pub_inputs = AirAdapterPublicInputs {
            winterfell_public_inputs: pub_inputs,
            transition_degrees: vec![0; 182], // Not used, but still has to have 182 things because of zip's.
            transition_exemptions: vec![2; 182],
            transition_offsets: vec![0, 1],
            trace: winter_trace.clone(),
            trace_info: winter_trace.get_info(),
            num_transition_exemptions: 1,
        };

        let trace =
            AirAdapter::<ProcessorAir, ExecutionTrace, Felt>::convert_winterfell_trace_table(
                winter_trace.main_segment().clone(),
            );

        let proof = MidenProver::prove::<AirAdapter<ProcessorAir, ExecutionTrace, Felt>>(
            &trace,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        )
        .unwrap();

        assert!(MidenVerifier::verify::<
            AirAdapter<ProcessorAir, ExecutionTrace, Felt>,
        >(
            &proof,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        ));
    }

    #[test]
    fn prove_and_verify_a_winterfell_fibonacci_2_terms_air() {
        let lambda_proof_options = ProofOptions::default_test_options();
        let winter_trace = fibonacci_2_terms::build_trace(16);
        let trace = AirAdapter::<FibAir2Terms, TraceTable<_>, Felt>::convert_winterfell_trace_table(
            winter_trace.main_segment().clone(),
        );
        let pub_inputs = AirAdapterPublicInputs {
            winterfell_public_inputs: trace.columns()[1][7].value().clone(),
            transition_degrees: vec![1, 1],
            transition_exemptions: vec![1, 1],
            transition_offsets: vec![0, 1],
            trace: winter_trace,
            trace_info: TraceInfo::new(2, 8),
            num_transition_exemptions: 1,
        };

        let proof = MidenProver::prove::<AirAdapter<FibAir2Terms, TraceTable<_>, Felt>>(
            &trace,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        )
        .unwrap();

        assert!(MidenVerifier::verify::<
            AirAdapter<FibAir2Terms, TraceTable<_>, Felt>,
        >(
            &proof,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        ));
    }

    #[test]
    fn prove_and_verify_a_winterfell_fibonacci_rap_air() {
        let lambda_proof_options = ProofOptions::default_test_options();
        let winter_trace = fibonacci_rap::build_trace(16);
        let trace =
            AirAdapter::<FibonacciRAP, RapTraceTable<_>, Felt>::convert_winterfell_trace_table(
                winter_trace.main_segment().clone(),
            );
        let trace_layout = TraceLayout::new(3, [1], [1]);
        let trace_info = TraceInfo::new_multi_segment(trace_layout, 16, vec![]);
        let fibonacci_result = trace.columns()[1][15];
        let pub_inputs = AirAdapterPublicInputs::<FibonacciRAP, RapTraceTable<_>, Felt> {
            winterfell_public_inputs: fibonacci_result.value().clone(),
            transition_degrees: vec![1, 1, 2],
            transition_exemptions: vec![1, 1, 1],
            transition_offsets: vec![0, 1],
            trace: RapTraceTable::from_cols(matrix_lambda2winter(&trace.columns())),
            trace_info,
            num_transition_exemptions: 1,
        };

        let proof = MidenProver::prove::<AirAdapter<FibonacciRAP, RapTraceTable<_>, Felt>>(
            &trace,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        )
        .unwrap();
        assert!(MidenVerifier::verify::<
            AirAdapter<FibonacciRAP, RapTraceTable<_>, Felt>,
        >(
            &proof,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        ));
    }

    #[test]
    fn prove_and_verify_a_winterfell_cubic_air() {
        let lambda_proof_options = ProofOptions::default_test_options();
        let winter_trace = cubic::build_trace(16);
        let trace = AirAdapter::<Cubic, TraceTable<_>, Felt>::convert_winterfell_trace_table(
            winter_trace.main_segment().clone(),
        );
        let pub_inputs = AirAdapterPublicInputs {
            winterfell_public_inputs: trace.columns()[0][15].value().clone(),
            transition_degrees: vec![3],
            transition_exemptions: vec![1],
            transition_offsets: vec![0, 1],
            trace: winter_trace,
            trace_info: TraceInfo::new(1, 16),
            num_transition_exemptions: 1,
        };

        let proof = MidenProver::prove::<AirAdapter<Cubic, TraceTable<_>, Felt>>(
            &trace,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        )
        .unwrap();
        assert!(MidenVerifier::verify::<
            AirAdapter<Cubic, TraceTable<_>, Felt>,
        >(
            &proof,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        ));
    }
}
