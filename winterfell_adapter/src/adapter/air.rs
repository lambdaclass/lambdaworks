use crate::utils::{
    matrix_lambda2winter, matrix_winter2lambda, vec_lambda2winter, vec_winter2lambda,
};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
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

pub trait FromColumns<A, M> {
    fn from_cols(columns: Vec<Vec<A>>, metadata: &M) -> Self;
}

impl FromColumns<Felt, ()> for TraceTable<Felt> {
    fn from_cols(columns: Vec<Vec<Felt>>, _: &()) -> Self {
        TraceTable::init(columns)
    }
}

#[derive(Clone)]
pub struct AirAdapter<A, T, FE, E, M>
where
    FE: IsWinterfellFieldElement
        + StarkField
        + ByteConversion
        + Unpin
        + IsFFTField
        + IsSubFieldOf<E>,
    E: IsField,
    A: Air<BaseField = FE>,
    A::PublicInputs: Clone,
    T: Trace<BaseField = FE> + Clone + FromColumns<FE, M>,
    M: Clone,
{
    winterfell_air: A,
    public_inputs: AirAdapterPublicInputs<A, M>,
    air_context: stark_platinum_prover::context::AirContext,
    trace: PhantomData<T>,
    extension: PhantomData<E>,
}

impl<A, T, FE, E, M> AirAdapter<A, T, FE, E, M>
where
    FE: IsWinterfellFieldElement
        + StarkField
        + ByteConversion
        + Unpin
        + IsFFTField
        + IsField<BaseType = FE>
        + IsSubFieldOf<E>,
    E: IsField<BaseType = E> + IsWinterfellFieldElement<BaseField = FE>,
    A: Air<BaseField = FE> + Clone,
    A::PublicInputs: Clone,
    T: Trace<BaseField = FE> + Clone + FromColumns<FE, M>,
    M: Clone,
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

impl<A, T, FE, E, M> AIR for AirAdapter<A, T, FE, E, M>
where
    FE: IsWinterfellFieldElement
        + StarkField
        + ByteConversion
        + Unpin
        + IsFFTField
        + IsField<BaseType = FE>
        + IsSubFieldOf<E>,
    E: IsField<BaseType = E> + IsWinterfellFieldElement<BaseField = FE>,
    A: Air<BaseField = FE> + Clone,
    A::PublicInputs: Clone,
    T: Trace<BaseField = FE> + Clone + FromColumns<FE, M>,
    M: Clone,
{
    type Field = FE;
    type FieldExtension = E;
    type RAPChallenges = Vec<E>;
    type PublicInputs = AirAdapterPublicInputs<A, M>;
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
            transition_exemptions: pub_inputs.transition_exemptions.to_owned(),
            transition_offsets: pub_inputs.transition_offsets.to_owned(),
            num_transition_constraints: winterfell_context.num_transition_constraints(),
            trace_columns: pub_inputs.trace_info.width(),
        };

        Self {
            winterfell_air,
            public_inputs: pub_inputs.clone(),
            air_context: lambda_context,
            trace: PhantomData,
            extension: PhantomData,
        }
    }

    fn build_auxiliary_trace(
        &self,
        main_trace: &stark_platinum_prover::trace::TraceTable<Self::Field>,
        rap_challenges: &Self::RAPChallenges,
    ) -> stark_platinum_prover::trace::TraceTable<Self::FieldExtension> {
        // We support at most a one-stage RAP. This covers most use cases.
        if let Some(winter_trace) = T::from_cols(
            matrix_lambda2winter(&main_trace.columns()),
            &self.pub_inputs().metadata,
        )
        .build_aux_segment(&[], rap_challenges)
        {
            let mut columns = Vec::new();
            for i in 0..winter_trace.num_cols() {
                columns.push(winter_trace.get_column(i).to_owned());
            }
            stark_platinum_prover::trace::TraceTable::<E>::from_columns(
                matrix_winter2lambda(&columns),
                1,
            )
        } else {
            stark_platinum_prover::trace::TraceTable::<E>::empty()
        }
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut impl stark_platinum_prover::transcript::IsStarkTranscript<Self::FieldExtension>,
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

    fn compute_transition_prover(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Self::Field, Self::FieldExtension>,
        periodic_values: &[FieldElement<Self::Field>],
        rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);

        let main_frame = EvaluationFrame::from_rows(
            vec_lambda2winter(first_step.get_row_main(0)),
            vec_lambda2winter(second_step.get_row_main(0)),
        );

        let periodic_values = vec_lambda2winter(periodic_values);

        let main_result = vec![
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
        );

        let mut result: Vec<_> = vec_winter2lambda(&main_result_winter)
            .into_iter()
            .map(|element| element.to_extension())
            .collect();

        if self.winterfell_air.trace_layout().num_aux_segments() == 1 {
            let mut rand_elements = AuxTraceRandElements::new();
            rand_elements.add_segment_elements(rap_challenges.clone());

            let first_step = frame.get_evaluation_step(0);
            let second_step = frame.get_evaluation_step(1);

            let aux_frame = EvaluationFrame::from_rows(
                vec_lambda2winter(first_step.get_row_aux(0)),
                vec_lambda2winter(second_step.get_row_aux(0)),
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
            result.extend_from_slice(&aux_result);
        }
        result
    }

    fn boundary_constraints(
        &self,
        rap_challenges: &Self::RAPChallenges,
    ) -> stark_platinum_prover::constraints::boundary::BoundaryConstraints<E> {
        let mut result = Vec::new();
        for assertion in self.winterfell_air.get_assertions() {
            assert!(assertion.is_single());
            result.push(BoundaryConstraint::new_main(
                assertion.column(),
                assertion.first_step(),
                FieldElement::<FE>::const_from_raw(assertion.values()[0]).to_extension(),
            ));
        }

        let mut rand_elements = AuxTraceRandElements::new();
        rand_elements.add_segment_elements(rap_challenges.clone());

        for assertion in self.winterfell_air.get_aux_assertions(&rand_elements) {
            assert!(assertion.is_single());
            result.push(BoundaryConstraint::new_aux(
                assertion.column(),
                assertion.first_step(),
                FieldElement::<E>::const_from_raw(assertion.values()[0]),
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

    fn compute_transition_verifier(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Self::FieldExtension, Self::FieldExtension>,
        periodic_values: &[FieldElement<Self::FieldExtension>],
        rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);

        let main_frame = EvaluationFrame::from_rows(
            vec_lambda2winter(first_step.get_row_main(0)),
            vec_lambda2winter(second_step.get_row_main(0)),
        );

        let periodic_values = vec_lambda2winter(periodic_values);

        let main_result = vec![
            FieldElement::zero();
            self.winterfell_air
                .context()
                .num_main_transition_constraints()
        ];

        let mut main_result_winter = vec_lambda2winter(&main_result);
        self.winterfell_air.evaluate_transition::<E>(
            &main_frame,
            &periodic_values,
            &mut main_result_winter,
        );

        let mut result: Vec<FieldElement<E>> = vec_winter2lambda(&main_result_winter);

        if self.winterfell_air.trace_layout().num_aux_segments() == 1 {
            let mut rand_elements = AuxTraceRandElements::new();
            rand_elements.add_segment_elements(rap_challenges.clone());

            let first_step = frame.get_evaluation_step(0);
            let second_step = frame.get_evaluation_step(1);

            let aux_frame = EvaluationFrame::from_rows(
                vec_lambda2winter(first_step.get_row_aux(0)),
                vec_lambda2winter(second_step.get_row_aux(0)),
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
            result.extend_from_slice(&aux_result);
        }
        result
    }
}
