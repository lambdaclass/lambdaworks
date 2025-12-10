use crate::{
    constraints::{boundary::BoundaryConstraints, transition::TransitionConstraint},
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{TransitionEvaluationContext, AIR},
};
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    },
};

type FE = FieldElement<Babybear31PrimeField>;

type ADDTraceTable = TraceTable<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>;

#[derive(Clone)]
pub enum CarryIndex {
    Zero,
    One,
}

#[derive(Clone)]
pub struct CarryBitConstraint {
    carry_idx: CarryIndex,
    constraint_idx: usize,
}

impl CarryBitConstraint {
    fn new(carry_idx: CarryIndex, constraint_idx: usize) -> Self {
        Self {
            carry_idx,
            constraint_idx,
        }
    }
}

impl TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>
    for CarryBitConstraint
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn exemptions_period(&self) -> Option<usize> {
        None
    }

    fn periodic_exemptions_offset(&self) -> Option<usize> {
        None
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<
            Babybear31PrimeField,
            Degree4BabyBearU32ExtensionField,
        >,
        transition_evaluations: &mut [FieldElement<Degree4BabyBearU32ExtensionField>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                // - `carry[0] := (lhs[0] + rhs[0] - res[0])/65536`
                // - `carry[1] := (lhs[1] + rhs[1] - res[1] + carry[0])/65536`
                let step = frame.get_evaluation_step(0);

                let lhs_0 = step.get_main_evaluation_element(0, 0);
                let rhs_0 = step.get_main_evaluation_element(0, 2);
                let res_0 = step.get_main_evaluation_element(0, 4);

                let one = FieldElement::<Babybear31PrimeField>::one();
                let inverse = FieldElement::<Babybear31PrimeField>::from(65536)
                    .inv()
                    .unwrap();
                let carry_0 = (lhs_0 + rhs_0 - res_0) * inverse;

                let bit_contraint = match self.carry_idx {
                    CarryIndex::Zero => carry_0 * (carry_0 - one),
                    CarryIndex::One => {
                        let lhs_1 = step.get_main_evaluation_element(0, 1);
                        let rhs_1 = step.get_main_evaluation_element(0, 3);
                        let res_1 = step.get_main_evaluation_element(0, 5);
                        let carry_1 = (lhs_1 + rhs_1 - res_1 + carry_0) * inverse;
                        carry_1 * (carry_1 - one)
                    }
                };

                transition_evaluations[self.constraint_idx()] = bit_contraint.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                // - `carry[0] := (lhs[0] + rhs[0] - res[0])/65536`
                // - `carry[1] := (lhs[1] + rhs[1] - res[1] + carry[0])/65536`
                let step = frame.get_evaluation_step(0);

                let lhs_0 = step.get_main_evaluation_element(0, 0);
                let rhs_0 = step.get_main_evaluation_element(0, 2);
                let res_0 = step.get_main_evaluation_element(0, 4);

                let one = FieldElement::<Degree4BabyBearU32ExtensionField>::one();
                let inverse = FieldElement::<Degree4BabyBearU32ExtensionField>::from(65536)
                    .inv()
                    .unwrap();
                let carry_0 = (lhs_0 + rhs_0 - res_0) * inverse;

                let bit_contraint = match self.carry_idx {
                    CarryIndex::Zero => carry_0 * (carry_0 - one),
                    CarryIndex::One => {
                        let lhs_1 = step.get_main_evaluation_element(0, 1);
                        let rhs_1 = step.get_main_evaluation_element(0, 3);
                        let res_1 = step.get_main_evaluation_element(0, 5);
                        let carry_1 = (lhs_1 + rhs_1 - res_1 + carry_0) * inverse;
                        carry_1 * (carry_1 - one)
                    }
                };

                transition_evaluations[self.constraint_idx()] = bit_contraint
            }
        }
    }
}

pub struct ADDTableAIR {
    context: AirContext,
    constraints:
        Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>>>,
    trace_length: usize,
}

impl AIR for ADDTableAIR {
    type Field = Babybear31PrimeField;
    type FieldExtension = Degree4BabyBearU32ExtensionField;
    type PublicInputs = ();

    const STEP_SIZE: usize = 1;

    fn new(
        trace_length: usize,
        _pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        let constraints: Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> = vec![
            Box::new(CarryBitConstraint::new(CarryIndex::Zero, 0)),
            Box::new(CarryBitConstraint::new(CarryIndex::One, 1)),
        ];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 6,
            transition_offsets: vec![0],
            num_transition_constraints: 2,
        };

        Self {
            context,
            trace_length,
            constraints,
        }
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> {
        &self.constraints
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::FieldExtension> {
        BoundaryConstraints::from_constraints(vec![])
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length * 2
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (6, 0)
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &()
    }
}

pub fn build_add_trace(columns: Vec<Vec<FE>>) -> ADDTraceTable {
    TraceTable::from_columns_main(columns, 1)
}
