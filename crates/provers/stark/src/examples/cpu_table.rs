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

type CPUTraceTable = TraceTable<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>;

#[derive(Clone)]
pub struct BitConstraint {
    column_idx: usize,
    constraint_idx: usize,
}

impl BitConstraint {
    fn new(column_idx: usize, constraint_idx: usize) -> Self {
        Self {
            column_idx,
            constraint_idx,
        }
    }
}

impl TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>
    for BitConstraint
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
                let step = frame.get_evaluation_step(0);

                let flag = step.get_main_evaluation_element(0, self.column_idx);

                let one = FieldElement::<Babybear31PrimeField>::one();

                let bit_constraint = flag * (flag - one);

                transition_evaluations[self.constraint_idx()] = bit_constraint.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                let step = frame.get_evaluation_step(0);

                let flag = step.get_main_evaluation_element(0, self.column_idx);

                let one = FieldElement::<Degree4BabyBearU32ExtensionField>::one();

                let bit_constraint = flag * (flag - one);

                transition_evaluations[self.constraint_idx()] = bit_constraint;
            }
        }
    }
}

pub struct CPUTableAIR {
    context: AirContext,
    constraints:
        Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>>>,
    trace_length: usize,
}

impl AIR for CPUTableAIR {
    type Field = Babybear31PrimeField;
    type FieldExtension = Degree4BabyBearU32ExtensionField;
    type PublicInputs = ();

    const STEP_SIZE: usize = 1;

    fn new(
        trace_length: usize,
        _pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        // Constraint IS_BIT[f[i]] where:
        // f = [write_register, memory_2bytes, memory_4bytes, signed, signed2, muldiv_selector, ADD, SUB, SLT, AND, OR, XOR, SL, SR, JALR, BEQ, BLT, LOAD, STORE, MUL, DIVREM, ECALL, EBREAK]
        let columns_index_to_constraint = [
            7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let constraints: Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> =
            columns_index_to_constraint
                .iter()
                .enumerate()
                .map(|(i, &column_idx)| {
                    Box::new(BitConstraint::new(column_idx, i))
                        as Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>
                })
                .collect();
        let num_transition_constraints = constraints.len();

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 54,
            transition_offsets: vec![0],
            num_transition_constraints,
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
        (54, 0)
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &()
    }
}

// We assume that the columns have a power of two number of rows.
pub fn build_cpu_trace(columns: Vec<Vec<FE>>) -> CPUTraceTable {
    TraceTable::from_columns_main(columns, 1)
}
