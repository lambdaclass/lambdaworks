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
        3
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

                let two_fifty_six = FieldElement::<Babybear31PrimeField>::from(256);

                let add = step.get_main_evaluation_element(0, 15);
                let load = step.get_main_evaluation_element(0, 26);
                let store = step.get_main_evaluation_element(0, 27);

                let flag = add + load + store;

                let lhs_0 = step.get_main_evaluation_element(0, 34)
                    + two_fifty_six * step.get_main_evaluation_element(0, 35);
                let rhs_0 = step.get_main_evaluation_element(0, 44)
                    + two_fifty_six * step.get_main_evaluation_element(0, 45);
                let res_0 = step.get_main_evaluation_element(0, 48)
                    + two_fifty_six * step.get_main_evaluation_element(0, 49);

                let one = FieldElement::<Babybear31PrimeField>::one();
                let inverse = FieldElement::<Babybear31PrimeField>::from(65536)
                    .inv()
                    .unwrap();
                let carry_0 = (lhs_0 + rhs_0 - res_0) * inverse;

                let bit_contraint = match self.carry_idx {
                    CarryIndex::Zero => flag * carry_0 * (carry_0 - one),
                    CarryIndex::One => {
                        let lhs_1 = step.get_main_evaluation_element(0, 36)
                            + two_fifty_six * step.get_main_evaluation_element(0, 37);
                        let rhs_1 = step.get_main_evaluation_element(0, 46)
                            + two_fifty_six * step.get_main_evaluation_element(0, 47);
                        let res_1 = step.get_main_evaluation_element(0, 50)
                            + two_fifty_six * step.get_main_evaluation_element(0, 51);
                        let carry_1 = (lhs_1 + rhs_1 - res_1 + carry_0) * inverse;
                        flag * carry_1 * (carry_1 - one)
                    }
                };

                transition_evaluations[self.constraint_idx()] = bit_contraint.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
            } => {
                let step = frame.get_evaluation_step(0);

                let two_fifty_six = FieldElement::<Babybear31PrimeField>::from(256);

                let add = step.get_main_evaluation_element(0, 15);
                let load = step.get_main_evaluation_element(0, 26);
                let store = step.get_main_evaluation_element(0, 27);

                let flag = add + load + store;

                let lhs_0 = step.get_main_evaluation_element(0, 34)
                    + two_fifty_six * step.get_main_evaluation_element(0, 35);
                let rhs_0 = step.get_main_evaluation_element(0, 44)
                    + two_fifty_six * step.get_main_evaluation_element(0, 45);
                let res_0 = step.get_main_evaluation_element(0, 48)
                    + two_fifty_six * step.get_main_evaluation_element(0, 49);

                let one = FieldElement::<Degree4BabyBearU32ExtensionField>::one();
                let inverse = FieldElement::<Degree4BabyBearU32ExtensionField>::from(65536)
                    .inv()
                    .unwrap();
                let carry_0 = (lhs_0 + rhs_0 - res_0) * inverse;

                let bit_contraint = match self.carry_idx {
                    CarryIndex::Zero => flag * carry_0 * (carry_0 - one),
                    CarryIndex::One => {
                        let lhs_1 = step.get_main_evaluation_element(0, 36)
                            + two_fifty_six * step.get_main_evaluation_element(0, 37);
                        let rhs_1 = step.get_main_evaluation_element(0, 46)
                            + two_fifty_six * step.get_main_evaluation_element(0, 47);
                        let res_1 = step.get_main_evaluation_element(0, 50)
                            + two_fifty_six * step.get_main_evaluation_element(0, 51);
                        let carry_1 = (lhs_1 + rhs_1 - res_1 + carry_0) * inverse;
                        flag * carry_1 * (carry_1 - one)
                    }
                };

                transition_evaluations[self.constraint_idx()] = bit_contraint
            }
        }
    }
}

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
        let bit_columns_index_to_constraint = [
            7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let bit_constraints: Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> =
            bit_columns_index_to_constraint
                .iter()
                .enumerate()
                .map(|(i, &column_idx)| {
                    Box::new(BitConstraint::new(column_idx, i))
                        as Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>
                })
                .collect();

        let last_index = bit_columns_index_to_constraint.len() - 1;

        // Add constraints:
        // Carry[0] := (lhs[0] + rhs[0] - res[0])/65536`
        // carry[1] := (lhs[1] + rhs[1] - res[1] + carry[0])/65536`
        let add_constraints: Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> = vec![
            Box::new(CarryBitConstraint::new(CarryIndex::Zero, last_index + 1)),
            Box::new(CarryBitConstraint::new(CarryIndex::One, last_index + 2)),
        ];

        let mut constraints = bit_constraints;
        constraints.extend(add_constraints);

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
