use crate::{constraints::transition::TransitionConstraint, traits::TransitionEvaluationContext};
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    },
};

pub struct BitConstraint {
    column_idx: usize,
    constraint_idx: usize,
}

impl BitConstraint {
    pub fn new(column_idx: usize, constraint_idx: usize) -> Self {
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

pub fn new_bit_constraints(
    column_idx: &[usize],
    constraint_idx_start: usize,
) -> Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>>> {
    column_idx
        .iter()
        .enumerate()
        .map(|(i, &column_idx)| {
            Box::new(BitConstraint::new(column_idx, constraint_idx_start + i))
                as Box<
                    dyn TransitionConstraint<
                        Babybear31PrimeField,
                        Degree4BabyBearU32ExtensionField,
                    >,
                >
        })
        .collect()
}

#[derive(Clone)]
pub enum CarryIndex {
    Zero,
    One,
}

#[derive(Clone)]
pub struct CarryBitConstraint {
    carry_idx: CarryIndex,
    flags_idx: Vec<usize>,
    lhs_start_idx: usize,
    rhs_start_idx: usize,
    res_start_idx: usize,
    constraint_idx: usize,
}

impl CarryBitConstraint {
    fn new(
        carry_idx: CarryIndex,
        flags_idx: Vec<usize>,
        lhs_start_idx: usize,
        rhs_start_idx: usize,
        res_start_idx: usize,
        constraint_idx: usize,
    ) -> Self {
        Self {
            carry_idx,
            flags_idx,
            lhs_start_idx,
            rhs_start_idx,
            res_start_idx,
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

                let flag = self
                    .flags_idx
                    .iter()
                    .fold(FieldElement::<Babybear31PrimeField>::zero(), |acc, &idx| {
                        acc + step.get_main_evaluation_element(0, idx)
                    });

                let lhs_0 = step.get_main_evaluation_element(0, self.lhs_start_idx)
                    + two_fifty_six * step.get_main_evaluation_element(0, self.lhs_start_idx + 1);
                let rhs_0 = step.get_main_evaluation_element(0, self.rhs_start_idx)
                    + two_fifty_six * step.get_main_evaluation_element(0, self.rhs_start_idx + 1);
                let res_0 = step.get_main_evaluation_element(0, self.res_start_idx)
                    + two_fifty_six * step.get_main_evaluation_element(0, self.res_start_idx + 1);

                let one = FieldElement::<Babybear31PrimeField>::one();
                let inverse = FieldElement::<Babybear31PrimeField>::from(65536)
                    .inv()
                    .unwrap();
                let carry_0 = (lhs_0 + rhs_0 - res_0) * inverse;

                let bit_contraint: FieldElement<Babybear31PrimeField> = match self.carry_idx {
                    CarryIndex::Zero => flag * carry_0 * (carry_0 - one),
                    CarryIndex::One => {
                        let lhs_1 = step.get_main_evaluation_element(0, self.lhs_start_idx + 2)
                            + two_fifty_six
                                * step.get_main_evaluation_element(0, self.lhs_start_idx + 3);
                        let rhs_1 = step.get_main_evaluation_element(0, self.rhs_start_idx + 2)
                            + two_fifty_six
                                * step.get_main_evaluation_element(0, self.rhs_start_idx + 3);
                        let res_1 = step.get_main_evaluation_element(0, self.res_start_idx + 2)
                            + two_fifty_six
                                * step.get_main_evaluation_element(0, self.res_start_idx + 3);
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

                let flag = self.flags_idx.iter().fold(
                    FieldElement::<Degree4BabyBearU32ExtensionField>::zero(),
                    |acc, &idx| acc + step.get_main_evaluation_element(0, idx),
                );

                let lhs_0 = step.get_main_evaluation_element(0, self.lhs_start_idx)
                    + two_fifty_six * step.get_main_evaluation_element(0, self.lhs_start_idx + 1);
                let rhs_0 = step.get_main_evaluation_element(0, self.rhs_start_idx)
                    + two_fifty_six * step.get_main_evaluation_element(0, self.rhs_start_idx + 1);
                let res_0 = step.get_main_evaluation_element(0, self.res_start_idx)
                    + two_fifty_six * step.get_main_evaluation_element(0, self.res_start_idx + 1);

                let one = FieldElement::<Degree4BabyBearU32ExtensionField>::one();
                let inverse = FieldElement::<Degree4BabyBearU32ExtensionField>::from(65536)
                    .inv()
                    .unwrap();
                let carry_0 = (lhs_0 + rhs_0 - res_0) * inverse;

                let bit_contraint = match self.carry_idx {
                    CarryIndex::Zero => flag * carry_0 * (carry_0 - one),
                    CarryIndex::One => {
                        let lhs_1 = step.get_main_evaluation_element(0, self.lhs_start_idx + 2)
                            + two_fifty_six
                                * step.get_main_evaluation_element(0, self.lhs_start_idx + 3);
                        let rhs_1 = step.get_main_evaluation_element(0, self.rhs_start_idx + 2)
                            + two_fifty_six
                                * step.get_main_evaluation_element(0, self.rhs_start_idx + 3);
                        let res_1 = step.get_main_evaluation_element(0, self.res_start_idx + 2)
                            + two_fifty_six
                                * step.get_main_evaluation_element(0, self.res_start_idx + 3);
                        let carry_1 = (lhs_1 + rhs_1 - res_1 + carry_0) * inverse;
                        flag * carry_1 * (carry_1 - one)
                    }
                };

                transition_evaluations[self.constraint_idx()] = bit_contraint
            }
        }
    }
}

pub fn new_add_constraint(
    flags_idx: Vec<usize>,
    lhs_start_idx: usize,
    rhs_start_idx: usize,
    res_start_idx: usize,
    constraint_idx_start: usize,
) -> Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Degree4BabyBearU32ExtensionField>>> {
    vec![
        Box::new(CarryBitConstraint::new(
            CarryIndex::Zero,
            flags_idx.clone(),
            lhs_start_idx,
            rhs_start_idx,
            res_start_idx,
            constraint_idx_start,
        )),
        Box::new(CarryBitConstraint::new(
            CarryIndex::One,
            flags_idx,
            lhs_start_idx,
            rhs_start_idx,
            res_start_idx,
            constraint_idx_start + 1,
        )),
    ]
}
