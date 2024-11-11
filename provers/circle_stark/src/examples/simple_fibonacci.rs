use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        transition::TransitionConstraint,
    },
    air_context::AirContext,
    frame::Frame,
    trace::TraceTable,
    air::AIR,
};
use lambdaworks_math::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field, traits::IsFFTField};
// use std::marker::PhantomData;
#[derive(Clone)]
struct FibConstraint;

impl FibConstraint {
    pub fn new() -> Self {
        Self {}
    }
}
impl TransitionConstraint for FibConstraint {
    fn degree(&self) -> usize {
        1
    }
    fn constraint_idx(&self) -> usize {
        0
    }
    fn end_exemptions(&self) -> usize {
        4
    }
    fn evaluate(
        &self,
        frame: &Frame,
        transition_evaluations: &mut [FieldElement<Mersenne31Field>],
    ) {
        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);
        let third_step = frame.get_evaluation_step(2);
        let a0 = first_step[0];
        let a1 = second_step[0];
        let a2 = third_step[0];
        let res = a2 - a1 - a0;
        transition_evaluations[self.constraint_idx()] = res;
    }
}
pub struct FibonacciAIR {
    context: AirContext,
    trace_length: usize,
    pub_inputs: FibonacciPublicInputs,
    constraints: Vec<Box<dyn TransitionConstraint>>,
}
#[derive(Clone, Debug)]
pub struct FibonacciPublicInputs{
    pub a0: FieldElement<Mersenne31Field>,
    pub a1: FieldElement<Mersenne31Field>,
}
impl AIR for FibonacciAIR
{
    type PublicInputs = FibonacciPublicInputs;
    fn new(trace_length: usize, pub_inputs: &Self::PublicInputs) -> Self {
        let constraints: Vec<Box<dyn TransitionConstraint>> =
            vec![Box::new(FibConstraint::new())];
        let context = AirContext {
            trace_columns: 1,
            transition_exemptions: vec![2],
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: constraints.len(),
        };
        Self {
            pub_inputs: pub_inputs.clone(),
            context,
            trace_length,
            constraints,
        }
    }
    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length()
    }
    fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraint>> {
        &self.constraints
    }
    fn boundary_constraints(&self) -> BoundaryConstraints {
        let a0 = BoundaryConstraint::new_simple(0, self.pub_inputs.a0.clone());
        let a1 = BoundaryConstraint::new_simple(1, self.pub_inputs.a1.clone());
        BoundaryConstraints::from_constraints(vec![a0, a1])
    }
    fn context(&self) -> &AirContext {
        &self.context
    }
    fn trace_length(&self) -> usize {
        self.trace_length
    }
    fn trace_layout(&self) -> usize {
        1
    }
    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
    fn compute_transition_verifier(
        &self,
        frame: &Frame,
    ) -> Vec<FieldElement<Mersenne31Field>> {
        self.compute_transition_prover(frame)
    }
}
pub fn fibonacci_trace(
    initial_values: [FieldElement<Mersenne31Field>; 2],
    trace_length: usize,
) -> TraceTable {
    let mut ret: Vec<FieldElement<Mersenne31Field>> = vec![];
    ret.push(initial_values[0].clone());
    ret.push(initial_values[1].clone());
    for i in 2..(trace_length) {
        ret.push(ret[i - 1].clone() + ret[i - 2].clone());
    }
    TraceTable::from_columns(vec![ret])
}
