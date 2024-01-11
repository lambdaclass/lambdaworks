// use std::iter;

// use lambdaworks_math::field::{
//     element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
// };

// use crate::{
//     constraints::{boundary::BoundaryConstraints, transition::TransitionConstraint},
//     context::AirContext,
//     frame::Frame,
//     proof::options::ProofOptions,
//     trace::TraceTable,
//     traits::AIR,
//     Felt252,
// };

// #[derive(Clone)]
// pub struct BitConstraint;
// impl BitConstraint {
//     fn new() -> Self {
//         Self
//     }
// }

// impl TransitionConstraint<Stark252PrimeField> for BitConstraint {
//     fn degree(&self) -> usize {
//         2
//     }

//     fn constraint_idx(&self) -> usize {
//         0
//     }

//     fn exemptions_period(&self) -> Option<usize> {
//         Some(16)
//     }

//     fn periodic_exemptions_offset(&self) -> Option<usize> {
//         Some(15)
//     }

//     fn end_exemptions(&self) -> usize {
//         0
//     }

//     fn evaluate(
//         &self,
//         frame: &Frame<Stark252PrimeField>,
//         transition_evaluations: &mut [Felt252],
//         _periodic_values: &[Felt252],
//         _rap_challenges: &[Felt252],
//     ) {
//         let step = frame.get_evaluation_step(0);

//         let prefix_flag = step.get_evaluation_element(0, 0);
//         let next_prefix_flag = step.get_evaluation_element(1, 0);

//         let two = Felt252::from(2);
//         let one = Felt252::one();
//         let bit_flag = prefix_flag - two * next_prefix_flag;

//         let bit_constraint = bit_flag * (bit_flag - one);

//         transition_evaluations[self.constraint_idx()] = bit_constraint;
//     }
// }

// #[derive(Clone)]
// pub struct ZeroFlagConstraint;
// impl ZeroFlagConstraint {
//     fn new() -> Self {
//         Self
//     }
// }

// impl TransitionConstraint<Stark252PrimeField> for ZeroFlagConstraint {
//     fn degree(&self) -> usize {
//         1
//     }

//     fn constraint_idx(&self) -> usize {
//         1
//     }

//     fn end_exemptions(&self) -> usize {
//         0
//     }

//     fn period(&self) -> usize {
//         16
//     }

//     fn offset(&self) -> usize {
//         15
//     }

//     fn evaluate(
//         &self,
//         frame: &Frame<Stark252PrimeField>,
//         transition_evaluations: &mut [FieldElement<Stark252PrimeField>],
//         _periodic_values: &[FieldElement<Stark252PrimeField>],
//         _rap_challenges: &[FieldElement<Stark252PrimeField>],
//     ) {
//         let step = frame.get_evaluation_step(0);
//         let zero_flag = step.get_evaluation_element(15, 0);

//         transition_evaluations[self.constraint_idx()] = *zero_flag;
//     }
// }

// pub struct BitFlagsAIR {
//     context: AirContext,
//     constraints: Vec<Box<dyn TransitionConstraint<Stark252PrimeField>>>,
//     trace_length: usize,
// }

// impl AIR for BitFlagsAIR {
//     type Field = Stark252PrimeField;
//     type PublicInputs = ();

//     const STEP_SIZE: usize = 16;

//     fn new(
//         trace_length: usize,
//         _pub_inputs: &Self::PublicInputs,
//         proof_options: &ProofOptions,
//     ) -> Self {
//         let bit_constraint = Box::new(BitConstraint::new());
//         let flag_constraint = Box::new(ZeroFlagConstraint::new());
//         let constraints: Vec<Box<dyn TransitionConstraint<Self::Field>>> =
//             vec![bit_constraint, flag_constraint];
//         // vec![flag_constraint];
//         // vec![bit_constraint];

//         let num_transition_constraints = constraints.len();
//         let transition_exemptions: Vec<_> =
//             constraints.iter().map(|c| c.end_exemptions()).collect();

//         let context = AirContext {
//             proof_options: proof_options.clone(),
//             trace_columns: 1,
//             transition_exemptions,
//             transition_offsets: vec![0],
//             num_transition_constraints,
//         };

//         Self {
//             context,
//             trace_length,
//             constraints,
//         }
//     }

//     fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraint<Self::Field>>> {
//         &self.constraints
//     }

//     fn boundary_constraints(
//         &self,
//         _rap_challenges: &[FieldElement<Self::Field>],
//     ) -> BoundaryConstraints<Self::Field> {
//         BoundaryConstraints::from_constraints(vec![])
//     }

//     fn context(&self) -> &AirContext {
//         &self.context
//     }

//     fn composition_poly_degree_bound(&self) -> usize {
//         self.trace_length * 2
//     }

//     fn trace_length(&self) -> usize {
//         self.trace_length
//     }

//     fn pub_inputs(&self) -> &Self::PublicInputs {
//         &()
//     }
// }

// pub fn bit_prefix_flag_trace(num_steps: usize) -> TraceTable<Stark252PrimeField> {
//     debug_assert!(num_steps.is_power_of_two());
//     let step: Vec<Felt252> = [
//         1031u64, 515, 257, 128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0,
//     ]
//     .iter()
//     .map(|t| Felt252::from(*t))
//     .collect();

//     let mut data: Vec<Felt252> = iter::repeat(step).take(num_steps).flatten().collect();
//     data[0] = Felt252::from(1030);

//     TraceTable::new(data, 1, 16)
// }
