use lambdaworks_math::polynomial::Polynomial;
use winterfell::{
    crypto::hashers::Blake3_256,
    math::{fields::f128::BaseElement, FieldElement, StarkField},
    Air, AirContext, Assertion, AuxTraceRandElements, ByteWriter, EvaluationFrame, FieldExtension,
    ProofOptions, Prover, Serializable, StarkProof, Trace, TraceInfo, TraceTable,
    TransitionConstraintDegree,
};

use winterfell::prover::build_trace_commitment_f;
use winterfell::prover::channel::ProverChannel;
use winterfell::prover::constraints::ConstraintEvaluator;
use winterfell::prover::domain::StarkDomain;
use winterfell::prover::trace::commitment::TraceCommitment;

// fn get_composition_poly<A, E, F>(
fn get_composition_poly<A, E>(
    air: A,
    trace: TraceTable<E>,
    pub_inputs: A::PublicInputs,
) -> Result<(), ()>
// ) -> Polynomial<F>
where
    A: Air<BaseField = E>,
    E: StarkField,
    //F: FieldElement,
{
    let mut pub_inputs_bytes = Vec::new();
    pub_inputs.write_into(&mut pub_inputs_bytes);

    let mut channel = ProverChannel::<A, E, Blake3_256<E>>::new(&air, pub_inputs_bytes);

    let domain = StarkDomain::new(&air);

    // extend the main execution trace and build a Merkle tree from the extended trace
    let (main_trace_lde, main_trace_tree, _main_trace_polys) =
        build_trace_commitment_f::<E, E, Blake3_256<E>>(trace.main_segment(), &domain);

    // commit to the LDE of the main trace by writing the root of its Merkle tree into
    // the channel
    channel.commit_trace(*main_trace_tree.root());

    // initialize trace commitment and trace polynomial table structs with the main trace
    // data; for multi-segment traces these structs will be used as accumulators of all
    // trace segments
    let trace_commitment = TraceCommitment::new(
        main_trace_lde,
        main_trace_tree,
        domain.trace_to_lde_blowup(),
    );

    // let mut trace_polys: TracePolyTable<BaseElement> = TracePolyTable::new(main_trace_polys);
    let aux_trace_rand_elements = AuxTraceRandElements::new();
    let constraint_coeffs = channel.get_constraint_composition_coeffs();
    let evaluator = ConstraintEvaluator::new(&air, aux_trace_rand_elements, constraint_coeffs);
    let constraint_evaluations = evaluator.evaluate(trace_commitment.trace_table(), &domain);

    let composition_poly = constraint_evaluations.into_poly().unwrap();

    println!("COMPOSITION POLYNOMIAL: {:?}", composition_poly);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // This test is made for the following computation, taken from the winterfell README:
    //    This computation starts with an element in a finite field and then, for the specified number of steps,
    //    cubes the element and adds value 42 to it.
    #[test]
    fn test_composition_poly_simple_computation() {
        pub fn build_do_work_trace(start: BaseElement, n: usize) -> TraceTable<BaseElement> {
            // Instantiate the trace with a given width and length; this will allocate all
            // required memory for the trace
            let trace_width = 1;
            let mut trace = TraceTable::new(trace_width, n);

            // Fill the trace with data; the first closure initializes the first state of the
            // computation; the second closure computes the next state of the computation based
            // on its current state.
            trace.fill(
                |state| {
                    state[0] = start;
                },
                |_, state| {
                    state[0] = state[0].exp(3u32.into()) + BaseElement::new(42);
                },
            );

            trace
        }

        // Public inputs for our computation will consist of the starting value and the end result.
        #[derive(Clone)]
        pub struct PublicInputs {
            start: BaseElement,
            result: BaseElement,
        }

        // We need to describe how public inputs can be converted to bytes.
        impl Serializable for PublicInputs {
            fn write_into<W: ByteWriter>(&self, target: &mut W) {
                target.write(self.start);
                target.write(self.result);
            }
        }

        // For a specific instance of our computation, we'll keep track of the public inputs and
        // the computation's context which we'll build in the constructor. The context is used
        // internally by the Winterfell prover/verifier when interpreting this AIR.
        pub struct WorkAir {
            context: AirContext<BaseElement>,
            start: BaseElement,
            result: BaseElement,
        }

        impl Air for WorkAir {
            // First, we'll specify which finite field to use for our computation, and also how
            // the public inputs must look like.
            type BaseField = BaseElement;
            type PublicInputs = PublicInputs;

            // Here, we'll construct a new instance of our computation which is defined by 3 parameters:
            // starting value, number of steps, and the end result. Another way to think about it is
            // that an instance of our computation is a specific invocation of the do_work() function.
            fn new(trace_info: TraceInfo, pub_inputs: PublicInputs, options: ProofOptions) -> Self {
                // our execution trace should have only one column.
                assert_eq!(1, trace_info.width());

                // Our computation requires a single transition constraint. The constraint itself
                // is defined in the evaluate_transition() method below, but here we need to specify
                // the expected degree of the constraint. If the expected and actual degrees of the
                // constraints don't match, an error will be thrown in the debug mode, but in release
                // mode, an invalid proof will be generated which will not be accepted by any verifier.
                let degrees = vec![TransitionConstraintDegree::new(3)];

                // We also need to specify the exact number of assertions we will place against the
                // execution trace. This number must be the same as the number of items in a vector
                // returned from the get_assertions() method below.
                let num_assertions = 2;

                WorkAir {
                    context: AirContext::new(trace_info, degrees, num_assertions, options),
                    start: pub_inputs.start,
                    result: pub_inputs.result,
                }
            }

            // In this method we'll define our transition constraints; a computation is considered to
            // be valid, if for all valid state transitions, transition constraints evaluate to all
            // zeros, and for any invalid transition, at least one constraint evaluates to a non-zero
            // value. The `frame` parameter will contain current and next states of the computation.
            fn evaluate_transition<E: FieldElement + From<Self::BaseField>>(
                &self,
                frame: &EvaluationFrame<E>,
                _periodic_values: &[E],
                result: &mut [E],
            ) {
                // First, we'll read the current state, and use it to compute the expected next state
                let current_state = &frame.current()[0];
                let next_state = current_state.exp(3u32.into()) + E::from(42u32);

                // Then, we'll subtract the expected next state from the actual next state; this will
                // evaluate to zero if and only if the expected and actual states are the same.
                result[0] = frame.next()[0] - next_state;
            }

            // Here, we'll define a set of assertions about the execution trace which must be satisfied
            // for the computation to be valid. Essentially, this ties computation's execution trace
            // to the public inputs.
            fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
                // for our computation to be valid, value in column 0 at step 0 must be equal to the
                // starting value, and at the last step it must be equal to the result.
                let last_step = self.trace_length() - 1;
                vec![
                    Assertion::single(0, 0, self.start),
                    Assertion::single(0, last_step, self.result),
                ]
            }

            // This is just boilerplate which is used by the Winterfell prover/verifier to retrieve
            // the context of the computation.
            fn context(&self) -> &AirContext<Self::BaseField> {
                &self.context
            }
        }

        let start = BaseElement::new(3);
        let n = 8;

        // Build the execution trace and get the result from the last step.
        let trace = build_do_work_trace(start, n);
        let pub_inputs = PublicInputs {
            start: trace.get(0, 0),
            result: trace.get(0, trace.length() - 1),
        };

        // Define proof options; these will be enough for ~96-bit security level.
        let options = ProofOptions::new(
            32, // number of queries
            8,  // blowup factor
            0,  // grinding factor
            FieldExtension::None,
            8,   // FRI folding factor
            128, // FRI max remainder length
        );

        let air = WorkAir::new(trace.get_info(), pub_inputs.clone(), options);

        assert!(get_composition_poly(air, trace, pub_inputs).is_ok());
    }
}
