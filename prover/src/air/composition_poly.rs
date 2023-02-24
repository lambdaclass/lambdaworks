use crypto::hashers::{Blake2s_256, Blake3_256};
use lambdaworks_math::field::fields::u384_prime_field::IsMontgomeryConfiguration;
use lambdaworks_math::unsigned_integer::element::U384;
use lambdaworks_math::{
    field::{element::FieldElement, fields::u384_prime_field::MontgomeryBackendPrimeField},
    polynomial::Polynomial,
};
use math::StarkField;
use winter_prover::trace::TracePolyTable;
use winter_prover::{
    constraints::CompositionPoly, Air, AuxTraceRandElements, Serializable, Trace, TraceTable,
};

use giza_core::Felt;
use winter_prover::{
    build_trace_commitment_f, channel::ProverChannel, constraints::ConstraintEvaluator,
    domain::StarkDomain, trace::commitment::TraceCommitment,
};

#[derive(Clone, Debug)]
pub struct MontgomeryConfig;
impl IsMontgomeryConfiguration for MontgomeryConfig {
    const MODULUS: U384 =
        U384::from("800000000000011000000000000000000000000000000000000000000000001");
    const MP: u64 = 18446744073709551615;
    const R2: U384 = U384::from("38e5f79873c0a6df47d84f8363000187545706677ffcc06cc7177d1406df18e");
}

type U384PrimeField = MontgomeryBackendPrimeField<MontgomeryConfig>;
type U384FieldElement = FieldElement<U384PrimeField>;

/// Given a CompositionPoly from winterfell, extract its coefficients
/// as a vector.
#[allow(dead_code)]
pub(crate) fn get_coefficients<E: StarkField>(poly: CompositionPoly<E>) -> Vec<E> {
    let data = poly.into_columns();

    let num_columns = data.len();
    let column_len = data[0].len();

    let mut coeffs = Vec::with_capacity(num_columns * column_len);

    for i in 0..column_len {
        for coeff_col in data.iter().take(num_columns) {
            coeffs.push(coeff_col[i]);
        }
    }

    coeffs
}

/// Given a trace and an AIR defined with winterfell data structures, outputs
/// a Polynomial lambdaworks native data structure.
#[allow(dead_code)]
pub(crate) fn get_composition_poly<A, T>(
    air: A,
    mut trace: T,
    pub_inputs: A::PublicInputs,
) -> Polynomial<U384FieldElement>
where
    A: Air<BaseField = Felt>,
    T: Trace<BaseField = A::BaseField>,
{
    let mut pub_inputs_bytes = Vec::new();
    pub_inputs.write_into(&mut pub_inputs_bytes);
    let mut channel =
        ProverChannel::<A, A::BaseField, Blake2s_256<A::BaseField>>::new(&air, pub_inputs_bytes);
    let domain = StarkDomain::new(&air);

    // extend the main execution trace and build a Merkle tree from the extended trace
    let (main_trace_lde, main_trace_tree, main_trace_polys) = build_trace_commitment_f::<
        A::BaseField,
        A::BaseField,
        Blake2s_256<A::BaseField>,
    >(trace.main_segment(), &domain);

    // commit to the LDE of the main trace by writing the root of its Merkle tree into
    // the channel
    channel.commit_trace(*main_trace_tree.root());

    let mut trace_commitment = TraceCommitment::new(
        main_trace_lde,
        main_trace_tree,
        domain.trace_to_lde_blowup(),
    );

    let mut trace_polys = TracePolyTable::new(main_trace_polys);

    let mut aux_trace_segments = Vec::new();
    let mut aux_trace_rand_elements = AuxTraceRandElements::new();
    for i in 0..trace.layout().num_aux_segments() {
        // draw a set of random elements required to build an auxiliary trace segment
        let rand_elements = channel.get_aux_trace_segment_rand_elements(i);

        // build the trace segment
        let aux_segment = trace
            .build_aux_segment(&aux_trace_segments, &rand_elements)
            .expect("failed build auxiliary trace segment");

        // extend the auxiliary trace segment and build a Merkle tree from the extended trace
        let (aux_segment_lde, aux_segment_tree, aux_segment_polys) =
            build_trace_commitment_f::<A::BaseField, A::BaseField, Blake2s_256<A::BaseField>>(
                &aux_segment,
                &domain,
            );

        // commit to the LDE of the extended auxiliary trace segment  by writing the root of
        // its Merkle tree into the channel
        channel.commit_trace(*aux_segment_tree.root());

        // append the segment to the trace commitment and trace polynomial table structs
        trace_commitment.add_segment(aux_segment_lde, aux_segment_tree);
        trace_polys.add_aux_segment(aux_segment_polys);
        aux_trace_rand_elements.add_segment_elements(rand_elements);
        aux_trace_segments.push(aux_segment);
    }

    let constraint_coeffs = channel.get_constraint_composition_coeffs();
    let evaluator = ConstraintEvaluator::new(&air, aux_trace_rand_elements, constraint_coeffs);
    let constraint_evaluations = evaluator.evaluate(trace_commitment.trace_table(), &domain);

    let composition_poly = constraint_evaluations.into_poly().unwrap();

    let coeffs: Vec<U384FieldElement> = get_coefficients(composition_poly)
        .into_iter()
        .map(|c| U384FieldElement::from(&U384::from(&c.to_string())))
        .collect();

    Polynomial::new(&coeffs)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use giza_air::{FieldExtension, HashFunction};
    use winter_air::ProofOptions;

    use super::*;

    #[test]
    fn test_cairo_air() {
        use giza_air::{ProcessorAir, ProofOptions};
        use runner::ExecutionTrace;

        let trace = ExecutionTrace::from_file(
            PathBuf::from("src/air/test/program.json"),
            PathBuf::from("src/air/test/trace.bin"),
            PathBuf::from("src/air/test/memory.bin"),
            Some(3),
        );

        // generate the proof of execution
        let proof_options = ProofOptions::with_proof_options(None, None, None, None, None);
        let (_proof, pub_inputs) = giza_prover::prove_trace(trace.clone(), &proof_options).unwrap();
        // let proof_bytes = proof.to_bytes();

        let air = ProcessorAir::new(
            trace.clone().get_info(),
            pub_inputs.clone(),
            proof_options.into_inner(),
        );

        let expected_poly =
            Polynomial::new(&vec![lambdaworks_math::field::element::FieldElement::from(
                0,
            )]);

        assert_eq!(get_composition_poly(air, trace, pub_inputs), expected_poly);
    }

    #[test]
    fn test_get_coefficients() {
        let coeffs = (0u128..16).map(Felt::from).collect::<Vec<_>>();
        let poly = CompositionPoly::new(coeffs.clone(), 2);
        let coeffs_res = get_coefficients(poly);

        assert_eq!(coeffs, coeffs_res);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // This test is made for the following computation, taken from the winterfell README example:
    //    "This computation starts with an element in a finite field and then, for the specified number of steps,
    //    cubes the element and adds value 42 to it"
    //
    // The test setup consists of the following:
    //     * Creating a trace of the computation
    //     * Implementing an AIR of the computation
    //
    // TODO: Check that the obtained polynomial is correct.
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #[test]
    fn test_composition_poly_simple_computation() {
        use simple_computation_test_utils::*;

        let start = Felt::from(3u64);
        let n = 8;

        // Build the execution trace and .
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
            HashFunction::Blake2s_256,
            FieldExtension::None,
            8,   // FRI folding factor
            128, // FRI max remainder length
        );

        let air = WorkAir::new(trace.get_info(), pub_inputs.clone(), options);

        // TODO: this coefficients should be checked correctly to know
        // the test is really passing
        let expected_coeffs: Vec<U384FieldElement> = vec![
            73805846515134368521942875729025268850u128,
            251094867283236114961184226859763993364,
            24104107408363517664638843184319555199,
            235849850267413452314892506985835977650,
            90060524782298155599732670785320526321,
            191672218871615916423281291477102427646,
            266219015768353823155348590781060058741,
            323575369761999186385729504758291491620,
            11584578295884344582449562404474773268,
            210204873954083390997653826263858894919,
            255852687493109162976897695176505210663,
            263909750263371532307415443077776653137,
            270439831904630486671154877882097540335,
            295423943150257863151191255913349696708,
            305657170959297052488312417688653377091,
            170130930667860970428731413388750994520,
        ]
        .into_iter()
        .map(|c| U384FieldElement::from(&U384::from(&c.to_string())))
        .collect();

        let expected_poly = Polynomial::new(&expected_coeffs);

        assert_eq!(get_composition_poly(air, trace, pub_inputs), expected_poly);
    }
}

#[cfg(test)]
pub(crate) mod simple_computation_test_utils {
    use math::FieldElement;
    use winter_air::{
        AirContext, Assertion, DefaultEvaluationFrame, ProofOptions, TraceInfo,
        TransitionConstraintDegree,
    };
    use winter_prover::ByteWriter;

    use super::*;

    pub(crate) fn build_do_work_trace(start: Felt, n: usize) -> TraceTable<Felt> {
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
                state[0] = state[0].exp(3u32.into()) + Felt::from(42u64);
            },
        );

        trace
    }

    // Public inputs for our computation will consist of the starting value and the end result.
    #[derive(Clone)]
    pub(crate) struct PublicInputs {
        pub start: Felt,
        pub result: Felt,
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
    pub(crate) struct WorkAir {
        context: AirContext<Felt>,
        start: Felt,
        result: Felt,
    }

    impl Air for WorkAir {
        // First, we'll specify which finite field to use for our computation, and also how
        // the public inputs must look like.
        type BaseField = Felt;
        type PublicInputs = PublicInputs;
        type Frame<E: FieldElement> = DefaultEvaluationFrame<E>;
        type AuxFrame<E: FieldElement> = DefaultEvaluationFrame<E>;

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
        fn evaluate_transition<E: math::FieldElement + From<Self::BaseField>>(
            &self,
            frame: &DefaultEvaluationFrame<E>,
            _periodic_values: &[E],
            result: &mut [E],
        ) {
            // First, we'll read the current state, and use it to compute the expected next state
            let current_state = &frame.current()[0];
            let next_state = current_state.exp(3u32.into()) + E::from(42u32);

            // Then, we'll subtract the expected next state from the actual next state; this will
            // evaluate to zero if and only if the expected and actual states are the same.
            result[0] = (frame.next()[0] - next_state).into();
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
}
