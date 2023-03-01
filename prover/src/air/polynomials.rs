use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::u384_prime_field::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField},
    },
    polynomial::Polynomial,
    unsigned_integer::element::U384,
};
use winterfell::{
    crypto::hashers::Blake3_256,
    math::{fields::f128::BaseElement, StarkField},
    Air, AuxTraceRandElements, Matrix, Serializable, Trace, TraceTable,
};

use crate::U384FieldElement;

use super::errors::ProverError;
use winterfell::prover::{
    build_trace_commitment_f, channel::ProverChannel, constraints::ConstraintEvaluator,
    domain::StarkDomain, trace::commitment::TraceCommitment,
};

/// Given a CompositionPoly from winterfell, extract its coefficients
/// as a vector.
#[allow(dead_code)]
pub(crate) fn get_coefficients<E: StarkField>(matrix_poly: Matrix<E>) -> Vec<E> {
    let data = matrix_poly.into_columns();

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

#[allow(dead_code)]
pub(crate) fn winter_2_lambda_felts(coeffs: Matrix<BaseElement>) -> Vec<U384FieldElement> {
    get_coefficients(coeffs)
        .into_iter()
        .map(|c| U384FieldElement::from(&U384::from(&c.to_string())))
        .collect()
}

/// Given a trace and an AIR defined with winterfell data structures, outputs
/// a tuple of lambdaworks Polynomial data structure:
/// (composition_polynomial, trace_polynomial)
#[allow(dead_code)]
pub fn get_cp_and_tp<A>(
    air: A,
    trace: TraceTable<A::BaseField>,
    pub_inputs: A::PublicInputs,
) -> Result<(Polynomial<U384FieldElement>, Polynomial<U384FieldElement>), ProverError>
where
    A: Air<BaseField = BaseElement>,
{
    let mut pub_inputs_bytes = Vec::new();
    pub_inputs.write_into(&mut pub_inputs_bytes);
    let mut channel =
        ProverChannel::<A, A::BaseField, Blake3_256<A::BaseField>>::new(&air, pub_inputs_bytes);
    let domain = StarkDomain::new(&air);

    // extend the main execution trace and build a Merkle tree from the extended trace
    let (main_trace_lde, main_trace_tree, main_trace_polys) = build_trace_commitment_f::<
        A::BaseField,
        A::BaseField,
        Blake3_256<A::BaseField>,
    >(trace.main_segment(), &domain);

    let tp_coeffs = winter_2_lambda_felts(main_trace_polys);

    // commit to the LDE of the main trace by writing the root of its Merkle tree into
    // the channel
    channel.commit_trace(*main_trace_tree.root());

    let trace_commitment = TraceCommitment::new(
        main_trace_lde,
        main_trace_tree,
        domain.trace_to_lde_blowup(),
    );

    let aux_trace_rand_elements = AuxTraceRandElements::new();
    let constraint_coeffs = channel.get_constraint_composition_coeffs();
    let evaluator = ConstraintEvaluator::new(&air, aux_trace_rand_elements, constraint_coeffs);
    let constraint_evaluations = evaluator.evaluate(trace_commitment.trace_table(), &domain);

    let composition_poly = constraint_evaluations
        .into_poly()
        .map_err(|e| ProverError::CompositionPolyError(e))?
        .data;

    let cp_coeffs: Vec<U384FieldElement> = winter_2_lambda_felts(composition_poly);

    Ok((Polynomial::new(&cp_coeffs), Polynomial::new(&tp_coeffs)))
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use simple_computation_test_utils::*;
//     use winterfell::{prover::constraints::CompositionPoly, FieldExtension, ProofOptions};

//     #[test]
//     fn test_get_coefficients() {
//         let coeffs = (0u128..16).map(BaseElement::new).collect::<Vec<_>>();
//         let poly = CompositionPoly::new(coeffs.clone(), 2);
//         let coeffs_res = get_coefficients(poly.data);

//         assert_eq!(coeffs, coeffs_res);
//     }

//     //////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     // This test is made for the following computation, taken from the winterfell README example:
//     //    "This computation starts with an element in a finite field and then, for the specified number
//     //    of steps, cubes the element and adds value 42 to it"
//     //
//     // The test setup consists of the following:
//     //     * Creating a trace of the computation
//     //     * Implementing an AIR of the computation
//     //
//     // TODO: Check that the obtained polynomial is correct.
//     //////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     #[test]
//     fn test_cp_and_tp_simple_computation() {
//         let start = BaseElement::new(3);
//         let n = 8;

//         // Build the execution trace and .
//         let trace = build_do_work_trace(start, n);
//         let pub_inputs = PublicInputs {
//             start: trace.get(0, 0),
//             result: trace.get(0, trace.length() - 1),
//         };

//         // Define proof options; these will be enough for ~96-bit security level.
//         let options = ProofOptions::new(
//             32, // number of queries
//             8,  // blowup factor
//             0,  // grinding factor
//             FieldExtension::None,
//             8,   // FRI folding factor
//             128, // FRI max remainder length
//         );

//         let air = WorkAir::new(trace.get_info(), pub_inputs.clone(), options);

//         // TODO: this coefficients should be checked correctly to know
//         // the test is really passing
//         let expected_cp_coeffs: Vec<U384FieldElement> = vec![
//             73805846515134368521942875729025268850u128,
//             251094867283236114961184226859763993364,
//             24104107408363517664638843184319555199,
//             235849850267413452314892506985835977650,
//             90060524782298155599732670785320526321,
//             191672218871615916423281291477102427646,
//             266219015768353823155348590781060058741,
//             323575369761999186385729504758291491620,
//             11584578295884344582449562404474773268,
//             210204873954083390997653826263858894919,
//             255852687493109162976897695176505210663,
//             263909750263371532307415443077776653137,
//             270439831904630486671154877882097540335,
//             295423943150257863151191255913349696708,
//             305657170959297052488312417688653377091,
//             170130930667860970428731413388750994520,
//         ]
//         .into_iter()
//         .map(BaseElement::new)
//         .map(|c| U384FieldElement::from(&U384::from(&c.to_string())))
//         .collect();

//         // TODO: this coefficients should be checked correctly to know
//         // the test is really passing
//         let expected_tp_coeffs: Vec<U384FieldElement> = vec![
//             191794735525862843530824769197527107708,
//             223814277178031177559040503778682093811,
//             330061736062598381779522724430379788786,
//             311785526618037647669831294599779434990,
//             318268825200495925654723848557842782850,
//             60163512837320141687261149502403704415,
//             299361964383055940508518867517619087030,
//             306443623720228722390524190138235769635,
//         ]
//         .into_iter()
//         .map(BaseElement::new)
//         .map(|c| U384FieldElement::from(&U384::from(&c.to_string())))
//         .collect();

//         let expected_cp = Polynomial::new(&expected_cp_coeffs);
//         let expected_tp = Polynomial::new(&expected_tp_coeffs);

//         let (result_cp, result_tp) = get_cp_and_tp(air, trace, pub_inputs).unwrap();

//         // assert polynomials are ok
//         assert_eq!(result_cp, expected_cp);
//         assert_eq!(result_tp, expected_tp);
//     }

//     #[test]
//     fn test_prove() {
//         let start = BaseElement::new(3);
//         let n = 8;

//         // Build the execution trace and .
//         let trace = build_do_work_trace(start, n);
//         let pub_inputs = PublicInputs {
//             start: trace.get(0, 0),
//             result: trace.get(0, trace.length() - 1),
//         };

//         // Define proof options; these will be enough for ~96-bit security level.
//         let options = ProofOptions::new(
//             32, // number of queries
//             8,  // blowup factor
//             0,  // grinding factor
//             FieldExtension::None,
//             8,   // FRI folding factor
//             128, // FRI max remainder length
//         );

//         let air = WorkAir::new(trace.get_info(), pub_inputs.clone(), options);

//         let ret_prove = crate::prove(air, trace, pub_inputs);

//         println!("{ret_prove:?}");
//     }
// }

// #[cfg(test)]
// pub(crate) mod simple_computation_test_utils {
//     use winterfell::{
//         math::FieldElement, AirContext, Assertion, ByteWriter, EvaluationFrame, ProofOptions,
//         TraceInfo, TransitionConstraintDegree,
//     };

//     use super::*;

//     pub(crate) fn build_do_work_trace(start: BaseElement, n: usize) -> TraceTable<BaseElement> {
//         // Instantiate the trace with a given width and length; this will allocate all
//         // required memory for the trace
//         let trace_width = 1;
//         let mut trace = TraceTable::new(trace_width, n);

//         // Fill the trace with data; the first closure initializes the first state of the
//         // computation; the second closure computes the next state of the computation based
//         // on its current state.
//         trace.fill(
//             |state| {
//                 state[0] = start;
//             },
//             |_, state| {
//                 state[0] = state[0].exp(3u32.into()) + BaseElement::new(42);
//             },
//         );

//         trace
//     }

//     // Public inputs for our computation will consist of the starting value and the end result.
//     #[derive(Clone)]
//     pub(crate) struct PublicInputs {
//         pub start: BaseElement,
//         pub result: BaseElement,
//     }

//     // We need to describe how public inputs can be converted to bytes.
//     impl Serializable for PublicInputs {
//         fn write_into<W: ByteWriter>(&self, target: &mut W) {
//             target.write(self.start);
//             target.write(self.result);
//         }
//     }

//     // For a specific instance of our computation, we'll keep track of the public inputs and
//     // the computation's context which we'll build in the constructor. The context is used
//     // internally by the Winterfell prover/verifier when interpreting this AIR.
//     pub(crate) struct WorkAir {
//         context: AirContext<BaseElement>,
//         start: BaseElement,
//         result: BaseElement,
//     }

//     impl Air for WorkAir {
//         // First, we'll specify which finite field to use for our computation, and also how
//         // the public inputs must look like.
//         type BaseField = BaseElement;
//         type PublicInputs = PublicInputs;

//         // Here, we'll construct a new instance of our computation which is defined by 3 parameters:
//         // starting value, number of steps, and the end result. Another way to think about it is
//         // that an instance of our computation is a specific invocation of the do_work() function.
//         fn new(trace_info: TraceInfo, pub_inputs: PublicInputs, options: ProofOptions) -> Self {
//             // our execution trace should have only one column.
//             assert_eq!(1, trace_info.width());

//             // Our computation requires a single transition constraint. The constraint itself
//             // is defined in the evaluate_transition() method below, but here we need to specify
//             // the expected degree of the constraint. If the expected and actual degrees of the
//             // constraints don't match, an error will be thrown in the debug mode, but in release
//             // mode, an invalid proof will be generated which will not be accepted by any verifier.
//             let degrees = vec![TransitionConstraintDegree::new(3)];

//             // We also need to specify the exact number of assertions we will place against the
//             // execution trace. This number must be the same as the number of items in a vector
//             // returned from the get_assertions() method below.
//             let num_assertions = 2;

//             WorkAir {
//                 context: AirContext::new(trace_info, degrees, num_assertions, options),
//                 start: pub_inputs.start,
//                 result: pub_inputs.result,
//             }
//         }

//         // In this method we'll define our transition constraints; a computation is considered to
//         // be valid, if for all valid state transitions, transition constraints evaluate to all
//         // zeros, and for any invalid transition, at least one constraint evaluates to a non-zero
//         // value. The `frame` parameter will contain current and next states of the computation.
//         fn evaluate_transition<E: winterfell::math::FieldElement + From<Self::BaseField>>(
//             &self,
//             frame: &EvaluationFrame<E>,
//             _periodic_values: &[E],
//             result: &mut [E],
//         ) {
//             // First, we'll read the current state, and use it to compute the expected next state
//             let current_state = &frame.current()[0];
//             let next_state = current_state.exp(3u32.into()) + E::from(42u32);

//             // Then, we'll subtract the expected next state from the actual next state; this will
//             // evaluate to zero if and only if the expected and actual states are the same.
//             result[0] = frame.next()[0] - next_state;
//         }

//         // Here, we'll define a set of assertions about the execution trace which must be satisfied
//         // for the computation to be valid. Essentially, this ties computation's execution trace
//         // to the public inputs.
//         fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
//             // for our computation to be valid, value in column 0 at step 0 must be equal to the
//             // starting value, and at the last step it must be equal to the result.
//             let last_step = self.trace_length() - 1;
//             vec![
//                 Assertion::single(0, 0, self.start),
//                 Assertion::single(0, last_step, self.result),
//             ]
//         }

//         // This is just boilerplate which is used by the Winterfell prover/verifier to retrieve
//         // the context of the computation.
//         fn context(&self) -> &AirContext<Self::BaseField> {
//             &self.context
//         }
//     }
// }
