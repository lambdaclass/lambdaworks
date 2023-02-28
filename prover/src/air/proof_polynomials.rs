use crypto::hashers::Blake2s_256;
use lambdaworks_math::field::fields::u384_prime_field::IsMontgomeryConfiguration;
use lambdaworks_math::unsigned_integer::element::U384;
use lambdaworks_math::{
    field::{element::FieldElement, fields::u384_prime_field::MontgomeryBackendPrimeField},
    polynomial::Polynomial,
};
use math::StarkField;

use giza_core::Felt;
use winter_prover::{
    build_trace_commitment_f, channel::ProverChannel, constraints::ConstraintEvaluator,
    domain::StarkDomain, trace::commitment::TraceCommitment, trace::TracePolyTable, Air,
    AuxTraceRandElements, Matrix, Serializable, Trace,
};

use crate::errors::ProverError;

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
pub(crate) fn get_cp_coeffs<E: StarkField>(matrix_poly: Matrix<E>) -> Vec<E> {
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
pub(crate) fn giza2lambda_felts(coeffs: &[Felt]) -> Vec<U384FieldElement> {
    coeffs
        .iter()
        .map(|c| U384FieldElement::from(&U384::from(&c.to_string())))
        .collect()
}

#[allow(dead_code)]
pub(crate) fn giza2lambda_tp(polys: TracePolyTable<Felt>) -> Vec<Polynomial<U384FieldElement>> {
    let polys_iter = polys.main_trace_polys();
    let mut lambda_polys: Vec<Polynomial<U384FieldElement>> = Vec::with_capacity(polys_iter.len());
    for poly in polys_iter {
        let coeffs = giza2lambda_felts(poly);
        let p = Polynomial::new(&coeffs);
        lambda_polys.push(p);
    }

    lambda_polys
}

/// Given a trace and an AIR defined with winterfell data structures, outputs
/// a Polynomial lambdaworks native data structure.
#[allow(dead_code)]
pub(crate) fn get_cp_and_tps<A, T>(
    air: A,
    mut trace: T,
    pub_inputs: A::PublicInputs,
) -> Result<
    (
        Polynomial<U384FieldElement>,
        Vec<Polynomial<U384FieldElement>>,
    ),
    ProverError,
>
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

    // TODO: Should we get the aux trace polys too?
    let main_trace_polys = giza2lambda_tp(trace_polys.clone());

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

    let composition_poly = constraint_evaluations
        .into_poly()
        .map_err(ProverError::CompositionPolyError)?
        .data;

    let giza_cp_coeffs: Vec<Felt> = get_cp_coeffs(composition_poly);
    let lambda_coeffs: Vec<U384FieldElement> = giza2lambda_felts(&giza_cp_coeffs);

    Ok((Polynomial::new(&lambda_coeffs), main_trace_polys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use giza_air::{FieldExtension, HashFunction, ProcessorAir};
    use runner::ExecutionTrace;
    use std::path::PathBuf;
    use winter_prover::{constraints::CompositionPoly, TraceTable};

    #[test]
    fn test_get_coefficients() {
        let coeffs = (0u128..16).map(Felt::from).collect::<Vec<_>>();
        let poly = CompositionPoly::new(coeffs.clone(), 2);
        let coeffs_res = get_cp_coeffs(poly.data);

        assert_eq!(coeffs, coeffs_res);
    }

    #[test]
    fn test_fibonacci_computation_verification() {
        use fibonacci_computation_test_utils::*;

        let (res, proof) = prove_work();
        let verification = verify_work(Felt::from(1u64), Felt::from(1u64), res, proof);

        assert!(verification.is_ok());
    }

    #[test]
    fn test_composition_poly_simple_computation() {
        use fibonacci_computation_test_utils::*;

        // Build the execution trace
        let t = vec![vec![
            Felt::from(1u64),
            Felt::from(1u64),
            Felt::from(2u64),
            Felt::from(3u64),
            Felt::from(5u64),
            Felt::from(8u64),
            Felt::from(13u64),
            Felt::from(21u64),
        ]];

        let trace = TraceTable::init(t);
        let pub_inputs = PublicInputs {
            a0: trace.get(0, 0),
            a1: trace.get(0, 1),
            result: trace.get(0, trace.length() - 1),
        };

        // Define proof options; these will be enough for ~96-bit security level.
        let options = winter_air::ProofOptions::new(
            32, // number of queries
            8,  // blowup factor
            0,  // grinding factor
            HashFunction::Blake2s_256,
            FieldExtension::None,
            8,   // FRI folding factor
            128, // FRI max remainder length
        );

        let air = FibAir::new(trace.get_info(), pub_inputs.clone(), options);

        let res = get_cp_and_tps(air, trace, pub_inputs);

        // TODO: Just asserting is_ok(), we should make the calculations about the coefficients of all the involved
        // polynomials for the test and then assert they are correct.
        assert!(res.is_ok());
    }

    #[test]
    fn test_cairo_air() {
        let trace = ExecutionTrace::from_file(
            PathBuf::from("src/air/test/program.json"),
            PathBuf::from("src/air/test/trace.bin"),
            PathBuf::from("src/air/test/memory.bin"),
            Some(3),
        );

        // generate the proof of execution
        let proof_options =
            giza_air::ProofOptions::with_proof_options(None, None, None, None, None);
        let (_proof, pub_inputs) = giza_prover::prove_trace(trace.clone(), &proof_options).unwrap();
        // let proof_bytes = proof.to_bytes();

        let air = ProcessorAir::new(
            trace.clone().get_info(),
            pub_inputs.clone(),
            proof_options.into_inner(),
        );

        let res = get_cp_and_tps(air, trace, pub_inputs);

        // TODO: Just asserting is_ok(), we should make the calculations about the coefficients of all the involved
        // polynomials for the test and then assert they are correct.
        assert!(res.is_ok());
    }
}

#[cfg(test)]
pub(crate) mod fibonacci_computation_test_utils {
    use giza_air::{EvaluationFrame, FieldExtension, HashFunction};
    use giza_prover::StarkProof;
    use math::FieldElement;
    use winter_air::{
        AirContext, Assertion, ProofOptions, Table, TraceInfo, TransitionConstraintDegree,
    };
    use winter_prover::{ByteWriter, Prover, TraceTable};
    use winter_utils::TableReader;

    use super::*;

    // Public inputs for our computation will consist of the starting value and the end result.
    #[derive(Clone)]
    pub(crate) struct PublicInputs {
        pub a0: Felt,
        pub a1: Felt,
        pub result: Felt,
    }

    // We need to describe how public inputs can be converted to bytes.
    impl Serializable for PublicInputs {
        fn write_into<W: ByteWriter>(&self, target: &mut W) {
            target.write(self.a0);
            target.write(self.a1);
            target.write(self.result);
        }
    }

    /// Contains rows of the execution trace
    #[derive(Debug, Clone)]
    pub struct FibEvaluationFrame<E: FieldElement> {
        table: Table<E>, // row-major indexing
    }

    // FIBONACCI EVALUATION FRAME
    // ================================================================================================

    impl<E: FieldElement> EvaluationFrame<E> for FibEvaluationFrame<E> {
        // CONSTRUCTORS
        // --------------------------------------------------------------------------------------------
        fn new<A: Air>(air: &A) -> Self {
            let num_cols = air.trace_layout().main_trace_width();
            let num_rows = Self::num_rows();
            FibEvaluationFrame {
                table: Table::new(num_rows, num_cols),
            }
        }

        fn from_table(table: Table<E>) -> Self {
            Self { table }
        }

        // ROW MUTATORS
        // --------------------------------------------------------------------------------------------
        fn read_from<R: TableReader<E>>(
            &mut self,
            data: R,
            step: usize,
            offset: usize,
            blowup: usize,
        ) {
            let trace_len = data.num_rows();
            for (row, row_idx) in self.table.rows_mut().zip(Self::offsets().into_iter()) {
                for col_idx in 0..data.num_cols() {
                    row[col_idx + offset] =
                        data.get(col_idx, (step + row_idx * blowup) % trace_len);
                }
            }
        }

        // ROW ACCESSORS
        // --------------------------------------------------------------------------------------------
        fn row<'a>(&'a self, row_idx: usize) -> &'a [E] {
            &self.table.get_row(row_idx)
        }

        fn to_table(&self) -> Table<E> {
            self.table.clone()
        }

        fn offsets() -> &'static [usize] {
            &[0, 1, 2]
        }
    }

    impl<E: FieldElement> FibEvaluationFrame<E> {
        pub fn current<'a>(&'a self) -> &'a [E] {
            &self.table.get_row(0)
        }
        pub fn next<'a>(&'a self) -> &'a [E] {
            &self.table.get_row(1)
        }
        pub fn next_2<'a>(&'a self) -> &'a [E] {
            &self.table.get_row(2)
        }
    }

    // For a specific instance of our computation, we'll keep track of the public inputs and
    // the computation's context which we'll build in the constructor. The context is used
    // internally by the Winterfell prover/verifier when interpreting this AIR.
    pub(crate) struct FibAir {
        context: AirContext<Felt>,
        a0: Felt,
        a1: Felt,
        result: Felt,
    }

    impl Air for FibAir {
        // First, we'll specify which finite field to use for our computation, and also how
        // the public inputs must look like.
        type BaseField = Felt;
        type PublicInputs = PublicInputs;
        type Frame<E: FieldElement> = FibEvaluationFrame<E>;
        type AuxFrame<E: FieldElement> = FibEvaluationFrame<E>;

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
            let degrees = vec![TransitionConstraintDegree::new(4)];

            // We also need to specify the exact number of assertions we will place against the
            // execution trace. This number must be the same as the number of items in a vector
            // returned from the get_assertions() method below.
            let num_assertions = 3;

            let mut air_context = AirContext::new(trace_info, degrees, num_assertions, options);
            air_context = air_context.set_num_transition_exemptions(2);

            FibAir {
                context: air_context,
                a0: pub_inputs.a0,
                a1: pub_inputs.a1,
                result: pub_inputs.result,
            }
        }

        // In this method we'll define our transition constraints; a computation is considered to
        // be valid, if for all valid state transitions, transition constraints evaluate to all
        // zeros, and for any invalid transition, at least one constraint evaluates to a non-zero
        // value. The `frame` parameter will contain current and next states of the computation.
        fn evaluate_transition<E: math::FieldElement + From<Self::BaseField>>(
            &self,
            frame: &FibEvaluationFrame<E>,
            _periodic_values: &[E],
            result: &mut [E],
        ) {
            // First, we'll read the current state, and use it to compute the expected next state
            let a0 = &frame.current()[0];
            let a1 = &frame.next()[0];
            let a2 = *a0 + *a1;

            // Then, we'll subtract the expected next state from the actual next state; this will
            // evaluate to zero if and only if the expected and actual states are the same.
            result[0] = (frame.next_2()[0] - a2).into();
        }

        // Here, we'll define a set of assertions about the execution trace which must be satisfied
        // for the computation to be valid. Essentially, this ties computation's execution trace
        // to the public inputs.
        fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
            // for our computation to be valid, value in column 0 at step 0 must be equal to the
            // starting value, and at the last step it must be equal to the result.
            let last_step = self.trace_length() - 1;
            vec![
                Assertion::single(0, 0, self.a0),
                Assertion::single(0, 1, self.a1),
                Assertion::single(0, last_step, self.result),
            ]
        }

        // This is just boilerplate which is used by the Winterfell prover/verifier to retrieve
        // the context of the computation.
        fn context(&self) -> &AirContext<Self::BaseField> {
            &self.context
        }
    }

    struct FibProver {
        options: ProofOptions,
    }

    impl FibProver {
        pub fn new(options: ProofOptions) -> Self {
            Self { options }
        }
    }

    // When implementing Prover trait we set the `Air` associated type to the AIR of the
    // computation we defined previously, and set the `Trace` associated type to `TraceTable`
    // struct as we don't need to define a custom trace for our computation.
    impl Prover for FibProver {
        type BaseField = Felt;
        type Air = FibAir;
        type Trace = TraceTable<Self::BaseField>;

        // Our public inputs consist of the first and last value in the execution trace.
        fn get_pub_inputs(&self, trace: &Self::Trace) -> PublicInputs {
            let last_step = trace.length() - 1;
            PublicInputs {
                a0: trace.get(0, 0),
                a1: trace.get(0, 1),
                result: trace.get(0, last_step),
            }
        }

        fn options(&self) -> &ProofOptions {
            &self.options
        }
    }

    pub fn prove_work() -> (Felt, StarkProof) {
        // We'll just hard-code the parameters here for this example.
        let t = vec![vec![
            Felt::from(1u64),
            Felt::from(1u64),
            Felt::from(2u64),
            Felt::from(3u64),
            Felt::from(5u64),
            Felt::from(8u64),
            Felt::from(13u64),
            Felt::from(21u64),
        ]];

        let trace = TraceTable::init(t);

        let result = trace.get(0, trace.length() - 1);

        // Define proof options; these will be enough for ~96-bit security level.
        let options = winter_air::ProofOptions::new(
            32, // number of queries
            8,  // blowup factor
            0,  // grinding factor
            HashFunction::Blake2s_256,
            FieldExtension::None,
            8,   // FRI folding factor
            128, // FRI max remainder length
        );

        // Instantiate the prover and generate the proof.
        let prover = FibProver::new(options);
        let proof = prover.prove(trace).unwrap();

        (result, proof)
    }

    pub fn verify_work(a0: Felt, a1: Felt, result: Felt, proof: StarkProof) -> Result<(), ()> {
        // The number of steps and options are encoded in the proof itself, so we
        // don't need to pass them explicitly to the verifier.
        let pub_inputs = PublicInputs { a0, a1, result };
        match winter_verifier::verify::<FibAir>(proof, pub_inputs) {
            // The verication passed
            Ok(_) => Ok(()),
            // The verication did not pass
            Err(_) => Err(()),
        }
    }
}
