use std::time::Instant;

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::u64_goldilocks_field::{Degree3GoldilocksExtensionField, Goldilocks64Field},
};
use lambdaworks_math::helpers::resize_to_next_power_of_two;
use lambdaworks_stark_gpu::metal::prover::{prove_gpu, prove_gpu_fp3, prove_gpu_optimized};
use stark_platinum_prover::{
    constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
    constraints::transition::TransitionConstraint,
    context::AirContext,
    examples::fibonacci_rap::{fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs},
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    trace::TraceTable,
    traits::{TransitionEvaluationContext, AIR},
};

type F = Goldilocks64Field;
type FpE = FieldElement<F>;
type Fp3 = Degree3GoldilocksExtensionField;
type Fp3E = FieldElement<Fp3>;

// =========================================================================
// Fp3 extension field AIR (same as test infrastructure in prover.rs)
// =========================================================================

#[derive(Clone)]
struct FibConstraintFp3;

impl TransitionConstraint<F, Fp3> for FibConstraintFp3 {
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        0
    }

    fn end_exemptions(&self) -> usize {
        3 + 32 - 16 - 1
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, Fp3>,
        transition_evaluations: &mut [Fp3E],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover { frame, .. } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                let s2 = frame.get_evaluation_step(2);
                let a0 = s0.get_main_evaluation_element(0, 0);
                let a1 = s1.get_main_evaluation_element(0, 0);
                let a2 = s2.get_main_evaluation_element(0, 0);
                transition_evaluations[0] = (a2 - a1 - a0).to_extension::<Fp3>();
            }
            TransitionEvaluationContext::Verifier { frame, .. } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                let s2 = frame.get_evaluation_step(2);
                let a0 = s0.get_main_evaluation_element(0, 0);
                let a1 = s1.get_main_evaluation_element(0, 0);
                let a2 = s2.get_main_evaluation_element(0, 0);
                transition_evaluations[0] = a2 - a1 - a0;
            }
        }
    }
}

#[derive(Clone)]
struct PermutationConstraintFp3;

impl TransitionConstraint<F, Fp3> for PermutationConstraintFp3 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        1
    }

    fn end_exemptions(&self) -> usize {
        1
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, Fp3>,
        transition_evaluations: &mut [Fp3E],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                rap_challenges,
                ..
            } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                let z_i = s0.get_aux_evaluation_element(0, 0);
                let z_i_plus_one = s1.get_aux_evaluation_element(0, 0);
                let gamma = &rap_challenges[0];
                let a_i = s0.get_main_evaluation_element(0, 0);
                let b_i = s0.get_main_evaluation_element(0, 1);
                transition_evaluations[1] = z_i_plus_one * (b_i + gamma) - z_i * (a_i + gamma);
            }
            TransitionEvaluationContext::Verifier {
                frame,
                rap_challenges,
                ..
            } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                let z_i = s0.get_aux_evaluation_element(0, 0);
                let z_i_plus_one = s1.get_aux_evaluation_element(0, 0);
                let gamma = &rap_challenges[0];
                let a_i = s0.get_main_evaluation_element(0, 0);
                let b_i = s0.get_main_evaluation_element(0, 1);
                transition_evaluations[1] = z_i_plus_one * (b_i + gamma) - z_i * (a_i + gamma);
            }
        }
    }
}

struct FibonacciRAPFp3 {
    context: AirContext,
    trace_length: usize,
    pub_inputs: FibonacciRAPPublicInputs<F>,
    transition_constraints: Vec<Box<dyn TransitionConstraint<F, Fp3>>>,
}

impl AIR for FibonacciRAPFp3 {
    type Field = F;
    type FieldExtension = Fp3;
    type PublicInputs = FibonacciRAPPublicInputs<F>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        let transition_constraints: Vec<Box<dyn TransitionConstraint<F, Fp3>>> = vec![
            Box::new(FibConstraintFp3),
            Box::new(PermutationConstraintFp3),
        ];
        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 3,
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: transition_constraints.len(),
        };
        Self {
            context,
            trace_length,
            pub_inputs: pub_inputs.clone(),
            transition_constraints,
        }
    }

    fn build_auxiliary_trace(&self, trace: &mut TraceTable<F, Fp3>, challenges: &[Fp3E]) {
        let main_segment_cols = trace.columns_main();
        let not_perm = &main_segment_cols[0];
        let perm = &main_segment_cols[1];
        let gamma = &challenges[0];
        let trace_len = trace.num_rows();

        let mut denominators: Vec<Fp3E> = (0..trace_len - 1).map(|i| &perm[i] + gamma).collect();
        FieldElement::inplace_batch_inverse(&mut denominators)
            .expect("denominators are non-zero with high probability (gamma is random)");

        let mut aux_col: Vec<Fp3E> = Vec::with_capacity(trace_len);
        aux_col.push(Fp3E::one());
        for i in 0..trace_len - 1 {
            let z_i = &aux_col[i];
            let n_p_term = not_perm[i] + gamma;
            aux_col.push(z_i * &n_p_term * &denominators[i]);
        }
        for (i, aux_elem) in aux_col.iter().enumerate().take(trace.num_rows()) {
            trace.set_aux(i, 0, aux_elem.clone());
        }
    }

    fn build_rap_challenges(&self, transcript: &mut dyn IsStarkTranscript<Fp3, F>) -> Vec<Fp3E> {
        vec![transcript.sample_field_element()]
    }

    fn trace_layout(&self) -> (usize, usize) {
        (2, 1)
    }

    fn boundary_constraints(&self, _rap_challenges: &[Fp3E]) -> BoundaryConstraints<Fp3> {
        let a0 = BoundaryConstraint::new_simple_main(0, Fp3E::one());
        let a1 = BoundaryConstraint::new_simple_main(1, Fp3E::one());
        let a0_aux = BoundaryConstraint::new_aux(0, 0, Fp3E::one());
        BoundaryConstraints::from_constraints(vec![a0, a1, a0_aux])
    }

    fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraint<F, Fp3>>> {
        &self.transition_constraints
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length()
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}

fn fibonacci_rap_trace_fp3(initial_values: [FpE; 2], trace_length: usize) -> TraceTable<F, Fp3> {
    let mut fib_seq: Vec<FpE> = vec![];
    fib_seq.push(initial_values[0]);
    fib_seq.push(initial_values[1]);
    for i in 2..trace_length {
        fib_seq.push(fib_seq[i - 1] + fib_seq[i - 2]);
    }

    let last_value = fib_seq[trace_length - 1];
    let mut fib_permuted = fib_seq.clone();
    fib_permuted[0] = last_value;
    fib_permuted[trace_length - 1] = initial_values[0];

    fib_seq.push(FpE::zero());
    fib_permuted.push(FpE::zero());
    let mut trace_cols = vec![fib_seq, fib_permuted];
    resize_to_next_power_of_two(&mut trace_cols);

    let aux_columns = vec![vec![Fp3E::zero(); trace_cols[0].len()]];
    TraceTable::from_columns(trace_cols, aux_columns, 1)
}

// =========================================================================

fn main() {
    println!(
        "{:<8} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "Trace", "CPU", "GPU", "GPU_opt", "GPU_fp3", "GPU/CPU", "opt/CPU", "fp3/CPU"
    );
    println!("{}", "-".repeat(100));

    let sizes: Vec<usize> = std::env::args()
        .nth(1)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![10, 12, 14, 16, 18, 20, 22]);
    for log_len in sizes {
        let trace_length: usize = 1 << log_len;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();

        // CPU (base field F==E)
        let start = Instant::now();
        let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
        let mut transcript = DefaultTranscript::<F>::new(&[]);
        Prover::<F, F, _>::prove(&air, &mut trace, &mut transcript).unwrap();
        let cpu_time = start.elapsed();

        // GPU (generic)
        let start = Instant::now();
        let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
        let mut transcript = DefaultTranscript::<F>::new(&[]);
        prove_gpu(&air, &mut trace, &mut transcript).unwrap();
        let gpu_time = start.elapsed();

        // GPU optimized (F==E)
        let start = Instant::now();
        let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
        let mut transcript = DefaultTranscript::<F>::new(&[]);
        prove_gpu_optimized(&air, &mut trace, &mut transcript).unwrap();
        let gpu_opt_time = start.elapsed();

        // GPU Fp3 (extension field)
        let start = Instant::now();
        let mut trace_fp3 = fibonacci_rap_trace_fp3([FpE::one(), FpE::one()], trace_length);
        let air_fp3 = FibonacciRAPFp3::new(trace_fp3.num_rows(), &pub_inputs, &proof_options);
        let mut transcript_fp3 = DefaultTranscript::<Fp3>::new(&[]);
        prove_gpu_fp3(&air_fp3, &mut trace_fp3, &mut transcript_fp3).unwrap();
        let gpu_fp3_time = start.elapsed();

        let ratio_gpu = gpu_time.as_secs_f64() / cpu_time.as_secs_f64();
        let ratio_opt = gpu_opt_time.as_secs_f64() / cpu_time.as_secs_f64();
        let ratio_fp3 = gpu_fp3_time.as_secs_f64() / cpu_time.as_secs_f64();
        println!(
            "2^{:<5} {:>12.2?} {:>12.2?} {:>12.2?} {:>12.2?} {:>10.2}x {:>10.2}x {:>10.2}x",
            log_len,
            cpu_time,
            gpu_time,
            gpu_opt_time,
            gpu_fp3_time,
            ratio_gpu,
            ratio_opt,
            ratio_fp3
        );
    }
}
