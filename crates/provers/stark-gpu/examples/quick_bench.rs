use std::time::Instant;

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::{
    element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field,
};
use lambdaworks_stark_gpu::metal::prover::{prove_gpu, prove_gpu_optimized};
use stark_platinum_prover::{
    examples::fibonacci_rap::{fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs},
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    traits::AIR,
};

type F = Goldilocks64Field;
type FpE = FieldElement<F>;

fn main() {
    println!(
        "{:<8} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "Trace", "CPU", "GPU", "GPU_opt", "GPU/CPU", "opt/CPU"
    );
    println!("{}", "-".repeat(76));

    for log_len in [10, 12, 14, 16, 18] {
        let trace_length: usize = 1 << log_len;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();

        // CPU
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

        // GPU optimized
        let start = Instant::now();
        let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
        let mut transcript = DefaultTranscript::<F>::new(&[]);
        prove_gpu_optimized(&air, &mut trace, &mut transcript).unwrap();
        let gpu_opt_time = start.elapsed();

        let ratio_gpu = gpu_time.as_secs_f64() / cpu_time.as_secs_f64();
        let ratio_opt = gpu_opt_time.as_secs_f64() / cpu_time.as_secs_f64();
        println!(
            "2^{:<5} {:>12.2?} {:>12.2?} {:>12.2?} {:>10.2}x {:>10.2}x",
            log_len, cpu_time, gpu_time, gpu_opt_time, ratio_gpu, ratio_opt
        );
    }
}
