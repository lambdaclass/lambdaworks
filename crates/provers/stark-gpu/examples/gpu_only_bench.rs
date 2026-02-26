//! GPU-only benchmark: runs ONLY prove_gpu_optimized (no CPU warmup).
//!
//! Run:
//!   cargo run -p lambdaworks-stark-gpu --features metal --release --example gpu_only_bench -- "14,16,18,20"

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::{
    element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field,
};
use stark_platinum_prover::{
    examples::fibonacci_rap::{fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs},
    proof::options::ProofOptions,
    traits::AIR,
};

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_stark_gpu::metal::prover::prove_gpu_optimized;

type F = Goldilocks64Field;
type FpE = FieldElement<F>;

fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        eprintln!("This benchmark requires macOS + metal feature");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        eprintln!("=== GPU-only benchmark (Keccak256 Merkle) ===\n");

        let sizes: Vec<usize> = std::env::args()
            .nth(1)
            .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| vec![14, 16, 18, 20]);

        // Warmup run (small) to compile Metal shaders
        {
            let trace_length: usize = 1 << 10;
            let pub_inputs = FibonacciRAPPublicInputs {
                steps: trace_length,
                a0: FpE::one(),
                a1: FpE::one(),
            };
            let proof_options = ProofOptions::default_test_options();
            let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
            let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
            let mut transcript = DefaultTranscript::<F>::new(&[]);
            let _ = prove_gpu_optimized(&air, &mut trace, &mut transcript);
            eprintln!("  (shader warmup done)\n");
        }

        eprintln!("{:<8} {:>12}", "Trace", "GPU_opt");
        eprintln!("{}", "-".repeat(24));

        for log_n in sizes {
            let trace_length: usize = 1 << log_n;
            let pub_inputs = FibonacciRAPPublicInputs {
                steps: trace_length,
                a0: FpE::one(),
                a1: FpE::one(),
            };
            let proof_options = ProofOptions::default_test_options();

            let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
            let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
            let mut transcript = DefaultTranscript::<F>::new(&[]);

            let start = std::time::Instant::now();
            let result = prove_gpu_optimized(&air, &mut trace, &mut transcript);
            let elapsed = start.elapsed();

            match result {
                Ok(_) => eprintln!("2^{:<5} {:>12.2?}", log_n, elapsed),
                Err(e) => eprintln!("2^{:<5} ERROR: {}", log_n, e),
            }
        }
    }
}
