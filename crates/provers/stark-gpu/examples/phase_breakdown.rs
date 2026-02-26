//! Quick phase-breakdown benchmark for GPU STARK prover.
//!
//! Run with:
//!   cargo run -p lambdaworks-stark-gpu --features metal --release --example phase_breakdown

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
        for log_n in [10, 12, 14, 16, 18, 20] {
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

            eprintln!("=== Trace 2^{log_n} ({trace_length} rows) ===");
            let total = std::time::Instant::now();
            let proof = prove_gpu_optimized(&air, &mut trace, &mut transcript);
            let elapsed = total.elapsed();
            match proof {
                Ok(_) => eprintln!("  TOTAL:                 {elapsed:>10.2?}"),
                Err(e) => eprintln!("  ERROR: {e}"),
            }
            eprintln!();
        }
    }
}
