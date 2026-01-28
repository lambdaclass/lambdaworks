//! STARK Prover Profiling Binary
//!
//! This binary can be used with various profiling tools to analyze the STARK prover's
//! performance characteristics including memory usage and CPU time.
//!
//! # Memory Profiling with dhat
//!
//! Build and run with dhat heap profiling enabled:
//! ```sh
//! cargo build --release -p stark-platinum-prover --bench prover_profile --features dhat-heap
//! ./target/release/deps/prover_profile-* --trace-length 16
//! ```
//! This generates `dhat-heap.json` which can be viewed at https://nnethercote.github.io/dh_view/dh_view.html
//!
//! # CPU Profiling with samply
//!
//! ```sh
//! cargo build --release -p stark-platinum-prover --bench prover_profile
//! samply record ./target/release/deps/prover_profile-* --trace-length 18
//! ```
//!
//! # Flamegraph generation
//!
//! ```sh
//! cargo build --release -p stark-platinum-prover --bench prover_profile
//! # On macOS, you may need to run with sudo or use dtrace
//! cargo flamegraph --bench prover_profile -p stark-platinum-prover -- --trace-length 18
//! ```
//!
//! # Command-line options
//!
//! - `--trace-length <N>`: Set trace length to 2^N (default: 16, meaning 65536 rows)
//! - `--iterations <N>`: Number of prove iterations (default: 1)
//! - `--parallel`: Enable parallel proving (requires `parallel` feature)

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use stark_platinum_prover::{
    examples::simple_fibonacci::{fibonacci_trace, FibonacciAIR, FibonacciPublicInputs},
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    traits::AIR,
    transcript::StoneProverTranscript,
    verifier::{IsStarkVerifier, Verifier},
};

use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

type FE = FieldElement<Stark252PrimeField>;

fn parse_args() -> (u32, usize) {
    let args: Vec<String> = std::env::args().collect();
    let mut trace_length_exp: u32 = 16; // Default: 2^16 = 65536 rows
    let mut iterations: usize = 1;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--trace-length" => {
                i += 1;
                if i < args.len() {
                    trace_length_exp = args[i].parse().expect("Invalid trace length exponent");
                }
            }
            "--iterations" => {
                i += 1;
                if i < args.len() {
                    iterations = args[i].parse().expect("Invalid iterations count");
                }
            }
            "--parallel" => {
                // Flag is handled by feature, just consume it
            }
            "--help" | "-h" => {
                eprintln!("STARK Prover Profiling Tool");
                eprintln!();
                eprintln!("Usage: prover_profile [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --trace-length <N>  Trace length exponent (2^N rows, default: 16)");
                eprintln!("  --iterations <N>    Number of iterations (default: 1)");
                eprintln!("  --parallel          Enable parallel mode (requires feature)");
                eprintln!("  --help, -h          Show this help");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    (trace_length_exp, iterations)
}

fn run_prover(trace_length: usize) {
    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciPublicInputs {
        a0: FE::one(),
        a1: FE::one(),
    };

    // Generate trace
    let mut trace = fibonacci_trace([FE::one(), FE::one()], trace_length);

    // Create AIR
    let air =
        FibonacciAIR::<Stark252PrimeField>::new(trace.num_rows(), &pub_inputs, &proof_options);

    // Create transcript
    let mut transcript = StoneProverTranscript::new(&[]);

    // Run prover
    let proof = Prover::prove(&air, &mut trace, &mut transcript).expect("Proof generation failed");

    // Verify (optional, to ensure correctness)
    let mut verify_transcript = StoneProverTranscript::new(&[]);
    assert!(
        Verifier::verify(&proof, &air, &mut verify_transcript),
        "Proof verification failed"
    );
}

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let (trace_length_exp, iterations) = parse_args();
    let trace_length = 1usize << trace_length_exp;

    eprintln!("STARK Prover Profiling");
    eprintln!("======================");
    eprintln!(
        "Trace length: 2^{} = {} rows",
        trace_length_exp, trace_length
    );
    eprintln!("Iterations: {}", iterations);
    #[cfg(feature = "parallel")]
    eprintln!("Parallel: enabled");
    #[cfg(not(feature = "parallel"))]
    eprintln!("Parallel: disabled");
    #[cfg(feature = "dhat-heap")]
    eprintln!("dhat heap profiling: enabled");
    eprintln!();

    let start = std::time::Instant::now();

    for i in 0..iterations {
        if iterations > 1 {
            eprintln!("Iteration {}/{}", i + 1, iterations);
        }
        run_prover(trace_length);
    }

    let elapsed = start.elapsed();
    eprintln!();
    eprintln!("Total time: {:?}", elapsed);
    eprintln!("Average time per proof: {:?}", elapsed / iterations as u32);

    #[cfg(feature = "dhat-heap")]
    eprintln!("\ndhat heap profile written to dhat-heap.json");
}
