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
//! - `--air <NAME>`: AIR to use: fibonacci (default), read_only_memory (3 transition + 6 boundary constraints)

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use stark_platinum_prover::{
    examples::simple_fibonacci::{fibonacci_trace, FibonacciAIR, FibonacciPublicInputs},
    examples::read_only_memory::{sort_rap_trace, ReadOnlyPublicInputs, ReadOnlyRAP},
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

#[derive(Clone, Copy, Debug)]
enum AirType {
    Fibonacci,
    ReadOnlyMemory,
}

fn parse_args() -> (u32, usize, AirType) {
    let args: Vec<String> = std::env::args().collect();
    let mut trace_length_exp: u32 = 16; // Default: 2^16 = 65536 rows
    let mut iterations: usize = 1;
    let mut air_type = AirType::Fibonacci;

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
            "--air" => {
                i += 1;
                if i < args.len() {
                    air_type = match args[i].as_str() {
                        "fibonacci" | "fib" => AirType::Fibonacci,
                        "read_only_memory" | "rom" => AirType::ReadOnlyMemory,
                        _ => {
                            eprintln!("Unknown AIR type: {}. Use 'fibonacci' or 'read_only_memory'", args[i]);
                            std::process::exit(1);
                        }
                    };
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
                eprintln!("  --air <NAME>        AIR type: fibonacci (default), read_only_memory");
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

    (trace_length_exp, iterations, air_type)
}

fn run_fibonacci_prover(trace_length: usize) {
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

fn run_read_only_memory_prover(trace_length: usize) {
    let proof_options = ProofOptions::default_test_options();

    // Generate address and value columns
    // Create a simple read-only memory trace where addresses are sequential
    let address: Vec<FE> = (0..trace_length as u64).map(FE::from).collect();
    let value: Vec<FE> = (0..trace_length as u64).map(|i| FE::from(i * 10)).collect();

    // Create sorted trace
    let mut trace = sort_rap_trace(address.clone(), value.clone());

    let pub_inputs = ReadOnlyPublicInputs {
        a0: address[0].clone(),
        v0: value[0].clone(),
        a_sorted0: FE::from(0u64), // Sorted addresses start at 0
        v_sorted0: FE::from(0u64), // Sorted values start at 0
    };

    // Create AIR
    let air = ReadOnlyRAP::<Stark252PrimeField>::new(trace.num_rows(), &pub_inputs, &proof_options);

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

fn run_prover(trace_length: usize, air_type: AirType) {
    match air_type {
        AirType::Fibonacci => run_fibonacci_prover(trace_length),
        AirType::ReadOnlyMemory => run_read_only_memory_prover(trace_length),
    }
}

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let (trace_length_exp, iterations, air_type) = parse_args();
    let trace_length = 1usize << trace_length_exp;

    eprintln!("STARK Prover Profiling");
    eprintln!("======================");
    eprintln!(
        "Trace length: 2^{} = {} rows",
        trace_length_exp, trace_length
    );
    eprintln!("AIR: {:?}", air_type);
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
        run_prover(trace_length, air_type);
    }

    let elapsed = start.elapsed();
    eprintln!();
    eprintln!("Total time: {:?}", elapsed);
    eprintln!("Average time per proof: {:?}", elapsed / iterations as u32);

    #[cfg(feature = "dhat-heap")]
    eprintln!("\ndhat heap profile written to dhat-heap.json");
}
