//! Differential fuzzer: GPU STARK prover vs CPU STARK prover.
//! Generates random-ish Fibonacci RAP traces and checks that GPU and CPU
//! produce identical proofs.
#![no_main]

use libfuzzer_sys::fuzz_target;

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::{element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field};
use stark_platinum_prover::{
    examples::fibonacci_rap::{fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs},
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    traits::AIR,
    verifier::{IsStarkVerifier, Verifier},
};
use lambdaworks_stark_gpu::metal::prover::prove_gpu;

type F = Goldilocks64Field;
type FpE = FieldElement<F>;

fuzz_target!(|data: (u64, u8)| {
    let (seed, trace_log2_raw) = data;

    // Cap trace length: min 2^3 = 8, max 2^8 = 256 (keep fuzzing fast)
    let trace_log2 = (trace_log2_raw % 6).max(3) + 3; // range: 3..8
    let trace_length: usize = 1 << trace_log2;

    // Derive a0, a1 from seed (just needs to be non-zero for valid trace)
    let a0 = FpE::from(seed.wrapping_add(1).max(1));
    let a1 = FpE::from(seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1).max(1));

    let pub_inputs = FibonacciRAPPublicInputs {
        steps: trace_length,
        a0: a0.clone(),
        a1: a1.clone(),
    };
    let proof_options = ProofOptions::default_test_options();

    // CPU proof
    let mut cpu_trace = fibonacci_rap_trace::<F>([a0.clone(), a1.clone()], trace_length);
    let air = FibonacciRAP::new(cpu_trace.num_rows(), &pub_inputs, &proof_options);
    let mut cpu_transcript = DefaultTranscript::<F>::new(&[]);
    let cpu_proof = match Prover::<F, F, _>::prove(&air, &mut cpu_trace, &mut cpu_transcript) {
        Ok(p) => p,
        Err(_) => return, // Invalid trace, skip
    };

    // GPU proof
    let mut gpu_trace = fibonacci_rap_trace::<F>([a0, a1], trace_length);
    let air = FibonacciRAP::new(gpu_trace.num_rows(), &pub_inputs, &proof_options);
    let mut gpu_transcript = DefaultTranscript::<F>::new(&[]);
    let gpu_proof = match prove_gpu(&air, &mut gpu_trace, &mut gpu_transcript) {
        Ok(p) => p,
        Err(e) => panic!("GPU prover failed but CPU succeeded: {:?}", e),
    };

    // Compare key proof fields
    assert_eq!(cpu_proof.trace_length, gpu_proof.trace_length, "trace_length mismatch");
    assert_eq!(cpu_proof.lde_trace_main_merkle_root, gpu_proof.lde_trace_main_merkle_root, "main merkle root mismatch");
    assert_eq!(cpu_proof.composition_poly_root, gpu_proof.composition_poly_root, "composition root mismatch");
    assert_eq!(cpu_proof.fri_layers_merkle_roots, gpu_proof.fri_layers_merkle_roots, "FRI roots mismatch");
    assert_eq!(cpu_proof.fri_last_value, gpu_proof.fri_last_value, "FRI last value mismatch");
    assert_eq!(cpu_proof.nonce, gpu_proof.nonce, "nonce mismatch");

    // Verify GPU proof
    let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
    assert!(
        Verifier::<F, F, _>::verify(&gpu_proof, &air, &mut verifier_transcript),
        "GPU proof failed verification"
    );
});
