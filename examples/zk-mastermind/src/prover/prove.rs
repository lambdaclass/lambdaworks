//! Prover for ZK Mastermind
//!
//! Generates STARK proofs that the feedback for a guess is correct
//! without revealing the secret code.

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use stark_platinum_prover::{
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
};

use crate::{
    circuit::{generate_computation_trace, MastermindAIR},
    game::{Feedback, Guess, MastermindPublicInputs, SecretCode},
};

/// Type alias for the field type
pub type F = Stark252PrimeField;

/// Type alias for StarkProof with F = E (no field extension)
pub type Proof = stark_platinum_prover::proof::stark::StarkProof<F, F>;

/// Generate a ZK proof that the feedback for a guess is correct
///
/// # Arguments
/// * `secret` - The secret code (private witness)
/// * `guess` - The guess (public input)
/// * `feedback` - The expected feedback (public input)
///
/// # Returns
/// A STARK proof that can be verified, along with the secret commitment
pub fn generate_proof(
    secret: &SecretCode,
    guess: &Guess,
    feedback: &Feedback,
) -> Result<Proof, String> {
    // Generate the trace table
    let mut trace = generate_computation_trace::<F>(secret, guess);

    // Create public inputs with secret commitment
    let pub_inputs = MastermindPublicInputs::new(secret, guess, feedback);

    // Create proof options
    let proof_options = ProofOptions::default_test_options();

    // Create the AIR
    let air = MastermindAIR::<F>::new(trace.num_rows(), &pub_inputs, &proof_options);

    // Generate the proof
    let transcript = &mut DefaultTranscript::<F>::new(&[]);
    Prover::prove(&air, &mut trace, transcript)
        .map_err(|e| format!("Proof generation failed: {:?}", e))
}

/// Generate a proof with custom options
///
/// # Arguments
/// * `secret` - The secret code
/// * `guess` - The guess
/// * `feedback` - The expected feedback
/// * `proof_options` - Custom proof options
pub fn generate_proof_with_options(
    secret: &SecretCode,
    guess: &Guess,
    feedback: &Feedback,
    proof_options: &ProofOptions,
) -> Result<Proof, String> {
    let mut trace = generate_computation_trace::<F>(secret, guess);
    let pub_inputs = MastermindPublicInputs::new(secret, guess, feedback);
    let air = MastermindAIR::<F>::new(trace.num_rows(), &pub_inputs, proof_options);

    let transcript = &mut DefaultTranscript::<F>::new(&[]);
    Prover::prove(&air, &mut trace, transcript)
        .map_err(|e| format!("Proof generation failed: {:?}", e))
}

/// Get the size of a proof in bytes
pub fn proof_size(proof: &Proof) -> usize {
    // Estimate: count the bytes in the serialized proof
    // This is a rough estimation based on proof structure
    let trace_commitment_size = 32; // Merkle root
    let composition_poly_size = 32; // Merkle root
    let fri_layers = 8 * 32; // FRI layer commitments
    let query_list_size = proof.query_list.len() * 256; // Approximate per query
    let deep_poly_size = proof.deep_poly_openings.len() * 512; // Approximate per opening

    trace_commitment_size + composition_poly_size + fri_layers + query_list_size + deep_poly_size
}

/// Serialize a proof to bytes (using serde)
pub fn serialize_proof(proof: &Proof) -> Vec<u8> {
    serde_json::to_vec(proof).unwrap_or_default()
}

/// Deserialize a proof from bytes
pub fn deserialize_proof(bytes: &[u8]) -> Result<Proof, String> {
    serde_json::from_slice(bytes).map_err(|e| format!("Deserialization failed: {:?}", e))
}
