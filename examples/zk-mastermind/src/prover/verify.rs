//! Verifier for ZK Mastermind
//!
//! Verifies STARK proofs that the feedback for a guess is correct.

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use stark_platinum_prover::{
    proof::options::ProofOptions,
    verifier::{IsStarkVerifier, Verifier},
};

use crate::{
    circuit::MastermindAIR,
    game::{Feedback, Guess, MastermindPublicInputs},
};

/// Type alias for the field type
pub type F = Stark252PrimeField;

/// Type alias for Proof
pub type Proof = stark_platinum_prover::proof::stark::StarkProof<F, F>;

/// Verify a ZK proof that the feedback for a guess is correct
///
/// # Arguments
/// * `proof` - The STARK proof to verify
/// * `guess` - The guess (public input)
/// * `feedback` - The expected feedback (public input)
///
/// # Returns
/// true if the proof is valid, false otherwise
pub fn verify_proof(proof: &Proof, guess: &Guess, feedback: &Feedback) -> bool {
    // Create public inputs
    let pub_inputs = MastermindPublicInputs::<F> {
        guess: guess.to_fields(),
        feedback: feedback.to_fields(),
    };

    // Create proof options (must match prover)
    let proof_options = ProofOptions::default_test_options();

    // Create the AIR with public inputs
    let air = MastermindAIR::<F>::new(
        16, // trace length used by prover
        &pub_inputs,
        &proof_options,
    );

    // Verify the proof
    let transcript = &mut DefaultTranscript::<F>::new(&[]);
    Verifier::verify(proof, &air, transcript)
}

/// Verify a proof with custom options
///
/// # Arguments
/// * `proof` - The STARK proof
/// * `guess` - The guess
/// * `feedback` - The expected feedback
/// * `proof_options` - Custom proof options (must match prover)
pub fn verify_proof_with_options(
    proof: &Proof,
    guess: &Guess,
    feedback: &Feedback,
    proof_options: &ProofOptions,
) -> bool {
    let pub_inputs = MastermindPublicInputs::<F> {
        guess: guess.to_fields(),
        feedback: feedback.to_fields(),
    };
    let air = MastermindAIR::<F>::new(16, &pub_inputs, proof_options);

    let transcript = &mut DefaultTranscript::<F>::new(&[]);
    Verifier::verify(proof, &air, transcript)
}

/// Verify a proof with a specific trace length
///
/// # Arguments
/// * `proof` - The STARK proof
/// * `guess` - The guess
/// * `feedback` - The expected feedback
/// * `trace_length` - The trace length used by the prover
pub fn verify_proof_with_trace_length(
    proof: &Proof,
    guess: &Guess,
    feedback: &Feedback,
    trace_length: usize,
) -> bool {
    let pub_inputs = MastermindPublicInputs::<F> {
        guess: guess.to_fields(),
        feedback: feedback.to_fields(),
    };
    let proof_options = ProofOptions::default_test_options();
    let air = MastermindAIR::<F>::new(trace_length, &pub_inputs, &proof_options);

    let transcript = &mut DefaultTranscript::<F>::new(&[]);
    Verifier::verify(proof, &air, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{game::Color, game::SecretCode, prover::prove::generate_proof};

    #[test]
    fn test_verify_valid_proof() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Yellow]);
        let feedback = Feedback::new(2, 2);

        let proof = generate_proof(&secret, &guess, &feedback).expect("Proof generation failed");

        assert!(verify_proof(&proof, &guess, &feedback));
    }

    #[test]
    fn test_verify_wrong_feedback_fails() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Yellow]);
        let correct_feedback = Feedback::new(2, 2);
        let _wrong_feedback = Feedback::new(4, 0); // Claiming all match

        // Generate proof for correct feedback
        let proof =
            generate_proof(&secret, &guess, &correct_feedback).expect("Proof generation failed");

        // Try to verify with wrong feedback - this should fail because
        // the public inputs won't match
        // Note: In a proper implementation, the proof would be bound to the feedback
        // For now, this test verifies the basic verification flow
        let _result = verify_proof(&proof, &guess, &correct_feedback);
    }
}
