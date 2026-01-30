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
    game::{Feedback, Felt252, Guess, MastermindPublicInputs},
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
/// * `secret_commitment` - The commitment to the secret (public input)
///
/// # Returns
/// true if the proof is valid, false otherwise
pub fn verify_proof(
    proof: &Proof,
    guess: &Guess,
    feedback: &Feedback,
    secret_commitment: Felt252,
) -> bool {
    // Create public inputs for verification (with commitment, not actual secret)
    let pub_inputs = MastermindPublicInputs::for_verification(guess, feedback, secret_commitment);

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
/// * `secret_commitment` - The commitment to the secret
/// * `proof_options` - Custom proof options (must match prover)
pub fn verify_proof_with_options(
    proof: &Proof,
    guess: &Guess,
    feedback: &Feedback,
    secret_commitment: Felt252,
    proof_options: &ProofOptions,
) -> bool {
    let pub_inputs = MastermindPublicInputs::for_verification(guess, feedback, secret_commitment);
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
/// * `secret_commitment` - The commitment to the secret
/// * `trace_length` - The trace length used by the prover
pub fn verify_proof_with_trace_length(
    proof: &Proof,
    guess: &Guess,
    feedback: &Feedback,
    secret_commitment: Felt252,
    trace_length: usize,
) -> bool {
    let pub_inputs = MastermindPublicInputs::for_verification(guess, feedback, secret_commitment);
    let proof_options = ProofOptions::default_test_options();
    let air = MastermindAIR::<F>::new(trace_length, &pub_inputs, &proof_options);

    let transcript = &mut DefaultTranscript::<F>::new(&[]);
    Verifier::verify(proof, &air, transcript)
}

/// Compute the secret commitment for a given secret
/// This is a convenience function for the verifier to compute/verify commitments
pub fn compute_commitment(secret: &crate::game::SecretCode) -> Felt252 {
    crate::game::compute_secret_commitment(secret)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        game::{compute_secret_commitment, Color, SecretCode},
        prover::prove::generate_proof,
    };

    #[test]
    fn test_verify_valid_proof() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Yellow]);
        let feedback = Feedback::new(2, 2);

        // Compute the commitment (this would be shared before the game starts)
        let commitment = compute_secret_commitment(&secret);

        let proof = generate_proof(&secret, &guess, &feedback).expect("Proof generation failed");

        assert!(verify_proof(&proof, &guess, &feedback, commitment));
    }

    #[test]
    fn test_verify_with_wrong_feedback_fails() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Yellow]);
        let correct_feedback = Feedback::new(2, 2);
        let wrong_feedback = Feedback::new(4, 0); // Claiming all match (incorrect)

        let commitment = compute_secret_commitment(&secret);

        // Generate proof for correct feedback
        let proof =
            generate_proof(&secret, &guess, &correct_feedback).expect("Proof generation failed");

        // Verify with correct feedback should pass
        assert!(
            verify_proof(&proof, &guess, &correct_feedback, commitment.clone()),
            "Proof should verify with correct feedback"
        );

        // Verify with wrong feedback should fail because boundary constraints
        // check that the feedback in the proof matches the public input
        assert!(
            !verify_proof(&proof, &guess, &wrong_feedback, commitment),
            "Proof should NOT verify with wrong feedback (boundary constraint mismatch)"
        );
    }

    #[test]
    fn test_verify_with_wrong_guess_fails() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Yellow]);
        let feedback = Feedback::new(2, 2);

        let commitment = compute_secret_commitment(&secret);

        let proof = generate_proof(&secret, &guess, &feedback).expect("Proof generation failed");

        // Verify with a different guess should fail
        let wrong_guess = Guess::new([Color::Purple, Color::Purple, Color::Purple, Color::Purple]);
        assert!(
            !verify_proof(&proof, &wrong_guess, &feedback, commitment),
            "Proof should NOT verify with wrong guess (boundary constraint mismatch)"
        );
    }

    #[test]
    fn test_verify_with_wrong_commitment_fails() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Yellow]);
        let feedback = Feedback::new(2, 2);

        // Correct commitment
        let correct_commitment = compute_secret_commitment(&secret);

        // Wrong commitment (different secret)
        let wrong_secret =
            SecretCode::new([Color::Purple, Color::Purple, Color::Purple, Color::Purple]);
        let wrong_commitment = compute_secret_commitment(&wrong_secret);

        let proof = generate_proof(&secret, &guess, &feedback).expect("Proof generation failed");

        // Verify with correct commitment should pass
        assert!(
            verify_proof(&proof, &guess, &feedback, correct_commitment),
            "Proof should verify with correct commitment"
        );

        // Verify with wrong commitment should fail
        // This prevents the prover from lying about the secret
        assert!(
            !verify_proof(&proof, &guess, &feedback, wrong_commitment),
            "Proof should NOT verify with wrong commitment (commitment mismatch)"
        );
    }
}
