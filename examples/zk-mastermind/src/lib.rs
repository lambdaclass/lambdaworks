//! ZK Mastermind - Zero-Knowledge Mastermind Game
//!
//! This crate implements a zero-knowledge version of the Mastermind game
//! using STARK proofs via the stark_platinum_prover from lambdaworks.
//!
//! # Educational Example
//!
//! This is an educational example demonstrating how to build STARK proofs
//! with lambdaworks.
//!
//! ## What this example demonstrates:
//! - How to define an AIR (Algebraic Intermediate Representation)
//! - How to generate execution traces
//! - How to use boundary and transition constraints
//! - How to generate and verify STARK proofs
//! - How to prevent prover cheating using commitments
//!
//! ## Security Model
//!
//! The circuit enforces:
//! - **Secret Commitment**: The prover commits to a secret before guesses begin.
//!   The commitment (s0 + s1*6 + s2*36 + s3*216) binds the prover to a specific secret.
//! - **Exact Match Verification**: Uses boolean equality indicators (eq_i) with:
//!   - Boolean constraint: eq_i * (1 - eq_i) = 0 (ensures 0 or 1)
//!   - Equality constraint: eq_i * (secret[i] - guess[i]) = 0 (if eq=1, must match)
//!   - Sum constraint: exact_count = eq_0 + eq_1 + eq_2 + eq_3
//! - **Range Checks**: All colors are validated to be in range [0, 5]
//!
//! ## Known Limitations
//!
//! - Partial match verification is NOT cryptographically enforced (would require
//!   additional color frequency counting constraints)
//! - No constant-time operations (timing side-channel vulnerable)
//! - Memory is not zeroized after use
//! - Uses `thread_rng()` instead of cryptographic RNG
//!
//! See [`circuit::air`] module documentation for constraint details.
//!
//! # Components
//!
//! - `game`: Game logic (types, rules for calculating feedback)
//! - `circuit`: ZK circuit (AIR, trace generation, constraints)
//! - `prover`: Prover and verifier for STARK proofs

pub mod circuit;
pub mod game;
pub mod prover;

// Re-export main types for convenience
pub use game::{
    calculate_feedback, compute_secret_commitment, verify_feedback, Color, Feedback, Felt252,
    Guess, MastermindPublicInputs, SecretCode,
};

pub use prover::{generate_proof, proof_size, verify_proof, Proof};

use rand::Rng;

/// Generate a random secret code
pub fn generate_random_secret() -> SecretCode {
    let mut rng = rand::thread_rng();
    let colors = [
        Color::Red,
        Color::Blue,
        Color::Green,
        Color::Yellow,
        Color::Orange,
        Color::Purple,
    ];

    SecretCode::new([
        colors[rng.gen_range(0..6)],
        colors[rng.gen_range(0..6)],
        colors[rng.gen_range(0..6)],
        colors[rng.gen_range(0..6)],
    ])
}

/// Game state for a Mastermind game
#[derive(Clone)]
pub struct GameState {
    pub secret: SecretCode,
}

impl GameState {
    /// Create a new game with the given secret
    pub fn new(secret: SecretCode) -> Self {
        Self { secret }
    }

    /// Create a new game with a random secret
    pub fn new_random() -> Self {
        Self::new(generate_random_secret())
    }

    /// Respond to a guess with feedback
    pub fn respond(&self, guess: &Guess) -> Feedback {
        calculate_feedback(&self.secret, guess)
    }

    /// Respond to a guess with feedback and ZK proof
    pub fn respond_with_proof(&self, guess: &Guess) -> Result<(Feedback, Proof), String> {
        let feedback = self.respond(guess);
        let proof = generate_proof(&self.secret, guess, &feedback)?;
        Ok((feedback, proof))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_state() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let game = GameState::new(secret);

        let guess = Guess::new([Color::Red, Color::Blue, Color::Orange, Color::Orange]);
        let feedback = game.respond(&guess);

        assert_eq!(feedback.exact, 2);
        assert_eq!(feedback.partial, 0);
    }

    #[test]
    fn test_random_secret() {
        let secret1 = generate_random_secret();
        let secret2 = generate_random_secret();

        // Very unlikely to be equal
        // This test mainly ensures the function doesn't panic
        assert_eq!(secret1.0.len(), 4);
        assert_eq!(secret2.0.len(), 4);
    }
}
