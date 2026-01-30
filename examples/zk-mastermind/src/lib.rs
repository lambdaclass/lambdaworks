//! ZK Mastermind - Zero-Knowledge Mastermind Game
//!
//! This crate implements a zero-knowledge version of the Mastermind game
//! using STARK proofs via the stark_platinum_prover from lambdaworks.
//!
//! # Overview
//!
//! In ZK Mastermind, the CodeMaker can prove that their feedback on a guess
//! is correct without revealing the secret code.
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
    calculate_feedback, verify_feedback, Color, Feedback, Felt252, Guess, MastermindPublicInputs,
    SecretCode,
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
