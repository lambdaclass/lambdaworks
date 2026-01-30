//! Game logic for ZK Mastermind
//!
//! This module contains the game types and rules for Mastermind.

pub mod rules;
pub mod types;

pub use rules::{calculate_feedback, verify_feedback};
pub use types::{Color, Feedback, Felt252, Guess, MastermindPublicInputs, SecretCode};
