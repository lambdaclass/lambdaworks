//! ZK Circuit for Mastermind
//!
//! This module contains the AIR (Algebraic Intermediate Representation),
//! trace generation, and constraint functions for the ZK Mastermind circuit.

pub mod air;
pub mod constraints;
pub mod trace;

pub use air::MastermindAIR;
pub use constraints::{
    compute_exact_matches, compute_partial_matches, is_equal, is_zero, range_check,
    verify_feedback_constraint,
};
pub use trace::{generate_computation_trace, generate_trace};
