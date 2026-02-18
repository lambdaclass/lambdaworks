//! LogUp-GKR: efficient lookup arguments using the GKR protocol.
//!
//! Based on "Improving logarithmic derivative lookups using GKR"
//! by Shahar Papini and Ulrich Haböck (<https://eprint.iacr.org/2023/1284>).

pub mod eq_evals;
pub mod fraction;
pub mod layer;
pub mod mle;
pub mod prover;
pub mod utils;
pub mod verifier;

pub use fraction::Fraction;
pub use layer::Layer;
pub use prover::{prove, prove_batch};
pub use verifier::{
    verify, verify_batch, BatchProof, BatchVerificationResult, Gate, Proof, SumcheckProof,
    VerificationResult,
};
