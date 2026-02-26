//! LogUp-GKR: efficient lookup arguments using the GKR protocol.
//!
//! Based on "Improving logarithmic derivative lookups using GKR"
//! by Shahar Papini and Ulrich Haböck (<https://eprint.iacr.org/2023/1284>).
//!
//! Implementation inspired by [stwo](https://github.com/starkware-libs/stwo/tree/dev/crates/stwo/src/prover/lookups).

pub mod eq_evals;
pub mod fraction;
pub mod fri;
pub mod layer;
pub mod prover;
pub mod univariate;
pub mod univariate_layer;

pub mod utils;
pub mod verifier;

/// Max degree of round polynomials in the GKR sumcheck.
pub const MAX_DEGREE: usize = 3;

pub use fraction::Fraction;
pub use lambdaworks_sumcheck::ProverError;
pub use layer::Layer;
pub use prover::{prove, prove_batch};
pub use univariate::iop::{prove_univariate, prove_with_pcs, verify_univariate, verify_with_pcs};
pub use univariate::pcs::{CommitmentSchemeError, IsUnivariateCommitmentScheme};
pub use univariate::types::{UnivariateIopError, UnivariateIopProof, UnivariateIopProofV2};
pub use verifier::{
    verify, verify_batch, BatchProof, BatchVerificationResult, Gate, Proof, SumcheckProof,
    VerificationResult,
};
