pub mod common;
pub mod scs;
pub mod ssp;
pub mod utils;

mod prover;
mod setup;
mod verifier;

pub use prover::{Proof, Prover};
pub use setup::{setup, ProvingKey, VerifyingKey};
pub use verifier::verify;
