//! Rescue-based hash functions over the Goldilocks field (p = 2^64 - 2^32 + 1).
//!
//! - **RPO** ([`Rpo256`]): Rescue Prime Optimized — 7 uniform full rounds.
//!   Paper: <https://eprint.iacr.org/2022/1577>
//! - **RPX** ([`Rpx256`]): Rescue Prime eXtension (XHash-12) — ~2x faster than RPO
//!   by replacing some full rounds with cubic extension field rounds.
//!   Paper: <https://eprint.iacr.org/2023/1045>
//!
//! Both share the same S-box (x^7 / x^{alpha_inv}), MDS circulant matrix,
//! and sponge construction via [`rescue_core::RescueCore`]. They differ only
//! in their permutation round structure.

mod parameters;
mod rescue_core;
mod rpo;
mod rpx;
mod utils;

pub use parameters::MdsMethod;
pub use parameters::SecurityLevel;
pub use rpo::Rpo256;
pub use rpx::Rpx256;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

pub type Fp = FieldElement<Goldilocks64Field>;
