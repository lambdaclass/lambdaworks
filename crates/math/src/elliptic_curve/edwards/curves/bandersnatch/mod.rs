//! Bandersnatch curve implementation.
//!
//! Bandersnatch is a twisted Edwards curve built over the scalar field of BLS12-381,
//! making it suitable for in-circuit operations when using BLS12-381 as the main curve.
//!
//! # Parameters
//!
//! - **Base field**: BLS12-381 scalar field (Fr) - reused from `bls12_381::default_types`
//! - **Subgroup order**: 13108968793781547619861935127046491459309155893440570251786403306729687672801
//! - **Cofactor**: 4
//! - **Curve equation**: -5x² + y² = 1 + dx²y²
//!
//! # References
//!
//! - [Bandersnatch paper (ePrint 2021/1152)](https://eprint.iacr.org/2021/1152)
//! - [Arkworks implementation](https://github.com/arkworks-rs/curves/tree/master/ed_on_bls12_381_bandersnatch)

pub mod compression;
pub mod curve;
pub mod scalar_field;

pub use compression::{compress, decompress, COMPRESSED_POINT_SIZE};
pub use curve::{BandersnatchBaseField, BandersnatchCurve, BANDERSNATCH_COFACTOR};
pub use scalar_field::BANDERSNATCH_SUBGROUP_ORDER;
