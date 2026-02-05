//! Bandersnatch curve implementation.
//!
//! Bandersnatch is a twisted Edwards curve designed for efficient zero-knowledge
//! proofs. It is built over the scalar field of BLS12-381, making it suitable
//! for in-circuit operations when using BLS12-381 as the main curve.
//!
//! # Key Features
//!
//! - **Base field**: BLS12-381 scalar field (Fq)
//! - **Scalar field order**: 13108968793781547619861935127046491459309155893440570251786403306729687672801
//! - **Cofactor**: 4
//! - **Curve equation**: -5x² + y² = 1 + dx²y²
//!
//! # Modules
//!
//! - [`curve`] - Curve definition and point operations
//! - [`field`] - Base field (Fq) implementation
//! - [`scalar_field`] - Scalar field (Fr) implementation
//! - [`compression`] - Point compression/decompression
//!
//! # Example
//!
//! ```rust
//! use lambdaworks_math::elliptic_curve::edwards::curves::bandersnatch::curve::BandersnatchCurve;
//! use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
//! use lambdaworks_math::cyclic_group::IsGroup;
//!
//! // Get the generator
//! let g = BandersnatchCurve::generator();
//!
//! // Scalar multiplication
//! let p = g.operate_with_self(42u64);
//!
//! // Check subgroup membership
//! assert!(p.is_in_subgroup());
//! ```
//!
//! # References
//!
//! - [Bandersnatch paper (ePrint 2021/1152)](https://eprint.iacr.org/2021/1152)
//! - [Arkworks implementation](https://github.com/arkworks-rs/curves/tree/master/ed_on_bls12_381_bandersnatch)

pub mod compression;
pub mod curve;
pub mod field;
pub mod scalar_field;

// Re-export commonly used types
pub use compression::{compress, decompress, COMPRESSED_POINT_SIZE};
pub use curve::{
    BandersnatchCurve, BaseBandersnatchFieldElement, BANDERSNATCH_COFACTOR,
    BANDERSNATCH_COFACTOR_INV,
};
pub use field::{FqElement, FqField, BANDERSNATCH_PRIME_FIELD_ORDER};
pub use scalar_field::{FrElement, FrField, BANDERSNATCH_SUBGROUP_ORDER};
