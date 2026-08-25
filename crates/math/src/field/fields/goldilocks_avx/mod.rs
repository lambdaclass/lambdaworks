//! Packed Goldilocks field arithmetic.
//!
//! This module provides packed (multi-element) implementations of Goldilocks
//! field operations.  On x86-64 with AVX2 or AVX-512, SIMD-accelerated paths
//! are used; on all other platforms a portable scalar fallback is selected
//! automatically via the [`PackedGoldilocks`] type alias.
//!
//! # Implementations
//!
//! - **AVX2** (`avx2`): Processes 4 field elements in parallel (256-bit registers)
//! - **AVX-512** (`avx512`): Processes 8 field elements in parallel (512-bit registers)
//! - **Scalar** (`scalar`): Portable fallback, processes 4 elements sequentially
//!
//! # Supported Fields
//!
//! - **Goldilocks (Fp)**: Base field p = 2^64 - 2^32 + 1
//! - **Goldilocks Fp2**: Quadratic extension using x² - 7
//! - **Goldilocks Fp3**: Cubic extension using x³ - 2
//!
//! # Features
//!
//! Compile with `RUSTFLAGS="-C target-feature=+avx2"` for AVX2, or
//! `RUSTFLAGS="-C target-cpu=native"` to enable all CPU features.
//! Without these flags (or on non-x86_64 targets), the scalar fallback is used.
//!
//! # Inspiration
//!
//! The SIMD implementations are based on [Plonky3's Goldilocks AVX implementation](https://github.com/Plonky3/Plonky3/tree/main/goldilocks).
//!
//! # Usage
//!
//! ```ignore
//! use lambdaworks_math::field::fields::goldilocks_avx::PackedGoldilocks;
//!
//! // Uses AVX2 on x86_64 when available, scalar fallback otherwise
//! let a = PackedGoldilocks::from_u64_array([1, 2, 3, 4]);
//! let b = PackedGoldilocks::from_u64_array([5, 6, 7, 8]);
//! let c = a + b;
//! ```

// Base field implementations
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub mod avx2;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub mod avx512;

// Extension field implementations (require base field)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub mod avx2_ext;

// Portable scalar fallback (always available)
pub mod scalar;

// Re-exports for base field
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use avx2::PackedGoldilocksAVX2;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use avx512::PackedGoldilocksAVX512;

// Re-exports for extension fields
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use avx2_ext::{PackedGoldilocksFp2AVX2, PackedGoldilocksFp3AVX2};

pub use scalar::PackedGoldilocksScalar;

/// Auto-selecting type alias: uses AVX2 on x86_64 when available, scalar otherwise.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub type PackedGoldilocks = PackedGoldilocksAVX2;

/// Auto-selecting type alias: scalar fallback on non-AVX2 platforms.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub type PackedGoldilocks = PackedGoldilocksScalar;
