//! AVX2/AVX-512 optimized Goldilocks field arithmetic.
//!
//! This module provides SIMD-accelerated implementations of Goldilocks field
//! operations using x86-64 AVX2 and AVX-512 instructions.
//!
//! # Performance
//!
//! - AVX2: Processes 4 field elements in parallel (256-bit registers)
//! - AVX-512: Processes 8 field elements in parallel (512-bit registers)
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
//!
//! # Inspiration
//!
//! This implementation is based on [Plonky3's Goldilocks AVX implementation](https://github.com/Plonky3/Plonky3/tree/main/goldilocks).
//!
//! # Usage
//!
//! ```ignore
//! use lambdaworks_math::field::fields::goldilocks_avx::{
//!     PackedGoldilocksAVX2, PackedGoldilocksFp2AVX2
//! };
//!
//! // Base field: 4 parallel operations
//! let a = PackedGoldilocksAVX2::from_u64_array([1, 2, 3, 4]);
//! let b = PackedGoldilocksAVX2::from_u64_array([5, 6, 7, 8]);
//! let c = a + b; // SIMD parallel addition
//!
//! // Extension field: 4 Fp2 elements in parallel
//! let fp2_a = PackedGoldilocksFp2AVX2::from_fp2_array([...]);
//! let fp2_b = PackedGoldilocksFp2AVX2::from_fp2_array([...]);
//! let fp2_c = fp2_a * fp2_b; // SIMD parallel Fp2 multiplication
//! ```

// Base field implementations
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub mod avx2;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub mod avx512;

// Extension field implementations (require base field)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub mod avx2_ext;

// Re-exports for base field
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use avx2::PackedGoldilocksAVX2;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use avx512::PackedGoldilocksAVX512;

// Re-exports for extension fields
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use avx2_ext::{PackedGoldilocksFp2AVX2, PackedGoldilocksFp3AVX2};
