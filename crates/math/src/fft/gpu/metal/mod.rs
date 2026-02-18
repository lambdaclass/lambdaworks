//! Metal GPU FFT module for lambdaworks.
//!
//! This module provides GPU-accelerated FFT operations using Apple's Metal framework.
//!
//! # Inspiration
//!
//! This implementation draws from:
//! - Original lambdaworks Metal implementation (removed in PR#993)
//! - [ICICLE](https://github.com/ingonyama-zk/icicle) - Multi-backend GPU acceleration
//! - [VkFFT](https://github.com/DTolm/VkFFT) - Efficient multi-backend FFT library
//! - [LambdaClass Metal FFT blog](https://blog.lambdaclass.com/using-metal-and-rust-to-make-fft-even-faster/)
//!
//! # Usage
//!
//! ## Base Field FFT
//!
//! ```ignore
//! use lambdaworks_math::fft::gpu::metal::ops::{fft, gen_twiddles};
//! use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
//! use lambdaworks_gpu::metal::abstractions::state::MetalState;
//!
//! // Initialize Metal state
//! let state = MetalState::new(None)?;
//!
//! // Generate twiddle factors on GPU
//! let twiddles = gen_twiddles::<Stark252PrimeField>(10, RootsConfig::BitReverse, &state)?;
//!
//! // Perform FFT on GPU
//! let result = fft(&input, &twiddles, &state)?;
//! ```
//!
//! ## Extension Field FFT
//!
//! For FFT with extension field coefficients and base field twiddles:
//!
//! ```ignore
//! use lambdaworks_math::fft::gpu::metal::ops::{fft_extension, gen_twiddles};
//! use lambdaworks_math::field::fields::u64_goldilocks_field::{
//!     Goldilocks64Field, Degree2GoldilocksExtensionField
//! };
//! use lambdaworks_gpu::metal::abstractions::state::MetalState;
//!
//! let state = MetalState::new(None)?;
//!
//! // Twiddles are in base Goldilocks field
//! let twiddles = gen_twiddles::<Goldilocks64Field>(10, RootsConfig::BitReverse, &state)?;
//!
//! // Input is in Fp2 extension field
//! let result = fft_extension::<Degree2GoldilocksExtensionField>(&fp2_input, &twiddles, &state)?;
//! ```

pub mod ops;

pub use ops::{
    bitrev_permutation, bitrev_permutation_extension, fft, fft_extension, fft_to_buffer,
    gen_twiddles, gen_twiddles_to_buffer, HasMetalExtensionKernel,
};
