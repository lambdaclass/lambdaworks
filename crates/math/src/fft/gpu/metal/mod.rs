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

pub mod ops;

pub use ops::{bitrev_permutation, fft, gen_twiddles};
