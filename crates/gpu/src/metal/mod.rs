//! Metal GPU backend for lambdaworks.
//!
//! This module provides GPU acceleration using Apple's Metal framework,
//! enabling high-performance cryptographic operations on Apple Silicon
//! and other Metal-compatible GPUs.
//!
//! # Inspiration
//!
//! This implementation draws from:
//! - Original lambdaworks Metal implementation (removed in PR#993)
//! - [ICICLE](https://github.com/ingonyama-zk/icicle) - Multi-backend GPU acceleration
//! - [ministark](https://github.com/andrewmilson/ministark) - Metal shader patterns
//! - [VkFFT](https://github.com/DTolm/VkFFT) - Multi-backend FFT library
//!
//! # Features
//!
//! - FFT/NTT operations on finite fields
//! - Twiddle factor generation
//! - Bit-reverse permutation
//! - Support for 256-bit fields (Stark252)
//!
//! # Usage
//!
//! Enable the `metal` feature in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! lambdaworks-gpu = { version = "...", features = ["metal"] }
//! ```

pub mod abstractions;
