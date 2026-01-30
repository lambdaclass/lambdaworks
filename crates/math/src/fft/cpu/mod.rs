pub mod bit_reversing;
/// Bowers G Network FFT implementation with optimizations:
/// - LayerTwiddles for cache-friendly sequential twiddle access
/// - 2-layer butterfly fusion to reduce memory traffic
/// - Internal parallelization via rayon for large inputs
///
/// Use `bowers_fft_opt_fused` for single-threaded workloads and
/// `bowers_fft_opt_fused_parallel` for multi-threaded workloads.
pub mod bowers_fft;
pub mod fft;
#[cfg(feature = "alloc")]
pub mod ops;
#[cfg(feature = "alloc")]
pub mod roots_of_unity;
