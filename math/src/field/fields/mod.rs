/// Implementation of two-adic prime fields to use with the Fast Fourier Transform (FFT).
pub mod fft_friendly;
/// Implementation of the 32-bit Mersenne Prime field (p = 2^31 - 1)
pub mod mersenne31;
pub mod montgomery_backed_prime_fields;
/// Implementation of the Goldilocks Prime field (p = 2^448 - 2^224 - 1)
pub mod p448_goldilocks_prime_field;
/// Implementation of the u64 Goldilocks Prime field (p = 2^64 - 2^32 + 1)
pub mod u64_goldilocks_field;
/// Implementation of prime fields over 64 bit unsigned integers.
pub mod u64_prime_field;

/// Winterfell and miden field compatibility
#[cfg(feature = "winter_compatibility")]
pub mod winterfell;
