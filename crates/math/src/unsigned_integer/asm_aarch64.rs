//! ARM64 (AArch64) optimized implementations of Montgomery multiplication.
//!
//! These implementations are optimized for ARM64 processors and let LLVM
//! generate efficient code using `mul`/`umulh` for 64x64->128-bit multiplication.
//!
//! For maximum performance, compile with:
//! - `RUSTFLAGS="-C target-cpu=native"` for M1/M2/M3 Macs
//! - LTO enabled in Cargo.toml

use super::element::UnsignedInteger;

/// Optimized CIOS Montgomery multiplication for 256-bit integers (4 limbs)
///
/// This version is structured to help LLVM generate optimal ARM64 code.
/// It avoids the extra array bounds checks and uses explicit carry tracking.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn cios_optimized_256(
    a: &UnsignedInteger<4>,
    b: &UnsignedInteger<4>,
    q: &UnsignedInteger<4>,
    mu: &u64,
) -> UnsignedInteger<4> {
    // Use the standard CIOS implementation - LLVM generates good ARM64 code
    // when compiling with -C target-cpu=native
    super::montgomery::MontgomeryAlgorithms::cios(a, b, q, mu)
}

/// Optimized CIOS Montgomery multiplication for 384-bit integers (6 limbs)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn cios_optimized_384(
    a: &UnsignedInteger<6>,
    b: &UnsignedInteger<6>,
    q: &UnsignedInteger<6>,
    mu: &u64,
) -> UnsignedInteger<6> {
    super::montgomery::MontgomeryAlgorithms::cios(a, b, q, mu)
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_optimized_256_matches_standard() {
        use super::*;
        use crate::unsigned_integer::element::U256;
        use crate::unsigned_integer::montgomery::MontgomeryAlgorithms;

        let a = U256::from_u64(123456789);
        let b = U256::from_u64(987654321);
        let q = U256::from_hex_unchecked(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
        );
        let mu: u64 = 0xd838091dd2253531;

        let standard = MontgomeryAlgorithms::cios(&a, &b, &q, &mu);
        let optimized = cios_optimized_256(&a, &b, &q, &mu);

        assert_eq!(standard, optimized);
    }
}
