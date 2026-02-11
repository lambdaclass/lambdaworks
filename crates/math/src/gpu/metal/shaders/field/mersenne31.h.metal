// Mersenne31 prime field element arithmetic for Metal shaders.
//
// This implementation uses direct representation (NOT Montgomery form).
// The Mersenne prime p = 2^31 - 1 enables fast reduction via bit manipulation.
//
// Matches the CPU implementation in field/fields/mersenne31/field.rs.

#ifndef mersenne31_h
#define mersenne31_h

#include <metal_stdlib>

/// The Mersenne31 prime: p = 2^31 - 1 = 0x7FFFFFFF
constant uint32_t MERSENNE31_P = 0x7FFFFFFF;

/// Mersenne31 prime field element.
///
/// Values are stored as u32 with the invariant that the 31st bit is clear
/// and value < p. Uses the Mersenne prime structure for fast arithmetic.
class FpMersenne31 {
public:
    FpMersenne31() = default;
    constexpr FpMersenne31(uint32_t v) : inner(v) {}

    constexpr explicit operator uint32_t() const { return inner; }

    /// Zero element.
    static FpMersenne31 zero() { return FpMersenne31(0); }

    /// One element.
    static FpMersenne31 one() { return FpMersenne31(1); }

    /// Field addition: (a + b) mod p
    /// Matches field.rs lines 88-91: weak_reduce(a + b)
    FpMersenne31 operator+(const FpMersenne31 rhs) const {
        return FpMersenne31(weak_reduce(inner + rhs.inner));
    }

    /// Field subtraction: (a - b) mod p
    /// Matches field.rs lines 102-103: weak_reduce(a + P - b)
    FpMersenne31 operator-(const FpMersenne31 rhs) const {
        return FpMersenne31(weak_reduce(inner + MERSENNE31_P - rhs.inner));
    }

    /// Field multiplication: (a * b) mod p
    /// Matches field.rs lines 97-98: from_u64(u64(a) * u64(b))
    FpMersenne31 operator*(const FpMersenne31 rhs) const {
        return FpMersenne31(from_u64((uint64_t)inner * (uint64_t)rhs.inner));
    }

private:
    uint32_t inner;

    /// Clear MSB, add it back reduced.
    /// Matches field.rs lines 23-31: weak_reduce
    static uint32_t weak_reduce(uint32_t n) {
        uint32_t msb = n & (1u << 31);
        uint32_t msb_reduced = msb >> 31;
        uint32_t res = msb ^ n;
        return res + msb_reduced;
    }

    /// Reduce a u64 to a Mersenne31 field element.
    /// Matches field.rs lines 159-161: from_u64
    static uint32_t from_u64(uint64_t x) {
        return (uint32_t)((((((x >> 31) + x + 1) >> 31) + x) & (uint64_t)MERSENNE31_P));
    }
};

#endif /* mersenne31_h */
