// 64-bit prime field element arithmetic for Metal shaders.
//
// This implementation is optimized for the Goldilocks prime p = 2^64 - 2^32 + 1.
// Uses direct representation (NOT Montgomery form) for efficiency.
//
// The key optimization exploits: 2^64 ≡ 2^32 - 1 (mod p)
// This allows fast reduction without expensive division.
//
// Inspired by Plonky3's Goldilocks implementation.

#ifndef fp_u64_h
#define fp_u64_h

#include <metal_stdlib>

/// EPSILON = 2^32 - 1, used for fast reduction since 2^64 ≡ EPSILON (mod p)
constant uint64_t GOLDILOCKS_EPSILON = 0xFFFFFFFF;

/// The Goldilocks prime: p = 2^64 - 2^32 + 1
constant uint64_t GOLDILOCKS_PRIME = 0xFFFFFFFF00000001;

/// Goldilocks 64-bit prime field element.
///
/// Values are stored as u64, canonicalized to [0, p) when needed.
/// Uses the special structure of p = 2^64 - 2^32 + 1 for fast arithmetic.
class Fp64Goldilocks {
public:
    Fp64Goldilocks() = default;
    constexpr Fp64Goldilocks(uint64_t v) : inner(v) {}

    constexpr explicit operator uint64_t() const { return inner; }

    /// Zero element.
    static Fp64Goldilocks zero() { return Fp64Goldilocks(0); }

    /// One element.
    static Fp64Goldilocks one() { return Fp64Goldilocks(1); }

    /// Field addition: (a + b) mod p
    /// If overflow occurs, we add EPSILON (since 2^64 ≡ EPSILON mod p)
    Fp64Goldilocks operator+(const Fp64Goldilocks rhs) const {
        uint64_t sum = inner + rhs.inner;
        bool over = sum < inner;  // Overflow detection

        // If overflow, add EPSILON
        uint64_t sum2 = sum + (over ? GOLDILOCKS_EPSILON : 0);
        bool over2 = over && (sum2 < sum);

        // Handle second overflow
        return Fp64Goldilocks(sum2 + (over2 ? GOLDILOCKS_EPSILON : 0));
    }

    /// Field subtraction: (a - b) mod p
    Fp64Goldilocks operator-(const Fp64Goldilocks rhs) const {
        uint64_t diff = inner - rhs.inner;
        bool under = inner < rhs.inner;  // Underflow detection

        // If underflow, subtract EPSILON (equivalent to adding p)
        uint64_t diff2 = diff - (under ? GOLDILOCKS_EPSILON : 0);
        bool under2 = under && (diff2 > diff);

        return Fp64Goldilocks(diff2 - (under2 ? GOLDILOCKS_EPSILON : 0));
    }

    /// Field multiplication: (a * b) mod p
    /// Uses 128-bit intermediate and fast reduction
    Fp64Goldilocks operator*(const Fp64Goldilocks rhs) const {
        // Compute 128-bit product
        uint64_t lo = inner * rhs.inner;
        uint64_t hi = metal::mulhi(inner, rhs.inner);

        // Reduce using: 2^64 ≡ EPSILON (mod p)
        return reduce128(lo, hi);
    }

    /// Exponentiation by squaring: a^exp mod p
    Fp64Goldilocks pow(uint32_t exp) const {
        Fp64Goldilocks base = *this;
        Fp64Goldilocks result = Fp64Goldilocks(1);

        while (exp > 0) {
            if (exp & 1) {
                result = result * base;
            }
            exp >>= 1;
            base = base * base;
        }

        return result;
    }

    /// Field inversion using Fermat's little theorem: a^(-1) = a^(p-2)
    /// Uses an optimized addition chain for p-2 = 2^64 - 2^32 - 1
    Fp64Goldilocks inverse() const {
        // Addition chain for a^(p-2) where p-2 = 0xFFFFFFFE_FFFFFFFF
        Fp64Goldilocks x = *this;
        Fp64Goldilocks x2 = x * x;
        Fp64Goldilocks x3 = x2 * x;
        Fp64Goldilocks x7 = exp_acc(x3, x, 1);
        Fp64Goldilocks x63 = exp_acc(x7, x7, 3);
        Fp64Goldilocks x12m1 = exp_acc(x63, x63, 6);
        Fp64Goldilocks x24m1 = exp_acc(x12m1, x12m1, 12);
        Fp64Goldilocks x30m1 = exp_acc(x24m1, x63, 6);
        Fp64Goldilocks x31m1 = exp_acc(x30m1, x, 1);
        Fp64Goldilocks x32m1 = exp_acc(x31m1, x, 1);

        Fp64Goldilocks t = x31m1;
        for (int i = 0; i < 33; i++) {
            t = t * t;
        }

        return t * x32m1;
    }

    /// Field negation: -a mod p
    Fp64Goldilocks neg() const {
        uint64_t canonical = canonicalize();
        if (canonical == 0) {
            return Fp64Goldilocks(0);
        }
        return Fp64Goldilocks(GOLDILOCKS_PRIME - canonical);
    }

    /// Canonicalize to [0, p)
    uint64_t canonicalize() const {
        if (inner >= GOLDILOCKS_PRIME) {
            return inner - GOLDILOCKS_PRIME;
        }
        return inner;
    }

private:
    uint64_t inner;

    /// Helper: square n times then multiply by tail
    Fp64Goldilocks exp_acc(Fp64Goldilocks base, Fp64Goldilocks tail, uint32_t n) const {
        Fp64Goldilocks result = base;
        for (uint32_t i = 0; i < n; i++) {
            result = result * result;
        }
        return result * tail;
    }

    /// Reduce a 128-bit value (lo, hi) to a 64-bit Goldilocks element.
    /// Uses: 2^64 ≡ 2^32 - 1 (mod p)
    static Fp64Goldilocks reduce128(uint64_t lo, uint64_t hi) {
        // Split hi into hi_hi (upper 32 bits) and hi_lo (lower 32 bits)
        uint64_t hi_hi = hi >> 32;
        uint64_t hi_lo = hi & GOLDILOCKS_EPSILON;

        // Step 1: t0 = lo - hi_hi (since hi_hi * 2^64 ≡ hi_hi * EPSILON)
        uint64_t t0 = lo - hi_hi;
        bool borrow = lo < hi_hi;
        t0 = borrow ? t0 - GOLDILOCKS_EPSILON : t0;

        // Step 2: t1 = hi_lo * EPSILON = (hi_lo << 32) - hi_lo
        uint64_t t1 = (hi_lo << 32) - hi_lo;

        // Step 3: result = t0 + t1
        uint64_t result = t0 + t1;
        bool carry = result < t0;
        result = carry ? result + GOLDILOCKS_EPSILON : result;

        return Fp64Goldilocks(result);
    }
};

#endif /* fp_u64_h */
