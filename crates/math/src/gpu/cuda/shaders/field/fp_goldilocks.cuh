// Goldilocks field arithmetic for CUDA
// p = 2^64 - 2^32 + 1 (the "Goldilocks" prime)
//
// Key optimization: Uses EPSILON trick for fast modular reduction
// 2^64 ≡ 2^32 - 1 (mod p), so we call EPSILON = 2^32 - 1
//
// This allows reduction of 128-bit products without expensive division:
// (hi * 2^64 + lo) ≡ (hi * EPSILON + lo) (mod p)

#ifndef fp_goldilocks_h
#define fp_goldilocks_h

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define GOLDILOCKS_PRIME 0xFFFFFFFF00000001ULL
#define EPSILON 0xFFFFFFFFULL  // 2^32 - 1

class Fp64 {
public:
    uint64_t inner;

    Fp64() = default;

    __device__ constexpr Fp64(uint64_t v) : inner(v) {}

    __device__ constexpr explicit operator uint64_t() const { return inner; }

    // Addition: a + b mod p
    __device__ Fp64 operator+(const Fp64 rhs) const {
        uint64_t sum = inner + rhs.inner;
        uint64_t overflow = sum < inner;  // 1 if wrapped past 2^64

        // If overflow, we added 2^64 which ≡ EPSILON (mod p)
        uint64_t reduced = sum + overflow * EPSILON;

        // Handle potential second overflow from adding EPSILON
        uint64_t overflow2 = reduced < sum;
        reduced = reduced + overflow2 * EPSILON;

        // Final canonical reduction if >= p
        return Fp64(reduced >= GOLDILOCKS_PRIME ? reduced - GOLDILOCKS_PRIME : reduced);
    }

    // Subtraction: a - b mod p
    __device__ Fp64 operator-(const Fp64 rhs) const {
        uint64_t diff = inner - rhs.inner;
        uint64_t underflow = inner < rhs.inner;  // 1 if borrowed

        // If underflow: diff = inner + 2^64 - rhs.inner
        // We want inner - rhs.inner + p (to get positive result)
        // Since 2^64 = p + EPSILON, we have:
        //   inner + 2^64 - rhs.inner - EPSILON = inner + p - rhs.inner ✓
        // So: correct = diff - EPSILON (no second underflow since result > 0)
        return Fp64(underflow ? diff - EPSILON : diff);
    }

    // Multiplication: a * b mod p using EPSILON reduction
    __device__ Fp64 operator*(const Fp64 rhs) const {
        uint64_t lo = inner * rhs.inner;
        uint64_t hi = __umul64hi(inner, rhs.inner);
        return Fp64(reduce128(lo, hi));
    }

    // Exponentiation by squaring
    __device__ Fp64 pow(unsigned exp) const {
        Fp64 result(1);
        Fp64 base = *this;

        while (exp > 0) {
            if (exp & 1) {
                result = result * base;
            }
            exp >>= 1;
            base = base * base;
        }
        return result;
    }

    // Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
    // p - 2 = 0xFFFFFFFEFFFFFFFF
    //
    // Note: This uses variable-time exponentiation which is acceptable for FFT
    // operations where the values being inverted are not secret. For applications
    // requiring constant-time inversion, use a different implementation.
    __device__ Fp64 inverse() const {
        return pow_u64(0xFFFFFFFEFFFFFFFFULL);
    }

    __device__ Fp64 neg() const {
        return inner == 0 ? Fp64(0) : Fp64(GOLDILOCKS_PRIME - inner);
    }

private:
    // Power with 64-bit exponent
    __device__ Fp64 pow_u64(uint64_t exp) const {
        Fp64 result(1);
        Fp64 base = *this;

        while (exp > 0) {
            if (exp & 1) {
                result = result * base;
            }
            exp >>= 1;
            base = base * base;
        }
        return result;
    }

    // Reduce 128-bit number (lo, hi) to 64-bit mod p using EPSILON trick
    // x = hi * 2^64 + lo ≡ hi * EPSILON + lo (mod p)
    __device__ static uint64_t reduce128(uint64_t lo, uint64_t hi) {
        // hi * EPSILON can be up to (2^64-1) * (2^32-1) ≈ 2^96
        // So we compute it in parts

        // Split hi into 32-bit halves
        uint64_t hi_lo = hi & 0xFFFFFFFFULL;
        uint64_t hi_hi = hi >> 32;

        // hi * EPSILON = hi * (2^32 - 1) = (hi << 32) - hi
        // As a 96-bit number: (hi_hi, hi_lo << 32) - (0, hi)
        //                   = (hi_hi - borrow, (hi_lo << 32) - hi)

        uint64_t term1 = hi_lo << 32;
        uint64_t hi_eps_lo = term1 - hi;
        uint64_t borrow = term1 < hi;
        uint64_t hi_eps_hi = hi_hi - borrow;  // At most 32 bits

        // Now: x ≡ lo + hi_eps_hi * 2^64 + hi_eps_lo (mod p)
        //      ≡ lo + hi_eps_hi * EPSILON + hi_eps_lo (mod p)

        // Add lo + hi_eps_lo
        uint64_t sum1 = lo + hi_eps_lo;
        uint64_t c1 = sum1 < lo;

        // Add hi_eps_hi * EPSILON (fits in 64 bits since both factors < 2^32)
        uint64_t term2 = hi_eps_hi * EPSILON;
        uint64_t sum2 = sum1 + term2;
        uint64_t c2 = sum2 < sum1;

        // Handle carries: each carry means we added 2^64 ≡ EPSILON (mod p)
        uint64_t carry_sum = (c1 + c2) * EPSILON;
        uint64_t sum3 = sum2 + carry_sum;
        uint64_t c3 = sum3 < sum2;

        // One more potential carry
        uint64_t sum4 = sum3 + c3 * EPSILON;
        uint64_t c4 = sum4 < sum3;
        uint64_t sum5 = sum4 + c4 * EPSILON;

        // Final canonical reduction if >= p
        return sum5 >= GOLDILOCKS_PRIME ? sum5 - GOLDILOCKS_PRIME : sum5;
    }
};

namespace goldilocks {
    using Fp = Fp64;
} // namespace goldilocks

#endif // fp_goldilocks_h
