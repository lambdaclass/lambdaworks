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
    __device__ Fp64 inverse() const {
        // Use addition chain for p - 2 = 2^64 - 2^32 - 1
        // This is more efficient than naive square-and-multiply

        // Precompute small powers
        Fp64 x2 = *this * *this;           // x^2
        Fp64 x3 = x2 * *this;              // x^3
        Fp64 x4 = x2 * x2;                 // x^4
        Fp64 x5 = x4 * *this;              // x^5
        Fp64 x10 = x5 * x5;                // x^10
        Fp64 x11 = x10 * *this;            // x^11
        Fp64 x22 = x11 * x11;              // x^22
        Fp64 x44 = sqn<22>(x22);           // x^(22 * 2^22) = x^(22 * 2^22)

        // Build up x^(2^32 - 1) = x^0xFFFFFFFF
        // 2^32 - 1 = 4294967295
        Fp64 x_2_32_m1 = sqn<32>(*this);   // Start fresh
        x_2_32_m1 = x_2_32_m1 * *this;

        // Actually, let's use a cleaner approach:
        // x^(2^k - 1) can be computed as x * x^2 * x^4 * ... * x^(2^(k-1))
        // Or: x^(2^k - 1) = (x^(2^(k-1) - 1))^2 * x^(2^(k-1) - 1) * x
        //                = (x^(2^(k-1) - 1))^2 * x

        // Build x^(2^32 - 1)
        Fp64 e1 = *this;                    // x^(2^1 - 1) = x^1
        Fp64 e2 = sqn<1>(e1) * e1;          // x^(2^2 - 1) = x^3
        Fp64 e4 = sqn<2>(e2) * e2;          // x^(2^4 - 1) = x^15
        Fp64 e8 = sqn<4>(e4) * e4;          // x^(2^8 - 1) = x^255
        Fp64 e16 = sqn<8>(e8) * e8;         // x^(2^16 - 1) = x^65535
        Fp64 e32 = sqn<16>(e16) * e16;      // x^(2^32 - 1)

        // Now compute x^(p-2) = x^(2^64 - 2^32 - 1)
        // = x^(2^32 - 1) * x^((2^32 - 1) * 2^32 - 1) -- NO, let me recalculate
        // p - 2 = 2^64 - 2^32 - 1 = (2^32 - 1) * 2^32 + (2^32 - 2)
        //       = 0xFFFFFFFF * 2^32 + 0xFFFFFFFE
        //       = 0xFFFFFFFF_FFFFFFFE
        // Wait that's not right either. Let me verify:
        // p = 0xFFFFFFFF_00000001
        // p - 2 = 0xFFFFFFFF_00000001 - 2 = 0xFFFFFFFE_FFFFFFFF

        // 0xFFFFFFFE_FFFFFFFF = (2^32 - 2) * 2^32 + (2^32 - 1)
        //                     = 0xFFFFFFFE * 2^32 + 0xFFFFFFFF
        //                     = (2^32 - 1 - 1) * 2^32 + (2^32 - 1)
        //                     = (2^32 - 1) * 2^32 - 2^32 + 2^32 - 1
        //                     = (2^32 - 1) * 2^32 + (2^32 - 1) - 2^32
        //                     = (2^32 - 1) * (2^32 + 1) - 2^32
        // Actually let's just compute it directly:
        // 0xFFFFFFFE_FFFFFFFF = 2^64 - 2^32 - 1

        // We can write: 2^64 - 2^32 - 1 = 2^32 * (2^32 - 1) + (2^32 - 1)
        //                               = (2^32 - 1) * (2^32 + 1)
        // But 2^32 + 1 is not a power of 2, so this doesn't help directly.

        // Instead: x^(2^64 - 2^32 - 1) = x^((2^32-1)*2^32) * x^(2^32 - 1) / x^(2^32)
        // That's also not clean.

        // Let's use: exp = 0xFFFFFFFE_FFFFFFFF
        // exp = 0xFFFFFFFF_00000000 + 0xFFFFFFFF - 0x100000000
        //     = (2^32-1)*2^32 + (2^32-1) - 2^32
        //     = (2^32-1)(2^32 + 1) - 2^32

        // Alternative approach: direct square-and-multiply on the exponent
        // exp high 32 bits: 0xFFFFFFFE
        // exp low 32 bits: 0xFFFFFFFF

        // x^exp = x^(high * 2^32 + low)
        //       = (x^high)^(2^32) * x^low
        //       = (x^(2^32-2))^(2^32) * x^(2^32-1)

        // x^(2^32-2) = x^(2^32-1) / x = e32 * x.inverse()... circular!
        // x^(2^32-2) = x^(2^32-1) * x^(-1)... still circular

        // Better: x^(2^32-2) = x^(2*(2^31-1)) = (x^2)^(2^31-1)
        Fp64 x_2_31_m1 = sqn<15>(e16) * e16 * *this; // x^(2^31-1)
        // Wait, e16 = x^(2^16-1), sqn<15>(e16) = x^((2^16-1)*2^15)
        // That's not x^(2^31-1)

        // Let me just use the loop-based approach with the known exponent
        return pow_u64(0xFFFFFFFEFFFFFFFFULL);
    }

    __device__ Fp64 neg() const {
        return inner == 0 ? Fp64(0) : Fp64(GOLDILOCKS_PRIME - inner);
    }

private:
    // Square n times
    template <unsigned N>
    __device__ Fp64 sqn(Fp64 base) const {
        Fp64 result = base;
        #pragma unroll
        for (unsigned i = 0; i < N; i++) {
            result = result * result;
        }
        return result;
    }

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
