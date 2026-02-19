// Generic 32-bit Montgomery prime field for Metal shaders.
//
// Template parameters:
//   P       - the prime modulus
//   MU      - Montgomery parameter: -P^(-1) mod 2^32
//   R2      - Montgomery R^2 = 2^64 mod P
//   ONE_MONT - Montgomery representation of 1 = R mod P
//
// Arithmetic is performed in Montgomery form for efficient modular multiplication.
// Conversion: to Montgomery form multiply by R2 via montmul, from Montgomery form
// multiply by 1.

#ifndef fp_u32_h
#define fp_u32_h

#include <metal_stdlib>

template <uint32_t P, uint32_t MU, uint32_t R2, uint32_t ONE_MONT>
class Fp32 {
public:
    using raw_type = uint32_t;

    Fp32() = default;
    constexpr Fp32(uint32_t v) : inner(v) {}

    constexpr explicit operator uint32_t() const { return inner; }

    /// Raw inner value (Montgomery form).
    uint32_t raw() const { return inner; }

    /// Field addition: (a + b) mod P, inputs in Montgomery form.
    Fp32 operator+(const Fp32 rhs) const {
        uint32_t sum = inner + rhs.inner;
        // Conditional subtraction if sum >= P
        return Fp32(sum >= P ? sum - P : sum);
    }

    /// Field subtraction: (a - b) mod P, inputs in Montgomery form.
    Fp32 operator-(const Fp32 rhs) const {
        uint32_t diff = inner - rhs.inner;
        // If underflow, add P
        return Fp32(inner < rhs.inner ? diff + P : diff);
    }

    /// Field multiplication via Montgomery reduction.
    Fp32 operator*(const Fp32 rhs) const {
        return montmul(inner, rhs.inner);
    }

    /// Zero element in Montgomery form.
    static Fp32 zero() { return Fp32(0); }

    /// One element in Montgomery form.
    static Fp32 one() { return Fp32(ONE_MONT); }

    /// Convert from canonical representation to Montgomery form.
    static Fp32 from_canonical(uint32_t v) {
        return montmul(v, R2);
    }

    /// Convert from Montgomery form to canonical representation.
    uint32_t to_canonical() const {
        return montmul(inner, 1).inner;
    }

    /// Field negation.
    Fp32 neg() const {
        if (inner == 0) return Fp32(0);
        return Fp32(P - inner);
    }

private:
    uint32_t inner;

    /// Montgomery multiplication: returns (a * b * R^{-1}) mod P
    static Fp32 montmul(uint32_t a, uint32_t b) {
        uint64_t product = (uint64_t)a * (uint64_t)b;
        uint32_t lo = (uint32_t)product;
        uint32_t hi = (uint32_t)(product >> 32);

        // Montgomery reduction: t = (lo * MU) mod 2^32
        uint32_t t = lo * MU;
        // m = (t * P) >> 32
        uint64_t m = (uint64_t)t * (uint64_t)P;
        uint32_t m_hi = (uint32_t)(m >> 32);

        // result = hi - m_hi (with correction)
        uint32_t result = hi - m_hi;
        if (hi < m_hi) {
            result += P;
        }

        // Final reduction
        if (result >= P) {
            result -= P;
        }

        return Fp32(result);
    }
};

#endif /* fp_u32_h */
