// Goldilocks quadratic extension field (Fp2) for Metal shaders.
//
// Extension field: Fp2 = Fp[x] / (x^2 - 7)
// Elements are represented as (a0, a1) meaning a0 + a1*w where w^2 = 7
//
// This shader supports FFT operations where:
// - Twiddles are in the base field (Goldilocks)
// - Coefficients are in the extension field (Fp2)

#ifndef fp2_goldilocks_h
#define fp2_goldilocks_h

#include "fp_u64.h.metal"

/// Quadratic extension of Goldilocks: Fp2 = Fp[w] / (w^2 - 7)
class Fp2Goldilocks {
public:
    Fp64Goldilocks c0;  // Coefficient of 1
    Fp64Goldilocks c1;  // Coefficient of w

    Fp2Goldilocks() = default;

    constexpr Fp2Goldilocks(Fp64Goldilocks _c0, Fp64Goldilocks _c1)
        : c0(_c0), c1(_c1) {}

    /// Construct from base field element (embed)
    constexpr explicit Fp2Goldilocks(Fp64Goldilocks x)
        : c0(x), c1(Fp64Goldilocks(0)) {}

    /// Extension field addition: (a0 + a1*w) + (b0 + b1*w) = (a0+b0) + (a1+b1)*w
    Fp2Goldilocks operator+(const Fp2Goldilocks rhs) const {
        return Fp2Goldilocks(c0 + rhs.c0, c1 + rhs.c1);
    }

    /// Extension field subtraction: (a0 + a1*w) - (b0 + b1*w) = (a0-b0) + (a1-b1)*w
    Fp2Goldilocks operator-(const Fp2Goldilocks rhs) const {
        return Fp2Goldilocks(c0 - rhs.c0, c1 - rhs.c1);
    }

    /// Multiplication by base field scalar: c * (a0 + a1*w) = (c*a0) + (c*a1)*w
    /// This is the key operation for FFT with base field twiddles
    Fp2Goldilocks scalar_mul(const Fp64Goldilocks scalar) const {
        return Fp2Goldilocks(c0 * scalar, c1 * scalar);
    }

    /// Full extension field multiplication (for completeness):
    /// (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
    Fp2Goldilocks operator*(const Fp2Goldilocks rhs) const {
        Fp64Goldilocks a0b0 = c0 * rhs.c0;
        Fp64Goldilocks a1b1 = c1 * rhs.c1;
        Fp64Goldilocks a0b1_plus_a1b0 = (c0 + c1) * (rhs.c0 + rhs.c1) - a0b0 - a1b1;

        // w^2 = 7, so a1*b1*w^2 = 7*a1*b1
        Fp64Goldilocks seven_a1b1 = mul_by_7(a1b1);

        return Fp2Goldilocks(a0b0 + seven_a1b1, a0b1_plus_a1b0);
    }

    /// Negation: -(a0 + a1*w) = (-a0) + (-a1)*w
    Fp2Goldilocks neg() const {
        return Fp2Goldilocks(c0.neg(), c1.neg());
    }

private:
    /// Multiply by 7 using shifts: 7*x = 8*x - x = (x << 3) - x
    static Fp64Goldilocks mul_by_7(Fp64Goldilocks x) {
        Fp64Goldilocks x2 = x + x;
        Fp64Goldilocks x4 = x2 + x2;
        Fp64Goldilocks x8 = x4 + x4;
        return x8 - x;
    }
};

#endif /* fp2_goldilocks_h */
