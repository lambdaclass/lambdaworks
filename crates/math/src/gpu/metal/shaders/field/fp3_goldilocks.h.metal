// Goldilocks cubic extension field (Fp3) for Metal shaders.
//
// Extension field: Fp3 = Fp[x] / (x^3 - 2)
// Elements are represented as (a0, a1, a2) meaning a0 + a1*w + a2*w^2 where w^3 = 2
//
// This shader supports FFT operations where:
// - Twiddles are in the base field (Goldilocks)
// - Coefficients are in the extension field (Fp3)

#ifndef fp3_goldilocks_h
#define fp3_goldilocks_h

#include "fp_u64.h.metal"

/// Cubic extension of Goldilocks: Fp3 = Fp[w] / (w^3 - 2)
class Fp3Goldilocks {
public:
    Fp64Goldilocks c0;  // Coefficient of 1
    Fp64Goldilocks c1;  // Coefficient of w
    Fp64Goldilocks c2;  // Coefficient of w^2

    Fp3Goldilocks() = default;

    constexpr Fp3Goldilocks(Fp64Goldilocks _c0, Fp64Goldilocks _c1, Fp64Goldilocks _c2)
        : c0(_c0), c1(_c1), c2(_c2) {}

    /// Construct from base field element (embed)
    constexpr explicit Fp3Goldilocks(Fp64Goldilocks x)
        : c0(x), c1(Fp64Goldilocks(0)), c2(Fp64Goldilocks(0)) {}

    /// Extension field addition: component-wise
    Fp3Goldilocks operator+(const Fp3Goldilocks rhs) const {
        return Fp3Goldilocks(c0 + rhs.c0, c1 + rhs.c1, c2 + rhs.c2);
    }

    /// Extension field subtraction: component-wise
    Fp3Goldilocks operator-(const Fp3Goldilocks rhs) const {
        return Fp3Goldilocks(c0 - rhs.c0, c1 - rhs.c1, c2 - rhs.c2);
    }

    /// Multiplication by base field scalar: c * (a0 + a1*w + a2*w^2)
    /// This is the key operation for FFT with base field twiddles
    Fp3Goldilocks scalar_mul(const Fp64Goldilocks scalar) const {
        return Fp3Goldilocks(c0 * scalar, c1 * scalar, c2 * scalar);
    }

    /// Full extension field multiplication (for completeness):
    /// Uses w^3 = 2 for reduction
    ///
    /// (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2)
    /// = a0*b0 + (a0*b1 + a1*b0)*w + (a0*b2 + a1*b1 + a2*b0)*w^2
    ///   + (a1*b2 + a2*b1)*w^3 + a2*b2*w^4
    /// = a0*b0 + 2*(a1*b2 + a2*b1)  [w^3 = 2]
    ///   + (a0*b1 + a1*b0 + 2*a2*b2)*w  [w^4 = 2*w]
    ///   + (a0*b2 + a1*b1 + a2*b0)*w^2
    Fp3Goldilocks operator*(const Fp3Goldilocks rhs) const {
        Fp64Goldilocks a0b0 = c0 * rhs.c0;
        Fp64Goldilocks a0b1 = c0 * rhs.c1;
        Fp64Goldilocks a0b2 = c0 * rhs.c2;
        Fp64Goldilocks a1b0 = c1 * rhs.c0;
        Fp64Goldilocks a1b1 = c1 * rhs.c1;
        Fp64Goldilocks a1b2 = c1 * rhs.c2;
        Fp64Goldilocks a2b0 = c2 * rhs.c0;
        Fp64Goldilocks a2b1 = c2 * rhs.c1;
        Fp64Goldilocks a2b2 = c2 * rhs.c2;

        // w^3 = 2, so we multiply by 2 using doubling
        Fp64Goldilocks two_a1b2_a2b1 = (a1b2 + a2b1) + (a1b2 + a2b1);
        Fp64Goldilocks two_a2b2 = a2b2 + a2b2;

        return Fp3Goldilocks(
            a0b0 + two_a1b2_a2b1,           // coefficient of 1
            a0b1 + a1b0 + two_a2b2,         // coefficient of w
            a0b2 + a1b1 + a2b0              // coefficient of w^2
        );
    }

    /// Negation: component-wise
    Fp3Goldilocks neg() const {
        return Fp3Goldilocks(c0.neg(), c1.neg(), c2.neg());
    }
};

#endif /* fp3_goldilocks_h */
