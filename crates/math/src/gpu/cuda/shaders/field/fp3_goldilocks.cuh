// Goldilocks cubic extension field arithmetic for CUDA
// Fp3 = Fp[w] / (w^3 - 2), where Fp is the Goldilocks prime field
// Elements are represented as a0 + a1*w + a2*w^2 where w^3 = 2
//
// 2 is a cubic non-residue in the Goldilocks field.

#ifndef fp3_goldilocks_h
#define fp3_goldilocks_h

#include "fp_goldilocks.cuh"

class Fp3_64 {
public:
    Fp64 c0;  // Coefficient of 1
    Fp64 c1;  // Coefficient of w
    Fp64 c2;  // Coefficient of w^2

    Fp3_64() = default;

    __device__ constexpr Fp3_64(Fp64 a0, Fp64 a1, Fp64 a2) : c0(a0), c1(a1), c2(a2) {}

    __device__ constexpr Fp3_64(uint64_t val) : c0(val), c1(0), c2(0) {}

    // Addition: component-wise
    __device__ Fp3_64 operator+(const Fp3_64 rhs) const {
        return Fp3_64(c0 + rhs.c0, c1 + rhs.c1, c2 + rhs.c2);
    }

    // Subtraction: component-wise
    __device__ Fp3_64 operator-(const Fp3_64 rhs) const {
        return Fp3_64(c0 - rhs.c0, c1 - rhs.c1, c2 - rhs.c2);
    }

    // Multiplication using Karatsuba-like method
    // (a0 + a1*w + a2*w^2)(b0 + b1*w + b2*w^2) mod (w^3 - 2)
    //
    // v0 = a0*b0
    // v1 = a1*b1
    // v2 = a2*b2
    //
    // c0 = v0 + 2*((a1+a2)(b1+b2) - v1 - v2)
    // c1 = (a0+a1)(b0+b1) - v0 - v1 + 2*v2
    // c2 = (a0+a2)(b0+b2) - v0 + v1 - v2
    __device__ Fp3_64 operator*(const Fp3_64 rhs) const {
        Fp64 v0 = c0 * rhs.c0;
        Fp64 v1 = c1 * rhs.c1;
        Fp64 v2 = c2 * rhs.c2;

        Fp64 t0 = (c1 + c2) * (rhs.c1 + rhs.c2) - v1 - v2;  // (a1+a2)(b1+b2) - v1 - v2
        Fp64 t1 = (c0 + c1) * (rhs.c0 + rhs.c1) - v0 - v1;  // (a0+a1)(b0+b1) - v0 - v1
        Fp64 t2 = (c0 + c2) * (rhs.c0 + rhs.c2) - v0 - v2;  // (a0+a2)(b0+b2) - v0 - v2

        // 2*x = x + x
        Fp64 t0_2 = t0 + t0;
        Fp64 v2_2 = v2 + v2;

        return Fp3_64(
            v0 + t0_2,       // c0 = v0 + 2*t0
            t1 + v2_2,       // c1 = t1 + 2*v2
            t2 + v1          // c2 = t2 + v1
        );
    }

    // Squaring
    // (a0 + a1*w + a2*w^2)^2 mod (w^3 - 2)
    //
    // s0 = a0^2
    // s1 = a1^2
    // s2 = a2^2
    // a01 = a0*a1
    // a02 = a0*a2
    // a12 = a1*a2
    //
    // c0 = s0 + 4*a12    (since 2*w^3 = 4, and a12*w^3 = a12*2)
    // c1 = 2*a01 + 2*s2
    // c2 = 2*a02 + s1
    __device__ Fp3_64 square() const {
        Fp64 s0 = c0 * c0;
        Fp64 s1 = c1 * c1;
        Fp64 s2 = c2 * c2;
        Fp64 a01 = c0 * c1;
        Fp64 a02 = c0 * c2;
        Fp64 a12 = c1 * c2;

        // 4*a12 = 2*(2*a12)
        Fp64 a12_2 = a12 + a12;
        Fp64 a12_4 = a12_2 + a12_2;

        // 2*a01, 2*s2, 2*a02
        Fp64 a01_2 = a01 + a01;
        Fp64 s2_2 = s2 + s2;
        Fp64 a02_2 = a02 + a02;

        return Fp3_64(
            s0 + a12_4,     // c0 = s0 + 4*a12
            a01_2 + s2_2,   // c1 = 2*a01 + 2*s2
            a02_2 + s1      // c2 = 2*a02 + s1
        );
    }

    // Exponentiation by squaring
    __device__ Fp3_64 pow(unsigned exp) const {
        Fp3_64 result(Fp64(1), Fp64(0), Fp64(0));
        Fp3_64 base = *this;

        while (exp > 0) {
            if (exp & 1) {
                result = result * base;
            }
            exp >>= 1;
            base = base.square();
        }
        return result;
    }

    // Multiplicative inverse
    // Using the formula for inverse in cubic extensions:
    // norm = a0^3 + 2*a1^3 + 4*a2^3 - 6*a0*a1*a2
    //
    // inv[0] = (a0^2 - 2*a1*a2) / norm
    // inv[1] = (2*a2^2 - a0*a1) / norm
    // inv[2] = (a1^2 - a0*a2) / norm
    //
    // The inverse of zero is undefined; caller must ensure the element is nonzero.
    __device__ Fp3_64 inverse() const {
        Fp64 a0_sq = c0 * c0;
        Fp64 a1_sq = c1 * c1;
        Fp64 a2_sq = c2 * c2;

        Fp64 a0_cubed = a0_sq * c0;
        Fp64 a1_cubed = a1_sq * c1;
        Fp64 a2_cubed = a2_sq * c2;

        Fp64 a0a1 = c0 * c1;
        Fp64 a0a2 = c0 * c2;
        Fp64 a1a2 = c1 * c2;
        Fp64 a0a1a2 = a0a1 * c2;

        // 2*a1^3, 4*a2^3, 6*a0a1a2
        Fp64 a1_cubed_2 = a1_cubed + a1_cubed;
        Fp64 a2_cubed_2 = a2_cubed + a2_cubed;
        Fp64 a2_cubed_4 = a2_cubed_2 + a2_cubed_2;

        Fp64 a0a1a2_2 = a0a1a2 + a0a1a2;
        Fp64 a0a1a2_4 = a0a1a2_2 + a0a1a2_2;
        Fp64 a0a1a2_6 = a0a1a2_2 + a0a1a2_4;

        // norm = a0^3 + 2*a1^3 + 4*a2^3 - 6*a0*a1*a2
        Fp64 norm = a0_cubed + a1_cubed_2 + a2_cubed_4 - a0a1a2_6;
        assert(norm.inner != 0 && "Fp3_64::inverse() called on zero element");
        Fp64 norm_inv = norm.inverse();

        // 2*a1a2, 2*a2_sq
        Fp64 a1a2_2 = a1a2 + a1a2;
        Fp64 a2_sq_2 = a2_sq + a2_sq;

        // inv[0] = (a0^2 - 2*a1*a2) / norm
        // inv[1] = (2*a2^2 - a0*a1) / norm
        // inv[2] = (a1^2 - a0*a2) / norm
        return Fp3_64(
            (a0_sq - a1a2_2) * norm_inv,
            (a2_sq_2 - a0a1) * norm_inv,
            (a1_sq - a0a2) * norm_inv
        );
    }

    // Negation: component-wise
    __device__ Fp3_64 neg() const {
        return Fp3_64(c0.neg(), c1.neg(), c2.neg());
    }

    // Multiply by base field element
    __device__ Fp3_64 mul_by_fp(const Fp64 scalar) const {
        return Fp3_64(c0 * scalar, c1 * scalar, c2 * scalar);
    }
};

namespace goldilocks {
    using Fp3 = Fp3_64;
} // namespace goldilocks

#endif // fp3_goldilocks_h
