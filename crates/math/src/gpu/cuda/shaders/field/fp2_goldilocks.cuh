// Goldilocks quadratic extension field arithmetic for CUDA
// Fp2 = Fp[w] / (w^2 - 7), where Fp is the Goldilocks prime field
// Elements are represented as a0 + a1*w where w^2 = 7
//
// 7 is a quadratic non-residue in the Goldilocks field.

#ifndef fp2_goldilocks_h
#define fp2_goldilocks_h

#include "fp_goldilocks.cuh"

class Fp2_64 {
public:
    Fp64 c0;  // Real part
    Fp64 c1;  // Imaginary part (coefficient of w)

    Fp2_64() = default;

    __device__ constexpr Fp2_64(Fp64 real, Fp64 imag) : c0(real), c1(imag) {}

    __device__ constexpr Fp2_64(uint64_t real) : c0(real), c1(0) {}

    // Addition: (a0 + a1*w) + (b0 + b1*w) = (a0+b0) + (a1+b1)*w
    __device__ Fp2_64 operator+(const Fp2_64 rhs) const {
        return Fp2_64(c0 + rhs.c0, c1 + rhs.c1);
    }

    // Subtraction: (a0 + a1*w) - (b0 + b1*w) = (a0-b0) + (a1-b1)*w
    __device__ Fp2_64 operator-(const Fp2_64 rhs) const {
        return Fp2_64(c0 - rhs.c0, c1 - rhs.c1);
    }

    // Multiplication: (a0 + a1*w)(b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
    // Uses Karatsuba-like optimization:
    //   z0 = a0*b0
    //   z1 = a1*b1
    //   z2 = (a0+a1)*(b0+b1) = a0*b0 + a0*b1 + a1*b0 + a1*b1
    //   c1 = z2 - z0 - z1 = a0*b1 + a1*b0
    //   c0 = z0 + 7*z1
    __device__ Fp2_64 operator*(const Fp2_64 rhs) const {
        Fp64 z0 = c0 * rhs.c0;
        Fp64 z1 = c1 * rhs.c1;
        Fp64 z2 = (c0 + c1) * (rhs.c0 + rhs.c1);

        // 7*z1 using 7 = 1 + 2 + 4
        Fp64 z1_2 = z1 + z1;
        Fp64 z1_4 = z1_2 + z1_2;
        Fp64 z1_7 = z1 + z1_2 + z1_4;

        return Fp2_64(z0 + z1_7, z2 - z0 - z1);
    }

    // Squaring: (a0 + a1*w)^2 = (a0^2 + 7*a1^2) + 2*a0*a1*w
    __device__ Fp2_64 square() const {
        Fp64 a0_sq = c0 * c0;
        Fp64 a1_sq = c1 * c1;
        Fp64 a0a1 = c0 * c1;

        // 7*a1_sq
        Fp64 a1_sq_2 = a1_sq + a1_sq;
        Fp64 a1_sq_4 = a1_sq_2 + a1_sq_2;
        Fp64 a1_sq_7 = a1_sq + a1_sq_2 + a1_sq_4;

        return Fp2_64(a0_sq + a1_sq_7, a0a1 + a0a1);
    }

    // Exponentiation by squaring
    __device__ Fp2_64 pow(unsigned exp) const {
        Fp2_64 result(Fp64(1), Fp64(0));
        Fp2_64 base = *this;

        while (exp > 0) {
            if (exp & 1) {
                result = result * base;
            }
            exp >>= 1;
            base = base.square();
        }
        return result;
    }

    // Multiplicative inverse: (a0 + a1*w)^-1 = (a0 - a1*w) / (a0^2 - 7*a1^2)
    __device__ Fp2_64 inverse() const {
        Fp64 a0_sq = c0 * c0;
        Fp64 a1_sq = c1 * c1;

        // 7*a1_sq
        Fp64 a1_sq_2 = a1_sq + a1_sq;
        Fp64 a1_sq_4 = a1_sq_2 + a1_sq_2;
        Fp64 a1_sq_7 = a1_sq + a1_sq_2 + a1_sq_4;

        // norm = a0^2 - 7*a1^2
        Fp64 norm = a0_sq - a1_sq_7;
        Fp64 norm_inv = norm.inverse();

        return Fp2_64(c0 * norm_inv, c1.neg() * norm_inv);
    }

    // Negation: -(a0 + a1*w) = (-a0) + (-a1)*w
    __device__ Fp2_64 neg() const {
        return Fp2_64(c0.neg(), c1.neg());
    }

    // Conjugate: conjugate(a0 + a1*w) = a0 - a1*w
    __device__ Fp2_64 conjugate() const {
        return Fp2_64(c0, c1.neg());
    }

    // Multiply by base field element
    __device__ Fp2_64 mul_by_fp(const Fp64 scalar) const {
        return Fp2_64(c0 * scalar, c1 * scalar);
    }
};

namespace goldilocks {
    using Fp2 = Fp2_64;
} // namespace goldilocks

#endif // fp2_goldilocks_h
