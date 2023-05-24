#pragma once

#include "unsigned_int.h.metal"


namespace {
    typedef UnsignedInteger<12> u384;
}

// taken from the Rust implementation
constexpr static const constant u384 N = {
    0x1a0111ea,0x397fe69a,
    0x4b1ba7b6,0x434bacd7,
    0x64774b84,0xf38512bf,
    0x6730d2a0,0xf6b0f624,
    0x1eabfffe,0xb153ffff,
    0xb9feffff,0xffffaaab
};
constexpr static const constant u384 R_SQUARED = {
    0x11988fe5,0x92cae3aa,
    0x9a793e85,0xb519952d,
    0x67eb88a9,0x939d83c0,
    0x8de5476c,0x4c95b6d5,
    0x0a76e6a6,0x09d104f1,
    0xf4df1f34,0x1c341746
};

// Equates to `(1 << 384) - N`
constexpr static const constant u384 R_SUB_N = {
    0xe5feee15,0xc6801965,
    0xb4e45849,0xbcb45328,
    0x9b88b47b,0x0c7aed40,
    0x98cf2d5f,0x094f09db,
    0xe1540001,0x4eac0000,
    0x46010000,0x00005555
};

// MU = -N^{-1} mod (2^32)
constexpr static const constant uint64_t MU = 4294770685;

class FpBLS12381 {
public:
    u384 inner;
    constexpr FpBLS12381() = default;
    constexpr FpBLS12381(uint64_t v) : inner{u384::from_int(v)} {}
    constexpr FpBLS12381(u384 v) : inner{v} {}

    constexpr explicit operator u384() const
    {
        return inner;
    }

    constexpr FpBLS12381 operator+(const FpBLS12381 rhs) const
    {
        return FpBLS12381(add(inner, rhs.inner));
    }

    constexpr FpBLS12381 operator-(const FpBLS12381 rhs) const
    {
        return FpBLS12381(sub(inner, rhs.inner));
    }

    constexpr FpBLS12381 operator*(const FpBLS12381 rhs) const
    {
        return FpBLS12381(mul(inner, rhs.inner));
    }

    constexpr bool operator==(const FpBLS12381 rhs) const
    {
        return inner == rhs.inner;
    }

    constexpr bool operator!=(const FpBLS12381 rhs) const
    {
        return !(inner == rhs.inner);
    }

    constexpr explicit operator uint32_t() const
    {
        return inner.m_limbs[11];
    }

    FpBLS12381 operator>>(const uint32_t rhs) const
    {
        return FpBLS12381(inner >> rhs);
    }

    FpBLS12381 operator<<(const uint32_t rhs) const
    {
        return FpBLS12381(inner << rhs);
    }

    constexpr static FpBLS12381 one()
    {
        // TODO find a way to generate on compile time
        const FpBLS12381 ONE = FpBLS12381::mul(u384::from_int((uint32_t) 1), R_SQUARED);
        return ONE;
    }

    constexpr FpBLS12381 to_montgomery()
    {
        return mul(inner, R_SQUARED);
    }

    // TODO: make method for all fields
    FpBLS12381 pow(uint32_t exp) const
    {
        // TODO find a way to generate on compile time
        FpBLS12381 const ONE = one();
        FpBLS12381 res = ONE;
        FpBLS12381 power = *this;

        while (exp > 0)
        {
            if (exp & 1)
            {
                res = res * power;
            }
            exp >>= 1;
            power = power * power;
        }

        return res;
    }

    FpBLS12381 inverse() 
    {
        // used addchain
        // https://github.com/mmcloughlin/addchain
        u384 _10 = mul(inner, inner);
        u384 _11 = mul(_10, inner);
        u384 _1100 = sqn<2>(_11);
        u384 _1101 = mul(inner, _1100);
        u384 _1111 = mul(_10, _1101);
        u384 _11001 = mul(_1100, _1101);
        u384 _110010 = mul(_11001, _11001);
        u384 _110011 = mul(inner, _110010);
        u384 _1000010 = mul(_1111, _110011);
        u384 _1001110 = mul(_1100, _1000010);
        u384 _10000001 = mul(_110011, _1001110);
        u384 _11001111 = mul(_1001110, _10000001);
        u384 i14 = mul(_11001111, _11001111);
        u384 i15 = mul(_10000001, i14);
        u384 i16 = mul(i14, i15);
        u384 x10 = mul(_1000010, i16);
        u384 i27 = sqn<10>(x10);
        u384 i28 = mul(i16, i27);
        u384 i38 = sqn<10>(i27);
        u384 i39 = mul(i28, i38);
        u384 i49 = sqn<10>(i38);
        u384 i50 = mul(i39, i49);
        u384 i60 = sqn<10>(i49);
        u384 i61 = mul(i50, i60);
        u384 i72 = mul(sqn<10>(i60), i61);
        u384 x60 = mul(_1000010, i72);
        u384 i76 = sqn<2>(mul(i72, x60));
        u384 x64 = mul(mul(i15, i76), i76);
        u384 i208 = mul(sqn<64>(mul(sqn<63>(mul(i15, x64)), x64)), x64);
        return FpBLS12381(mul(sqn<60>(i208), x60));
    }

    FpBLS12381 neg()
    {
        // TODO: can improve
        return FpBLS12381(sub(u384::from_int((uint32_t)0), inner));
    }

private:

    template<uint32_t N_ACC>
    u384 sqn(u384 base) const {
        u384 result = base;
#pragma unroll
        for (uint32_t i = 0; i < N_ACC; i++) {
            result = mul(result, result);
        }
        return result;
    }

    // Computes `lhs + rhs mod N`
    // Returns value in range [0,N)
    inline u384 add(const u384 lhs, const u384 rhs) const
    {
        u384 addition = lhs + rhs;
        u384 res = addition;
        // TODO: determine if an if statement here are more optimal

        return res - u384::from_int((uint64_t)(addition >= N)) * N + u384::from_int((uint64_t)(addition < lhs)) * R_SUB_N;
    }

    // Computes `lhs - rhs mod N`
    // Assumes `rhs` value in range [0,N)
    inline u384 sub(const u384 lhs, const u384 rhs) const
    {
        return add(lhs, ((u384)N) - rhs);
    }

    // Computes `lhs * rhs mod M`
    //
    // Essential that inputs are already in the range [0,N) and are in montgomery
    // form. Multiplication performs single round of montgomery reduction.
    //
    // Reference:
    // - https://en.wikipedia.org/wiki/Montgomery_modular_multiplication (REDC)
    // - https://www.youtube.com/watch?v=2UmQDKcelBQ
    constexpr static u384 mul(const u384 a, const u384 b)
    {
        constexpr uint64_t NUM_LIMBS = 12;
        metal::array<uint32_t, NUM_LIMBS> t = {};
        metal::array<uint32_t, 2> t_extra = {};

        u384 q = N;

        uint64_t i = NUM_LIMBS;

        while (i > 0) {
            i -= 1;
            // C := 0
            uint64_t c = 0;

            // for j=0 to N-1
            //    (C,t[j]) := t[j] + a[j]*b[i] + C
            uint64_t cs = 0;
            uint64_t j = NUM_LIMBS;
            while (j > 0) {
                j -= 1;
                cs = (uint64_t)t[j] + (uint64_t)a.m_limbs[j] * (uint64_t)b.m_limbs[i] + c;
                c = cs >> 32;
                t[j] = (uint32_t)((cs << 32) >> 32);
            }

            // (t[N+1],t[N]) := t[N] + C
            cs = (uint64_t)t_extra[1] + c;
            t_extra[0] = (uint32_t)(cs >> 32);
            t_extra[1] = (uint32_t)((cs << 32) >> 32);

            // m := t[0]*q'[0] mod D
            uint64_t m = (((uint64_t)t[NUM_LIMBS - 1] * MU) << 32) >> 32;

            // (C,_) := t[0] + m*q[0]
            c = ((uint64_t)t[NUM_LIMBS - 1] + m * (uint64_t)q.m_limbs[NUM_LIMBS - 1]) >> 32;

            // for j=1 to N-1
            //    (C,t[j-1]) := t[j] + m*q[j] + C

            j = NUM_LIMBS - 1;
            while (j > 0) {
                j -= 1;
                cs = (uint64_t)t[j] + m * (uint64_t)q.m_limbs[j] + c;
                c = cs >> 32;
                t[j + 1] = (uint32_t)((cs << 32) >> 32);
            }

            // (C,t[N-1]) := t[N] + C
            cs = (uint64_t)t_extra[1] + c;
            c = cs >> 32;
            t[0] = (uint32_t)((cs << 32) >> 32);

            // t[N] := t[N+1] + C
            t_extra[1] = t_extra[0] + (uint32_t)c;
        }

        u384 result {t};

        uint64_t overflow = t_extra[0] > 0;
        // TODO: assuming the integer represented by
        // [t_extra[1], t[0], ..., t[NUM_LIMBS - 1]] is at most
        // 2q in any case.
        if (overflow || q <= result) {
            result = result - q;
        }

        return result;
    }
};
