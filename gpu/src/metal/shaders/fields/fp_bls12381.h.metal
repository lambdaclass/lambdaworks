#pragma once

#include "fp_u256.h.metal"
#include "unsigned_int.h.metal"
#include "ec_point.h.metal"
#include "../test/test_bls12381.h.metal"

// Prime Field of U256 with modulus 0x800000000000011000000000000000000000000000000000000000000000001, used for Starks
namespace {
    typedef Fp256<
    /* =N **/ /*u256(*/ 576460752303423505, 0, 0, 1 /*)*/,
    /* =R_SQUARED **/ /*u256(*/ 576413109808302096, 18446744073700081664, 5151653887, 18446741271209837569 /*)*/,
    /* =N_PRIME **/ /*u256(*/ 576460752303423504, 18446744073709551615, 18446744073709551615, 18446744073709551615 /*)*/
    > Fp;

    typedef ECPoint<Fp, 5> BLS12381;
    typedef UnsignedInteger<12> u384;
}

class FpBLS12381 {
public:
    FpBLS12381() = default;
    //constexpr FpBLS12381(uint32_t v) : inner{} {}
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

    FpBLS12381 operator*(const FpBLS12381 rhs) const
    {
        return FpBLS12381(mul(inner, rhs.inner));
    }

    // TODO: make method for all fields
    FpBLS12381 pow(uint32_t exp)
    {
        // TODO find a way to generate on compile time
        FpBLS12381 const ONE = mul(1, R_SQUARED);
        FpBLS12381 res = ONE;

        while (exp > 0)
        {
            if (exp & 1)
            {
                res = res * *this;
            }
            exp >>= 1;
            *this = *this * *this;
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
        return FpBLS12381(sub(0, inner));
    }

private:
    u384 inner;

    constexpr static const constant u384 N = {
              436277738, 964683418,
              1260103606, 1129032919,
              1685539716, 4085584575,
              1731252896, 4138792484,
              514588670, 2975072255,
              3120496639, 4294945451};
    constexpr static const constant uint32_t R_SQUARED[12] {
            295210981, 2462770090,
            2591637125, 3038352685,
            1743489193, 2476573632,
            2380613484, 1284880085,
            175564454, 164693233,
            4108263220, 473175878};
    //constexpr static const constant u384 N_PRIME = u384(N_PRIME_0, N_PRIME_1, N_PRIME_2, N_PRIME_3);

    // Equates to `(1 << 256) - N`
    //constexpr static const constant u384 R_SUB_N =
    //   u384(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF) - N + u384(1);

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
        u384 addition = (lhs + rhs);
        u384 res = addition;
        // TODO: determine if an if statement here are more optimal
        return res - u384(addition >= N) * N + u384(addition < lhs) * R_SUB_N;
    }

    // Computes `lhs - rhs mod N`
    // Assumes `rhs` value in range [0,N)
    inline u384 sub(const u384 lhs, const u384 rhs) const
    {
        // TODO: figure what goes on here with "constant" scope variables
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
    u384 mul(const u384 lhs, const u384 rhs) const
    {
        u384 lhs_low = lhs.low();
        u384 lhs_high = lhs.high();
        u384 rhs_low = rhs.low();
        u384 rhs_high = rhs.high();

        u384 partial_t_high = lhs_high * rhs_high;
        u384 partial_t_mid_a = lhs_high * rhs_low;
        u384 partial_t_mid_a_low = partial_t_mid_a.low();
        u384 partial_t_mid_a_high = partial_t_mid_a.high();
        u384 partial_t_mid_b = rhs_high * lhs_low;
        u384 partial_t_mid_b_low = partial_t_mid_b.low();
        u384 partial_t_mid_b_high = partial_t_mid_b.high();
        u384 partial_t_low = lhs_low * rhs_low;

        u384 tmp = partial_t_mid_a_low +
                   partial_t_mid_b_low + partial_t_low.high();
        u384 carry = tmp.high();
        u384 t_low = u384(tmp.low(), partial_t_low.low());
        u384 t_high = partial_t_high + partial_t_mid_a_high + partial_t_mid_b_high + carry;

        // Compute `m = T * N' mod R`
        u384 m = t_low * N_PRIME;

        // Compute `t = (T + m * N) / R`
        u384 n = N;
        u384 n_low = n.low();
        u384 n_high = n.high();
        u384 m_low = m.low();
        u384 m_high = m.high();

        u384 partial_mn_high = m_high * n_high;
        u384 partial_mn_mid_a = m_high * n_low;
        u384 partial_mn_mid_a_low = partial_mn_mid_a.low();
        u384 partial_mn_mid_a_high = partial_mn_mid_a.high();
        u384 partial_mn_mid_b = n_high * m_low;
        u384 partial_mn_mid_b_low = partial_mn_mid_b.low();
        u384 partial_mn_mid_b_high = partial_mn_mid_b.high();
        u384 partial_mn_low = m_low * n_low;

        tmp = partial_mn_mid_a_low + partial_mn_mid_b_low + u384(partial_mn_low.high());
        carry = tmp.high();
        u384 mn_low = u384(tmp.low(), partial_mn_low.low());
        u384 mn_high = partial_mn_high + partial_mn_mid_a_high + partial_mn_mid_b_high + carry;

        u384 overflow = mn_low + t_low < mn_low;
        u384 t_tmp = t_high + overflow;
        u384 t = t_tmp + mn_high;
        u384 overflows_r = t < t_tmp;
        u384 overflows_modulus = t >= N;

        return t + overflows_r * R_SUB_N - overflows_modulus * N;
    }
};



#endif /* bls12381_h */
