// https://github.com/andrewmilson/ministark/blob/main/gpu-poly/src/metal/felt_u256.h.metal

#ifndef felt_u256_h
#define felt_u256_h

#include "../unsigned_integer/u256.h.metal"

template <
    /* =N **/ unsigned long N_0, unsigned long N_1, unsigned long N_2, unsigned long N_3,
    /* =R_SQUARED **/ unsigned long R_SQUARED_0, unsigned long R_SQUARED_1, unsigned long R_SQUARED_2, unsigned long R_SQUARED_3,
    /* =N_PRIME **/ unsigned long N_PRIME_0, unsigned long N_PRIME_1, unsigned long N_PRIME_2, unsigned long N_PRIME_3>
class Fp256 {
public:
    Fp256() = default;
    constexpr Fp256(unsigned long v) : inner(v) {}
    constexpr Fp256(u256 v) : inner(v) {}

    constexpr explicit operator u256() const
    {
        return inner;
    }

    constexpr Fp256 operator+(const Fp256 rhs) const
    {
        return Fp256(add(inner, rhs.inner));
    }

    constexpr Fp256 operator-(const Fp256 rhs) const
    {
        return Fp256(sub(inner, rhs.inner));
    }

    Fp256 operator*(const Fp256 rhs) const
    {
        return Fp256(mul(inner, rhs.inner));
    }

    // TODO: make method for all fields
    Fp256 pow(unsigned exp)
    {
        // TODO find a way to generate on compile time
        Fp256 const ONE = mul(u256(1), R_SQUARED);
        Fp256 res = ONE;

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

    Fp256 inverse() 
    {
        // used addchain
        // https://github.com/mmcloughlin/addchain
        u256 _10 = mul(inner, inner);
        u256 _11 = mul(_10, inner);
        u256 _1100 = sqn<2>(_11);
        u256 _1101 = mul(inner, _1100);
        u256 _1111 = mul(_10, _1101);
        u256 _11001 = mul(_1100, _1101);
        u256 _110010 = mul(_11001, _11001);
        u256 _110011 = mul(inner, _110010);
        u256 _1000010 = mul(_1111, _110011);
        u256 _1001110 = mul(_1100, _1000010);
        u256 _10000001 = mul(_110011, _1001110);
        u256 _11001111 = mul(_1001110, _10000001);
        u256 i14 = mul(_11001111, _11001111);
        u256 i15 = mul(_10000001, i14);
        u256 i16 = mul(i14, i15);
        u256 x10 = mul(_1000010, i16);
        u256 i27 = sqn<10>(x10);
        u256 i28 = mul(i16, i27);
        u256 i38 = sqn<10>(i27);
        u256 i39 = mul(i28, i38);
        u256 i49 = sqn<10>(i38);
        u256 i50 = mul(i39, i49);
        u256 i60 = sqn<10>(i49);
        u256 i61 = mul(i50, i60);
        u256 i72 = mul(sqn<10>(i60), i61);
        u256 x60 = mul(_1000010, i72);
        u256 i76 = sqn<2>(mul(i72, x60));
        u256 x64 = mul(mul(i15, i76), i76);
        u256 i208 = mul(sqn<64>(mul(sqn<63>(mul(i15, x64)), x64)), x64);
        return Fp256(mul(sqn<60>(i208), x60));
    }

    Fp256 neg()
    {
        // TODO: can improve
        return Fp256(sub(0, inner));
    }

private:
    u256 inner;

    constexpr static const constant u256 N = u256(N_0, N_1, N_2, N_3);
    constexpr static const constant u256 R_SQUARED = u256(R_SQUARED_0, R_SQUARED_1, R_SQUARED_2, R_SQUARED_3);
    constexpr static const constant u256 N_PRIME = u256(N_PRIME_0, N_PRIME_1, N_PRIME_2, N_PRIME_3);

    // Equates to `(1 << 256) - N`
    constexpr static const constant u256 R_SUB_N =
        u256(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF) - N + u256(1);

    template<unsigned N_ACC>
    u256 sqn(u256 base) const {
        u256 result = base;
#pragma unroll
        for (unsigned i = 0; i < N_ACC; i++) {
            result = mul(result, result);
        }
        return result;
    }

    // Computes `lhs + rhs mod N`
    // Returns value in range [0,N)
    inline u256 add(const u256 lhs, const u256 rhs) const
    {
        u256 addition = (lhs + rhs);
        u256 res = addition;
        // TODO: determine if an if statement here are more optimal
        return res - u256(addition >= N) * N + u256(addition < lhs) * R_SUB_N;
    }

    // Computes `lhs - rhs mod N`
    // Assumes `rhs` value in range [0,N)
    inline u256 sub(const u256 lhs, const u256 rhs) const
    {
        // TODO: figure what goes on here with "constant" scope variables
        return add(lhs, ((u256)N) - rhs);
    }

    // Computes `lhs * rhs mod M`
    //
    // Essential that inputs are already in the range [0,N) and are in montgomery
    // form. Multiplication performs single round of montgomery reduction.
    //
    // Reference:
    // - https://en.wikipedia.org/wiki/Montgomery_modular_multiplication (REDC)
    // - https://www.youtube.com/watch?v=2UmQDKcelBQ
    u256 mul(const u256 lhs, const u256 rhs) const
    {
        u256 lhs_low = lhs.low;
        u256 lhs_high = lhs.high;
        u256 rhs_low = rhs.low;
        u256 rhs_high = rhs.high;

        u256 partial_t_high = lhs_high * rhs_high;
        u256 partial_t_mid_a = lhs_high * rhs_low;
        u256 partial_t_mid_a_low = partial_t_mid_a.low;
        u256 partial_t_mid_a_high = partial_t_mid_a.high;
        u256 partial_t_mid_b = rhs_high * lhs_low;
        u256 partial_t_mid_b_low = partial_t_mid_b.low;
        u256 partial_t_mid_b_high = partial_t_mid_b.high;
        u256 partial_t_low = lhs_low * rhs_low;

        u256 tmp = partial_t_mid_a_low +
                   partial_t_mid_b_low + partial_t_low.high;
        u256 carry = tmp.high;
        u256 t_low = u256(tmp.low, partial_t_low.low);
        u256 t_high = partial_t_high + partial_t_mid_a_high + partial_t_mid_b_high + carry;

        // Compute `m = T * N' mod R`
        u256 m = t_low * N_PRIME;

        // Compute `t = (T + m * N) / R`
        u256 n = N;
        u256 n_low = n.low;
        u256 n_high = n.high;
        u256 m_low = m.low;
        u256 m_high = m.high;

        u256 partial_mn_high = m_high * n_high;
        u256 partial_mn_mid_a = m_high * n_low;
        u256 partial_mn_mid_a_low = partial_mn_mid_a.low;
        u256 partial_mn_mid_a_high = partial_mn_mid_a.high;
        u256 partial_mn_mid_b = n_high * m_low;
        u256 partial_mn_mid_b_low = partial_mn_mid_b.low;
        u256 partial_mn_mid_b_high = partial_mn_mid_b.high;
        u256 partial_mn_low = m_low * n_low;

        tmp = partial_mn_mid_a_low + partial_mn_mid_b_low + u256(partial_mn_low.high);
        carry = tmp.high;
        u256 mn_low = u256(tmp.low, partial_mn_low.low);
        u256 mn_high = partial_mn_high + partial_mn_mid_a_high + partial_mn_mid_b_high + carry;

        u256 overflow = mn_low + t_low < mn_low;
        u256 t_tmp = t_high + overflow;
        u256 t = t_tmp + mn_high;
        u256 overflows_r = t < t_tmp;
        u256 overflows_modulus = t >= N;

        return t + overflows_r * R_SUB_N - overflows_modulus * N;
    }
};



#endif
