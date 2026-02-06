// 256-bit unsigned integer arithmetic for Metal shaders.
//
// Based on ministark's implementation:
// https://github.com/andrewmilson/ministark/blob/main/gpu-poly/src/metal/u256.h.metal

#ifndef u256_h
#define u256_h

#include <metal_stdlib>
#include "u128.h.metal"

/// 256-bit unsigned integer type for GPU field arithmetic.
///
/// Provides basic arithmetic operations needed for 256-bit finite field computations.
class u256
{
public:
    u256() = default;
    constexpr u256(int l) : low(l), high(0) {}
    constexpr u256(unsigned long l) : low(u128(l)), high(0) {}
    constexpr u256(u128 l) : low(l), high(0) {}
    constexpr u256(bool b) : low(b), high(0) {}
    constexpr u256(u128 h, u128 l) : low(l), high(h) {}
    constexpr u256(unsigned long hh, unsigned long hl, unsigned long lh, unsigned long ll) :
        low(u128(lh, ll)), high(u128(hh, hl)) {}

    constexpr u256 operator+(const u256 rhs) const
    {
        return u256(high + rhs.high + ((low + rhs.low) < low), low + rhs.low);
    }

    constexpr u256 operator+=(const u256 rhs)
    {
        *this = *this + rhs;
        return *this;
    }

    constexpr inline u256 operator-(const u256 rhs) const
    {
        return u256(high - rhs.high - ((low - rhs.low) > low), low - rhs.low);
    }

    constexpr u256 operator-=(const u256 rhs)
    {
        *this = *this - rhs;
        return *this;
    }

    constexpr bool operator==(const u256 rhs) const
    {
        return high == rhs.high && low == rhs.low;
    }

    constexpr bool operator!=(const u256 rhs) const
    {
        return !(*this == rhs);
    }

    constexpr bool operator<(const u256 rhs) const
    {
        return ((high == rhs.high) && (low < rhs.low)) || (high < rhs.high);
    }

    constexpr u256 operator&(const u256 rhs) const
    {
        return u256(high & rhs.high, low & rhs.low);
    }

    constexpr bool operator>(const u256 rhs) const
    {
        return ((high == rhs.high) && (low > rhs.low)) || (high > rhs.high);
    }

    constexpr bool operator>=(const u256 rhs) const
    {
        return !(*this < rhs);
    }

    constexpr bool operator<=(const u256 rhs) const
    {
        return !(*this > rhs);
    }

    /// Right shift operator optimized for GPU (branchless).
    inline u256 operator>>(const unsigned shift) const
    {
        u128 new_low = low * (shift == 0)
                     | high * (shift == 128)
                     | (high << (128 - shift) | (low >> shift)) * ((shift < 128) ^ (shift == 0))
                     | (high >> (shift - 128)) * ((shift < 256) & (shift > 128));

        u128 new_high = high * (shift == 0)
                      | (high >> shift) * ((shift < 128) ^ (shift == 0));

        return u256(new_high, new_low);
    }

    constexpr u256 operator>>=(unsigned rhs)
    {
        *this = *this >> rhs;
        return *this;
    }

    u256 operator*(const bool rhs) const
    {
        return u256(high * rhs, low * rhs);
    }

    /// 256-bit multiplication using Karatsuba-style decomposition.
    u256 operator*(const u256 rhs) const
    {
        // Split values into 4 64-bit parts for efficient multiplication
        u128 top[2] = {u128(low.high), u128(low.low)};
        u128 bottom[3] = {u128(rhs.high.low), u128(rhs.low.high), u128(rhs.low.low)};

        unsigned long tmp3_3 = high.high * rhs.low.low;
        unsigned long tmp0_0 = low.low * rhs.high.high;
        unsigned long tmp2_2 = high.low * rhs.low.high;

        u128 tmp2_3 = u128(high.low) * bottom[2];
        u128 tmp0_3 = top[1] * bottom[2];
        u128 tmp1_3 = top[0] * bottom[2];

        u128 tmp0_2 = top[1] * bottom[1];
        u128 third64 = u128(tmp0_2.low) + u128(tmp0_3.high);
        u128 tmp1_2 = top[0] * bottom[1];

        u128 tmp0_1 = top[1] * bottom[0];
        u128 second64 = u128(tmp0_1.low) + u128(tmp0_2.high);
        unsigned long first64 = tmp0_0 + tmp0_1.high;

        u128 tmp1_1 = top[0] * bottom[0];
        first64 += tmp1_1.low + tmp1_2.high;

        // Second row
        third64 += u128(tmp1_3.low);
        second64 += u128(tmp1_2.low) + u128(tmp1_3.high);

        // Third row
        second64 += u128(tmp2_3.low);
        first64 += tmp2_2 + tmp2_3.high;

        // Fourth row
        first64 += tmp3_3;
        second64 += u128(third64.high);
        first64 += second64.high;

        return u256(u128(first64, second64.low), u128(third64.low, tmp0_3.low));
    }

    u256 operator*=(const u256 rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    u128 high;
    u128 low;
};

#endif /* u256_h */
