// 128-bit unsigned integer arithmetic for Metal shaders.
//
// Based on ministark's implementation:
// https://github.com/andrewmilson/ministark/blob/main/gpu-poly/src/metal/u128.h.metal

#ifndef u128_h
#define u128_h

#include <metal_stdlib>

/// 128-bit unsigned integer type for GPU field arithmetic.
///
/// Provides basic arithmetic operations needed for finite field computations.
class u128
{
public:
    u128() = default;
    constexpr u128(int l) : low(l), high(0) {}
    constexpr u128(unsigned long l) : low(l), high(0) {}
    constexpr u128(bool b) : low(b), high(0) {}
    constexpr u128(unsigned long h, unsigned long l) : low(l), high(h) {}

    constexpr u128 operator+(const u128 rhs) const
    {
        return u128(high + rhs.high + ((low + rhs.low) < low), low + rhs.low);
    }

    constexpr u128 operator+=(const u128 rhs)
    {
        *this = *this + rhs;
        return *this;
    }

    constexpr inline u128 operator-(const u128 rhs) const
    {
        return u128(high - rhs.high - ((low - rhs.low) > low), low - rhs.low);
    }

    constexpr u128 operator-=(const u128 rhs)
    {
        *this = *this - rhs;
        return *this;
    }

    constexpr bool operator==(const u128 rhs) const
    {
        return high == rhs.high && low == rhs.low;
    }

    constexpr bool operator!=(const u128 rhs) const
    {
        return !(*this == rhs);
    }

    constexpr bool operator<(const u128 rhs) const
    {
        return ((high == rhs.high) && (low < rhs.low)) || (high < rhs.high);
    }

    constexpr u128 operator&(const u128 rhs) const
    {
        return u128(high & rhs.high, low & rhs.low);
    }

    constexpr u128 operator|(const u128 rhs) const
    {
        return u128(high | rhs.high, low | rhs.low);
    }

    constexpr bool operator>(const u128 rhs) const
    {
        return ((high == rhs.high) && (low > rhs.low)) || (high > rhs.high);
    }

    constexpr bool operator>=(const u128 rhs) const
    {
        return !(*this < rhs);
    }

    constexpr bool operator<=(const u128 rhs) const
    {
        return !(*this > rhs);
    }

    /// Right shift operator optimized for GPU (branchless).
    constexpr inline u128 operator>>(unsigned shift) const
    {
        uint64_t new_low = (shift == 0) * low
                         | (shift == 64) * high
                         | ((shift < 64) ^ (shift == 0)) * ((high << (64 - shift)) | (low >> shift))
                         | ((shift > 64) & (shift < 128)) * (high >> (shift - 64));

        uint64_t new_high = (shift == 0) * high
                          | ((shift < 64) ^ (shift == 0)) * (high >> shift);

        return u128(new_high, new_low);
    }

    /// Left shift operator optimized for GPU (branchless).
    constexpr inline u128 operator<<(unsigned shift) const
    {
        unsigned long new_low = (shift == 0) * low
                            | ((shift < 64) ^ (shift == 0)) * (low << shift);

        unsigned long new_high = (shift == 0) * high
                            | (shift == 64) * low
                            | ((shift < 64) ^ (shift == 0)) * (high << shift) | (low >> (64 - shift))
                            | ((shift > 64) & (shift < 128)) * (low >> (shift - 64));

        return u128(new_high, new_low);
    }

    constexpr u128 operator>>=(unsigned rhs)
    {
        *this = *this >> rhs;
        return *this;
    }

    u128 operator*(const bool rhs) const
    {
        return u128(high * rhs, low * rhs);
    }

    /// Multiplication using Metal's mulhi for efficient 64x64→128 bit multiplication.
    u128 operator*(const u128 rhs) const
    {
        unsigned long t_low_high = low * rhs.high;
        unsigned long t_high = metal::mulhi(low, rhs.low);
        unsigned long t_high_low = high * rhs.low;
        unsigned long t_low = low * rhs.low;
        return u128(t_low_high + t_high_low + t_high, t_low);
    }

    u128 operator*=(const u128 rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    unsigned long high;
    unsigned long low;
};

#endif /* u128_h */
