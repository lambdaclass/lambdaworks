// https://github.com/andrewmilson/ministark/blob/main/gpu-poly/src/metal/u128.h.metal

#ifndef u128_h
#define u128_h

typedef unsigned long long uint64_t;

class u128
{
public:
  u128() = default;
  __device__ constexpr u128(int l) : low(l), high(0) {}
  __device__ constexpr u128(uint64_t l) : low(l), high(0) {}
  __device__ constexpr u128(bool b) : low(b), high(0) {}
  __device__ constexpr u128(uint64_t h, uint64_t l)
      : low(l), high(h) {}

  __device__ constexpr u128 operator+(const u128 rhs) const
  {
    return u128(high + rhs.high + ((low + rhs.low) < low), low + rhs.low);
  }

  __device__ constexpr u128 operator+=(const u128 rhs)
  {
    *this = *this + rhs;
    return *this;
  }

  __device__ constexpr inline u128 operator-(const u128 rhs) const
  {
    return u128(high - rhs.high - ((low - rhs.low) > low), low - rhs.low);
  }

  __device__ constexpr u128 operator-=(const u128 rhs)
  {
    *this = *this - rhs;
    return *this;
  }

  __device__ constexpr bool operator==(const u128 rhs) const
  {
    return high == rhs.high && low == rhs.low;
  }

  __device__ constexpr bool operator!=(const u128 rhs) const
  {
    return !(*this == rhs);
  }

  __device__ constexpr bool operator<(const u128 rhs) const
  {
    return ((high == rhs.high) && (low < rhs.low)) || (high < rhs.high);
  }

  __device__ constexpr u128 operator&(const u128 rhs) const
  {
    return u128(high & rhs.high, low & rhs.low);
  }

  __device__ constexpr u128 operator|(const u128 rhs) const
  {
    return u128(high | rhs.high, low | rhs.low);
  }

  __device__ constexpr bool operator>(const u128 rhs) const
  {
    return ((high == rhs.high) && (low > rhs.low)) || (high > rhs.high);
  }

  __device__ constexpr bool operator>=(const u128 rhs) const
  {
    return !(*this < rhs);
  }

  __device__ constexpr bool operator<=(const u128 rhs) const
  {
    return !(*this > rhs);
  }

  __device__ constexpr inline u128 operator>>(unsigned shift) const
  {
    uint64_t new_low = (shift == 0) * low
                          | (shift == 64) * high
                          | ((shift < 64) ^ (shift == 0)) * ((high << (64 - shift)) | (low >> shift))
                          | ((shift > 64) & (shift < 128)) * (high >> (shift - 64));

    uint64_t new_high = (shift == 0) * high
                          | ((shift < 64) ^ (shift == 0)) * (high >> shift);

    return u128(new_high, new_low);

    // Unoptimized form:
    // if (shift >= 128)
    //   return u128(0);
    // else if (shift == 64)
    //   return u128(0, high);
    // else if (shift == 0)
    //   return *this;
    // else if (shift < 64)
    //   return u128(high >> shift, (high << (64 - shift)) | (low >> shift));
    // else if ((128 > shift) && (shift > 64))
    //   return u128(0, (high >> (shift - 64)));
    // else
    //   return u128(0);
  }

  __device__ constexpr inline u128 operator<<(unsigned shift) const
  {
    uint64_t new_low = (shift == 0) * low
                          | ((shift < 64) ^ (shift == 0)) * (low << shift);

    uint64_t new_high = (shift == 0) * high
                          | (shift == 64) * low
                          | ((shift < 64) ^ (shift == 0)) * (high << shift) | (low >> (64 - shift))
                          | ((shift > 64) & (shift < 128)) * (low >> (shift - 64));

    return u128(new_high, new_low);

    // Unoptimized form:
    // if (shift >= 128)
    //   return u128(0);
    // else if (shift == 64)
    //   return u128(low, 0);
    // else if (shift == 0)
    //   return *this;
    // else if (shift < 64)
    //   return u128((high << shift) | (low >> (64 - shift)), low << shift);
    // else if ((128 > shift) && (shift > 64))
    //   return u128((low >> (shift - 64)), 0);
    // else
    //   return u128(0);
  }

  __device__ constexpr u128 operator>>=(unsigned rhs)
  {
    *this = *this >> rhs;
    return *this;
  }

  __device__ u128 operator*(const bool rhs) const
  {
    return u128(high * rhs, low * rhs);
  }

  __device__ u128 operator*(const u128 rhs) const
  {
    uint64_t t_low_high = low * rhs.high;
    uint64_t t_high = __umul64hi(low, rhs.low);
    uint64_t t_high_low = high * rhs.low;
    uint64_t t_low = low * rhs.low;
    return u128(t_low_high + t_high_low + t_high, t_low);
  }

  __device__ u128 operator*=(const u128 rhs)
  {
    *this = *this * rhs;
    return *this;
  }

  // TODO: check if performance improves with a different limb size
  uint64_t high;
  uint64_t low;
};

#endif /* u128_h */
