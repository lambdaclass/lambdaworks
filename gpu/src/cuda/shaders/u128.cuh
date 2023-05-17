// https://github.com/andrewmilson/ministark/blob/main/gpu-poly/src/metal/u128.h.metal

#ifndef u128_h
#define u128_h

class u128
{
public:
  u128() = default;
  __device__ constexpr u128(int l) : low(l), high(0) {}
  __device__ constexpr u128(unsigned long l) : low(l), high(0) {}
  __device__ constexpr u128(bool b) : low(b), high(0) {}
  __device__ constexpr u128(unsigned long h, unsigned long l)
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
    // TODO: reduce branch conditions
    if (shift >= 128)
    {
      return u128(0);
    }
    else if (shift == 64)
    {
      return u128(0, high);
    }
    else if (shift == 0)
    {
      return *this;
    }
    else if (shift < 64)
    {
      return u128(high >> shift, (high << (64 - shift)) | (low >> shift));
    }
    else if ((128 > shift) && (shift > 64))
    {
      return u128(0, (high >> (shift - 64)));
    }
    else
    {
      return u128(0);
    }
  }

  __device__ constexpr inline u128 operator<<(unsigned shift) const
  {
    // TODO: reduce branch conditions
    if (shift >= 128)
    {
      return u128(0);
    }
    else if (shift == 64)
    {
      return u128(low, 0);
    }
    else if (shift == 0)
    {
      return *this;
    }
    else if (shift < 64)
    {
      return u128((high << shift) | (low >> (64 - shift)), low << shift);
    }
    else if ((128 > shift) && (shift > 64))
    {
      return u128((low >> (shift - 64)), 0);
    }
    else
    {
      return u128(0);
    }
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
    unsigned long t_low_high = __umul64hi(low, rhs.high);
    unsigned long t_high = __umul64hi(low, rhs.low);
    unsigned long t_high_low = __umul64hi(high, rhs.low);
    unsigned long t_low = low * rhs.low;
    return u128(t_low_high + t_high_low + t_high, t_low);
  }

  __device__ u128 operator*=(const u128 rhs)
  {
    *this = *this * rhs;
    return *this;
  }

  // TODO: check if performance improves with a different limb size
  unsigned long high;
  unsigned long low;
};

#endif /* u128_h */
