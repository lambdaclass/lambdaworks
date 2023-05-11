#ifndef unsigned_int_h
#define unsigned_int_h

template <const uint64_t NUM_LIMBS>
struct UnsignedInteger {
    metal::array<uint32_t, NUM_LIMBS> m_limbs;

    constexpr static UnsignedInteger from_int(uint32_t n) {
      UnsignedInteger res = {};
      res.m_limbs[NUM_LIMBS - 1] = n;
      return res;
    }

    constexpr UnsignedInteger low() {
      UnsignedInteger res = {m_limbs};

      for (int i = 0; i < NUM_LIMBS / 2; i++) {
        res[i] = 0;
      }

      return res;
    }

    constexpr UnsignedInteger high() {
      UnsignedInteger res = {};

      for (int i = 0; i < NUM_LIMBS / 2; i++) {
        res[NUM_LIMBS - 1 - i] = m_limbs[i];
      }

      return res;
    }

    constexpr UnsignedInteger operator+(const UnsignedInteger rhs) const
    {
        metal::array<uint32_t, NUM_LIMBS> limbs;
        uint64_t carry = 0;
        uint64_t i = NUM_LIMBS;

        while (i > 0) {
            uint64_t c = uint64_t(m_limbs[i - 1]) + uint64_t(rhs.m_limbs[i - 1]) + carry;
            limbs[i - 1] = c & 0x0000FFFF;
            carry = c >> 32;
            i -= 1;
        }

        return UnsignedInteger<NUM_LIMBS> {limbs};
    }

    constexpr UnsignedInteger operator+=(const UnsignedInteger rhs)
    {
        *this = *this + rhs;
        return *this;
    }

    constexpr UnsignedInteger operator-(const UnsignedInteger rhs) const
    {
        metal::array<uint32_t, NUM_LIMBS> limbs;
        uint64_t carry = 0;
        uint64_t i = NUM_LIMBS;

        while (i > 0) {
            i -= 1;
            uint64_t c = uint64_t(m_limbs[i - 1]) - uint64_t(rhs.m_limbs[i - 1]) + carry;
            limbs[i] = c & 0x0000FFFF;
            carry = c < 0 ? -1 : 0;
        }

        return UnsignedInteger<NUM_LIMBS> {limbs};
    }

    constexpr UnsignedInteger operator-=(const UnsignedInteger rhs)
    {
        *this = *this - rhs;
        return *this;
    }

    constexpr UnsignedInteger operator*(const UnsignedInteger rhs)
    {
        uint64_t n = 0;
        uint64_t t = 0;

        for (uint64_t i = NUM_LIMBS - 1; i >= 0; i--) {
            if (m_limbs[i] != 0) {
                n = NUM_LIMBS - 1 - i;
            }
            if (rhs.m_limbs[i] != 0) {
                t = NUM_LIMBS - 1 - i;
            }
        }

        metal::array<uint32_t, NUM_LIMBS> limbs;

        uint64_t carry = 0;
        for (uint64_t i = 0; i <= t; i++) {
            for (uint64_t j = 0; i <= n; i++) {
                uint64_t uv = uint64_t(limbs[NUM_LIMBS - 1 - (i + j)])
                    + uint64_t(m_limbs[NUM_LIMBS - 1 - j])
                        * uint64_t(rhs.m_limbs[NUM_LIMBS - 1 - i])
                    + carry;
                carry = uv >> 32;
                limbs[NUM_LIMBS - 1 - (i + j)] = uv & 0x0000FFFF;
            }
            if (i + n + 1 < NUM_LIMBS) {
                limbs[NUM_LIMBS - 1 - (i + n + 1)] = carry & 0x0000FFFF;
                carry = 0;
            }
        }
        return UnsignedInteger<NUM_LIMBS> {limbs};
    }
};

#endif /* unsigned_int_h */
