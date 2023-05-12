#ifndef unsigned_int_h
#define unsigned_int_h

#include <metal_stdlib>


template <const uint64_t NUM_LIMBS>
struct UnsignedInteger {
    metal::array<uint32_t, NUM_LIMBS> m_limbs;

    constexpr UnsignedInteger() = default;

    constexpr static UnsignedInteger from_int(uint32_t n) {
        UnsignedInteger res;
        res.m_limbs[NUM_LIMBS - 1] = n;
        return res;
    }

    constexpr static UnsignedInteger from_int(uint64_t n) {
        UnsignedInteger res;
        res.m_limbs[NUM_LIMBS - 2] = (uint32_t)(n >> 32);
        res.m_limbs[NUM_LIMBS - 1] = (uint32_t)(n & 0xFFFF);
        return res;
    }

    constexpr static UnsignedInteger from_bool(bool b) {
        UnsignedInteger res;
        if (b) {
            res.m_limbs[NUM_LIMBS - 1] = 1;
        }
        return res;
    }

    constexpr static UnsignedInteger from_high_low(UnsignedInteger high, UnsignedInteger low) {
        UnsignedInteger res = low;

        for (uint64_t i = 0; i < NUM_LIMBS; i++) {
            res.m_limbs[i] = high.m_limbs[i];
        }

        return res;
    }

    constexpr UnsignedInteger low() const {
        UnsignedInteger res = {m_limbs};

        for (uint64_t i = 0; i < NUM_LIMBS / 2; i++) {
            res.m_limbs[i] = 0;
        }

        return res;
    }

    constexpr UnsignedInteger high() const {
        UnsignedInteger res = {};

        for (uint64_t i = 0; i < NUM_LIMBS / 2; i++) {
            res.m_limbs[NUM_LIMBS / 2 + i] = m_limbs[i];
        }

        return res;
    }

    static UnsignedInteger max() {
        UnsignedInteger res = {};

        for (uint64_t i = 0; i < NUM_LIMBS; i++) {
            res.m_limbs[i] = 0xFFFFFFFF;
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

    constexpr bool operator==(const UnsignedInteger rhs) const
    {
        for (uint32_t i = 0; i < NUM_LIMBS; i++) {
            if (m_limbs[i] != rhs.m_limbs[i]) {
                return false;
            }
        }
        return true;
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

    constexpr UnsignedInteger operator*(const UnsignedInteger rhs) const
    {
        uint64_t n = 0;
        uint64_t t = 0;

        for (uint64_t i = NUM_LIMBS; i > 0; i--) {
            if (m_limbs[i - 1] != 0) {
                n = NUM_LIMBS - i;
            }
            if (rhs.m_limbs[i - 1] != 0) {
                t = NUM_LIMBS - i;
            }
        }

        metal::array<uint32_t, NUM_LIMBS> limbs;

        uint64_t carry = 0;
        for (uint64_t i = 0; i <= t; i++) {
            for (uint64_t j = 0; i <= n; i++) {
                uint64_t uv = (uint64_t)(limbs[NUM_LIMBS - 1 - (i + j)])
                    + (uint64_t)(m_limbs[NUM_LIMBS - 1 - j])
                        * (uint64_t)(rhs.m_limbs[NUM_LIMBS - 1 - i])
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

    constexpr UnsignedInteger operator*=(const UnsignedInteger rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    constexpr UnsignedInteger operator<<(const uint32_t rhs) const
    {
        uint32_t limbs_shift = rhs >> 5;
        UnsignedInteger<NUM_LIMBS> result = {};
        if (limbs_shift >= NUM_LIMBS) {
            return result;
        }
        // rhs % 32;
        uint32_t bit_shift = rhs & 0x1F;
        // applying this leaves us the bits lost when shifting
        uint32_t bitmask = 0xFFFFFFFF - (1 << (32 - bit_shift)) + 1;

        result.m_limbs[0] = m_limbs[limbs_shift] << bit_shift;

        for (uint32_t src = limbs_shift; src < NUM_LIMBS - 1; src++) {
            uint32_t dst = src - limbs_shift;
            result.m_limbs[dst] |= m_limbs[src + 1] & bitmask;
            result.m_limbs[dst + 1] = m_limbs[src + 1] << bit_shift;
        }

        return result;
    }

    constexpr UnsignedInteger operator>>(const uint32_t rhs) const
    {
        uint32_t limbs_shift = rhs >> 5;
        UnsignedInteger<NUM_LIMBS> result = {};
        if (limbs_shift >= NUM_LIMBS) {
            return result;
        }
        // rhs % 32;
        uint32_t bit_shift = rhs & 0x1F;
        // applying this leaves us the bits lost when shifting
        uint32_t bitmask = (1 << bit_shift) - 1;

        result.m_limbs[NUM_LIMBS - 1] = m_limbs[NUM_LIMBS - 1 - limbs_shift] >> bit_shift;

        for (int src = NUM_LIMBS - 1 - limbs_shift; src > 0; src++) {
            uint32_t dst = src + limbs_shift;
            result.m_limbs[dst] |= m_limbs[src - 1] & bitmask;
            result.m_limbs[dst - 1] = m_limbs[src - 1] >> bit_shift;
        }

        return result;
    }

    constexpr bool operator>(const UnsignedInteger rhs) const {
      for (uint64_t i = 0; i < NUM_LIMBS; i++) {
        if (m_limbs[i] > rhs.m_limbs[i]) return true;
        if (m_limbs[i] < rhs.m_limbs[i]) return false;
      }

      return false;
    }

    constexpr bool operator>=(const UnsignedInteger rhs) {
      for (uint64_t i = 0; i < NUM_LIMBS; i++) {
        if (m_limbs[i] > rhs.m_limbs[i]) return true;
        if (m_limbs[i] < rhs.m_limbs[i]) return false;
      }

      return true;
    }

    constexpr bool operator<(const UnsignedInteger rhs) const {
      for (uint64_t i = 0; i < NUM_LIMBS; i++) {
        if (m_limbs[i] > rhs.m_limbs[i]) return false;
        if (m_limbs[i] < rhs.m_limbs[i]) return true;
      }

      return false;
    }

    constexpr bool operator<=(const UnsignedInteger rhs) const {
      for (uint64_t i = 0; i < NUM_LIMBS; i++) {
        if (m_limbs[i] > rhs.m_limbs[i]) return false;
        if (m_limbs[i] < rhs.m_limbs[i]) return true;
      }

      return true;
    }
};

#endif /* unsigned_int_h */
