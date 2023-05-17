#pragma once

template<typename Fp, const uint64_t A_CURVE>
class ECPoint {
public:

    Fp x;
    Fp y;
    Fp z;

    constexpr ECPoint() : ECPoint(ECPoint::neutral_element()) {}
    constexpr ECPoint(Fp _x, Fp _y, Fp _z) : x(_x), y(_y), z(_z) {}

    constexpr ECPoint operator+(const ECPoint other) const
    {
        if (is_neutral_element(*this)) {
            return other;
        }
        if (is_neutral_element(other)) {
            return *this;
        }

        Fp u1 = other.y * z;
        Fp u2 = y * other.z;
        Fp v1 = other.x * z;
        Fp v2 = x * other.z;

        if (v1 == v2) {
            if (u1 != u2 || y == Fp(0)) {
                return neutral_element();
            }

            Fp a_fp = Fp(A_CURVE);

            Fp w = a_fp * z * z + Fp(3) * x * x;
            Fp s = y * z;
            Fp b = x * y * s;
            Fp h = w * w - Fp(8) * b;
            Fp xp = Fp(2) * h * s;
            Fp yp = w * (Fp(4) * b - h) - Fp(8) * y * y * s * s;
            Fp zp = Fp(8) * s * s * s;
              
            return ECPoint(xp, yp, zp);
        }

        Fp u = u1 - u2;
        Fp v = v1 - v2;
        Fp w = z * other.z;
        Fp a = u * u * w - v * v * v - Fp(2) * v * v * v2;
        Fp xp = v * a;
        Fp yp = u * (v * v * v2 - a) - v * v * v * u2;
        Fp zp = v * v * v * w;

        return ECPoint(xp, yp, zp);
    }

    void operator+=(const ECPoint other)
    {
        *this = *this + other;
    }

    ECPoint neutral_element() const
    {
        return ECPoint(Fp(0), Fp().one(), Fp(0));
    }

    ECPoint operate_with_self(uint64_t exponent) const
    {
        ECPoint result = neutral_element();
        ECPoint base = ECPoint(x, y, z);

        while (exponent > 0) {
            if ((exponent & 1) == 1) {
                result = result + base;
            }
            exponent = exponent >> 1;
            base = base + base;
        }

        return result;
    }

    constexpr ECPoint operator*(uint64_t exponent) const
    {
        return (*this).operate_with_self(exponent);
    }

    constexpr void operator*=(uint64_t exponent)
    {
        *this = (*this).operate_with_self(exponent);
    }

    constexpr ECPoint neg()
    {
        return ECPoint(x, y.neg(), z);
    }

    constexpr bool is_neutral_element(const ECPoint a_point) const
    {
        return a_point.x == Fp(0) && a_point.y == Fp().one() && a_point.z == Fp(0);
    }
};
