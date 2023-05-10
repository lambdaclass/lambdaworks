#ifndef ec_point_h
#define ec_point_h

#include "ec_point.h.metal"

template<typename Fp, const uint64_t a_curve>
class ECPoint {
public:

    Fp x;
    Fp y;
    Fp z;
    Fp a_fp;

    ECPoint() = default;
    constexpr ECPoint(Fp _x, Fp _y, Fp _z) : x(_x), y(_y), z(_z), a_fp(a_curve){}

    constexpr ECPoint operator+(const ECPoint other) const
    {
        if(is_neutral_element(*this)){
            return other;
        }
        if(is_neutral_element(other)){
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

            Fp w = a_fp * z.pow(2) + Fp(3) * x.pow(2);
            Fp s = y * z;
            Fp b = x * y * s;
            Fp h = w.pow(2) - Fp(8) * b;
            Fp xp = Fp(2) * h * s;
            Fp yp = w * (Fp(4) * b - h) - Fp(8) * y.pow(2) * s.pow(2);
            Fp zp = Fp(8) * s.pow(3);
              
            return ECPoint(xp, yp, zp);
        }

        Fp u = u1 - u2;
        Fp v = v1 - v2;
        Fp w = z * other.z;
        Fp a = u.pow(2) * w - v.pow(3) - Fp(2) * v.pow(2) * v2;
        Fp xp = v * a;
        Fp yp = u * (v.pow(2) * v2 - a) - v.pow(3) * u2;
        Fp zp = v.pow(3) * w;

        return ECPoint(xp, yp, zp);
    }

    ECPoint neutral_element() const
    {
        return ECPoint(Fp(0), Fp(1), Fp(0));
    }

    bool is_neutral_element(const ECPoint a_point) const
    {
        return a_point.x == Fp(0) && a_point.y == Fp(1) && a_point.z == Fp(0);
    }
};

#endif
