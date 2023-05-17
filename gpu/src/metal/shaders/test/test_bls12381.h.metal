#pragma once

template<typename BLS12381, typename Fp>
[[kernel]] void add(
    constant Fp* p [[ buffer(0) ]],
    constant Fp* q [[ buffer(1) ]],
    device Fp* result [[ buffer(2) ]]
)
{
    BLS12381 P = BLS12381(p[0], p[1], p[2]);
    BLS12381 Q = BLS12381(q[0], q[1], q[2]);
    BLS12381 res = P + Q;

    result[0] = res.x;
    result[1] = res.y;
    result[2] = res.z;
}

template<typename Fp>
[[kernel]] void add_fp(
    constant Fp &_p [[ buffer(0) ]],
    constant Fp &_q [[ buffer(1) ]],
    device Fp &result [[ buffer(2) ]]
)
{
    Fp p = _p;
    Fp q = _q;
    Fp res = p + q;
    result = res;
}

template<typename Fp>
[[kernel]] void sub_fp(
    constant Fp &_p [[ buffer(0) ]],
    constant Fp &_q [[ buffer(1) ]],
    device Fp &result [[ buffer(2) ]]
)
{
    Fp p = _p;
    Fp q = _q;
    Fp res = p - q;
    result = res;
}

template<typename Fp>
[[kernel]] void mul_fp(
    constant Fp &_p [[ buffer(0) ]],
    constant Fp &_q [[ buffer(1) ]],
    device Fp &result [[ buffer(2) ]]
)
{
    Fp p = _p;
    Fp q = _q;
    Fp res = p * q;
    result = res;
}

template<typename Fp>
[[kernel]] void pow_fp(
    constant Fp &_p [[ buffer(0) ]],
    constant uint32_t &_a [[ buffer(1) ]],
    device Fp &result [[ buffer(2) ]]
)
{
    Fp p = _p;
    uint32_t a = _a;
    Fp res = p.pow(a);
    result = res;
}
