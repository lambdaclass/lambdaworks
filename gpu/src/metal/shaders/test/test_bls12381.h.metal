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

    result[0] = P.x;
    result[1] = P.y;
    result[2] = P.z;

    result[0] = p[0];
    result[1] = p[1];
    result[2] = p[2];
}

