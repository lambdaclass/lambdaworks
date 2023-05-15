#pragma once

template<typename BLS12381, typename Fp>
[[kernel]] void add(
    constant Fp* p [[ buffer(0) ]],
    constant Fp* q [[ buffer(1) ]],
    device Fp* result [[ buffer(2) ]]
)
{
    // BLS12381 P = BLS12381(p[0], p[1], p[2]);
    // BLS12381 Q = BLS12381(q[0], q[1], q[2]);
    // BLS12381 res = P + Q;

    //result[0] = res.x;
    //result[1] = res.y;
    //result[2] = res.z;
    result[0] = Fp::one();
}

template<typename Fp>
[[kernel]] void add_fp(
    constant Fp* p [[ buffer(0) ]],
    constant Fp* q [[ buffer(1) ]],
    device Fp* result [[ buffer(2) ]]
)
{
    Fp fp_p = p[0];
    Fp fp_q = q[0];
    Fp res = fp_p + fp_q;
    result[0] = res;
}
