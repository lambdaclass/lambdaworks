#include "./fields/fp_u256.cuh"
#include "./utils.h"

namespace p256
{
    // StarkWare field for Cairo
    // P =
    // 3618502788666131213697322783095070105623107215331596699973092056135872020481
    using Fp = Fp256<
        /* =N **/ /*u256(*/ 576460752303423505, 0, 0, 1 /*)*/,
        /* =R_SQUARED **/ /*u256(*/ 576413109808302096, 18446744073700081664,
        5151653887, 18446741271209837569 /*)*/,
        /* =N_PRIME **/ /*u256(*/ 576460752303423504, 18446744073709551615,
        18446744073709551615, 18446744073709551615 /*)*/
        >;
} // namespace p256

extern "C"
{
    // NOTE: In order to calculate the inverse twiddles, call with _omega = _omega.inverse()
    __global__ void calc_twiddles(p256::Fp *result,
                                  const p256::Fp &_omega)
    {
        int index = threadIdx.x;

        p256::Fp omega = _omega;
        result[index] = omega.pow((unsigned)index);
    };

    // NOTE: In order to calculate the inverse twiddles, call with _omega = _omega.inverse()
    __global__ void calc_twiddles_bitrev(p256::Fp *result,
                                         const p256::Fp &_omega)
    {
        int index = threadIdx.x;
        int size = blockDim.x;

        p256::Fp omega = _omega;
        result[index] = omega.pow(reverse_index((unsigned)index, (unsigned)size));
    };

}
