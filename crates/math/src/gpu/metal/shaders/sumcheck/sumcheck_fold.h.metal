// Sumcheck fold kernel: folds evaluation tables after receiving a challenge.
//
// Each thread folds one position:
//   table[tid] = table[tid] + challenge * (table[tid + half_len] - table[tid])

#ifndef sumcheck_fold_h
#define sumcheck_fold_h

#include <metal_stdlib>
using namespace metal;

template <typename Fp>
[[kernel]] void sumcheck_fold(
    device Fp* table          [[buffer(0)]],  // Factor evaluation table (modified in-place)
    constant Fp& challenge    [[buffer(1)]],  // Challenge value r
    constant uint32_t& half_len   [[buffer(2)]],  // Half of current table length
    uint tid                  [[thread_position_in_grid]]
) {
    // Copy from constant to thread address space for operator overloads.
    Fp chal = challenge;
    if (tid < half_len) {
        Fp lo = table[tid];
        Fp hi = table[tid + half_len];
        Fp diff = hi - lo;
        table[tid] = lo + chal * diff;
    }
}

#endif /* sumcheck_fold_h */
