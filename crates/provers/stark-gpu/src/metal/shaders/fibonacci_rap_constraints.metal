#include <metal_stdlib>
using namespace metal;

// Goldilocks field arithmetic
// Prime p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
// GL_EPSILON = 2^32 - 1 = 0xFFFFFFFF (useful for reduction)
constant uint64_t GL_EPSILON = 0xFFFFFFFF;
constant uint64_t GL_PRIME = 0xFFFFFFFF00000001UL;

struct GlFp {
    uint64_t inner;

    GlFp() = default;
    constexpr GlFp(uint64_t v) : inner(v) {}

    static GlFp zero() { return GlFp(0); }

    GlFp operator+(const GlFp rhs) const {
        uint64_t sum = inner + rhs.inner;
        bool over = sum < inner;
        uint64_t sum2 = sum + (over ? GL_EPSILON : 0);
        bool over2 = over && (sum2 < sum);
        return GlFp(sum2 + (over2 ? GL_EPSILON : 0));
    }

    GlFp operator-(const GlFp rhs) const {
        uint64_t diff = inner - rhs.inner;
        bool under = inner < rhs.inner;
        uint64_t diff2 = diff - (under ? GL_EPSILON : 0);
        bool under2 = under && (diff2 > diff);
        return GlFp(diff2 - (under2 ? GL_EPSILON : 0));
    }

    GlFp operator*(const GlFp rhs) const {
        uint64_t lo = inner * rhs.inner;
        uint64_t hi = metal::mulhi(inner, rhs.inner);
        return reduce128(lo, hi);
    }

private:
    static GlFp reduce128(uint64_t lo, uint64_t hi) {
        // Reduce a 128-bit product (hi:lo) modulo the Goldilocks prime.
        // p = 2^64 - 2^32 + 1, so 2^64 = 2^32 - 1 + p (mod p)
        // hi * 2^64 + lo = hi * (2^32 - 1) + lo  (mod p)
        uint64_t hi_hi = hi >> 32;
        uint64_t hi_lo = hi & GL_EPSILON;
        // t0 = lo - hi_hi (mod p)
        uint64_t t0 = lo - hi_hi;
        bool borrow = lo < hi_hi;
        t0 = borrow ? t0 - GL_EPSILON : t0;
        // t1 = hi_lo * 2^32 - hi_lo = hi_lo * (2^32 - 1)
        uint64_t t1 = (hi_lo << 32) - hi_lo;
        uint64_t result = t0 + t1;
        bool carry = result < t0;
        result = carry ? result + GL_EPSILON : result;
        return GlFp(result);
    }
};

struct FibRapParams {
    uint32_t lde_step_size;   // = step_size * blowup_factor (typically 4)
    uint32_t num_rows;        // total LDE domain size
    uint32_t zerofier_0_len; // length of transition zerofier 0
    uint32_t zerofier_1_len; // length of transition zerofier 1
    GlFp gamma;               // RAP challenge
    GlFp transition_coeff_0;  // coefficient for Fibonacci constraint
    GlFp transition_coeff_1;  // coefficient for permutation constraint
};

[[kernel]] void fibonacci_rap_constraint_eval(
    device const GlFp* main_col_0     [[ buffer(0) ]],  // column a (Fibonacci sequence)
    device const GlFp* main_col_1     [[ buffer(1) ]],  // column b (permuted sequence)
    device const GlFp* aux_col_0      [[ buffer(2) ]],  // column z (permutation argument)
    device const GlFp* zerofier_0     [[ buffer(3) ]],  // transition zerofier for constraint 0
    device const GlFp* zerofier_1     [[ buffer(4) ]],  // transition zerofier for constraint 1
    constant FibRapParams& params     [[ buffer(5) ]],
    device const GlFp* boundary_evals [[ buffer(6) ]],  // pre-computed boundary evaluations
    device GlFp* output               [[ buffer(7) ]],  // result
    uint tid                          [[ thread_position_in_grid ]]
) {
    if (tid >= params.num_rows) return;

    uint row0 = tid;
    uint row1 = (tid + params.lde_step_size) % params.num_rows;
    uint row2 = (tid + 2 * params.lde_step_size) % params.num_rows;

    // Read trace values at the three offsets
    GlFp a0 = main_col_0[row0];
    GlFp a1 = main_col_0[row1];
    GlFp a2 = main_col_0[row2];
    GlFp b0 = main_col_1[row0];
    GlFp z0 = aux_col_0[row0];
    GlFp z1 = aux_col_0[row1];

    // Transition constraint 0 (Fibonacci): a[i+2] - a[i+1] - a[i] = 0
    GlFp fib_eval = a2 - a1 - a0;

    // Transition constraint 1 (Permutation): z[i+1]*(b[i]+gamma) - z[i]*(a[i]+gamma) = 0
    GlFp perm_eval = z1 * (b0 + params.gamma) - z0 * (a0 + params.gamma);

    // Multiply each constraint evaluation by its zerofier and coefficient
    GlFp z0_val = zerofier_0[tid % params.zerofier_0_len];
    GlFp z1_val = zerofier_1[tid % params.zerofier_1_len];

    GlFp acc = fib_eval * z0_val * params.transition_coeff_0
             + perm_eval * z1_val * params.transition_coeff_1;

    // Add pre-computed boundary evaluations (computed on CPU)
    output[tid] = acc + boundary_evals[tid];
}
