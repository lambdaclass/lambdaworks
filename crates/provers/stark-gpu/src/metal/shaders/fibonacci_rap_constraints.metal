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

    // Modular inverse via Fermat's little theorem: a^(-1) = a^(p-2) mod p.
    // Uses an optimized addition chain for p-2 = 0xFFFFFFFE_FFFFFFFF.
    // Adapted from the Fp64Goldilocks::inverse() in fp_u64.h.metal.
    GlFp inv() const {
        GlFp x = *this;
        GlFp x2 = x * x;
        GlFp x3 = x2 * x;
        GlFp x7 = exp_acc(x3, x, 1);
        GlFp x63 = exp_acc(x7, x7, 3);
        GlFp x12m1 = exp_acc(x63, x63, 6);
        GlFp x24m1 = exp_acc(x12m1, x12m1, 12);
        GlFp x30m1 = exp_acc(x24m1, x63, 6);
        GlFp x31m1 = exp_acc(x30m1, x, 1);
        GlFp x32m1 = exp_acc(x31m1, x, 1);

        GlFp t = x31m1;
        for (int i = 0; i < 33; i++) {
            t = t * t;
        }
        return t * x32m1;
    }

private:
    // Helper: square `base` n times, then multiply by `tail`.
    static GlFp exp_acc(GlFp base, GlFp tail, uint32_t n) {
        GlFp result = base;
        for (uint32_t i = 0; i < n; i++) {
            result = result * result;
        }
        return result * tail;
    }

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

// Parameters for a single boundary constraint
struct BoundaryConstraintParam {
    GlFp g_pow_step;    // g^step (trace primitive root raised to the step)
    GlFp value;         // expected trace value at that step
    GlFp coefficient;   // alpha_k (random coefficient)
    uint32_t col;       // which trace column (0=main_col_0, 1=main_col_1, 2=aux_col_0)
    uint32_t _pad;      // alignment padding
};

struct BoundaryEvalParams {
    uint32_t num_rows;           // total LDE domain points
    uint32_t num_constraints;    // number of boundary constraints
};

// GPU boundary evaluation kernel.
//
// For each LDE point x_i, computes:
//   sum_k coeff_k * (trace_col_k[i] - value_k) / (x_i - g^step_k)
//
// The inverse 1/(x_i - g^step_k) is computed per-thread via Fermat's little theorem.
[[kernel]] void goldilocks_boundary_eval(
    device const GlFp* lde_coset_points  [[ buffer(0) ]],  // x_i for each LDE point
    device const GlFp* main_col_0       [[ buffer(1) ]],  // main trace column 0
    device const GlFp* main_col_1       [[ buffer(2) ]],  // main trace column 1
    device const GlFp* aux_col_0        [[ buffer(3) ]],  // aux trace column 0
    device const BoundaryConstraintParam* constraints [[ buffer(4) ]],
    constant BoundaryEvalParams& params  [[ buffer(5) ]],
    device GlFp* output                 [[ buffer(6) ]],
    uint tid                            [[ thread_position_in_grid ]]
) {
    if (tid >= params.num_rows) return;

    GlFp x_i = lde_coset_points[tid];
    GlFp acc = GlFp::zero();

    for (uint k = 0; k < params.num_constraints; k++) {
        BoundaryConstraintParam bc = constraints[k];

        // Compute zerofier inverse: 1 / (x_i - g^step)
        GlFp denom = x_i - bc.g_pow_step;
        GlFp inv_denom = denom.inv();

        // Read trace value at this point from the appropriate column
        GlFp trace_val;
        if (bc.col == 0)      trace_val = main_col_0[tid];
        else if (bc.col == 1) trace_val = main_col_1[tid];
        else                  trace_val = aux_col_0[tid];

        // Accumulate: coeff * (trace_val - value) * inv(x_i - g^step)
        acc = acc + bc.coefficient * (trace_val - bc.value) * inv_denom;
    }

    output[tid] = acc;
}

// Parameters for a single boundary constraint in the fused kernel.
struct FusedBoundaryParam {
    GlFp g_pow_step;    // g^step (trace primitive root raised to the step)
    GlFp value;         // expected trace value at that step
    GlFp coefficient;   // alpha_k (random coefficient)
    uint32_t col;       // which trace column (0=main_col_0, 1=main_col_1, 2=aux_col_0)
    uint32_t _pad;
};

// Parameters for the fused transition + boundary kernel.
struct FusedParams {
    uint32_t lde_step_size;   // = step_size * blowup_factor
    uint32_t num_rows;        // total LDE domain size
    uint32_t zerofier_0_len;  // length of transition zerofier 0
    uint32_t zerofier_1_len;  // length of transition zerofier 1
    GlFp gamma;               // RAP challenge
    GlFp transition_coeff_0;  // coefficient for Fibonacci constraint
    GlFp transition_coeff_1;  // coefficient for permutation constraint
    uint32_t num_boundary_constraints;
    uint32_t _pad2;
};

// Fused transition + boundary constraint evaluation kernel.
//
// Combines both evaluation steps into a single GPU dispatch, eliminating:
// - Separate boundary kernel launch overhead
// - Intermediate boundary_evals buffer (num_rows * 8 bytes)
// - Redundant trace column reads (shared between transition + boundary)
//
// Boundary zerofier inversions are computed inline via Fermat's little theorem.
[[kernel]] void fibonacci_rap_fused_eval(
    device const GlFp* main_col_0       [[ buffer(0) ]],
    device const GlFp* main_col_1       [[ buffer(1) ]],
    device const GlFp* aux_col_0        [[ buffer(2) ]],
    device const GlFp* zerofier_0       [[ buffer(3) ]],
    device const GlFp* zerofier_1       [[ buffer(4) ]],
    constant FusedParams& params        [[ buffer(5) ]],
    device const FusedBoundaryParam* bc_params [[ buffer(6) ]],
    device const GlFp* lde_coset_points [[ buffer(7) ]],  // x_i for each LDE point
    device GlFp* output                 [[ buffer(8) ]],
    uint tid                            [[ thread_position_in_grid ]]
) {
    if (tid >= params.num_rows) return;

    uint row0 = tid;
    uint row1 = (tid + params.lde_step_size) % params.num_rows;
    uint row2 = (tid + 2 * params.lde_step_size) % params.num_rows;

    // Read trace values at the three offsets (shared between transition + boundary)
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

    // Boundary constraints: inline Fermat inversion on GPU.
    // Reuses trace values already loaded above.
    GlFp x_i = lde_coset_points[tid];

    for (uint k = 0; k < params.num_boundary_constraints; k++) {
        FusedBoundaryParam bc = bc_params[k];

        GlFp denom = x_i - bc.g_pow_step;
        GlFp inv_denom = denom.inv();

        GlFp trace_val;
        if (bc.col == 0)      trace_val = a0;
        else if (bc.col == 1) trace_val = b0;
        else                  trace_val = z0;

        acc = acc + bc.coefficient * (trace_val - bc.value) * inv_denom;
    }

    output[tid] = acc;
}
