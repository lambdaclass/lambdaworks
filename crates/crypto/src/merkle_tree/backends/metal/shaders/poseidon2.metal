// Poseidon2 hash function for Goldilocks field (p = 2^64 - 2^32 + 1).
//
// Metal GPU compute shader implementing the Poseidon2 permutation with width 8,
// x^7 S-box, and Horizen Labs / Plonky3 variant linear layers.
//
// Configuration:
//   - Field: Goldilocks (p = 2^64 - 2^32 + 1)
//   - Width: 8
//   - External rounds: 4 (initial) + 4 (terminal) = 8
//   - Internal rounds: 22
//   - S-box: x^7
//   - Output: 2 field elements (128-bit digest)

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Goldilocks field arithmetic (Fp64Goldilocks)
// ============================================================================
// Copied from fp_u64.h.metal because DynamicMetalState compiles from source
// at runtime and cannot #include headers from other crates.

constant uint64_t GOLDILOCKS_EPSILON = 0xFFFFFFFF;
constant uint64_t GOLDILOCKS_PRIME = 0xFFFFFFFF00000001;

class Fp64Goldilocks {
public:
    using raw_type = uint64_t;

    Fp64Goldilocks() = default;
    constexpr Fp64Goldilocks(uint64_t v) : inner(v) {}

    constexpr explicit operator uint64_t() const { return inner; }

    static Fp64Goldilocks zero() { return Fp64Goldilocks(0); }
    static Fp64Goldilocks one() { return Fp64Goldilocks(1); }

    Fp64Goldilocks operator+(const Fp64Goldilocks rhs) const {
        uint64_t sum = inner + rhs.inner;
        bool over = sum < inner;
        uint64_t sum2 = sum + (over ? GOLDILOCKS_EPSILON : 0);
        bool over2 = over && (sum2 < sum);
        return Fp64Goldilocks(sum2 + (over2 ? GOLDILOCKS_EPSILON : 0));
    }

    Fp64Goldilocks operator-(const Fp64Goldilocks rhs) const {
        uint64_t diff = inner - rhs.inner;
        bool under = inner < rhs.inner;
        uint64_t diff2 = diff - (under ? GOLDILOCKS_EPSILON : 0);
        bool under2 = under && (diff2 > diff);
        return Fp64Goldilocks(diff2 - (under2 ? GOLDILOCKS_EPSILON : 0));
    }

    Fp64Goldilocks operator*(const Fp64Goldilocks rhs) const {
        uint64_t lo = inner * rhs.inner;
        uint64_t hi = metal::mulhi(inner, rhs.inner);
        return reduce128(lo, hi);
    }

private:
    uint64_t inner;

    static Fp64Goldilocks reduce128(uint64_t lo, uint64_t hi) {
        uint64_t hi_hi = hi >> 32;
        uint64_t hi_lo = hi & GOLDILOCKS_EPSILON;

        uint64_t t0 = lo - hi_hi;
        bool borrow = lo < hi_hi;
        t0 = borrow ? t0 - GOLDILOCKS_EPSILON : t0;

        uint64_t t1 = (hi_lo << 32) - hi_lo;

        uint64_t result = t0 + t1;
        bool carry = result < t0;
        result = carry ? result + GOLDILOCKS_EPSILON : result;

        return Fp64Goldilocks(result);
    }
};

using Fp = Fp64Goldilocks;

// ============================================================================
// Poseidon2 constants
// ============================================================================

// Diagonal matrix for internal linear layer (width 8)
constant uint64_t MATRIX_DIAG_8[8] = {
    0xa98811a1fed4e3a5, 0x1cc48b54f377e2a0, 0xe40cd4f6c5609a26, 0x11de79ebca97a4a3,
    0x9177c73d8b7e929c, 0x2a6fe8085797e791, 0x3de6e93329f8d5ad, 0x3f7af9125da962fe,
};

// External round constants - initial 4 rounds (4 x 8)
constant uint64_t EXTERNAL_RC_INIT[4][8] = {
    {0xdd5743e7f2a5a5d9, 0xcb3a864e58ada44b, 0xffa2449ed32f8cdc, 0x42025f65d6bd13ee,
     0x7889175e25506323, 0x34b98bb03d24b737, 0xbdcc535ecc4faa2a, 0x5b20ad869fc0d033},
    {0xf1dda5b9259dfcb4, 0x27515210be112d59, 0x4227d1718c766c3f, 0x26d333161a5bd794,
     0x49b938957bf4b026, 0x4a56b5938b213669, 0x1120426b48c8353d, 0x6b323c3f10a56cad},
    {0xce57d6245ddca6b2, 0xb1fc8d402bba1eb1, 0xb5c5096ca959bd04, 0x6db55cd306d31f7f,
     0xc49d293a81cb9641, 0x1ce55a4fe979719f, 0xa92e60a9d178a4d1, 0x002cc64973bcfd8c},
    {0xcea721cce82fb11b, 0xe5b55eb8098ece81, 0x4e30525c6f1ddd66, 0x43c6702827070987,
     0xaca68430a7b5762a, 0x3674238634df9c93, 0x88cee1c825e33433, 0xde99ae8d74b57176},
};

// External round constants - terminal 4 rounds (4 x 8)
constant uint64_t EXTERNAL_RC_TERM[4][8] = {
    {0x014ef1197d341346, 0x9725e20825d07394, 0xfdb25aef2c5bae3b, 0xbe5402dc598c971e,
     0x93a5711f04cdca3d, 0xc45a9a5b2f8fb97b, 0xfe8946a924933545, 0x2af997a27369091c},
    {0xaa62c88e0b294011, 0x058eb9d810ce9f74, 0xb3cb23eced349ae4, 0xa3648177a77b4a84,
     0x43153d905992d95d, 0xf4e2a97cda44aa4b, 0x5baa2702b908682f, 0x082923bdf4f750d1},
    {0x98ae09a325893803, 0xf8a6475077968838, 0xceb0735bf00b2c5f, 0x0a1a5d953888e072,
     0x2fcb190489f94475, 0xb5be06270dec69fc, 0x739cb934b09acf8b, 0x537750b75ec7f25b},
    {0xe9dd318bae1f3961, 0xf7462137299efe1a, 0xb1f6b8eee9adb940, 0xbdebcc8a809dfe6b,
     0x40fc1f791b178113, 0x3ac1c3362d014864, 0x9a016184bdb8aeba, 0x95f2394459fbc25e},
};

// Internal round constants (22 rounds, applied to state[0] only)
constant uint64_t INTERNAL_RC[22] = {
    0x488897d85ff51f56, 0x1140737ccb162218, 0xa7eeb9215866ed35, 0x9bd2976fee49fcc9,
    0xc0c8f0de580a3fcc, 0x4fb2dae6ee8fc793, 0x343a89f35f37395b, 0x223b525a77ca72c8,
    0x56ccb62574aaa918, 0xc4d507d8027af9ed, 0xa080673cf0b7e95c, 0xf0184884eb70dcf8,
    0x044f10b0cb3d5c69, 0xe9e3f7993938f186, 0x1b761c80e772f459, 0x606cec607a1b5fac,
    0x14a0c2e1d45f03cd, 0x4eace8855398574f, 0xf905ca7103eff3e6, 0xf8c8f8d20862c059,
    0xb524fe8bdd678e5a, 0xfbb7865901a1ec41,
};

// ============================================================================
// Poseidon2 functions
// ============================================================================

// S-box: x^7 = x * x^2 * x^4
inline Fp sbox(Fp x) {
    Fp x2 = x * x;
    Fp x4 = x2 * x2;
    return x * x2 * x4;
}

// Horizen Labs 4x4 MDS matrix applied in-place
// Matrix: [[5,7,1,3], [4,6,1,1], [1,3,5,7], [1,1,4,6]]
inline void apply_hl_mat4(thread Fp* x) {
    Fp t0 = x[0] + x[1];
    Fp t1 = x[2] + x[3];
    Fp t2 = (x[1] + x[1]) + t1;
    Fp t3 = (x[3] + x[3]) + t0;
    Fp t1d = t1 + t1;
    Fp t4 = (t1d + t1d) + t3;
    Fp t0d = t0 + t0;
    Fp t5 = (t0d + t0d) + t2;
    Fp t6 = t3 + t5;
    Fp t7 = t2 + t4;

    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

// External linear layer: apply 4x4 MDS to each half, then cross-diffuse
inline void external_linear_layer(thread Fp* state) {
    apply_hl_mat4(state);
    apply_hl_mat4(state + 4);

    for (int i = 0; i < 4; i++) {
        Fp sum = state[i] + state[i + 4];
        state[i] = state[i] + sum;
        state[i + 4] = state[i + 4] + sum;
    }
}

// Internal linear layer: y_i = diag_i * x_i + sum(x_j)
inline void internal_linear_layer(thread Fp* state) {
    Fp sum = Fp::zero();
    for (int i = 0; i < 8; i++) {
        sum = sum + state[i];
    }

    for (int i = 0; i < 8; i++) {
        state[i] = Fp(MATRIX_DIAG_8[i]) * state[i] + sum;
    }
}

// External round: add constants + S-box all elements + external linear layer
inline void external_round(thread Fp* state, constant uint64_t rc[8]) {
    for (int i = 0; i < 8; i++) {
        state[i] = state[i] + Fp(rc[i]);
    }
    for (int i = 0; i < 8; i++) {
        state[i] = sbox(state[i]);
    }
    external_linear_layer(state);
}

// Internal round: add constant to state[0] + S-box state[0] + internal linear layer
inline void internal_round(thread Fp* state, uint64_t rc) {
    state[0] = state[0] + Fp(rc);
    state[0] = sbox(state[0]);
    internal_linear_layer(state);
}

// Full Poseidon2 permutation (Plonky3 variant: initial linear layer before rounds)
inline void poseidon2_permute(thread Fp* state) {
    // Initial external linear layer (Plonky3-specific)
    external_linear_layer(state);

    // Initial 4 external rounds
    for (int r = 0; r < 4; r++) {
        external_round(state, EXTERNAL_RC_INIT[r]);
    }

    // 22 internal rounds
    for (int r = 0; r < 22; r++) {
        internal_round(state, INTERNAL_RC[r]);
    }

    // Terminal 4 external rounds
    for (int r = 0; r < 4; r++) {
        external_round(state, EXTERNAL_RC_TERM[r]);
    }
}

// ============================================================================
// Compute kernels
// ============================================================================

// Hash single field element: state = [input, 0, 0, 0, 0, 0, 0, 1], permute, return [state[0], state[1]]
kernel void hash_single_kernel(
    device const uint64_t* input [[buffer(0)]],
    device uint64_t* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    Fp state[8];
    state[0] = Fp(input[gid]);
    state[1] = Fp::zero();
    state[2] = Fp::zero();
    state[3] = Fp::zero();
    state[4] = Fp::zero();
    state[5] = Fp::zero();
    state[6] = Fp::zero();
    state[7] = Fp(1);  // Domain tag = 1

    poseidon2_permute(state);

    output[gid * 2]     = (uint64_t)state[0];
    output[gid * 2 + 1] = (uint64_t)state[1];
}

// Compress two digests: state = [left0, left1, right0, right1, 0, 0, 0, 4], permute, return [state[0], state[1]]
kernel void compress_kernel(
    device const uint64_t* input [[buffer(0)]],
    device uint64_t* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint base = gid * 4;
    Fp state[8];
    state[0] = Fp(input[base]);
    state[1] = Fp(input[base + 1]);
    state[2] = Fp(input[base + 2]);
    state[3] = Fp(input[base + 3]);
    state[4] = Fp::zero();
    state[5] = Fp::zero();
    state[6] = Fp::zero();
    state[7] = Fp(4);  // Domain tag = 4

    poseidon2_permute(state);

    output[gid * 2]     = (uint64_t)state[0];
    output[gid * 2 + 1] = (uint64_t)state[1];
}
