#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Keccak-256 implementation for Metal GPU (optimized for register pressure)
//
// Uses scalar state variables instead of arrays to avoid register spills
// on Apple Silicon GPUs. The keccak_f1600 permutation is fully unrolled
// with combined ρ+π+χ steps to minimize live variable count.
// =============================================================================

// Keccak-f[1600] round constants (iota step)
constant uint64_t KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL
};

inline uint64_t rotl64(uint64_t x, uint n) {
    return (x << n) | (x >> (64 - n));
}

// Byte-swap u64 for big-endian ↔ little-endian conversion.
// Goldilocks as_bytes() is big-endian; Keccak lanes are little-endian.
inline uint64_t bswap64(uint64_t x) {
    x = ((x & 0x00FF00FF00FF00FFULL) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFULL);
    x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFULL);
    return (x << 32) | (x >> 32);
}

// Keccak-f[1600] permutation: 24 rounds on a 5x5 state of uint64_t.
// Uses scalar variables to help the Metal compiler keep state in registers.
//
// PI mapping (source index → B destination index):
//   PI_INDICES = [0, 10, 20, 5, 15, 16, 1, 11, 21, 6,
//                 7, 17,  2, 12, 22, 23,  8, 18,  3, 13,
//                14, 24,  9, 19,  4]
// ROT_OFFSETS = [0,  1, 62, 28, 27, 36, 44,  6, 55, 20,
//                3, 10, 43, 25, 39, 41, 45, 15, 21,  8,
//               18,  2, 61, 56, 14]
inline void keccak_f1600(thread uint64_t* s) {
    uint64_t s00=s[0],  s01=s[1],  s02=s[2],  s03=s[3],  s04=s[4];
    uint64_t s05=s[5],  s06=s[6],  s07=s[7],  s08=s[8],  s09=s[9];
    uint64_t s10=s[10], s11=s[11], s12=s[12], s13=s[13], s14=s[14];
    uint64_t s15=s[15], s16=s[16], s17=s[17], s18=s[18], s19=s[19];
    uint64_t s20=s[20], s21=s[21], s22=s[22], s23=s[23], s24=s[24];

    for (uint round = 0; round < 24; round++) {
        // θ (theta)
        uint64_t c0 = s00 ^ s05 ^ s10 ^ s15 ^ s20;
        uint64_t c1 = s01 ^ s06 ^ s11 ^ s16 ^ s21;
        uint64_t c2 = s02 ^ s07 ^ s12 ^ s17 ^ s22;
        uint64_t c3 = s03 ^ s08 ^ s13 ^ s18 ^ s23;
        uint64_t c4 = s04 ^ s09 ^ s14 ^ s19 ^ s24;

        uint64_t d0 = c4 ^ rotl64(c1, 1);
        uint64_t d1 = c0 ^ rotl64(c2, 1);
        uint64_t d2 = c1 ^ rotl64(c3, 1);
        uint64_t d3 = c2 ^ rotl64(c4, 1);
        uint64_t d4 = c3 ^ rotl64(c0, 1);

        s00 ^= d0; s01 ^= d1; s02 ^= d2; s03 ^= d3; s04 ^= d4;
        s05 ^= d0; s06 ^= d1; s07 ^= d2; s08 ^= d3; s09 ^= d4;
        s10 ^= d0; s11 ^= d1; s12 ^= d2; s13 ^= d3; s14 ^= d4;
        s15 ^= d0; s16 ^= d1; s17 ^= d2; s18 ^= d3; s19 ^= d4;
        s20 ^= d0; s21 ^= d1; s22 ^= d2; s23 ^= d3; s24 ^= d4;

        // ρ+π+χ combined, row by row.
        // Each row reads from the post-θ state via ρ+π, then writes χ output.
        // Since χ output overwrites state variables that later rows need,
        // we save conflicting values before each row's writes.

        // Save values that Row 0 will overwrite (s00..s04)
        uint64_t t01 = s01; // Row 2 needs A[1]
        uint64_t t02 = s02; // Row 4 needs A[2]
        uint64_t t03 = s03; // Row 1 needs A[3]
        uint64_t t04 = s04; // Row 3 needs A[4]

        // Row 0: B[0,1,2,3,4] ← A[0], A[6], A[12], A[18], A[24]
        uint64_t b0 = s00;
        uint64_t b1 = rotl64(s06, 44);
        uint64_t b2 = rotl64(s12, 43);
        uint64_t b3 = rotl64(s18, 21);
        uint64_t b4 = rotl64(s24, 14);
        s00 = b0 ^ ((~b1) & b2);
        s01 = b1 ^ ((~b2) & b3);
        s02 = b2 ^ ((~b3) & b4);
        s03 = b3 ^ ((~b4) & b0);
        s04 = b4 ^ ((~b0) & b1);

        // Save values that Row 1 will overwrite (s05..s09)
        uint64_t t05 = s05; // Row 3 needs A[5]
        uint64_t t07 = s07; // Row 2 needs A[7]
        uint64_t t08 = s08; // Row 4 needs A[8]

        // Row 1: B[5,6,7,8,9] ← A[3], A[9], A[10], A[16], A[22]
        b0 = rotl64(t03, 28);
        b1 = rotl64(s09, 20);
        b2 = rotl64(s10, 3);
        b3 = rotl64(s16, 45);
        b4 = rotl64(s22, 61);
        s05 = b0 ^ ((~b1) & b2);
        s06 = b1 ^ ((~b2) & b3);
        s07 = b2 ^ ((~b3) & b4);
        s08 = b3 ^ ((~b4) & b0);
        s09 = b4 ^ ((~b0) & b1);

        // Save values that Row 2 will overwrite (s10..s14)
        uint64_t t11 = s11; // Row 3 needs A[11]
        uint64_t t14 = s14; // Row 4 needs A[14]

        // Row 2: B[10,11,12,13,14] ← A[1], A[7], A[13], A[19], A[20]
        b0 = rotl64(t01, 1);
        b1 = rotl64(t07, 6);
        b2 = rotl64(s13, 25);
        b3 = rotl64(s19, 8);
        b4 = rotl64(s20, 18);
        s10 = b0 ^ ((~b1) & b2);
        s11 = b1 ^ ((~b2) & b3);
        s12 = b2 ^ ((~b3) & b4);
        s13 = b3 ^ ((~b4) & b0);
        s14 = b4 ^ ((~b0) & b1);

        // Save values that Row 3 will overwrite (s15..s19)
        uint64_t t15 = s15; // Row 4 needs A[15]

        // Row 3: B[15,16,17,18,19] ← A[4], A[5], A[11], A[17], A[23]
        b0 = rotl64(t04, 27);
        b1 = rotl64(t05, 36);
        b2 = rotl64(t11, 10);
        b3 = rotl64(s17, 15);
        b4 = rotl64(s23, 56);
        s15 = b0 ^ ((~b1) & b2);
        s16 = b1 ^ ((~b2) & b3);
        s17 = b2 ^ ((~b3) & b4);
        s18 = b3 ^ ((~b4) & b0);
        s19 = b4 ^ ((~b0) & b1);

        // Row 4: B[20,21,22,23,24] ← A[2], A[8], A[14], A[15], A[21]
        b0 = rotl64(t02, 62);
        b1 = rotl64(t08, 55);
        b2 = rotl64(t14, 39);
        b3 = rotl64(t15, 41);
        b4 = rotl64(s21, 2);
        s20 = b0 ^ ((~b1) & b2);
        s21 = b1 ^ ((~b2) & b3);
        s22 = b2 ^ ((~b3) & b4);
        s23 = b3 ^ ((~b4) & b0);
        s24 = b4 ^ ((~b0) & b1);

        // ι (iota)
        s00 ^= KECCAK_RC[round];
    }

    s[0]=s00; s[1]=s01; s[2]=s02; s[3]=s03; s[4]=s04;
    s[5]=s05; s[6]=s06; s[7]=s07; s[8]=s08; s[9]=s09;
    s[10]=s10; s[11]=s11; s[12]=s12; s[13]=s13; s[14]=s14;
    s[15]=s15; s[16]=s16; s[17]=s17; s[18]=s18; s[19]=s19;
    s[20]=s20; s[21]=s21; s[22]=s22; s[23]=s23; s[24]=s24;
}

// =============================================================================
// Kernel 1: Hash leaves
//
// Each thread hashes one row of LDE data (one leaf).
// Input: flat row-major u64 array, where row i starts at lde_data[i * num_cols]
// Each u64 is serialized as 8 bytes big-endian (matching Goldilocks as_bytes())
// Output: 32 bytes per leaf at output[gid * 32]
//
// Optimized: since each field element is exactly one Keccak lane (8 bytes),
// we pack lanes directly using bswap64 instead of byte-by-byte absorption.
// =============================================================================

[[kernel]] void keccak256_hash_leaves(
    device const ulong* lde_data    [[ buffer(0) ]],
    device uchar*        output     [[ buffer(1) ]],
    constant uint&       num_cols   [[ buffer(2) ]],
    constant uint&       num_rows   [[ buffer(3) ]],
    uint gid                        [[ thread_position_in_grid ]]
) {
    if (gid >= num_rows) return;

    uint64_t state[25] = {0};
    uint data_offset = gid * num_cols;

    // Each u64 field element = 8 big-endian bytes = 1 Keccak lane (byte-swapped).
    // Rate = 17 lanes (136 bytes). Process in chunks of 17 columns.

    uint col = 0;

    // Full rate blocks (17 columns per block)
    while (col + 17 <= num_cols) {
        for (uint i = 0; i < 17; i++) {
            state[i] ^= bswap64(lde_data[data_offset + col + i]);
        }
        keccak_f1600(state);
        col += 17;
    }

    // Remaining columns (partial block, always < 17)
    uint remaining = num_cols - col;
    for (uint i = 0; i < remaining; i++) {
        state[i] ^= bswap64(lde_data[data_offset + col + i]);
    }

    // Padding at byte position remaining * 8
    // Since remaining < 17, pad_lane = remaining, pad_byte = 0
    state[remaining] ^= (uint64_t)0x01;  // 0x01 at byte (remaining * 8)

    // Last byte of rate block (byte 135 = lane 16, byte 7)
    state[16] ^= (uint64_t)0x80 << 56;

    keccak_f1600(state);

    // Squeeze 32 bytes: write 4 lanes as little-endian uint64_t
    device ulong* out64 = (device ulong*)(output + gid * 32);
    out64[0] = state[0];
    out64[1] = state[1];
    out64[2] = state[2];
    out64[3] = state[3];
}

// =============================================================================
// Kernel 2: Hash pairs of child nodes into parent nodes
//
// Each thread takes two consecutive 32-byte child hashes and produces
// one 32-byte parent hash: parent[gid] = Keccak256(child[2*gid] || child[2*gid+1])
//
// Optimized: children hashes are in Keccak lane format (little-endian u64),
// so we load them as uint64_t directly without byte-by-byte packing.
// =============================================================================

[[kernel]] void keccak256_hash_pairs(
    device const uchar* children   [[ buffer(0) ]],
    device uchar*       parents    [[ buffer(1) ]],
    constant uint&      num_pairs  [[ buffer(2) ]],
    uint gid                       [[ thread_position_in_grid ]]
) {
    if (gid >= num_pairs) return;

    // Load 64 bytes (8 lanes) directly as uint64_t
    device const ulong* in64 = (device const ulong*)(children + gid * 64);

    uint64_t state[25] = {0};
    state[0] = in64[0];
    state[1] = in64[1];
    state[2] = in64[2];
    state[3] = in64[3];
    state[4] = in64[4];
    state[5] = in64[5];
    state[6] = in64[6];
    state[7] = in64[7];

    // Padding at byte 64 (lane 8, byte 0)
    state[8] ^= (uint64_t)0x01;

    // Last byte of rate block (byte 135 = lane 16, byte 7)
    state[16] ^= (uint64_t)0x80 << 56;

    keccak_f1600(state);

    // Squeeze 32 bytes as 4 uint64_t lanes
    device ulong* out64 = (device ulong*)(parents + gid * 32);
    out64[0] = state[0];
    out64[1] = state[1];
    out64[2] = state[2];
    out64[3] = state[3];
}

// =============================================================================
// Kernel 3: Grinding nonce search
// =============================================================================

[[kernel]] void keccak256_grind_nonce(
    device const uchar*   inner_hash    [[ buffer(0) ]],
    device atomic_uint*   result        [[ buffer(1) ]],
    constant ulong&       limit         [[ buffer(2) ]],
    constant ulong&       batch_offset  [[ buffer(3) ]],
    uint gid                            [[ thread_position_in_grid ]]
) {
    ulong nonce = batch_offset + (ulong)gid;

    uint64_t state[25] = {0};

    // Absorb inner_hash[32] as 4 lanes
    device const ulong* ih64 = (device const ulong*)inner_hash;
    state[0] = ih64[0];
    state[1] = ih64[1];
    state[2] = ih64[2];
    state[3] = ih64[3];

    // Nonce as 8 bytes big-endian → 1 lane byte-swapped
    state[4] ^= bswap64(nonce);

    // Padding at byte 40 (lane 5, byte 0)
    state[5] ^= (uint64_t)0x01;
    // Last byte of rate block (byte 135 = lane 16, byte 7)
    state[16] ^= (uint64_t)0x80 << 56;

    keccak_f1600(state);

    // Extract first 8 bytes of digest as big-endian u64
    ulong digest_head = bswap64(state[0]);

    if (digest_head < limit) {
        atomic_fetch_min_explicit(&result[0], gid, memory_order_relaxed);
    }
}

// =============================================================================
// Kernel 4: Column-to-row transpose with bit-reversal
// =============================================================================

[[kernel]] void transpose_bitrev_goldilocks(
    device const ulong* columns  [[ buffer(0) ]],
    device ulong*       rows     [[ buffer(1) ]],
    constant uint&      num_cols [[ buffer(2) ]],
    constant uint&      num_rows [[ buffer(3) ]],
    uint gid                     [[ thread_position_in_grid ]]
) {
    if (gid >= num_rows) return;

    // Compute log2(num_rows)
    uint log2_n = 0;
    {
        uint tmp = num_rows;
        while (tmp > 1) { tmp >>= 1; log2_n++; }
    }

    // Bit-reverse gid to get source row
    uint src_row = 0;
    {
        uint val = gid;
        for (uint i = 0; i < log2_n; i++) {
            src_row = (src_row << 1) | (val & 1);
            val >>= 1;
        }
    }

    uint out_base = gid * num_cols;
    for (uint c = 0; c < num_cols; c++) {
        rows[out_base + c] = columns[c * num_rows + src_row];
    }
}
