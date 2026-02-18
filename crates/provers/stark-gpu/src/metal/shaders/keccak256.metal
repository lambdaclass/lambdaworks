#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Keccak-256 implementation for Metal GPU
//
// Implements the full Keccak sponge construction matching the Rust sha3::Keccak256
// crate (which uses Keccak-256 with NIST padding variant: 0x01 ... 0x80).
//
// Three components:
// 1. keccak_f1600() - The Keccak-f[1600] permutation (24 rounds)
// 2. keccak256_hash_leaves - Kernel: hash field element rows into 32-byte digests
// 3. keccak256_hash_pairs - Kernel: hash pairs of 32-byte children into parent nodes
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

// Keccak rotation offsets for rho step
constant uint ROT_OFFSETS[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

// Pi step permutation indices
constant uint PI_INDICES[25] = {
     0, 10, 20,  5, 15,
    16,  1, 11, 21,  6,
     7, 17,  2, 12, 22,
    23,  8, 18,  3, 13,
    14, 24,  9, 19,  4
};

// Rotate left for uint64_t
inline uint64_t rotl64(uint64_t x, uint n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] permutation: 24 rounds on a 5x5 state of uint64_t
inline void keccak_f1600(thread uint64_t* state) {
    for (uint round = 0; round < 24; round++) {
        // θ (theta) step
        uint64_t C[5];
        for (uint x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        uint64_t D[5];
        for (uint x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
        for (uint i = 0; i < 25; i++) {
            state[i] ^= D[i % 5];
        }

        // ρ (rho) and π (pi) steps (combined)
        uint64_t B[25];
        for (uint i = 0; i < 25; i++) {
            B[PI_INDICES[i]] = rotl64(state[i], ROT_OFFSETS[i]);
        }

        // χ (chi) step
        for (uint y = 0; y < 5; y++) {
            for (uint x = 0; x < 5; x++) {
                state[y * 5 + x] = B[y * 5 + x] ^ ((~B[y * 5 + (x + 1) % 5]) & B[y * 5 + (x + 2) % 5]);
            }
        }

        // ι (iota) step
        state[0] ^= KECCAK_RC[round];
    }
}

// Absorb bytes into the Keccak sponge state.
// `data` points to the input bytes, `len` is the total byte count.
// `state` is the 25 x uint64_t Keccak state (modified in-place).
// Uses rate = 136 bytes (17 uint64_t lanes) for Keccak-256.
inline void keccak_absorb(thread uint64_t* state, const device uchar* data, uint len) {
    const uint RATE_BYTES = 136;  // 1088 bits = 17 lanes
    const uint RATE_LANES = 17;

    uint offset = 0;

    // Process full blocks
    while (offset + RATE_BYTES <= len) {
        for (uint i = 0; i < RATE_LANES; i++) {
            uint64_t lane = 0;
            for (uint b = 0; b < 8; b++) {
                lane |= (uint64_t)data[offset + i * 8 + b] << (b * 8);
            }
            state[i] ^= lane;
        }
        keccak_f1600(state);
        offset += RATE_BYTES;
    }

    // Process remaining bytes (partial block)
    uint remaining = len - offset;
    // XOR remaining bytes into state (little-endian lane packing)
    for (uint i = 0; i < remaining; i++) {
        uint lane_idx = i / 8;
        uint byte_idx = i % 8;
        state[lane_idx] ^= (uint64_t)data[offset + i] << (byte_idx * 8);
    }

    // Multi-rate padding: pad byte = 0x01, last byte of block = 0x80
    // For Keccak-256 (not SHA3-256): domain separation byte is 0x01
    uint pad_pos = remaining;
    uint pad_lane = pad_pos / 8;
    uint pad_byte = pad_pos % 8;
    state[pad_lane] ^= (uint64_t)0x01 << (pad_byte * 8);

    // Set the last bit of the rate block
    uint last_pos = RATE_BYTES - 1;
    uint last_lane = last_pos / 8;
    uint last_byte = last_pos % 8;
    state[last_lane] ^= (uint64_t)0x80 << (last_byte * 8);

    // Final permutation
    keccak_f1600(state);
}

// Squeeze 32 bytes (256 bits) from the sponge state (little-endian lane extraction).
inline void keccak_squeeze(thread uint64_t* state, device uchar* out) {
    for (uint i = 0; i < 4; i++) {  // 4 lanes * 8 bytes = 32 bytes
        uint64_t lane = state[i];
        for (uint b = 0; b < 8; b++) {
            out[i * 8 + b] = (uchar)(lane >> (b * 8));
        }
    }
}

// =============================================================================
// Variant: absorb from a thread-local buffer (for leaf hashing where we
// first serialize field elements into a thread-local byte array)
// =============================================================================

inline void keccak_absorb_local(thread uint64_t* state, thread const uchar* data, uint len) {
    const uint RATE_BYTES = 136;
    const uint RATE_LANES = 17;

    uint offset = 0;

    while (offset + RATE_BYTES <= len) {
        for (uint i = 0; i < RATE_LANES; i++) {
            uint64_t lane = 0;
            for (uint b = 0; b < 8; b++) {
                lane |= (uint64_t)data[offset + i * 8 + b] << (b * 8);
            }
            state[i] ^= lane;
        }
        keccak_f1600(state);
        offset += RATE_BYTES;
    }

    uint remaining = len - offset;
    for (uint i = 0; i < remaining; i++) {
        uint lane_idx = i / 8;
        uint byte_idx = i % 8;
        state[lane_idx] ^= (uint64_t)data[offset + i] << (byte_idx * 8);
    }

    uint pad_pos = remaining;
    uint pad_lane = pad_pos / 8;
    uint pad_byte = pad_pos % 8;
    state[pad_lane] ^= (uint64_t)0x01 << (pad_byte * 8);

    uint last_pos = RATE_BYTES - 1;
    uint last_lane = last_pos / 8;
    uint last_byte = last_pos % 8;
    state[last_lane] ^= (uint64_t)0x80 << (last_byte * 8);

    keccak_f1600(state);
}

// Squeeze to thread-local buffer
inline void keccak_squeeze_local(thread uint64_t* state, thread uchar* out) {
    for (uint i = 0; i < 4; i++) {
        uint64_t lane = state[i];
        for (uint b = 0; b < 8; b++) {
            out[i * 8 + b] = (uchar)(lane >> (b * 8));
        }
    }
}

// =============================================================================
// Kernel 1: Hash leaves
//
// Each thread hashes one row of LDE data (one leaf).
// Input: flat row-major u64 array, where row i starts at lde_data[i * num_cols]
// Each u64 is serialized as 8 bytes big-endian (matching Goldilocks as_bytes())
// Output: 32 bytes per leaf at output[gid * 32]
// =============================================================================

[[kernel]] void keccak256_hash_leaves(
    device const ulong* lde_data    [[ buffer(0) ]],
    device uchar*        output     [[ buffer(1) ]],
    constant uint&       num_cols   [[ buffer(2) ]],
    constant uint&       num_rows   [[ buffer(3) ]],
    uint gid                        [[ thread_position_in_grid ]]
) {
    if (gid >= num_rows) return;

    // Initialize Keccak state to zero
    uint64_t state[25] = {0};

    // Serialize field elements to bytes in a thread-local buffer.
    // Max supported: 256 columns * 8 bytes = 2048 bytes.
    // For larger inputs, we absorb in chunks.
    const uint RATE_BYTES = 136;
    const uint RATE_LANES = 17;

    uint total_bytes = num_cols * 8;
    uint data_offset = gid * num_cols;

    // We process the data by absorbing it directly into the sponge
    // without materializing the full byte array. We build lanes on-the-fly.

    // Track position within the current rate block
    uint block_pos = 0;  // byte position within current block

    for (uint col = 0; col < num_cols; col++) {
        uint64_t val = lde_data[data_offset + col];

        // Serialize as big-endian 8 bytes
        uchar bytes[8];
        bytes[0] = (uchar)(val >> 56);
        bytes[1] = (uchar)(val >> 48);
        bytes[2] = (uchar)(val >> 40);
        bytes[3] = (uchar)(val >> 32);
        bytes[4] = (uchar)(val >> 24);
        bytes[5] = (uchar)(val >> 16);
        bytes[6] = (uchar)(val >> 8);
        bytes[7] = (uchar)(val);

        // XOR each byte into the appropriate lane in the state
        for (uint b = 0; b < 8; b++) {
            uint lane_idx = block_pos / 8;
            uint byte_idx = block_pos % 8;
            state[lane_idx] ^= (uint64_t)bytes[b] << (byte_idx * 8);
            block_pos++;

            if (block_pos == RATE_BYTES) {
                keccak_f1600(state);
                block_pos = 0;
            }
        }
    }

    // Padding
    // Multi-rate padding: 0x01 at current position, 0x80 at last byte of block
    uint pad_lane = block_pos / 8;
    uint pad_byte = block_pos % 8;
    state[pad_lane] ^= (uint64_t)0x01 << (pad_byte * 8);

    uint last_pos = RATE_BYTES - 1;
    uint last_lane = last_pos / 8;
    uint last_byte_pos = last_pos % 8;
    state[last_lane] ^= (uint64_t)0x80 << (last_byte_pos * 8);

    keccak_f1600(state);

    // Squeeze 32 bytes
    device uchar* out = output + gid * 32;
    for (uint i = 0; i < 4; i++) {
        uint64_t lane = state[i];
        for (uint b = 0; b < 8; b++) {
            out[i * 8 + b] = (uchar)(lane >> (b * 8));
        }
    }
}

// =============================================================================
// Kernel 2: Hash pairs of child nodes into parent nodes
//
// Each thread takes two consecutive 32-byte child hashes and produces
// one 32-byte parent hash: parent[gid] = Keccak256(child[2*gid] || child[2*gid+1])
// =============================================================================

[[kernel]] void keccak256_hash_pairs(
    device const uchar* children   [[ buffer(0) ]],
    device uchar*       parents    [[ buffer(1) ]],
    constant uint&      num_pairs  [[ buffer(2) ]],
    uint gid                       [[ thread_position_in_grid ]]
) {
    if (gid >= num_pairs) return;

    // Read 64 bytes (two 32-byte children) from device memory
    device const uchar* input = children + gid * 64;

    // Initialize Keccak state
    uint64_t state[25] = {0};

    // Absorb 64 bytes (less than rate=136, so single block)
    // Pack bytes into lanes (little-endian)
    for (uint i = 0; i < 8; i++) {  // 8 lanes = 64 bytes
        uint64_t lane = 0;
        for (uint b = 0; b < 8; b++) {
            lane |= (uint64_t)input[i * 8 + b] << (b * 8);
        }
        state[i] ^= lane;
    }

    // Padding at byte 64
    uint pad_lane = 64 / 8;  // lane 8
    uint pad_byte = 64 % 8;  // byte 0
    state[pad_lane] ^= (uint64_t)0x01 << (pad_byte * 8);

    uint last_pos = 135;  // RATE_BYTES - 1
    uint last_lane = last_pos / 8;  // lane 16
    uint last_byte_pos = last_pos % 8;  // byte 7
    state[last_lane] ^= (uint64_t)0x80 << (last_byte_pos * 8);

    keccak_f1600(state);

    // Squeeze 32 bytes
    device uchar* out = parents + gid * 32;
    for (uint i = 0; i < 4; i++) {
        uint64_t lane = state[i];
        for (uint b = 0; b < 8; b++) {
            out[i * 8 + b] = (uchar)(lane >> (b * 8));
        }
    }
}

// =============================================================================
// Kernel 3: Column-to-row transpose with bit-reversal
//
// Converts column-major LDE data to row-major bit-reversed layout.
// Input: flat column-major: col0[0..M], col1[0..M], ... (N columns, M rows)
// Output: flat row-major: row_br(0)[0..N], row_br(1)[0..N], ...
// where br(i) = bit_reverse(i, log2(M))
//
// Thread gid = output row index.
// =============================================================================

[[kernel]] void transpose_bitrev_goldilocks(
    device const ulong* columns  [[ buffer(0) ]],
    device ulong*       rows     [[ buffer(1) ]],
    constant uint&      num_cols [[ buffer(2) ]],
    constant uint&      num_rows [[ buffer(3) ]],
    uint gid                     [[ thread_position_in_grid ]]
) {
    if (gid >= num_rows) return;

    // Compute log2(num_rows) - num_rows must be power of two
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

    // Copy N elements: for each column c, read columns[c * num_rows + src_row]
    // and write to rows[gid * num_cols + c]
    uint out_base = gid * num_cols;
    for (uint c = 0; c < num_cols; c++) {
        rows[out_base + c] = columns[c * num_rows + src_row];
    }
}
