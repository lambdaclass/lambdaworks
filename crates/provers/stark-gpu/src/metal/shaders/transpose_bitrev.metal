#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Transpose column-major to row-major with bit-reversed row indices.
//
// Input:  columns concatenated as [col0[0..M], col1[0..M], ...]
// Output: [row_br(0)[0..N], row_br(1)[0..N], ...]
// where br(i) = bit_reverse(i, log2(M))
//
// NOTE: This is a standalone shader (no Goldilocks header needed).
// It operates on raw ulong (uint64_t) values.
// =============================================================================

// Reverse the bottom `bits` bits of `val`.
inline uint reverse_bits(uint val, uint bits) {
    uint result = 0;
    for (uint i = 0; i < bits; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

/// Transpose column-major to row-major with bit-reversed row indices.
///
/// Thread gid = output row index.
/// For each output row gid, the source row is bit_reverse(gid, log_n).
/// Copies num_cols elements per row.
kernel void transpose_bitrev(
    device const ulong* columns  [[ buffer(0) ]],
    device ulong* rows           [[ buffer(1) ]],
    constant uint& num_cols      [[ buffer(2) ]],
    constant uint& num_rows      [[ buffer(3) ]],
    constant uint& log_n         [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= num_rows) return;

    uint src_row = reverse_bits(gid, log_n);
    uint out_base = gid * num_cols;
    for (uint col = 0; col < num_cols; col++) {
        rows[out_base + col] = columns[col * num_rows + src_row];
    }
}

/// Transpose column-major to paired-row-major with bit-reversed row indices.
///
/// Merges consecutive bit-reversed rows into paired leaves:
///   merged_row[i] = row[br(2*i)] ++ row[br(2*i+1)]
///
/// Thread gid = output merged row index (0..num_rows/2).
/// Output width per merged row = 2 * num_cols.
kernel void transpose_bitrev_paired(
    device const ulong* columns  [[ buffer(0) ]],
    device ulong* rows           [[ buffer(1) ]],
    constant uint& num_cols      [[ buffer(2) ]],
    constant uint& num_rows      [[ buffer(3) ]],
    constant uint& log_n         [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= num_rows / 2) return;

    uint pair_width = 2 * num_cols;
    uint row0 = 2 * gid;
    uint row1 = 2 * gid + 1;
    uint src_row0 = reverse_bits(row0, log_n);
    uint src_row1 = reverse_bits(row1, log_n);
    uint out_base = gid * pair_width;

    for (uint col = 0; col < num_cols; col++) {
        rows[out_base + col] = columns[col * num_rows + src_row0];
        rows[out_base + num_cols + col] = columns[col * num_rows + src_row1];
    }
}
