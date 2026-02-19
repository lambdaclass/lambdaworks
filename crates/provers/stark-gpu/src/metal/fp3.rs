//! GPU Fp3 (degree-3 Goldilocks extension) arithmetic.
//!
//! Provides helpers to compile Metal shaders that use the `Fp3Goldilocks` class
//! from `crates/math/src/gpu/metal/shaders/field/fp3_goldilocks.h.metal`.
//!
//! The `Fp3Goldilocks` struct mirrors `Degree3GoldilocksExtensionField` from
//! `lambdaworks_math`, using irreducible polynomial `w^3 = 2`.

/// Source code for the Goldilocks base-field header (Fp64Goldilocks class).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const FP_U64_HEADER_SOURCE: &str =
    include_str!("../../../../math/src/gpu/metal/shaders/field/fp_u64.h.metal");

/// Source code for the Fp3 Goldilocks extension header (Fp3Goldilocks class).
///
/// NOTE: This header contains `#include "fp_u64.h.metal"` which does not work
/// with runtime compilation via `new_library_with_source`. When concatenating
/// sources at runtime, we strip the `#include` and prepend `FP_U64_HEADER_SOURCE`.
#[cfg(all(target_os = "macos", feature = "metal"))]
const FP3_HEADER_RAW: &str =
    include_str!("../../../../math/src/gpu/metal/shaders/field/fp3_goldilocks.h.metal");

/// Build the combined source for any shader that needs both Fp64Goldilocks and Fp3Goldilocks.
///
/// Concatenates the base-field header, the Fp3 header (with `#include` stripped),
/// and the caller's shader body.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn combined_fp3_source(shader_body: &str) -> String {
    // Strip the #include directive since we concatenate manually
    let fp3_clean = FP3_HEADER_RAW.replace("#include \"fp_u64.h.metal\"", "// (included above)");
    format!("{}\n{}\n{}", FP_U64_HEADER_SOURCE, fp3_clean, shader_body)
}

// =============================================================================
// Tests: validate GPU Fp3 arithmetic against CPU Degree3GoldilocksExtensionField
// =============================================================================

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use lambdaworks_gpu::metal::abstractions::state::{DynamicMetalState, MetalState};
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_goldilocks_field::{
        Degree3GoldilocksExtensionField, Goldilocks64Field,
    };

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;
    type Fp3 = Degree3GoldilocksExtensionField;
    type Fp3E = FieldElement<Fp3>;

    /// Metal shader that tests all Fp3 operations.
    ///
    /// Input buffer 0: pairs of Fp3 elements (6 u64s each pair: a0,a1,a2,b0,b1,b2)
    /// Output buffer 1: results of 7 operations per pair (21 u64s each):
    ///   [add, sub, mul, mul_scalar, square, inv, neg] — 3 u64s each
    /// Buffer 2: num_pairs (constant uint)
    const FP3_TEST_KERNEL: &str = r#"
kernel void fp3_test(
    device const uint64_t* input   [[ buffer(0) ]],
    device uint64_t* output        [[ buffer(1) ]],
    constant uint& num_pairs       [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= num_pairs) return;

    // Read pair (6 u64s)
    uint base_in = gid * 6;
    Fp3Goldilocks a = Fp3Goldilocks(
        Fp64Goldilocks(input[base_in + 0]),
        Fp64Goldilocks(input[base_in + 1]),
        Fp64Goldilocks(input[base_in + 2])
    );
    Fp3Goldilocks b = Fp3Goldilocks(
        Fp64Goldilocks(input[base_in + 3]),
        Fp64Goldilocks(input[base_in + 4]),
        Fp64Goldilocks(input[base_in + 5])
    );

    // Compute all operations
    Fp3Goldilocks r_add = a + b;
    Fp3Goldilocks r_sub = a - b;
    Fp3Goldilocks r_mul = a * b;
    Fp3Goldilocks r_mul_scalar = a.scalar_mul(b.c0);
    Fp3Goldilocks r_square = a.square();
    Fp3Goldilocks r_inv = a.inverse();
    Fp3Goldilocks r_neg = a.neg();

    // Write results (21 u64s per pair)
    uint base_out = gid * 21;
    output[base_out +  0] = (uint64_t)r_add.c0;
    output[base_out +  1] = (uint64_t)r_add.c1;
    output[base_out +  2] = (uint64_t)r_add.c2;
    output[base_out +  3] = (uint64_t)r_sub.c0;
    output[base_out +  4] = (uint64_t)r_sub.c1;
    output[base_out +  5] = (uint64_t)r_sub.c2;
    output[base_out +  6] = (uint64_t)r_mul.c0;
    output[base_out +  7] = (uint64_t)r_mul.c1;
    output[base_out +  8] = (uint64_t)r_mul.c2;
    output[base_out +  9] = (uint64_t)r_mul_scalar.c0;
    output[base_out + 10] = (uint64_t)r_mul_scalar.c1;
    output[base_out + 11] = (uint64_t)r_mul_scalar.c2;
    output[base_out + 12] = (uint64_t)r_square.c0;
    output[base_out + 13] = (uint64_t)r_square.c1;
    output[base_out + 14] = (uint64_t)r_square.c2;
    output[base_out + 15] = (uint64_t)r_inv.c0;
    output[base_out + 16] = (uint64_t)r_inv.c1;
    output[base_out + 17] = (uint64_t)r_inv.c2;
    output[base_out + 18] = (uint64_t)r_neg.c0;
    output[base_out + 19] = (uint64_t)r_neg.c1;
    output[base_out + 20] = (uint64_t)r_neg.c2;
}
"#;

    /// Extract the raw u64 components from a Fp3E (for sending to GPU buffers).
    fn fp3_to_u64s(e: &Fp3E) -> [u64; 3] {
        let comps = e.value();
        [*comps[0].value(), *comps[1].value(), *comps[2].value()]
    }

    /// Create a Fp3E from u64 components via `FpE::from()` (applies modular reduction).
    fn fp3_from_u64s(vals: [u64; 3]) -> Fp3E {
        FieldElement::new([
            FpE::from(vals[0]),
            FpE::from(vals[1]),
            FpE::from(vals[2]),
        ])
    }

    /// Create a Fp3E from raw u64 values (as returned from GPU buffers).
    /// Uses `from_raw` to preserve the exact value; FieldElement::eq handles
    /// canonical comparison even if the raw value is >= p.
    fn fp3_from_raw_u64s(vals: [u64; 3]) -> Fp3E {
        FieldElement::new([
            FieldElement::from_raw(vals[0]),
            FieldElement::from_raw(vals[1]),
            FieldElement::from_raw(vals[2]),
        ])
    }

    #[test]
    fn gpu_fp3_arithmetic_matches_cpu() {
        let test_pairs: Vec<(Fp3E, Fp3E)> = vec![
            // Simple values
            (fp3_from_u64s([1, 2, 3]), fp3_from_u64s([4, 5, 6])),
            // Larger values
            (
                fp3_from_u64s([0x123456789abcdef0, 0xfedcba9876543210, 42]),
                fp3_from_u64s([0xdeadbeefcafebabe, 7, 0x1111111111111111]),
            ),
            // One element is one
            (Fp3E::one(), fp3_from_u64s([100, 200, 300])),
            // Large components near prime
            (
                fp3_from_u64s([0xFFFFFFFF00000000, 0xFFFFFFFE00000002, 1]),
                fp3_from_u64s([0xFFFFFFFF00000000, 1, 0xFFFFFFFE00000002]),
            ),
            // Non-trivial for inverse
            (fp3_from_u64s([42, 0, 0]), fp3_from_u64s([0, 0, 1])),
        ];

        // Build input buffer: 6 u64s per pair
        let mut input_data: Vec<u64> = Vec::new();
        for (a, b) in &test_pairs {
            let a_vals = fp3_to_u64s(a);
            let b_vals = fp3_to_u64s(b);
            input_data.extend_from_slice(&a_vals);
            input_data.extend_from_slice(&b_vals);
        }

        let num_pairs = test_pairs.len() as u32;

        // Compile shader
        let combined_source = combined_fp3_source(FP3_TEST_KERNEL);
        let mut state = DynamicMetalState::new().expect("Failed to create Metal state");
        state
            .load_library(&combined_source)
            .expect("Failed to compile Fp3 test shader");
        let max_threads = state
            .prepare_pipeline("fp3_test")
            .expect("Failed to prepare pipeline");

        // Create buffers via DynamicMetalState API
        let input_buffer = state
            .alloc_buffer_with_data(&input_data)
            .expect("Failed to create input buffer");
        let output_buffer = state
            .alloc_buffer(num_pairs as usize * 21 * std::mem::size_of::<u64>())
            .expect("Failed to create output buffer");
        let num_pairs_buf = state
            .alloc_buffer_with_data(&[num_pairs])
            .expect("Failed to create params buffer");

        // Dispatch
        state
            .execute_compute(
                "fp3_test",
                &[&input_buffer, &output_buffer, &num_pairs_buf],
                num_pairs as u64,
                max_threads,
            )
            .expect("Failed to execute fp3_test kernel");

        // Read results
        let results: Vec<u64> = MetalState::retrieve_contents(&output_buffer);

        // Helper to read a Fp3 result from GPU output (3 consecutive u64s).
        // We construct via FieldElement::from_raw to preserve the exact GPU value,
        // then rely on FieldElement::eq for canonical comparison.
        let read_fp3 = |offset: usize| -> Fp3E {
            fp3_from_raw_u64s([results[offset], results[offset + 1], results[offset + 2]])
        };

        // Validate each pair
        for (pair_idx, (a, b)) in test_pairs.iter().enumerate() {
            let base = pair_idx * 21;

            // 1. Add
            let gpu_add = read_fp3(base);
            assert_eq!(gpu_add, a + b, "Fp3 add mismatch at pair {pair_idx}");

            // 2. Sub
            let gpu_sub = read_fp3(base + 3);
            assert_eq!(gpu_sub, a - b, "Fp3 sub mismatch at pair {pair_idx}");

            // 3. Mul
            let gpu_mul = read_fp3(base + 6);
            assert_eq!(gpu_mul, a * b, "Fp3 mul mismatch at pair {pair_idx}");

            // 4. Mul scalar: a.scalar_mul(b.c0)
            let gpu_mul_scalar = read_fp3(base + 9);
            let b_c0 = &b.value()[0];
            let cpu_mul_scalar: Fp3E = b_c0 * a;
            assert_eq!(
                gpu_mul_scalar, cpu_mul_scalar,
                "Fp3 mul_scalar mismatch at pair {pair_idx}"
            );

            // 5. Square of a
            let gpu_sq = read_fp3(base + 12);
            assert_eq!(gpu_sq, a * a, "Fp3 square mismatch at pair {pair_idx}");

            // 6. Inverse of a
            let gpu_inv = read_fp3(base + 15);
            let cpu_inv = a.inv().expect("inverse should exist for non-zero");
            assert_eq!(gpu_inv, cpu_inv, "Fp3 inv mismatch at pair {pair_idx}");

            // Verify: a * inv(a) == 1
            let product = a * &gpu_inv;
            assert_eq!(
                product,
                Fp3E::one(),
                "Fp3 inv verification failed at pair {pair_idx}: a * inv(a) != 1"
            );

            // 7. Negation of a
            let gpu_neg = read_fp3(base + 18);
            assert_eq!(gpu_neg, -a, "Fp3 neg mismatch at pair {pair_idx}");
        }
    }
}
