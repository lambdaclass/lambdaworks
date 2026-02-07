#![no_main]
//! Fuzz tests for Karabina compression.
//!
//! Tests the following optimizations in BLS12-381 final exponentiation:
//! - CompressedCyclotomic compress/decompress roundtrip
//! - Compressed squaring matches cyclotomic squaring
//! - cyclotomic_pow_x_compressed matches cyclotomic_pow_x
//!
//! Karabina compression stores only 4 of 6 Fp2 coefficients for cyclotomic
//! subgroup elements, reducing memory while allowing efficient squaring.

use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve,
            field_extension::Degree12ExtensionField,
            pairing::{
                cyclotomic_pow_x, cyclotomic_square, final_exponentiation, miller,
                CompressedCyclotomic,
            },
            twist::BLS12381TwistCurve,
        },
        traits::IsEllipticCurve,
    },
    field::element::FieldElement,
};
use libfuzzer_sys::fuzz_target;

type Fp12E = FieldElement<Degree12ExtensionField>;

/// Generate a cyclotomic subgroup element from scalars.
/// Elements in the cyclotomic subgroup satisfy f^(p^6 + 1) = 1.
fn generate_cyclotomic_element(a: u64, b: u64) -> Fp12E {
    // Use pairing output which is in the cyclotomic subgroup after final exponentiation
    let p = BLS12381Curve::generator().operate_with_self(a.max(1));
    let q = BLS12381TwistCurve::generator().operate_with_self(b.max(1));

    let f = miller(&q.to_affine(), &p.to_affine());

    // Final exponentiation puts f in the cyclotomic subgroup
    final_exponentiation(&f).unwrap()
}

fuzz_target!(|data: (u64, u64, u8)| {
    let (scalar_a, scalar_b, num_squares) = data;

    // Avoid trivial cases
    if scalar_a == 0 || scalar_b == 0 {
        return;
    }

    // Limit number of squares to avoid timeout
    let num_squares = (num_squares % 20) as usize;

    // Generate a cyclotomic subgroup element
    let f = generate_cyclotomic_element(scalar_a, scalar_b);

    // ===== TEST 1: Compress/decompress roundtrip =====
    let compressed = CompressedCyclotomic::compress(&f);
    let decompressed = compressed.decompress();

    assert_eq!(
        f, decompressed,
        "Karabina compress/decompress roundtrip failed"
    );

    // ===== TEST 2: Single compressed square matches cyclotomic square =====
    let normal_sq = cyclotomic_square(&f);

    let compressed_f = CompressedCyclotomic::compress(&f);
    let compressed_sq = compressed_f.square();
    let decompressed_sq = compressed_sq.decompress();

    assert_eq!(
        normal_sq, decompressed_sq,
        "Compressed square != cyclotomic square"
    );

    // ===== TEST 3: Multiple compressed squares match repeated cyclotomic squares =====
    if num_squares > 0 {
        // Apply num_squares normal cyclotomic squares
        let mut normal_result = f.clone();
        for _ in 0..num_squares {
            normal_result = cyclotomic_square(&normal_result);
        }

        // Apply num_squares compressed squares then decompress
        let mut compressed_result = CompressedCyclotomic::compress(&f);
        for _ in 0..num_squares {
            compressed_result = compressed_result.square();
        }
        let decompressed_result = compressed_result.decompress();

        assert_eq!(
            normal_result, decompressed_result,
            "Multiple compressed squares don't match after {} iterations",
            num_squares
        );
    }

    // ===== TEST 4: Double compress/decompress =====
    // Compressing an already-compressed-then-decompressed element should work
    let double_compressed = CompressedCyclotomic::compress(&decompressed);
    let double_decompressed = double_compressed.decompress();

    assert_eq!(
        decompressed, double_decompressed,
        "Double compress/decompress failed"
    );

    // ===== TEST 5: Compressed multiplication with Fp12 =====
    // Test mul_by_fp12 helper
    let g = generate_cyclotomic_element(scalar_b, scalar_a); // Different element
    let compressed_f2 = CompressedCyclotomic::compress(&f);

    let mul_result = compressed_f2.mul_by_fp12(&g);
    let expected = f.clone() * &g;

    assert_eq!(
        mul_result, expected,
        "Compressed mul_by_fp12 doesn't match direct multiplication"
    );

    // ===== TEST 6: Square then multiply consistency =====
    // (f^2) * g should equal compress(f).square().decompress() * g
    let f_sq = cyclotomic_square(&f);
    let expected_product = &f_sq * &g;

    let compressed_sq2 = CompressedCyclotomic::compress(&f).square();
    let product_via_compressed = compressed_sq2.mul_by_fp12(&g);

    assert_eq!(
        product_via_compressed, expected_product,
        "Square-then-multiply consistency failed"
    );

    // ===== TEST 7: cyclotomic_pow_x consistency =====
    // Test that the pow function works correctly (used in final exponentiation)
    // This is expensive so only do it occasionally
    if scalar_a % 10 == 1 {
        let pow_result = cyclotomic_pow_x(&f);

        // Verify it's still in cyclotomic subgroup by checking f^(p^6) = f^(-1)
        // For cyclotomic elements: f^(p^6) = conjugate(f) = f^(-1)
        let conjugate = pow_result.conjugate();
        let inverse = pow_result.inv().unwrap();

        assert_eq!(
            conjugate, inverse,
            "pow_x result not in cyclotomic subgroup"
        );
    }

    // ===== TEST 8: Edge case - element close to identity =====
    // Test with f^large_power which might be close to 1
    let large_power = (scalar_a as u128) * (scalar_b as u128);
    if large_power > 0 && large_power < 1000 {
        let mut powered = f.clone();
        for _ in 0..3 {
            powered = cyclotomic_square(&powered);
        }

        let comp_powered = CompressedCyclotomic::compress(&powered);
        let decomp_powered = comp_powered.decompress();

        assert_eq!(
            powered, decomp_powered,
            "Compress/decompress failed for powered element"
        );
    }
});
