#![no_main]
//! Fuzz tests for G2Prepared precomputation.
//!
//! Tests the following optimizations:
//! - G2Prepared::from_g2_affine precomputes Miller loop coefficients
//! - miller_with_prepared matches standard miller function
//! - Pairing bilinearity with prepared points
//!
//! G2Prepared allows faster repeated pairings with the same G2 point.

use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_381::{
                curve::BLS12381Curve,
                field_extension::Degree12ExtensionField,
                pairing::{final_exponentiation, miller, miller_with_prepared, G2Prepared},
                twist::BLS12381TwistCurve,
            },
            point::ShortWeierstrassJacobianPoint,
        },
        traits::IsEllipticCurve,
    },
    field::element::FieldElement,
};
use libfuzzer_sys::fuzz_target;

type G2Point = ShortWeierstrassJacobianPoint<BLS12381TwistCurve>;
type Fp12E = FieldElement<Degree12ExtensionField>;

fuzz_target!(|data: (u64, u64, u64, u64)| {
    let (scalar_p, scalar_q, scalar_a, scalar_b) = data;

    // Avoid trivial cases
    if scalar_p == 0 || scalar_q == 0 {
        return;
    }

    let g1 = BLS12381Curve::generator();
    let g2 = BLS12381TwistCurve::generator();

    // Create test points
    let p = g1.operate_with_self(scalar_p).to_affine();
    let q = g2.operate_with_self(scalar_q);
    let q_affine = q.to_affine();

    // ===== TEST 1: Prepared miller matches standard miller =====
    let standard_result = miller(&q_affine, &p);

    let q_prepared = G2Prepared::from_g2_affine(&q);
    let prepared_result = miller_with_prepared(&q_prepared, &p);

    assert_eq!(
        standard_result, prepared_result,
        "Prepared miller != standard miller for scalars ({}, {})",
        scalar_p, scalar_q
    );

    // ===== TEST 2: Full pairing with prepared points =====
    let standard_pairing = final_exponentiation(&standard_result).unwrap();
    let prepared_pairing = final_exponentiation(&prepared_result).unwrap();

    assert_eq!(
        standard_pairing, prepared_pairing,
        "Prepared pairing != standard pairing"
    );

    // ===== TEST 3: Bilinearity with prepared points =====
    // e(aP, bQ) = e(P, Q)^(ab)
    if scalar_a > 0 && scalar_b > 0 && scalar_a < 1000 && scalar_b < 1000 {
        let ap = g1.operate_with_self(scalar_a).to_affine();
        let bq = g2.operate_with_self(scalar_b);
        let bq_prepared = G2Prepared::from_g2_affine(&bq);

        let f1 = miller_with_prepared(&bq_prepared, &ap);
        let result1 = final_exponentiation(&f1).unwrap();

        // e(P, Q)^(ab)
        let pq_miller = miller(&g2.to_affine(), &g1.to_affine());
        let pq_pairing = final_exponentiation(&pq_miller).unwrap();
        let result2 = pq_pairing.pow(scalar_a as u64 * scalar_b as u64);

        assert_eq!(result1, result2, "Bilinearity failed with prepared points");
    }

    // ===== TEST 4: G2Prepared with neutral element =====
    let neutral_g2 = G2Point::neutral_element();
    let neutral_prepared = G2Prepared::from_g2_affine(&neutral_g2);

    assert!(
        neutral_prepared.infinity,
        "G2Prepared of neutral element should have infinity=true"
    );
    assert!(
        neutral_prepared.coefficients.is_empty(),
        "G2Prepared of neutral element should have empty coefficients"
    );

    // Miller with prepared neutral should return 1
    let neutral_miller = miller_with_prepared(&neutral_prepared, &p);
    assert_eq!(
        neutral_miller,
        Fp12E::one(),
        "Miller with prepared neutral G2 should return 1"
    );

    // ===== TEST 5: Reusing G2Prepared for multiple G1 points =====
    // This tests the main use case for G2Prepared
    let p1 = g1.operate_with_self(scalar_p).to_affine();
    let p2 = g1.operate_with_self(scalar_p.wrapping_add(1)).to_affine();
    let p3 = g1.operate_with_self(scalar_p.wrapping_mul(2)).to_affine();

    // Standard miller for each
    let m1_std = miller(&q_affine, &p1);
    let m2_std = miller(&q_affine, &p2);
    let m3_std = miller(&q_affine, &p3);

    // Prepared miller reusing same G2Prepared
    let m1_prep = miller_with_prepared(&q_prepared, &p1);
    let m2_prep = miller_with_prepared(&q_prepared, &p2);
    let m3_prep = miller_with_prepared(&q_prepared, &p3);

    assert_eq!(m1_std, m1_prep, "Reused prepared miller mismatch for p1");
    assert_eq!(m2_std, m2_prep, "Reused prepared miller mismatch for p2");
    assert_eq!(m3_std, m3_prep, "Reused prepared miller mismatch for p3");

    // ===== TEST 6: G2Prepared coefficients count =====
    // The number of coefficients should be deterministic based on X_BINARY
    // X_BINARY has 64 bits, with 6 set bits, so we expect:
    // - 63 doubling steps (one per bit after the first)
    // - 5 addition steps (one per set bit after the first)
    // Total: approximately 68 coefficients
    if !q_prepared.infinity {
        // Just verify we have some coefficients (exact count depends on X_BINARY)
        assert!(
            !q_prepared.coefficients.is_empty(),
            "Non-neutral G2Prepared should have coefficients"
        );
        assert!(
            q_prepared.coefficients.len() > 50,
            "G2Prepared should have many coefficients"
        );
    }

    // ===== TEST 7: Prepared point at different scalars =====
    // Create multiple prepared points and verify they're different
    let q1_prepared = G2Prepared::from_g2_affine(&g2.operate_with_self(1_u64));
    let q2_prepared = G2Prepared::from_g2_affine(&g2.operate_with_self(2_u64));

    // They should produce different miller outputs with same P
    let m1 = miller_with_prepared(&q1_prepared, &g1.to_affine());
    let m2 = miller_with_prepared(&q2_prepared, &g1.to_affine());

    assert_ne!(
        m1, m2,
        "Different G2 points should produce different miller outputs"
    );

    // ===== TEST 8: Associativity via prepared points =====
    // e(P1 + P2, Q) = e(P1, Q) * e(P2, Q)
    if scalar_a > 0 && scalar_b > 0 {
        let p1_point = g1.operate_with_self(scalar_a).to_affine();
        let p2_point = g1.operate_with_self(scalar_b).to_affine();
        let p_sum = g1
            .operate_with_self(scalar_a)
            .operate_with(&g1.operate_with_self(scalar_b))
            .to_affine();

        // e(P1 + P2, Q)
        let left = miller_with_prepared(&q_prepared, &p_sum);
        let left_final = final_exponentiation(&left).unwrap();

        // e(P1, Q) * e(P2, Q)
        let m1 = miller_with_prepared(&q_prepared, &p1_point);
        let m2 = miller_with_prepared(&q_prepared, &p2_point);
        let right_final = final_exponentiation(&(&m1 * &m2)).unwrap();

        assert_eq!(
            left_final, right_final,
            "Pairing associativity failed with prepared points"
        );
    }
});
