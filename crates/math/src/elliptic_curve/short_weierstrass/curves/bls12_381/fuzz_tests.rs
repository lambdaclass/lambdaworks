//! Differential fuzzing tests: Lambdaworks BLS12-381 vs Arkworks.
//!
//! Uses proptest to generate random inputs and verify that our pairing,
//! field arithmetic, and final exponentiation match Arkworks results.

use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use crate::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::{
    BLS12381PrimeField, Degree12ExtensionField, Degree2ExtensionField, Degree6ExtensionField,
};
use crate::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::{
    final_exponentiation, miller, BLS12381AtePairing,
};
use crate::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
use crate::elliptic_curve::traits::{IsEllipticCurve, IsPairing};
use crate::field::element::FieldElement;
use crate::traits::ByteConversion;
use crate::unsigned_integer::element::U384;

use ark_bls12_381::{
    Bls12_381, Fq as ArkFq, Fq12 as ArkFq12, Fq2 as ArkFq2, Fq6 as ArkFq6, G1Projective as ArkG1,
    G2Projective as ArkG2,
};
use ark_ec::{pairing::Pairing, CurveGroup, PrimeGroup};
use ark_ff::{BigInteger384, Field as ArkField, PrimeField as ArkPrimeField};

use proptest::prelude::*;

type FpE = FieldElement<BLS12381PrimeField>;
type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp6E = FieldElement<Degree6ExtensionField>;
type Fp12E = FieldElement<Degree12ExtensionField>;

// ============================================
// CONVERSION HELPERS
// ============================================

/// Convert a lambdaworks Fp element to an arkworks Fq element
fn lw_fp_to_ark(elem: &FpE) -> ArkFq {
    let bytes = elem.to_bytes_be();
    ArkFq::from_be_bytes_mod_order(&bytes)
}

/// Convert a lambdaworks Fp2 element to an arkworks Fq2 element
fn lw_fp2_to_ark(elem: &Fp2E) -> ArkFq2 {
    let [c0, c1] = elem.value();
    ArkFq2::new(lw_fp_to_ark(c0), lw_fp_to_ark(c1))
}

/// Convert a lambdaworks Fp6 element to an arkworks Fq6 element
fn lw_fp6_to_ark(elem: &Fp6E) -> ArkFq6 {
    let [c0, c1, c2] = elem.value();
    ArkFq6::new(lw_fp2_to_ark(c0), lw_fp2_to_ark(c1), lw_fp2_to_ark(c2))
}

/// Convert a lambdaworks Fp12 element to an arkworks Fq12 element
fn lw_fp12_to_ark(elem: &Fp12E) -> ArkFq12 {
    let [c0, c1] = elem.value();
    ArkFq12::new(lw_fp6_to_ark(c0), lw_fp6_to_ark(c1))
}

/// Convert an arkworks Fq to lambdaworks Fp
#[allow(dead_code)]
fn ark_fp_to_lw(elem: &ArkFq) -> FpE {
    let bigint: BigInteger384 = (*elem).into();
    let limbs = bigint.0;
    FpE::new(U384::from_limbs(limbs))
}

/// Convert an arkworks Fq2 to lambdaworks Fp2
#[allow(dead_code)]
fn ark_fp2_to_lw(elem: &ArkFq2) -> Fp2E {
    Fp2E::new([ark_fp_to_lw(&elem.c0), ark_fp_to_lw(&elem.c1)])
}

/// Convert an arkworks Fq12 to lambdaworks Fp12
#[allow(dead_code)]
fn ark_fp12_to_lw(elem: &ArkFq12) -> Fp12E {
    let c0 = &elem.c0;
    let c1 = &elem.c1;
    Fp12E::new([
        Fp6E::new([
            ark_fp2_to_lw(&c0.c0),
            ark_fp2_to_lw(&c0.c1),
            ark_fp2_to_lw(&c0.c2),
        ]),
        Fp6E::new([
            ark_fp2_to_lw(&c1.c0),
            ark_fp2_to_lw(&c1.c1),
            ark_fp2_to_lw(&c1.c2),
        ]),
    ])
}

// ============================================
// PROPTEST STRATEGIES
// ============================================

/// Strategy for random u64 scalars (used to derive curve points)
fn scalar_strategy() -> impl Strategy<Value = u64> {
    1u64..=1_000_000u64
}

// ============================================
// PAIRING TESTS
// ============================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn fuzz_pairing_bilinearity(a in scalar_strategy(), b in scalar_strategy()) {
        // Lambdaworks: e(aP, bQ) should equal e(P, Q)^(ab)
        let lw_p = BLS12381Curve::generator();
        let lw_q = BLS12381TwistCurve::generator();

        let lw_ap = lw_p.operate_with_self(a).to_affine();
        let lw_bq = lw_q.operate_with_self(b).to_affine();

        let lw_result = BLS12381AtePairing::compute_batch(&[(&lw_ap, &lw_bq)])
            .expect("pairing should not fail for valid subgroup points");

        let lw_base = BLS12381AtePairing::compute_batch(&[
            (&lw_p.to_affine(), &lw_q.to_affine()),
        ]).expect("pairing should not fail for generator points");

        let lw_base_ab = lw_base.pow(a * b);
        prop_assert_eq!(lw_result, lw_base_ab);
    }

    #[test]
    fn fuzz_pairing_matches_arkworks(s in scalar_strategy()) {
        // Test that e(sP, Q) matches between lambdaworks and arkworks
        let lw_p = BLS12381Curve::generator().operate_with_self(s).to_affine();
        let lw_q = BLS12381TwistCurve::generator().to_affine();
        let lw_result = BLS12381AtePairing::compute_batch(&[(&lw_p, &lw_q)])
            .expect("pairing should not fail for valid subgroup points");

        let ark_p = (ArkG1::generator() * <ark_bls12_381::Fr as From<u64>>::from(s)).into_affine();
        let ark_q = ArkG2::generator().into_affine();
        let ark_result = Bls12_381::pairing(ark_p, ark_q);

        let lw_as_ark = lw_fp12_to_ark(&lw_result);
        prop_assert_eq!(lw_as_ark, ark_result.0);
    }

    #[test]
    fn fuzz_fp2_mul(
        a0 in scalar_strategy(), a1 in scalar_strategy(),
        b0 in scalar_strategy(), b1 in scalar_strategy()
    ) {
        let lw_a = Fp2E::new([FpE::from(a0), FpE::from(a1)]);
        let lw_b = Fp2E::new([FpE::from(b0), FpE::from(b1)]);
        let lw_result = &lw_a * &lw_b;

        let ark_a = lw_fp2_to_ark(&lw_a);
        let ark_b = lw_fp2_to_ark(&lw_b);
        let ark_result = ark_a * ark_b;

        prop_assert_eq!(lw_fp2_to_ark(&lw_result), ark_result);
    }

    #[test]
    fn fuzz_fp2_square(a0 in scalar_strategy(), a1 in scalar_strategy()) {
        let lw_a = Fp2E::new([FpE::from(a0), FpE::from(a1)]);
        let lw_result = lw_a.square();

        let ark_a = lw_fp2_to_ark(&lw_a);
        let ark_result = ark_a.square();

        prop_assert_eq!(lw_fp2_to_ark(&lw_result), ark_result);
    }

    #[test]
    fn fuzz_fp12_mul(
        a0 in scalar_strategy(), a1 in scalar_strategy(),
        b0 in scalar_strategy(), b1 in scalar_strategy()
    ) {
        // Build Fp12 elements from scalars (embed in the base component)
        let lw_a = Fp12E::new([
            Fp6E::new([Fp2E::new([FpE::from(a0), FpE::from(a1)]), Fp2E::zero(), Fp2E::zero()]),
            Fp6E::zero(),
        ]);
        let lw_b = Fp12E::new([
            Fp6E::new([Fp2E::new([FpE::from(b0), FpE::from(b1)]), Fp2E::zero(), Fp2E::zero()]),
            Fp6E::zero(),
        ]);
        let lw_result = &lw_a * &lw_b;

        let ark_a = lw_fp12_to_ark(&lw_a);
        let ark_b = lw_fp12_to_ark(&lw_b);
        let ark_result = ark_a * ark_b;

        prop_assert_eq!(lw_fp12_to_ark(&lw_result), ark_result);
    }

    #[test]
    fn fuzz_fp12_square(a0 in scalar_strategy(), a1 in scalar_strategy()) {
        let lw_a = Fp12E::new([
            Fp6E::new([Fp2E::new([FpE::from(a0), FpE::from(a1)]), Fp2E::zero(), Fp2E::zero()]),
            Fp6E::zero(),
        ]);
        let lw_result = lw_a.square();

        let ark_a = lw_fp12_to_ark(&lw_a);
        let ark_result = ark_a.square();

        prop_assert_eq!(lw_fp12_to_ark(&lw_result), ark_result);
    }

    #[test]
    fn fuzz_fp2_inv(a0 in scalar_strategy(), a1 in scalar_strategy()) {
        let lw_a = Fp2E::new([FpE::from(a0), FpE::from(a1)]);
        let lw_inv = lw_a.inv().expect("nonzero element should be invertible");
        let lw_product = &lw_a * &lw_inv;
        prop_assert_eq!(lw_product, Fp2E::one());

        let ark_a = lw_fp2_to_ark(&lw_a);
        let ark_inv = ark_a.inverse().expect("nonzero element should be invertible");
        prop_assert_eq!(lw_fp2_to_ark(&lw_inv), ark_inv);
    }
}

// ============================================
// NON-PROPTEST DIFFERENTIAL TESTS
// ============================================

#[test]
fn fuzz_final_exponentiation_matches_arkworks() {
    // Use a realistic miller loop output
    let lw_p = BLS12381Curve::generator().to_affine();
    let lw_q = BLS12381TwistCurve::generator().to_affine();
    let lw_f = miller(&lw_q, &lw_p);
    let lw_result = final_exponentiation(&lw_f)
        .expect("final exponentiation should succeed for valid miller output");

    let ark_p = ArkG1::generator().into_affine();
    let ark_q = ArkG2::generator().into_affine();
    let ark_f = Bls12_381::multi_miller_loop([ark_p], [ark_q]);
    let ark_result = Bls12_381::final_exponentiation(ark_f)
        .expect("arkworks final exponentiation should succeed");

    assert_eq!(lw_fp12_to_ark(&lw_result), ark_result.0);
}

#[test]
fn fuzz_miller_loop_matches_arkworks() {
    let lw_p = BLS12381Curve::generator().to_affine();
    let lw_q = BLS12381TwistCurve::generator().to_affine();
    let lw_f = miller(&lw_q, &lw_p);

    let ark_p = ArkG1::generator().into_affine();
    let ark_q = ArkG2::generator().into_affine();
    let ark_f = Bls12_381::multi_miller_loop([ark_p], [ark_q]);

    // Miller loop outputs may differ by a factor in the kernel of the final exponentiation.
    // So we verify via final exponentiation.
    let lw_result = final_exponentiation(&lw_f)
        .expect("final exponentiation should succeed for valid miller output");
    let ark_result = Bls12_381::final_exponentiation(ark_f)
        .expect("arkworks final exponentiation should succeed");

    assert_eq!(lw_fp12_to_ark(&lw_result), ark_result.0);
}
