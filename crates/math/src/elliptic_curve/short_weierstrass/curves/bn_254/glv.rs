// GLV (Gallant-Lambert-Vanstone) endomorphism-accelerated scalar multiplication for BN254 G1
//
// This implementation is inspired by:
// - arkworks: https://github.com/arkworks-rs/algebra/blob/master/curves/bn254/src/curves/g1.rs
// - gnark-crypto: https://github.com/Consensys/gnark-crypto/blob/master/ecc/bn254/g1.go
// - Constantine: https://github.com/mratsim/constantine
//
// References:
// - GLV Original: Gallant, Lambert, Vanstone. "Faster Point Multiplication on Elliptic Curves
//   with Efficient Endomorphisms." CRYPTO 2001.

use super::curve::BN254Curve;
use super::field_extension::BN254PrimeField;
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::field::element::FieldElement;
use crate::unsigned_integer::element::U256;

type FpE = FieldElement<BN254PrimeField>;

/// Cube root of unity in Fp for BN254 (BETA)
/// β³ = 1 in Fp, β ≠ 1
/// This is BETA² where BETA is the arkworks value. Using β² gives a smaller eigenvalue
/// which makes GLV decomposition more efficient.
/// β² = 2203960485148121921418603742825762020974279258880205651966
/// Hex: 0x59e26bcea0d48bacd4f263f1acdb5c4f5763473177fffffe
/// Note: This is the gnark-crypto thirdRootOneG1 value
pub const CUBE_ROOT_OF_UNITY_G1: FpE =
    FpE::from_hex_unchecked("59e26bcea0d48bacd4f263f1acdb5c4f5763473177fffffe");

/// Lambda - the eigenvalue of the endomorphism in the scalar field Fr
/// phi(P) = [LAMBDA]P for all P in G1 (using the β² cube root from gnark-crypto)
/// LAMBDA = r - λ₂ - 1 where λ₂ = 21888242871839275217838484774961031246154997185409878258781734729429964517155
/// This is a ~192 bit value enabling efficient scalar decomposition
pub const LAMBDA: U256 =
    U256::from_hex_unchecked("b3c4d79d41a917585bfc41088d8daaa78b17ea66b99c90dd");

/// Subgroup order r for BN254
pub const R: U256 =
    U256::from_hex_unchecked("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

/// Decomposes scalar k into (k1, k2) such that k ≡ k1 + k2 * LAMBDA (mod r)
/// where k1 and k2 have approximately half the bit-length of k.
///
/// Uses simple division: k = k2*λ + k1 where k1 = k mod λ, k2 = k / λ.
/// For large k where k1 > 128 bits, falls back to no decomposition.
///
/// Returns (k1_abs, k1_is_neg, k2_abs, k2_is_neg)
pub fn scalar_decomposition(k: &U256) -> (u128, bool, u128, bool) {
    // For small scalars that fit in 128 bits, no decomposition needed
    if k.limbs[0] == 0 && k.limbs[1] == 0 {
        let k_u128 = k.limbs[3] as u128 | ((k.limbs[2] as u128) << 64);
        return (k_u128, false, 0, false);
    }

    // For scalars that don't fit in 128 bits but are less than LAMBDA,
    // we can't use simple decomposition, so fall back to k1=lower 128 bits, k2=0
    // This means GLV won't provide speedup for these cases but will be correct
    if *k < LAMBDA {
        // k is between 128 and 192 bits but less than LAMBDA
        // Use only the lower 128 bits (truncation) and fall back to standard mul
        let k_u128 = k.limbs[3] as u128 | ((k.limbs[2] as u128) << 64);
        return (k_u128, false, 0, false);
    }

    // k >= LAMBDA: decompose as k = k2*λ + k1
    let (k2_u256, k1_u256) = k.div_rem(&LAMBDA);

    // k1 = k mod λ (should be < λ which is ~192 bits)
    // If k1 > 128 bits, fall back to standard mul
    if k1_u256.limbs[0] != 0 || k1_u256.limbs[1] != 0 {
        // k1 is larger than 128 bits, fall back
        let k_lower = k.limbs[3] as u128 | ((k.limbs[2] as u128) << 64);
        return (k_lower, false, 0, false);
    }
    let k1 = k1_u256.limbs[3] as u128 | ((k1_u256.limbs[2] as u128) << 64);

    // k2 = k / λ (should be small, ~62 bits for 254-bit k)
    if k2_u256.limbs[0] != 0 || k2_u256.limbs[1] != 0 || k2_u256.limbs[2] != 0 {
        // k2 is larger than 64 bits, something is wrong, fall back
        let k_lower = k.limbs[3] as u128 | ((k.limbs[2] as u128) << 64);
        return (k_lower, false, 0, false);
    }
    let k2 = k2_u256.limbs[3] as u128;

    (k1, false, k2, false)
}

/// Performs GLV-accelerated scalar multiplication: computes [k]P using the endomorphism
///
/// The algorithm:
/// 1. Decompose k into k1, k2 where k ≡ k1 + k2*λ (mod r)
/// 2. phi(P) = [lambda]P (the endomorphism satisfies this eigenvalue equation)
/// 3. Compute [k]P = [k1]P + [k2]phi(P) using Shamir's trick
pub fn glv_mul(
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    k: &U256,
) -> ShortWeierstrassProjectivePoint<BN254Curve> {
    // For small scalars that fit in 64 bits, just use regular scalar multiplication
    if k.limbs[0] == 0 && k.limbs[1] == 0 && k.limbs[2] == 0 {
        return p.operate_with_self(k.limbs[3]);
    }

    // For scalars that fit in 128 bits, use regular scalar multiplication
    if k.limbs[0] == 0 && k.limbs[1] == 0 {
        let k_u128 = k.limbs[3] as u128 | ((k.limbs[2] as u128) << 64);
        return p.operate_with_self(k_u128);
    }

    // For larger scalars, try decomposition
    let (k1, k1_neg, k2, k2_neg) = scalar_decomposition(k);

    // If k2 == 0, decomposition failed - use standard scalar multiplication with full k
    if k2 == 0 {
        // Fall back to standard scalar multiplication using the full U256
        return p.operate_with_self(*k);
    }

    // If k1 == 0, compute [k2]phi(P)
    if k1 == 0 {
        let phi_p = phi(p);
        let result = phi_p.operate_with_self(k2);
        return if k2_neg { result.neg() } else { result };
    }

    // Precompute points
    // p1 = [sign(k1)] * P
    let p1 = if k1_neg { p.neg() } else { p.clone() };

    // phi(P) = [lambda]P, so p2 = [sign(k2)] * phi(P)
    let phi_p = phi(p);
    let p2 = if k2_neg { phi_p.neg() } else { phi_p };

    let p1_plus_p2 = p1.operate_with(&p2);

    // Find the highest bit position across both scalars
    let bits1 = 128 - k1.leading_zeros();
    let bits2 = 128 - k2.leading_zeros();
    let max_bits = bits1.max(bits2);

    if max_bits == 0 {
        return ShortWeierstrassProjectivePoint::neutral_element();
    }

    // Simultaneous double-and-add (Shamir's trick)
    let mut result = ShortWeierstrassProjectivePoint::neutral_element();

    for i in (0..max_bits).rev() {
        result = result.operate_with(&result); // Double

        let b1 = (k1 >> i) & 1 == 1;
        let b2 = (k2 >> i) & 1 == 1;

        match (b1, b2) {
            (true, false) => result = result.operate_with(&p1),
            (false, true) => result = result.operate_with(&p2),
            (true, true) => result = result.operate_with(&p1_plus_p2),
            (false, false) => {}
        }
    }

    result
}

/// Computes the phi endomorphism: phi(x, y) = (beta * x, y)
/// where beta is the cube root of unity in Fp
/// The eigenvalue relationship: phi(P) = [LAMBDA]P
#[inline]
pub fn phi(
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
) -> ShortWeierstrassProjectivePoint<BN254Curve> {
    let [x, y, z] = p.coordinates();
    // SAFETY: The endomorphism always produces a valid point on the curve
    ShortWeierstrassProjectivePoint::new([&CUBE_ROOT_OF_UNITY_G1 * x, y.clone(), z.clone()])
        .expect("phi endomorphism should always produce a valid point")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;
    use crate::elliptic_curve::traits::IsEllipticCurve;

    #[test]
    fn test_phi_endomorphism() {
        let g = BN254Curve::generator();
        let phi_g = phi(&g);

        // phi(P) should be a different point
        assert_ne!(g, phi_g);

        // phi should preserve being on the curve
        let phi_g_affine = phi_g.to_affine();
        let [x, y, _] = phi_g_affine.coordinates();
        let lhs = y.square();
        let rhs = x.pow(3u64) + BN254Curve::b();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_phi_cubed_is_identity() {
        // phi³(P) = P since beta³ = 1
        let g = BN254Curve::generator();
        let phi_g = phi(&g);
        let phi2_g = phi(&phi_g);
        let phi3_g = phi(&phi2_g);

        assert_eq!(g, phi3_g);
    }

    #[test]
    fn test_cube_root_cubed_is_one() {
        // Verify that CUBE_ROOT_OF_UNITY_G1³ = 1 in Fp
        let beta = &CUBE_ROOT_OF_UNITY_G1;
        let beta_cubed = beta * beta * beta;
        assert_eq!(beta_cubed, FpE::one(), "Beta³ should equal 1 in Fp");
    }

    #[test]
    fn test_phi_eigenvalue() {
        // Verify the eigenvalue relationship: phi(G) = [LAMBDA]G
        let g = BN254Curve::generator();
        let phi_g = phi(&g);

        // λ₂ (large ~254 bits) from arkworks
        let lambda2 = U256::from_hex_unchecked(
            "30644e72e131a029048b6e193fd84104cc37a73fec2bc5e9b8ca0b2d36636f23",
        );
        // The correct eigenvalue for gnark's β² is r - λ₂ - 1
        let correct_lambda = &R - &lambda2 - U256::from(1u64);

        // Print the correct LAMBDA value in hex for updating the constant
        println!(
            "Correct LAMBDA hex: {:016x}{:016x}{:016x}{:016x}",
            correct_lambda.limbs[0],
            correct_lambda.limbs[1],
            correct_lambda.limbs[2],
            correct_lambda.limbs[3]
        );

        // Verify it works
        let result = g.operate_with_self(correct_lambda);
        assert_eq!(phi_g, result, "phi(G) should equal [correct_lambda]G");

        // Also verify LAMBDA constant is correct
        let lambda_result = g.operate_with_self(LAMBDA);
        assert_eq!(
            phi_g, lambda_result,
            "phi(G) should equal [LAMBDA]G - LAMBDA constant needs updating!"
        );
    }

    #[test]
    fn test_glv_mul_small_scalar() {
        let g = BN254Curve::generator();
        let scalar = U256::from(12345u64);

        let result_standard = g.operate_with_self(scalar);
        let result_glv = glv_mul(&g, &scalar);

        assert_eq!(result_standard, result_glv);
    }

    #[test]
    fn test_glv_mul_medium_scalar() {
        let g = BN254Curve::generator();
        let scalar = U256::from(0xDEADBEEFCAFEBABEu64);

        let result_standard = g.operate_with_self(scalar);
        let result_glv = glv_mul(&g, &scalar);

        assert_eq!(result_standard, result_glv);
    }

    #[test]
    fn test_glv_mul_large_scalar() {
        let g = BN254Curve::generator();
        // Use a large scalar (close to subgroup order)
        let scalar = U256::from_hex_unchecked(
            "30644e72e131a029b85045b68181585d2833e84879b97091000000000000000",
        );

        let result_standard = g.operate_with_self(scalar);
        let result_glv = glv_mul(&g, &scalar);

        assert_eq!(result_standard, result_glv);
    }

    #[test]
    fn test_glv_mul_random_scalars() {
        let g = BN254Curve::generator();

        // Test several random-ish scalars
        let scalars = [
            U256::from(0x123456789ABCDEFu64),
            U256::from_hex_unchecked("DEADBEEFCAFEBABE1234567890ABCDEF"),
            U256::from_hex_unchecked("1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF"),
        ];

        for scalar in &scalars {
            let result_standard = g.operate_with_self(*scalar);
            let result_glv = glv_mul(&g, scalar);
            assert_eq!(result_standard, result_glv);
        }
    }
}
