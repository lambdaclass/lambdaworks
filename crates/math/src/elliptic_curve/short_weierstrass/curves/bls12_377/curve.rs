use super::{
    field_extension::BLS12377PrimeField,
    pairing::{GAMMA_12, GAMMA_13},
    twist::BLS12377TwistCurve,
};
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::short_weierstrass::utils::{
    jac_to_proj, proj_to_jac, shamir_two_scalar_mul,
};
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::element::U256;

use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

pub const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001");

pub const CURVE_COFACTOR: U256 = U256::from_hex_unchecked("0x170b5d44300000000000000000000000");

pub type BLS12377FieldElement = FieldElement<BLS12377PrimeField>;

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12377Curve;

impl IsEllipticCurve for BLS12377Curve {
    type BaseField = BLS12377PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    /// Returns the generator point of the BLS12-377 curve.
    ///
    /// Generator values are taken from [Neuromancer's BLS12-377 page](https://neuromancer.sk/std/bls/BLS12-377).
    ///
    /// # Safety
    ///
    /// - The generator point `(x, y, 1)` is predefined and is **known to be a valid point** on the curve.
    /// - `unwrap` is used because this point is **mathematically verified**.
    /// - Do **not** modify this function unless a new generator has been **mathematically verified**.
    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - These values are mathematically verified and known to be valid points on BLS12-377.
        // - `unwrap()` is safe because we **ensure** the input values satisfy the curve equation.
        let point= Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new_base("8848defe740a67c8fc6225bf87ff5485951e2caa9d41bb188282c8bd37cb5cd5481512ffcd394eeab9b16eb21be9ef"),
            FieldElement::<Self::BaseField>::new_base("1914a69c5102eff1f674f5d30afeec4bd7fb348ca3e52d96d182ad44fb82305c2fe3d3634a9591afd82de55559c8ea6"),
            FieldElement::one()
        ]);
        point.unwrap()
    }
}

impl IsShortWeierstrass for BLS12377Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(1)
    }
}

/// This is equal to the frobenius trace of the BLS12 377 curve minus one or seed value z.
pub const MILLER_LOOP_CONSTANT: u64 = 0x8508c00000000001;

/// MILLER_LOOP_CONSTANT², used for faster subgroup checks: φ(P) = -u²P.
const MILLER_LOOP_CONSTANT_SQ: u128 =
    (MILLER_LOOP_CONSTANT as u128) * (MILLER_LOOP_CONSTANT as u128);

/// 𝛽 : primitive cube root of unity of 𝐹ₚ that satisfies the minimal equation
/// 𝛽² + 𝛽 + 1 = 0 mod 𝑝
pub const CUBE_ROOT_OF_UNITY_G1: BLS12377FieldElement = FieldElement::from_hex_unchecked(
    "0x1ae3a4617c510eabc8756ba8f8c524eb8882a75cc9bc8e359064ee822fb5bffd1e945779fffffffffffffffffffffff",
);

// GLV (Gallant-Lambert-Vanstone) Scalar Multiplication Constants for G1
//
// The endomorphism φ(x, y) = (βx, y) satisfies φ(P) = [λ]P for all P in the r-torsion subgroup.
// GLV decomposition splits scalar k into k₁ + k₂·λ where |k₁|, |k₂| < √r.

/// The eigenvalue λ of the GLV endomorphism, satisfying λ² + λ + 1 ≡ 0 (mod r).
pub const GLV_LAMBDA: U256 =
    U256::from_hex_unchecked("12ab655e9a2ca55660b44d1e5c37b00114885f32400000000000000000000000");

/// The small cube root of unity ω in Fr (≈ 2^126), used for scalar decomposition.
const GLV_OMEGA: U256 = U256::from_hex_unchecked("452217cc900000010a11800000000000");

/// ω + 1, used in the decomposition formula.
const GLV_OMEGA_PLUS_ONE: U256 = U256::from_hex_unchecked("452217cc900000010a11800000000001");

impl ShortWeierstrassProjectivePoint<BLS12377Curve> {
    /// Applies the GLV endomorphism: φ(x, y) = (βx, y) where β is the cube root of unity.
    /// Satisfies `φ(P) = [λ]P` where `λ² + λ + 1 ≡ 0 (mod r)`.
    /// See <https://eprint.iacr.org/2022/352.pdf> Section 2 Preliminaries.
    pub fn phi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        Self::new_unchecked([x * CUBE_ROOT_OF_UNITY_G1, y.clone(), z.clone()])
    }

    /// 𝜙(P) = −𝑢²P.
    /// See <https://eprint.iacr.org/2022/352.pdf> Section 4.3 Prop. 4.
    pub fn is_in_subgroup(&self) -> bool {
        if self.is_neutral_element() {
            return true;
        }
        self.operate_with_self(MILLER_LOOP_CONSTANT_SQ).neg() == self.phi()
    }

    /// GLV scalar multiplication: computes [k]P using the endomorphism.
    ///
    /// Decomposes k = k1 + k2*ω with k1, k2 (~126 bits each), then uses
    /// Shamir's trick for joint scalar multiplication.
    /// Measured speedup: ~1.4x for 192-bit scalars, ~2.5x for 253-bit scalars.
    ///
    /// # Security Note
    ///
    /// This implementation is **not constant-time** and may be vulnerable to
    /// timing side-channel attacks. Do not use with secret scalars in applications
    /// requiring side-channel resistance.
    pub fn glv_mul(&self, k: &U256) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }
        if *k == U256::from_u64(0) {
            return Self::neutral_element();
        }

        let (k1_neg, k1, k2_neg, k2) = glv_decompose_377(k);
        let phi_p = self.phi();

        let p1 = if k1_neg { self.neg() } else { self.clone() };
        let p2 = if k2_neg { phi_p.neg() } else { phi_p };

        // Use Jacobian coordinates for faster doubling (2M+5S vs 7M+5S in projective)
        let p1_jac = proj_to_jac(&p1);
        let p2_jac = proj_to_jac(&p2);
        let result_jac = shamir_two_scalar_mul(&p1_jac, &k1, &p2_jac, &k2);
        jac_to_proj(result_jac)
    }
}

impl ShortWeierstrassProjectivePoint<BLS12377TwistCurve> {
    /// Computes 𝜓(P) = 𝜁 ∘ 𝜋ₚ ∘ 𝜁⁻¹, where 𝜁 is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E, 𝜋ₚ is the p-power frobenius endomorphism
    /// and 𝜓 satisfies minimal equation 𝑋² + 𝑡𝑋 + 𝑞 = 𝑂.
    /// See <https://eprint.iacr.org/2022/352.pdf> Section 4.2 (7).
    /// ψ(P) = (ψ_x * conjugate(x), ψ_y * conjugate(y), conjugate(z))
    ///
    /// Crucially: ψ(P) = [u]P where u = MILLER_LOOP_CONSTANT (curve seed).
    ///
    /// # Safety
    ///
    /// - This function assumes `self` is a valid point on the BLS12-377 **twist** curve.
    /// - The conjugation operation preserves validity.
    /// - `unwrap()` is used because `psi()` is defined to **always return a valid point**.
    pub fn psi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        // SAFETY:
        // - `conjugate()` preserves the validity of the field element.
        // - `GAMMA_12` and `GAMMA_13` are precomputed constants that ensure the
        //   resulting point satisfies the curve equation.
        // - `unwrap()` is safe because the transformation follows
        //   **a known valid isomorphism** between the twist and E.
        let point = Self::new([
            x.conjugate() * GAMMA_12,
            y.conjugate() * GAMMA_13,
            z.conjugate(),
        ]);
        point.unwrap()
    }

    /// 𝜓(P) = 𝑢P, where 𝑢 = SEED of the curve.
    /// See <https://eprint.iacr.org/2022/352.pdf> Section 4.2.
    pub fn is_in_subgroup(&self) -> bool {
        self.psi() == self.operate_with_self(MILLER_LOOP_CONSTANT)
    }

    /// GLS scalar multiplication: computes [k]P using the Frobenius endomorphism.
    ///
    /// Decomposes k = k₁ + k₂·u where u is the curve seed (64-bit), then uses
    /// Shamir's trick: [k]P = [k₁]P + [k₂]ψ(P).
    ///
    /// Since u is 64 bits, k₂ ≈ 192 bits and k₁ ≤ 64 bits.
    /// Measured speedup: ~1.3x for 192-bit scalars, ~1.5x for 253-bit scalars.
    ///
    /// # Security Note
    ///
    /// This implementation is **not constant-time** and may be vulnerable to
    /// timing side-channel attacks. Do not use with secret scalars in applications
    /// requiring side-channel resistance.
    pub fn gls_mul(&self, k: &U256) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }

        let zero = U256::from_u64(0);
        if *k == zero {
            return Self::neutral_element();
        }

        let (k1_neg, k1, k2_neg, k2) = gls_decompose_377(k);
        let psi_p = self.psi();

        // [k]P = [k₁]P + [k₂]ψ(P)
        // Since ψ(P) = [u]P, we have [k₂]ψ(P) = [k₂·u]P (positive, no sign flip)
        let p1 = if k1_neg { self.neg() } else { self.clone() };
        let p2 = if k2_neg { psi_p.neg() } else { psi_p };

        // Use Jacobian coordinates for faster doubling (2M+5S vs 7M+5S in projective)
        let p1_jac = proj_to_jac(&p1);
        let p2_jac = proj_to_jac(&p2);
        let result_jac = shamir_two_scalar_mul(&p1_jac, &k1, &p2_jac, &k2);
        jac_to_proj(result_jac)
    }
}

/// The curve seed u as U256 for GLS division.
const GLS_X_377: U256 = U256::from_u64(MILLER_LOOP_CONSTANT);

/// Decomposes scalar k into (a, k₂) where k ≡ a − k₂·λ (mod r), with |a|, |k₂| ≈ √r.
///
/// Returns (a_neg, |a|, b_neg, |b|) for the GLV formula: [k]P = [a]P + [b]φ(P).
fn glv_decompose_377(k: &U256) -> (bool, U256, bool, U256) {
    let zero = U256::from_u64(0);

    if *k < GLV_OMEGA {
        return (false, *k, false, zero);
    }

    let (k2, _) = k.div_rem(&GLV_OMEGA_PLUS_ONE);
    let (k2_omega_hi, k2_omega_lo) = U256::mul(&k2, &GLV_OMEGA);

    if k2_omega_hi != zero {
        return (false, *k, false, zero);
    }

    // k - k2*ω ≥ 0 always (since k2 = floor(k/(ω+1)) implies k2*ω ≤ k - k2 ≤ k).
    debug_assert!(
        *k >= k2_omega_lo,
        "k1 underflow is mathematically impossible"
    );
    let k1 = U256::sub(k, &k2_omega_lo).0;

    let (a, a_neg) = if k1 >= k2 {
        (U256::sub(&k1, &k2).0, false)
    } else {
        (U256::sub(&k2, &k1).0, true)
    };

    // k2_neg=true: [k]P = [a]P - [k2]φ(P), consistent with gls_mul sign convention.
    (a_neg, a, true, k2)
}

/// Decomposes scalar k for GLS: k = k₁ + k₂·u (mod r)
///
/// BLS12-377: ψ(P) = [+u]P, so [k]P = [k₁]P + [k₂]ψ(P) directly (no sign flip).
///
/// Since u is 64 bits:
/// - k₂ = k / u (approximately 192 bits)
/// - k₁ = k mod u (at most 64 bits)
fn gls_decompose_377(k: &U256) -> (bool, U256, bool, U256) {
    let zero = U256::from_u64(0);

    if *k < GLS_X_377 {
        return (false, *k, false, zero);
    }

    let (k2, k1) = k.div_rem(&GLS_X_377);
    // ψ(P) = [+u]P, so [k]P = [k₁ + k₂·u]P = [k₁]P + [k₂]ψ(P)
    (false, k1, false, k2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::curves::bls12_377::field_extension::Degree2ExtensionField,
            traits::EllipticCurveError,
        },
        field::element::FieldElement,
    };

    use super::{BLS12377Curve, GLV_LAMBDA, MILLER_LOOP_CONSTANT};

    #[allow(clippy::upper_case_acronyms)]
    type FpE = FieldElement<BLS12377PrimeField>;
    type Fp2 = FieldElement<Degree2ExtensionField>;

    fn point_1() -> ShortWeierstrassProjectivePoint<BLS12377Curve> {
        let x = FpE::new_base("134e4cc122cb62a06767fb98e86f2d5f77e2a12fefe23bb0c4c31d1bd5348b88d6f5e5dee2b54db4a2146cc9f249eea");
        let y = FpE::new_base("17949c29effee7a9f13f69b1c28eccd78c1ed12b47068836473481ff818856594fd9c1935e3d9e621901a2d500257a2");
        BLS12377Curve::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> ShortWeierstrassProjectivePoint<BLS12377Curve> {
        let x = FpE::new_base("3c852d5aab73fbb51e57fbf5a0a8b5d6513ec922b2611b7547bfed74cba0dcdfc3ad2eac2733a4f55d198ec82b9964");
        let y = FpE::new_base("a71425e68e55299c64d7eada9ae9c3fb87a9626b941d17128b64685fc07d0e635f3c3a512903b4e0a43e464045967b");
        BLS12377Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn adding_five_times_point_1_works() {
        let point_1 = point_1();
        let point_1_times_5 = point_1_times_5();
        assert_eq!(point_1.operate_with_self(5_u16), point_1_times_5);
    }

    #[test]
    fn add_point1_2point1_with_both_algorithms_matches() {
        let point_1 = point_1();
        let point_2 = &point_1.operate_with(&point_1).to_affine();

        let first_algorithm_result = point_1.operate_with(point_2).to_affine();
        let second_algorithm_result = point_1.operate_with_affine(point_2).to_affine();

        assert_eq!(first_algorithm_result, second_algorithm_result);
    }

    #[test]
    fn add_point1_and_42424242point1_with_both_algorithms_matches() {
        let point_1 = point_1();
        let point_2 = &point_1.operate_with_self(42424242u128).to_affine();

        let first_algorithm_result = point_1.operate_with(point_2).to_affine();
        let second_algorithm_result = point_1.operate_with_affine(point_2).to_affine();

        assert_eq!(first_algorithm_result, second_algorithm_result);
    }

    #[test]
    fn add_point1_with_point1_both_algorithms_matches() {
        let point_1 = point_1().to_affine();

        let first_algorithm_result = point_1.operate_with(&point_1).to_affine();
        let second_algorithm_result = point_1.operate_with_affine(&point_1).to_affine();

        assert_eq!(first_algorithm_result, second_algorithm_result);
    }

    #[test]
    fn add_point2_with_point1_both_algorithms_matches() {
        let point_1 = point_1().to_affine();

        // Create point 2
        let x = FpE::new_base("134e4cc122cb62a06767fb98e86f2d5f77e2a12fefe23bb0c4c31d1bd5348b88d6f5e5dee2b54db4a2146cc9f249eea") * FpE::from(2);
        let y = FpE::new_base("17949c29effee7a9f13f69b1c28eccd78c1ed12b47068836473481ff818856594fd9c1935e3d9e621901a2d500257a2") * FpE::from(2);
        let z = FpE::from(2);
        let point_2 = ShortWeierstrassProjectivePoint::<BLS12377Curve>::new([x, y, z]).unwrap();

        let first_algorithm_result = point_2.operate_with(&point_1).to_affine();
        let second_algorithm_result = point_2.operate_with_affine(&point_1).to_affine();

        assert_eq!(first_algorithm_result, second_algorithm_result);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(*p.x(), FpE::new_base("134e4cc122cb62a06767fb98e86f2d5f77e2a12fefe23bb0c4c31d1bd5348b88d6f5e5dee2b54db4a2146cc9f249eea"));
        assert_eq!(*p.y(), FpE::new_base("17949c29effee7a9f13f69b1c28eccd78c1ed12b47068836473481ff818856594fd9c1935e3d9e621901a2d500257a2"));
        assert_eq!(*p.z(), FpE::new_base("1"));
    }

    #[test]
    fn create_invalid_points_panics() {
        assert_eq!(
            BLS12377Curve::create_point_from_affine(FpE::from(1), FpE::from(1)).unwrap_err(),
            EllipticCurveError::InvalidPoint
        )
    }

    #[test]
    fn equality_works() {
        let g = BLS12377Curve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = BLS12377Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }

    #[test]
    fn generator_g1_is_in_subgroup() {
        let g = BLS12377Curve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn point1_is_in_subgroup() {
        let p = point_1();
        assert!(p.is_in_subgroup())
    }

    #[test]
    fn arbitrary_g1_point_is_in_subgroup() {
        let g = BLS12377Curve::generator().operate_with_self(32u64);
        assert!(g.is_in_subgroup())
    }
    #[test]
    fn generator_g2_is_in_subgroup() {
        let g = BLS12377TwistCurve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn g2_conjugate_works() {
        let a = Fp2::zero();
        let mut expected = a.conjugate();
        expected = expected.conjugate();

        assert_eq!(a, expected);
    }

    // GLV scalar multiplication tests for G1

    #[test]
    fn glv_decompose_splits_large_scalar() {
        // For a scalar k > ω, the decomposition must produce non-zero k2 and
        // reduce the max bit length below k. This catches the U256::mul (hi,lo) swap bug
        // which caused k2 to always be zero.
        let k = U256::from_hex_unchecked(
            "12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000000",
        );
        let (_, k1, _, k2) = glv_decompose_377(&k);
        assert!(
            k2 > U256::from_u64(0),
            "k2 should be non-zero for a large scalar"
        );
        let max_bits = core::cmp::max(k1.bits_le(), k2.bits_le());
        assert!(
            max_bits < k.bits_le(),
            "decomposition should reduce max bit length"
        );
    }

    #[test]
    fn glv_mul_subgroup_order_is_neutral() {
        // [r]P = O for any point P in the subgroup
        let g = BLS12377Curve::generator();
        assert!(g.glv_mul(&SUBGROUP_ORDER).is_neutral_element());
    }

    #[test]
    fn glv_mul_omega_scalar() {
        // Triggers the refinement branch (k mod (ω+1) == ω); must equal operate_with_self
        let g = BLS12377Curve::generator();
        let expected = g.operate_with_self(GLV_OMEGA);
        assert_eq!(g.glv_mul(&GLV_OMEGA), expected);
    }

    #[test]
    fn glv_mul_small_scalar() {
        let g = BLS12377Curve::generator();
        let k = U256::from_u64(12345);
        let expected = g.operate_with_self(12345u64);
        assert_eq!(g.glv_mul(&k), expected);
    }

    #[test]
    fn glv_mul_medium_scalar() {
        let g = BLS12377Curve::generator();
        let k = U256::from_hex_unchecked("123456789abcdef0123456789abcdef0");
        let expected = g.operate_with_self(k);
        assert_eq!(g.glv_mul(&k), expected);
    }

    #[test]
    fn glv_mul_large_scalar() {
        let g = BLS12377Curve::generator();
        let k = U256::from_hex_unchecked(
            "12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000000",
        );
        let expected = g.operate_with_self(k);
        assert_eq!(g.glv_mul(&k), expected);
    }

    #[test]
    fn glv_mul_neutral_element() {
        let neutral = ShortWeierstrassProjectivePoint::<BLS12377Curve>::neutral_element();
        let k = U256::from_u64(12345);
        assert!(neutral.glv_mul(&k).is_neutral_element());
    }

    #[test]
    fn glv_mul_zero_scalar() {
        let g = BLS12377Curve::generator();
        let k = U256::from_u64(0);
        assert!(g.glv_mul(&k).is_neutral_element());
    }

    #[test]
    fn phi_endomorphism_property() {
        // Verify φ(P) = [λ]P
        let g = BLS12377Curve::generator();
        let phi_g = g.phi();
        let lambda_g = g.operate_with_self(GLV_LAMBDA);
        assert_eq!(phi_g.to_affine().x(), lambda_g.to_affine().x());
        assert_eq!(phi_g.to_affine().y(), lambda_g.to_affine().y());
    }

    #[test]
    fn phi_cube_is_identity() {
        // φ³ = identity
        let g = BLS12377Curve::generator();
        let phi3_g = g.phi().phi().phi();
        assert_eq!(g.to_affine().x(), phi3_g.to_affine().x());
        assert_eq!(g.to_affine().y(), phi3_g.to_affine().y());
    }

    // GLS scalar multiplication tests for G2

    #[test]
    fn gls_mul_subgroup_order_is_neutral() {
        // [r]P = O for any point P in the subgroup
        let g = BLS12377TwistCurve::generator();
        assert!(g.gls_mul(&SUBGROUP_ORDER).is_neutral_element());
    }

    #[test]
    fn gls_mul_small_scalar() {
        let g = BLS12377TwistCurve::generator();
        let k = U256::from_u64(12345);
        let expected = g.operate_with_self(12345u64);
        assert_eq!(g.gls_mul(&k), expected);
    }

    #[test]
    fn gls_mul_medium_scalar() {
        let g = BLS12377TwistCurve::generator();
        let k = U256::from_hex_unchecked("123456789abcdef0123456789abcdef0");
        let expected = g.operate_with_self(k);
        assert_eq!(g.gls_mul(&k), expected);
    }

    #[test]
    fn gls_mul_large_scalar() {
        let g = BLS12377TwistCurve::generator();
        let k = U256::from_hex_unchecked(
            "12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000000",
        );
        let expected = g.operate_with_self(k);
        assert_eq!(g.gls_mul(&k), expected);
    }

    #[test]
    fn gls_mul_neutral_element() {
        let neutral = ShortWeierstrassProjectivePoint::<BLS12377TwistCurve>::neutral_element();
        let k = U256::from_u64(12345);
        assert!(neutral.gls_mul(&k).is_neutral_element());
    }

    #[test]
    fn gls_mul_zero_scalar() {
        let g = BLS12377TwistCurve::generator();
        let k = U256::from_u64(0);
        assert!(g.gls_mul(&k).is_neutral_element());
    }

    #[test]
    fn psi_endomorphism_property() {
        // Verify ψ(P) = [u]P where u is the curve seed
        let g = BLS12377TwistCurve::generator();
        let psi_g = g.psi();
        let u_g = g.operate_with_self(MILLER_LOOP_CONSTANT);
        assert_eq!(psi_g.to_affine().x(), u_g.to_affine().x());
        assert_eq!(psi_g.to_affine().y(), u_g.to_affine().y());
    }
}
