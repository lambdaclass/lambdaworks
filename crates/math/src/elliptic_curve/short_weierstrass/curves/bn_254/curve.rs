use super::{
    field_extension::{BN254PrimeField, Degree2ExtensionField},
    pairing::{GAMMA_12, GAMMA_13, X},
    twist::BN254TwistCurve,
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

pub type BN254FieldElement = FieldElement<BN254PrimeField>;
pub type BN254TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

#[derive(Clone, Debug)]
pub struct BN254Curve;

impl IsEllipticCurve for BN254Curve {
    type BaseField = BN254PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    /// Returns the generator point of the BN254 curve.
    ///
    /// # Safety
    ///
    /// - The generator point is mathematically verified to be a valid point on the curve.
    /// - `unwrap()` is safe because the provided coordinates satisfy the curve equation.
    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - The generator coordinates `(1, 2, 1)` are **predefined** and belong to the BN254 curve.
        // - `unwrap()` is safe because we **ensure** the input values satisfy the curve equation.
        let point = Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::one(),
            FieldElement::<Self::BaseField>::from(2),
            FieldElement::one(),
        ]);
        point.unwrap()
    }
}

impl IsShortWeierstrass for BN254Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(3)
    }
}

// GLV (Gallant-Lambert-Vanstone) Scalar Multiplication Constants for G1
//
// The endomorphism φ(x, y) = (βx, y) satisfies φ(P) = [λ]P for all P in the r-torsion.
// β = β_large is the cube root of unity in Fp with eigenvalue λ_large in Fr.

/// β: primitive cube root of unity of Fp satisfying β² + β + 1 = 0 mod p.
/// Uses the large root β = β_small² so that φ(P) = [GLV_LAMBDA]P with large lambda.
pub const CUBE_ROOT_OF_UNITY_G1: BN254FieldElement = FieldElement::from_hex_unchecked(
    "30644e72e131a0295e6dd9e7e0acccb0c28f069fbb966e3de4bd44e5607cfd48",
);

/// The eigenvalue λ of the GLV endomorphism, satisfying λ² + λ + 1 ≡ 0 (mod r).
pub const GLV_LAMBDA: U256 =
    U256::from_hex_unchecked("30644e72e131a029048b6e193fd84104cc37a73fec2bc5e9b8ca0b2d36636f23");

/// The small cube root of unity ω in Fr (≈ 2^192), used for scalar decomposition.
const GLV_OMEGA: U256 =
    U256::from_hex_unchecked("b3c4d79d41a917585bfc41088d8daaa78b17ea66b99c90dd");

/// ω + 1, used in the decomposition formula.
const GLV_OMEGA_PLUS_ONE: U256 =
    U256::from_hex_unchecked("b3c4d79d41a917585bfc41088d8daaa78b17ea66b99c90de");

/// Frobenius eigenvalue for GLS on G2: φ(Q) = [p mod r]Q.
/// p mod r = t - 1 = 6x² where x is the BN254 seed.
const GLS_X_BN254: U256 = U256::from_hex_unchecked("6f4d8248eeb859fbf83e9682e87cfd46");

impl ShortWeierstrassProjectivePoint<BN254Curve> {
    pub fn is_in_subgroup(&self) -> bool {
        true
    }

    /// Applies the GLV endomorphism: φ(x, y) = (βx, y) where β is the cube root of unity in Fp.
    /// Satisfies `φ(P) = [GLV_LAMBDA]P`.
    pub fn phi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        Self::new_unchecked([x * CUBE_ROOT_OF_UNITY_G1, y.clone(), z.clone()])
    }

    /// GLV scalar multiplication: computes [k]P using the endomorphism.
    ///
    /// Decomposes k = k1 + k2*ω with k2 ≈ 62 bits and k1 ≈ 192 bits, then uses
    /// Shamir's trick. Reduces iterations from 254 to ~192 bits.
    /// Measured speedup: ~1.1x for 254-bit scalars.
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

        let (k1_neg, k1, k2_neg, k2) = glv_decompose_bn254(k);
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

impl ShortWeierstrassProjectivePoint<BN254TwistCurve> {
    /// phi morphism used to G2 subgroup check for twisted curve.
    /// We also use phi at the last lines of the Miller Loop of the pairing.
    /// phi(q) = (x^p, y^p, z^p), where (x, y, z) are the projective coordinates of q.
    /// See <https://hackmd.io/@Wimet/ry7z1Xj-2#Subgroup-Checks>.
    ///
    /// # Safety
    ///
    /// - The function assumes `self` is a valid point on the BN254 twist curve.
    /// - The transformation follows a known isomorphism and preserves validity.
    pub fn phi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        // SAFETY:
        // - `conjugate()` preserves the validity of the field element.
        // - `unwrap()` is safe because the transformation follows
        //   **a known valid isomorphism** between the twist and E.
        let point = Self::new([
            x.conjugate() * GAMMA_12,
            y.conjugate() * GAMMA_13,
            z.conjugate(),
        ]);
        point.unwrap()
    }

    // Checks if a G2 point is in the subgroup of the twisted curve.
    pub fn is_in_subgroup(&self) -> bool {
        let q_times_x = &self.operate_with_self(X);
        let q_times_x_plus_1 = &self.operate_with(q_times_x);
        let q_times_2x = q_times_x.double();

        // (x+1)Q + phi(xQ) + phi^2(xQ) == phi^3(2xQ)
        let phi_xq = q_times_x.phi();
        let phi2_xq = phi_xq.phi();
        q_times_x_plus_1.operate_with(&phi_xq.operate_with(&phi2_xq))
            == q_times_2x.phi().phi().phi()
    }

    /// GLS scalar multiplication: computes [k]P using the Frobenius endomorphism φ.
    ///
    /// φ(Q) = [p mod r]Q where p is the base field prime (~127 bits).
    /// Decomposes k = k₁ + k₂·(p mod r), so [k]Q = [k₁]Q + [k₂]φ(Q).
    /// Measured speedup: ~1.4x for 192-bit scalars, ~1.6x for 254-bit scalars.
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

        let (k1_neg, k1, k2_neg, k2) = gls_decompose_bn254(k);
        let phi_p = self.phi();

        // φ(Q) = [p mod r]Q (positive eigenvalue)
        let p1 = if k1_neg { self.neg() } else { self.clone() };
        let p2 = if k2_neg { phi_p.neg() } else { phi_p };

        // Use Jacobian coordinates for faster doubling (2M+5S vs 7M+5S in projective)
        let p1_jac = proj_to_jac(&p1);
        let p2_jac = proj_to_jac(&p2);
        let result_jac = shamir_two_scalar_mul(&p1_jac, &k1, &p2_jac, &k2);
        jac_to_proj(result_jac)
    }
}

/// Decomposes scalar k into k₁ + k₂*ω (mod r) for G1 GLV.
fn glv_decompose_bn254(k: &U256) -> (bool, U256, bool, U256) {
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
    if a >= GLV_OMEGA && !a_neg {
        let (a_adj, _) = U256::sub(&a, &GLV_OMEGA);
        let (b_adj, _) = U256::add(&k2, &U256::from_u64(1));
        (false, a_adj, true, b_adj)
    } else {
        (a_neg, a, true, k2)
    }
}

/// Decomposes scalar k for GLS: k = k₁ + k₂·(p mod r) (mod r).
///
/// φ(Q) = [p mod r]Q (positive eigenvalue, ~127 bits), giving ~50% speedup.
fn gls_decompose_bn254(k: &U256) -> (bool, U256, bool, U256) {
    let zero = U256::from_u64(0);

    if *k < GLS_X_BN254 {
        return (false, *k, false, zero);
    }

    let (k2, k1) = k.div_rem(&GLS_X_BN254);
    // φ(Q) = [p mod r]Q (positive), so [k]Q = [k₁]Q + [k₂]φ(Q)
    (false, k1, false, k2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement, unsigned_integer::element::U256,
    };

    use super::BN254Curve;

    #[allow(clippy::upper_case_acronyms)]
    type FpE = FieldElement<BN254PrimeField>;
    type Fp2E = FieldElement<Degree2ExtensionField>;

    /*
    Sage script:

    p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    Fbn128base = GF(p)
    bn128 = EllipticCurve(Fbn128base,[0,3])
    bn128.random_point()
    (17846236917809265466108795494334003231858579470112820692700477163012827709147 :
    17004516321005754027668809192838483252304167776681765357426682819242643291917 :
    1)
    */
    fn point() -> ShortWeierstrassProjectivePoint<BN254Curve> {
        let x = FpE::from_hex_unchecked(
            "27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb",
        );
        let y = FpE::from_hex_unchecked(
            "2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d",
        );
        BN254Curve::create_point_from_affine(x, y).unwrap()
    }

    /*
    Sage script:

    p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    a = 0
    b = 3
    Fp = GF(p)
    G1 = EllipticCurve(Fp, [a, b])

    P = G1(17846236917809265466108795494334003231858579470112820692700477163012827709147,17004516321005754027668809192838483252304167776681765357426682819242643291917)

    P * 5

    (10253039145495711056399135467328321588927131913042076209148619870699206197155 : 16767740621810149881158172518644598727924612864724721353109859494126614321586 : 1)

    hex(10253039145495711056399135467328321588927131913042076209148619870699206197155)
    = 0x16ab03b69dfb4f870b0143ebf6a71b7b2e4053ca7a4421d09a913b8b834bbfa3

    hex(16767740621810149881158172518644598727924612864724721353109859494126614321586) =
    0x2512347279ba1049ef97d4ec348d838f939d2b7623e88f4826643cf3889599b2
    */

    fn point_times_5() -> ShortWeierstrassProjectivePoint<BN254Curve> {
        let x = FpE::from_hex_unchecked(
            "16ab03b69dfb4f870b0143ebf6a71b7b2e4053ca7a4421d09a913b8b834bbfa3",
        );
        let y = FpE::from_hex_unchecked(
            "2512347279ba1049ef97d4ec348d838f939d2b7623e88f4826643cf3889599b2",
        );
        BN254Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn adding_five_times_point_works() {
        let point = point();
        let point_times_5 = point_times_5();
        assert_eq!(point.operate_with_self(5_u16), point_times_5);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point();
        assert_eq!(
            *p.x(),
            FpE::new_base("27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb")
        );
        assert_eq!(
            *p.y(),
            FpE::new_base("2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d")
        );
        assert_eq!(*p.z(), FpE::one());
    }

    #[test]
    fn addition_with_neutral_element_returns_same_element() {
        let p = point();
        assert_eq!(
            *p.x(),
            FpE::new_base("27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb")
        );
        assert_eq!(
            *p.y(),
            FpE::new_base("2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d")
        );

        let neutral_element = ShortWeierstrassProjectivePoint::<BN254Curve>::neutral_element();

        assert_eq!(p.operate_with(&neutral_element), p);
    }

    #[test]
    fn neutral_element_plus_neutral_element_is_neutral_element() {
        let neutral_element = ShortWeierstrassProjectivePoint::<BN254Curve>::neutral_element();

        assert_eq!(
            neutral_element.operate_with(&neutral_element),
            neutral_element
        );
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            BN254Curve::create_point_from_affine(FpE::from(0), FpE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = BN254Curve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = BN254Curve::generator();
        let g2 = g.operate_with_self(2_u64);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        // calculate both sides of BN254 equation
        let three = FpE::from(3);
        let y_sq_0 = x.pow(3_u16) + three;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = BN254Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }

    #[test]
    fn operate_with_self_works_2() {
        let g = BN254TwistCurve::generator();
        assert_eq!(
            (g.operate_with_self(X)).double(),
            (g.operate_with_self(2 * X))
        )
    }

    #[test]
    fn operate_with_self_works_3() {
        let g = BN254TwistCurve::generator();
        assert_eq!(
            (g.operate_with_self(X)).operate_with(&g),
            (g.operate_with_self(X + 1))
        )
    }

    #[test]
    fn generator_g2_is_in_subgroup() {
        let g = BN254TwistCurve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn other_g2_point_is_in_subgroup() {
        let g = BN254TwistCurve::generator().operate_with_self(32u64);
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn invalid_g2_is_not_in_subgroup() {
        let q = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edaddde46bd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920daef312c20b9f1099ecefa8b45575d349b0a6f04c16d0d58af900",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "22376289c558493c1d6cc413a5f07dcb54526a964e4e687b65a881aa9752faa2",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "05a7a5759338c23ca603c1c4adf979e004c2f3e3c5bad6f07693c59a85d600a9",
                )),
            ]),
            Fp2E::one(),
        ])
        .unwrap();
        assert!(!q.is_in_subgroup())
    }

    #[test]
    fn g2_conjugate_two_times_is_identity() {
        let a = Fp2E::zero();
        let mut expected = a.conjugate();
        expected = expected.conjugate();

        assert_eq!(a, expected);
    }

    #[test]
    fn apply_12_times_phi_is_identity() {
        let q = BN254TwistCurve::generator();
        let mut result = q.phi();
        for _ in 1..12 {
            result = result.phi();
        }
        assert_eq!(q, result)
    }

    // GLV scalar multiplication tests for G1

    #[test]
    fn glv_decompose_splits_large_scalar() {
        // For a scalar k > ω, the decomposition must produce non-zero k2 and
        // reduce the max bit length below k. This catches the U256::mul (hi,lo) swap bug
        // which caused k2 to always be zero.
        let k = U256::from_hex_unchecked(
            "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000",
        );
        let (_, k1, _, k2) = glv_decompose_bn254(&k);
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
    fn glv_mul_g1_subgroup_order_is_neutral() {
        // [r]P = O for any point P in the subgroup
        let g = BN254Curve::generator();
        // SUBGROUP_ORDER is not in scope here, use the known value
        let r = U256::from_hex_unchecked(
            "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
        );
        assert!(g.glv_mul(&r).is_neutral_element());
    }

    #[test]
    fn glv_mul_g1_small_scalar() {
        let g = BN254Curve::generator();
        let k = U256::from_u64(12345);
        let expected = g.operate_with_self(12345u64);
        assert_eq!(g.glv_mul(&k), expected);
    }

    #[test]
    fn glv_mul_g1_medium_scalar() {
        let g = BN254Curve::generator();
        let k = U256::from_hex_unchecked("123456789abcdef0123456789abcdef0");
        let expected = g.operate_with_self(k);
        assert_eq!(g.glv_mul(&k), expected);
    }

    #[test]
    fn glv_mul_g1_large_scalar() {
        let g = BN254Curve::generator();
        let k = U256::from_hex_unchecked(
            "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000",
        );
        let expected = g.operate_with_self(k);
        assert_eq!(g.glv_mul(&k), expected);
    }

    #[test]
    fn glv_mul_g1_neutral_element() {
        let neutral = ShortWeierstrassProjectivePoint::<BN254Curve>::neutral_element();
        let k = U256::from_u64(12345);
        assert!(neutral.glv_mul(&k).is_neutral_element());
    }

    #[test]
    fn glv_mul_g1_zero_scalar() {
        let g = BN254Curve::generator();
        let k = U256::from_u64(0);
        assert!(g.glv_mul(&k).is_neutral_element());
    }

    #[test]
    fn phi_g1_endomorphism_property() {
        // Verify φ(P) = [λ]P
        let g = BN254Curve::generator();
        let phi_g = g.phi();
        let lambda_g = g.operate_with_self(GLV_LAMBDA);
        assert_eq!(phi_g.to_affine().x(), lambda_g.to_affine().x());
        assert_eq!(phi_g.to_affine().y(), lambda_g.to_affine().y());
    }

    #[test]
    fn phi_g1_cube_is_identity() {
        // φ³ = identity
        let g = BN254Curve::generator();
        let phi3_g = g.phi().phi().phi();
        assert_eq!(g.to_affine().x(), phi3_g.to_affine().x());
        assert_eq!(g.to_affine().y(), phi3_g.to_affine().y());
    }

    // GLS scalar multiplication tests for G2

    #[test]
    fn gls_mul_g2_small_scalar() {
        let g = BN254TwistCurve::generator();
        let k = U256::from_u64(12345);
        let expected = g.operate_with_self(12345u64);
        assert_eq!(g.gls_mul(&k), expected);
    }

    #[test]
    fn gls_mul_g2_medium_scalar() {
        let g = BN254TwistCurve::generator();
        let k = U256::from_hex_unchecked("123456789abcdef0123456789abcdef0");
        let expected = g.operate_with_self(k);
        assert_eq!(g.gls_mul(&k), expected);
    }

    #[test]
    fn gls_mul_g2_large_scalar() {
        let g = BN254TwistCurve::generator();
        let k = U256::from_hex_unchecked(
            "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000",
        );
        let expected = g.operate_with_self(k);
        assert_eq!(g.gls_mul(&k), expected);
    }

    #[test]
    fn gls_mul_g2_neutral_element() {
        let neutral = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::neutral_element();
        let k = U256::from_u64(12345);
        assert!(neutral.gls_mul(&k).is_neutral_element());
    }

    #[test]
    fn gls_mul_g2_zero_scalar() {
        let g = BN254TwistCurve::generator();
        let k = U256::from_u64(0);
        assert!(g.gls_mul(&k).is_neutral_element());
    }

    #[test]
    fn phi_g2_frobenius_eigenvalue() {
        // Verify φ(Q) = [p mod r]Q where p mod r = GLS_X_BN254
        let g = BN254TwistCurve::generator();
        let phi_g = g.phi();
        let eigenval_g = g.operate_with_self(GLS_X_BN254);
        assert_eq!(phi_g.to_affine().x(), eigenval_g.to_affine().x());
        assert_eq!(phi_g.to_affine().y(), eigenval_g.to_affine().y());
    }
}
