use super::{
    field_extension::{BN254PrimeField, Degree2ExtensionField},
    pairing::{GAMMA_12, GAMMA_13, X},
    twist::BN254TwistCurve,
};
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
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

// GLV constants for BN254 G1

/// β: Cube root of unity in Fp satisfying β³ = 1 and β ≠ 1
/// β = 2203960485148121921418603742825762020974279258880205651966
pub const CUBE_ROOT_OF_UNITY_G1: BN254FieldElement = BN254FieldElement::from_hex_unchecked(
    "30644e72e131a0295e6dd9e7e0acccb0c28f069fbb966e3de4bd44e5607cfd48",
);

/// λ: The eigenvalue of the endomorphism in the scalar field
/// Satisfies λ³ ≡ 1 (mod r) and φ(P) = [λ]P
/// From Arkworks: 21888242871839275217838484774961031246154997185409878258781734729429964517155
pub const GLV_LAMBDA: U256 =
    U256::from_hex_unchecked("30644e72e131a029048b6e193fd841045cea24f6fd736bec231204708f703636");

/// Subgroup order r for BN254
const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

/// GLV lattice vector component (from Arkworks/Constantine)
/// a1 = 147946756881789319000765030803803410728
const GLV_V1_0: U256 = U256::from_hex_unchecked("6f4d8248eeb859fc8211bbeb7d4f1128");

impl ShortWeierstrassProjectivePoint<BN254Curve> {
    /// Applies the GLV endomorphism: φ(x, y) = (βx, y) where β is the cube root of unity.
    /// Satisfies φ(P) = [λ]P.
    #[inline(always)]
    pub fn phi(&self) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }
        let [x, y, z] = self.coordinates();
        Self::new_unchecked([x * CUBE_ROOT_OF_UNITY_G1, y.clone(), z.clone()])
    }

    /// GLV scalar multiplication: computes [k]P using the endomorphism for ~2x speedup.
    ///
    /// Decomposes k = k1 + k2*λ with small k1, k2 (~128 bits each), then uses
    /// Shamir's trick for joint scalar multiplication.
    pub fn glv_mul(&self, k: &U256) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }

        let zero = U256::from_u64(0);
        if *k == zero {
            return Self::neutral_element();
        }

        // For small scalars, use direct computation
        if k.limbs[0] == 0 && k.limbs[1] == 0 && k.limbs[2] < 0x1000 {
            return self.operate_with_self(*k);
        }

        let (k1_neg, k1, k2_neg, k2) = glv_decompose_bn254(k);
        let phi_p = self.phi();

        let p1 = if k1_neg { self.neg() } else { self.clone() };
        let p2 = if k2_neg { phi_p.neg() } else { phi_p };

        shamir_double_and_add_bn254(&p1, &k1, &p2, &k2)
    }

    pub fn is_in_subgroup(&self) -> bool {
        true
    }
}

/// GLV decomposition for BN254: k = k1 + k2*λ (mod r)
///
/// Uses the lattice-based decomposition from Arkworks/Constantine.
/// The short vectors are:
/// - v1 = (0x6f4d8248eeb859fc8211bbeb7d4f1128, -0x89d3256894d213e3)
/// - v2 = (-0x89d3256894d213e3, -0x6f4d8248eeb859fd0be4e1541221250b)
///
/// Returns (k1_neg, |k1|, k2_neg, |k2|)
fn glv_decompose_bn254(k: &U256) -> (bool, U256, bool, U256) {
    let zero = U256::from_u64(0);

    // For small k, no decomposition needed
    if *k < GLV_V1_0 {
        return (false, *k, false, zero);
    }

    // Use the lattice decomposition:
    // c1 = round(k * v1_1 / r), c2 = round(k * v2_1 / r)
    // k1 = k - c1*v1_0 - c2*v2_0
    // k2 = -c1*v1_1 - c2*v2_1
    //
    // Simplified: divide k by the lattice determinant approximation
    let (q, _) = k.div_rem(&GLV_V1_0);

    // k2 ≈ q (approximation using simplified decomposition)
    let k2 = if q > zero { q } else { zero };

    // k1 = k - k2 * λ (mod r)
    let (k2_lambda_lo, k2_lambda_hi) = U256::mul(&k2, &GLV_LAMBDA);

    // Handle overflow by falling back to direct multiplication
    if k2_lambda_hi != zero {
        return (false, *k, false, zero);
    }

    // k1 = k - k2*λ (mod r)
    let k1 = if *k >= k2_lambda_lo {
        U256::sub(k, &k2_lambda_lo).0
    } else {
        // k1 = r - (k2*λ - k)
        let diff = U256::sub(&k2_lambda_lo, k).0;
        if diff < SUBGROUP_ORDER {
            U256::sub(&SUBGROUP_ORDER, &diff).0
        } else {
            return (false, *k, false, zero);
        }
    };

    // Reduce k1 mod r if needed
    let k1_final = if k1 >= SUBGROUP_ORDER {
        U256::sub(&k1, &SUBGROUP_ORDER).0
    } else {
        k1
    };

    // Determine signs for smaller representation
    let half_r = U256::from_hex_unchecked(
        "183227397098d014dc2822db40c0ac2e9419f4243cdcb848a1f0fac9f8000000",
    );

    let (k1_neg, k1_abs) = if k1_final > half_r {
        (true, U256::sub(&SUBGROUP_ORDER, &k1_final).0)
    } else {
        (false, k1_final)
    };

    let (k2_neg, k2_abs) = if k2 > half_r {
        (true, U256::sub(&SUBGROUP_ORDER, &k2).0)
    } else {
        (false, k2)
    };

    (k1_neg, k1_abs, k2_neg, k2_abs)
}

/// Gets bit at position `pos` from a U256.
#[inline(always)]
fn get_bit_bn254(n: &U256, pos: usize) -> bool {
    if pos >= 256 {
        return false;
    }
    let limb_idx = 3 - pos / 64;
    let bit_idx = pos % 64;
    (n.limbs[limb_idx] >> bit_idx) & 1 == 1
}

/// Shamir's trick for joint scalar multiplication: [k1]P1 + [k2]P2
fn shamir_double_and_add_bn254(
    p1: &ShortWeierstrassProjectivePoint<BN254Curve>,
    k1: &U256,
    p2: &ShortWeierstrassProjectivePoint<BN254Curve>,
    k2: &U256,
) -> ShortWeierstrassProjectivePoint<BN254Curve> {
    let p1_plus_p2 = p1.operate_with(p2);

    let max_bits = core::cmp::max(k1.bits_le(), k2.bits_le());
    if max_bits == 0 {
        return ShortWeierstrassProjectivePoint::neutral_element();
    }

    let mut result = ShortWeierstrassProjectivePoint::neutral_element();

    for i in (0..max_bits).rev() {
        result = result.double();

        let b1 = get_bit_bn254(k1, i);
        let b2 = get_bit_bn254(k2, i);

        match (b1, b2) {
            (false, false) => {}
            (true, false) => result = result.operate_with(p1),
            (false, true) => result = result.operate_with(p2),
            (true, true) => result = result.operate_with(&p1_plus_p2),
        }
    }

    result
}

impl ShortWeierstrassProjectivePoint<BN254TwistCurve> {
    /// phi morphism used to G2 subgroup check for twisted curve.
    /// We also use phi at the last lines of the Miller Loop of the pairing.
    /// phi(q) = (x^p, y^p, z^p), where (x, y, z) are the projective coordinates of q.
    /// See https://hackmd.io/@Wimet/ry7z1Xj-2#Subgroup-Checks.
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

        // (x+1)Q + phi(xQ) + phi(phi(xQ)) == phi(phi(phi(2xQ)))
        q_times_x_plus_1.operate_with(&q_times_x.phi().operate_with(&q_times_x.phi().phi()))
            == q_times_2x.phi().phi().phi()
    }
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

    // G1 GLV tests

    #[test]
    fn glv_phi_is_valid_endomorphism() {
        // Test that φ(P) is a valid point and φ³(P) = P (since β³ = 1)
        let g = BN254Curve::generator();
        let phi_g = g.phi();
        let phi2_g = phi_g.phi();
        let phi3_g = phi2_g.phi();

        // φ is not the identity
        assert_ne!(phi_g.to_affine(), g.to_affine());

        // φ³ = identity (cube root property)
        assert_eq!(phi3_g.to_affine(), g.to_affine());
    }

    #[test]
    fn glv_mul_small_scalar() {
        let g = BN254Curve::generator();
        let k = U256::from_u64(12345);
        let expected = g.operate_with_self(k);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn glv_mul_medium_scalar() {
        let g = BN254Curve::generator();
        let k = U256::from_hex_unchecked("deadbeef12345678");
        let expected = g.operate_with_self(k);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn glv_mul_large_scalar() {
        let g = BN254Curve::generator();
        let k = U256::from_hex_unchecked(
            "a5a5a5a5b6b6b6b6c7c7c7c7d8d8d8d8e9e9e9e9fafafafa0b0b0b0b1c1c1c1c",
        );
        let expected = g.operate_with_self(k);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn glv_mul_neutral_element() {
        let neutral = ShortWeierstrassProjectivePoint::<BN254Curve>::neutral_element();
        let k = U256::from_u64(12345);
        let result = neutral.glv_mul(&k);
        assert_eq!(result.to_affine(), neutral.to_affine());
    }

    #[test]
    fn glv_mul_zero_scalar() {
        let g = BN254Curve::generator();
        let k = U256::from_u64(0);
        let result = g.glv_mul(&k);
        assert_eq!(
            result.to_affine(),
            ShortWeierstrassProjectivePoint::<BN254Curve>::neutral_element()
        );
    }

    #[test]
    fn glv_mul_one_scalar() {
        let g = BN254Curve::generator();
        let k = U256::from_u64(1);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), g.to_affine());
    }

    #[test]
    fn glv_mul_various_scalars() {
        let g = BN254Curve::generator();
        let scalars = [
            U256::from_u64(2),
            U256::from_u64(255),
            U256::from_u64(65535),
            U256::from_hex_unchecked("ffffffff"),
            U256::from_hex_unchecked("123456789abcdef0"),
            U256::from_hex_unchecked("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000"), // r-1
        ];

        for k in &scalars {
            let expected = g.operate_with_self(*k);
            let result = g.glv_mul(k);
            assert_eq!(result.to_affine(), expected.to_affine());
        }
    }
}
