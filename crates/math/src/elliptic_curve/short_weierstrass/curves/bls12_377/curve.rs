use super::{
    field_extension::{BLS12377PrimeField, Degree2ExtensionField},
    pairing::{GAMMA_12, GAMMA_13},
    twist::BLS12377TwistCurve,
};
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::element::U256;

use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

pub const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001");

pub const CURVE_COFACTOR: U256 = U256::from_hex_unchecked("0x170b5d44300000000000000000000000");

pub type BLS12377FieldElement = FieldElement<BLS12377PrimeField>;
pub type BLS12377TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

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

/// 𝛽 : primitive cube root of unity of 𝐹ₚ that §satisfies the minimal equation
/// 𝛽² + 𝛽 + 1 = 0 mod 𝑝
pub const CUBE_ROOT_OF_UNITY_G1: BLS12377FieldElement = FieldElement::from_hex_unchecked(
    "0x1ae3a4617c510eabc8756ba8f8c524eb8882a75cc9bc8e359064ee822fb5bffd1e945779fffffffffffffffffffffff",
);

/// x-coordinate of 𝜁 ∘ 𝜋_q ∘ 𝜁⁻¹, where 𝜁 is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E
pub const ENDO_U: BLS12377TwistCurveFieldElement =
    BLS12377TwistCurveFieldElement::const_from_raw([
        FieldElement::from_hex_unchecked(
            "9B3AF05DD14F6EC619AAF7D34594AABC5ED1347970DEC00452217CC900000008508C00000000002",
        ),
        FieldElement::from_hex_unchecked("0"),
    ]);

/// y-coordinate of 𝜁 ∘ 𝜋_q ∘ 𝜁⁻¹, where 𝜁 is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E
pub const ENDO_V: BLS12377TwistCurveFieldElement =
    BLS12377TwistCurveFieldElement::const_from_raw([
        FieldElement::from_hex_unchecked("1680A40796537CAC0C534DB1A79BEB1400398F50AD1DEC1BCE649CF436B0F6299588459BFF27D8E6E76D5ECF1391C63"),
        FieldElement::from_hex_unchecked("0"),
    ]);

impl ShortWeierstrassProjectivePoint<BLS12377Curve> {
    /// Returns 𝜙(P) = (𝑥, 𝑦) ⇒ (𝛽𝑥, 𝑦), where 𝛽 is the Cube Root of Unity in the base prime field
    /// https://eprint.iacr.org/2022/352.pdf 2 Preliminaries
    pub fn phi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        let new_x = x * CUBE_ROOT_OF_UNITY_G1;
        // SAFETY: The value `x` is computed correctly, so the point is in the curve.
        Self::new_unchecked([new_x, y.clone(), z.clone()])
    }

    /// 𝜙(P) = −𝑢²P
    /// https://eprint.iacr.org/2022/352.pdf 4.3 Prop. 4
    pub fn is_in_subgroup(&self) -> bool {
        self.operate_with_self(MILLER_LOOP_CONSTANT)
            .operate_with_self(MILLER_LOOP_CONSTANT)
            .neg()
            == self.phi()
    }

    /// GLV scalar multiplication: computes [k]P using the endomorphism.
    ///
    /// Note: BLS12-377 G1 has φ(P) = -u²P where u is the curve seed.
    /// This requires lattice-based decomposition which is not yet implemented.
    /// Currently falls back to direct multiplication for correctness.
    ///
    /// TODO: Implement proper GLV lattice decomposition for BLS12-377 G1.
    pub fn glv_mul(&self, k: &U256) -> Self {
        // For now, use direct multiplication
        // Proper GLV requires lattice-based decomposition with λ = -u² mod r
        self.operate_with_self(*k)
    }
}

/// Precomputed constants for ψ² endomorphism on G2.
/// Since ψ is the Frobenius, ψ² applies conjugate twice which is identity,
/// so we only need multiplication by the constants.
const ENDO_U_2: BLS12377TwistCurveFieldElement = BLS12377TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked(
        "9b3af05dd14f6ec619aaf7d34594aabc5ed1347970dec00452217cc900000008508c00000000001",
    ),
    FieldElement::from_hex_unchecked("0"),
]);

const ENDO_V_2: BLS12377TwistCurveFieldElement = BLS12377TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked(
        "1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000000",
    ),
    FieldElement::from_hex_unchecked("0"),
]);

impl ShortWeierstrassProjectivePoint<BLS12377TwistCurve> {
    /// Computes 𝜓(P) = 𝜁 ∘ 𝜋ₚ ∘ 𝜁⁻¹, the Frobenius endomorphism on G2.
    /// 𝜁 is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E.
    /// 𝜋ₚ is the p-power Frobenius endomorphism.
    /// 𝜓 satisfies minimal equation 𝑋² + 𝑡𝑋 + 𝑞 = 𝑂
    /// https://eprint.iacr.org/2022/352.pdf 4.2 (7)
    ///
    /// Crucially: 𝜓(P) = [x]P where x = MILLER_LOOP_CONSTANT (curve seed).
    /// Note: Unlike BLS12-381, BLS12-377 has ψ(P) = [x]P (positive).
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

    /// Computes ψ²(P), the square of the Frobenius endomorphism.
    /// Since conjugate² = identity, ψ² only needs multiplication by constants.
    pub fn psi2(&self) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }
        let [x, y, z] = self.coordinates();
        // SAFETY: ENDO_U_2 and ENDO_V_2 are precomputed to preserve curve validity.
        Self::new_unchecked([x * ENDO_U_2, y * ENDO_V_2, z.clone()])
    }

    /// Computes ψ³(P) = ψ(ψ²(P)).
    pub fn psi3(&self) -> Self {
        self.psi2().psi()
    }

    /// 𝜓(P) = [x]P, where x = SEED of the curve
    /// https://eprint.iacr.org/2022/352.pdf 4.2
    pub fn is_in_subgroup(&self) -> bool {
        self.psi() == self.operate_with_self(MILLER_LOOP_CONSTANT)
    }

    /// GLS scalar multiplication: computes [k]P using the Frobenius endomorphism.
    ///
    /// Uses 4-dimensional decomposition:
    /// k·P = k₀·P + k₁·ψ(P) + k₂·ψ²(P) + k₃·ψ³(P)
    ///
    /// Each mini-scalar is approximately 64 bits (vs 256 bits for full scalar),
    /// providing ~3-4x speedup over standard double-and-add.
    pub fn gls_mul(&self, k: &U256) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }

        let zero = U256::from_u64(0);
        if *k == zero {
            return Self::neutral_element();
        }

        // For small scalars, direct computation is faster
        if k.limbs[0] == 0 && k.limbs[1] == 0 && k.limbs[2] == 0 && k.limbs[3] < 1024 {
            return self.operate_with_self(*k);
        }

        // Decompose k into 4 mini-scalars
        let (k0_neg, k0, k1_neg, k1, k2_neg, k2, k3_neg, k3) = gls_decompose_4d_377(k);

        // Compute ψ(P), ψ²(P), ψ³(P)
        let psi_p = self.psi();
        let psi2_p = self.psi2();
        let psi3_p = self.psi3();

        // Apply signs
        let p0 = if k0_neg { self.neg() } else { self.clone() };
        let p1 = if k1_neg { psi_p.neg() } else { psi_p };
        let p2 = if k2_neg { psi2_p.neg() } else { psi2_p };
        let p3 = if k3_neg { psi3_p.neg() } else { psi3_p };

        // 4-way Shamir's trick
        shamir_4way_377(&p0, &k0, &p1, &k1, &p2, &k2, &p3, &k3)
    }
}

/// The curve seed x as U256 for division operations.
const GLS_X_377: U256 = U256::from_u64(MILLER_LOOP_CONSTANT);

/// GLS 4-dimensional scalar decomposition for BLS12-377 G2.
///
/// Decomposes scalar k into k₀ + k₁·x + k₂·x² + k₃·x³
/// where x is the curve seed.
///
/// For BLS12-377, ψ(P) = [x]P (positive), so:
/// k·P = k₀·P + k₁·ψ(P) + k₂·ψ²(P) + k₃·ψ³(P)
///     = k₀·P + k₁·[x]P + k₂·[x²]P + k₃·[x³]P
///
/// Returns (k0_neg, |k0|, k1_neg, |k1|, k2_neg, |k2|, k3_neg, |k3|)
fn gls_decompose_4d_377(k: &U256) -> (bool, U256, bool, U256, bool, U256, bool, U256) {
    let zero = U256::from_u64(0);

    // Small scalars: no decomposition needed
    if *k < GLS_X_377 {
        return (false, *k, false, zero, false, zero, false, zero);
    }

    // Decompose in base x: k = k₀ + k₁·x + k₂·x² + k₃·x³
    let (q1, k0) = k.div_rem(&GLS_X_377);

    if q1 < GLS_X_377 {
        // k fits in 2 limbs - all positive for BLS12-377
        return (false, k0, false, q1, false, zero, false, zero);
    }

    let (q2, k1) = q1.div_rem(&GLS_X_377);

    if q2 < GLS_X_377 {
        // k fits in 3 limbs
        return (false, k0, false, k1, false, q2, false, zero);
    }

    let (k3, k2) = q2.div_rem(&GLS_X_377);

    // Full 4-dimensional decomposition
    // All positive since ψ(P) = [x]P (not negative like BLS12-381)
    (false, k0, false, k1, false, k2, false, k3)
}

/// Gets bit at position `pos` from a U256 (little-endian bit indexing).
#[inline(always)]
fn get_bit_377(n: &U256, pos: usize) -> bool {
    if pos >= 256 {
        return false;
    }
    let limb_idx = 3 - pos / 64;
    let bit_idx = pos % 64;
    (n.limbs[limb_idx] >> bit_idx) & 1 == 1
}

/// 4-way Shamir's trick for joint scalar multiplication.
/// Computes [k0]P0 + [k1]P1 + [k2]P2 + [k3]P3 sharing doublings.
fn shamir_4way_377(
    p0: &ShortWeierstrassProjectivePoint<BLS12377TwistCurve>,
    k0: &U256,
    p1: &ShortWeierstrassProjectivePoint<BLS12377TwistCurve>,
    k1: &U256,
    p2: &ShortWeierstrassProjectivePoint<BLS12377TwistCurve>,
    k2: &U256,
    p3: &ShortWeierstrassProjectivePoint<BLS12377TwistCurve>,
    k3: &U256,
) -> ShortWeierstrassProjectivePoint<BLS12377TwistCurve> {
    // Precompute all 15 non-trivial sums for 4-bit lookup
    let p01 = p0.operate_with(p1);
    let p02 = p0.operate_with(p2);
    let p03 = p0.operate_with(p3);
    let p12 = p1.operate_with(p2);
    let p13 = p1.operate_with(p3);
    let p23 = p2.operate_with(p3);
    let p012 = p01.operate_with(p2);
    let p013 = p01.operate_with(p3);
    let p023 = p02.operate_with(p3);
    let p123 = p12.operate_with(p3);
    let p0123 = p012.operate_with(p3);

    let max_len = core::cmp::max(
        core::cmp::max(k0.bits_le(), k1.bits_le()),
        core::cmp::max(k2.bits_le(), k3.bits_le()),
    );

    if max_len == 0 {
        return ShortWeierstrassProjectivePoint::neutral_element();
    }

    let mut result = ShortWeierstrassProjectivePoint::neutral_element();

    for i in (0..max_len).rev() {
        result = result.double();

        let b0 = get_bit_377(k0, i);
        let b1 = get_bit_377(k1, i);
        let b2 = get_bit_377(k2, i);
        let b3 = get_bit_377(k3, i);

        // 4-bit lookup table
        let point = match (b0, b1, b2, b3) {
            (false, false, false, false) => continue,
            (true, false, false, false) => p0,
            (false, true, false, false) => p1,
            (true, true, false, false) => &p01,
            (false, false, true, false) => p2,
            (true, false, true, false) => &p02,
            (false, true, true, false) => &p12,
            (true, true, true, false) => &p012,
            (false, false, false, true) => p3,
            (true, false, false, true) => &p03,
            (false, true, false, true) => &p13,
            (true, true, false, true) => &p013,
            (false, false, true, true) => &p23,
            (true, false, true, true) => &p023,
            (false, true, true, true) => &p123,
            (true, true, true, true) => &p0123,
        };
        result = result.operate_with(point);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement,
    };

    use super::BLS12377Curve;

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

    // G2 GLS scalar multiplication tests

    #[test]
    fn g2_gls_mul_small_scalar() {
        let g = BLS12377TwistCurve::generator();
        let k = U256::from_u64(12345);
        let expected = g.operate_with_self(12345u64);
        let result = g.gls_mul(&k);
        assert_eq!(result, expected);
    }

    #[test]
    fn g2_gls_mul_medium_scalar() {
        let g = BLS12377TwistCurve::generator();
        let k = U256::from_hex_unchecked("123456789abcdef0123456789abcdef0");
        let expected = g.operate_with_self(k);
        let result = g.gls_mul(&k);
        assert_eq!(result, expected);
    }

    #[test]
    fn g2_gls_mul_large_scalar() {
        let g = BLS12377TwistCurve::generator();
        let k = U256::from_hex_unchecked(
            "12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11700000000000",
        );
        let expected = g.operate_with_self(k);
        let result = g.gls_mul(&k);
        assert_eq!(result, expected);
    }

    #[test]
    fn g2_gls_mul_neutral_element() {
        let neutral = ShortWeierstrassProjectivePoint::<BLS12377TwistCurve>::neutral_element();
        let k = U256::from_u64(12345);
        let result = neutral.gls_mul(&k);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn g2_gls_mul_zero_scalar() {
        let g = BLS12377TwistCurve::generator();
        let k = U256::from_u64(0);
        let result = g.gls_mul(&k);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn g2_psi_endomorphism_property() {
        // Verify ψ(P) = [x]P where x is the curve seed
        let g = BLS12377TwistCurve::generator();
        let psi_g = g.psi();
        let x_g = g.operate_with_self(MILLER_LOOP_CONSTANT);

        // Convert to affine for comparison
        let psi_g_affine = psi_g.to_affine();
        let x_g_affine = x_g.to_affine();

        assert_eq!(psi_g_affine.x(), x_g_affine.x(), "ψ(G).x should equal [x]G.x");
        assert_eq!(psi_g_affine.y(), x_g_affine.y(), "ψ(G).y should equal [x]G.y");
    }

    #[test]
    fn g2_psi2_endomorphism() {
        let g = BLS12377TwistCurve::generator();
        let psi2_g = g.psi2();
        // ψ² can also be computed as ψ(ψ(P))
        let psi_psi_g = g.psi().psi();

        let psi2_g_affine = psi2_g.to_affine();
        let psi_psi_g_affine = psi_psi_g.to_affine();

        assert_eq!(
            psi2_g_affine.x(),
            psi_psi_g_affine.x(),
            "ψ²(G).x should equal ψ(ψ(G)).x"
        );
        assert_eq!(
            psi2_g_affine.y(),
            psi_psi_g_affine.y(),
            "ψ²(G).y should equal ψ(ψ(G)).y"
        );
    }

    #[test]
    fn g2_gls_mul_various_scalars() {
        let g = BLS12377TwistCurve::generator();
        let scalars = [
            U256::from_u64(1),
            U256::from_u64(2),
            U256::from_u64(1000),
            U256::from_hex_unchecked("8508c00000000001"), // = x
            U256::from_hex_unchecked("8508c00000000002"), // = x + 1
            U256::from_hex_unchecked("ffffffffffffffff"),
            U256::from_hex_unchecked("ffffffffffffffffffffffffffffffff"),
        ];

        for k in scalars.iter() {
            let expected = g.operate_with_self(*k);
            let result = g.gls_mul(k);
            assert_eq!(result, expected, "GLS mul failed for scalar {:?}", k);
        }
    }

    // G1 GLV scalar multiplication tests

    #[test]
    fn g1_glv_phi_endomorphism_property() {
        // Test that φ(P) = [-u²]P where u is the curve seed
        let g = BLS12377Curve::generator();
        let phi_g = g.phi();
        // φ(P) = -u²P
        let u_squared_g = g
            .operate_with_self(MILLER_LOOP_CONSTANT)
            .operate_with_self(MILLER_LOOP_CONSTANT);
        let expected = u_squared_g.neg();
        assert_eq!(phi_g.to_affine(), expected.to_affine());
    }

    #[test]
    fn g1_glv_mul_small_scalar() {
        let g = BLS12377Curve::generator();
        let k = U256::from_u64(12345);
        let expected = g.operate_with_self(k);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn g1_glv_mul_medium_scalar() {
        let g = BLS12377Curve::generator();
        let k = U256::from_hex_unchecked("deadbeef12345678abcdef");
        let expected = g.operate_with_self(k);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn g1_glv_mul_large_scalar() {
        let g = BLS12377Curve::generator();
        // Use a 256-bit scalar (64 hex chars max)
        let k = U256::from_hex_unchecked(
            "a5a5a5a5b6b6b6b6c7c7c7c7d8d8d8d8e9e9e9e9fafafafa0b0b0b0b1c1c1c1c",
        );
        let expected = g.operate_with_self(k);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn g1_glv_mul_neutral_element() {
        let neutral = ShortWeierstrassProjectivePoint::<BLS12377Curve>::neutral_element();
        let k = U256::from_u64(12345);
        let result = neutral.glv_mul(&k);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn g1_glv_mul_zero_scalar() {
        let g = BLS12377Curve::generator();
        let k = U256::from_u64(0);
        let result = g.glv_mul(&k);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn g1_glv_mul_one_scalar() {
        let g = BLS12377Curve::generator();
        let k = U256::from_u64(1);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), g.to_affine());
    }

    #[test]
    fn g1_glv_mul_various_scalars() {
        let g = BLS12377Curve::generator();
        let scalars = [
            U256::from_u64(2),
            U256::from_u64(255),
            U256::from_u64(65535),
            U256::from_hex_unchecked("ffffffff"),
            U256::from_hex_unchecked("123456789abcdef0"),
            U256::from_hex_unchecked("ffffffffffffffffffffffffffffffff"),
        ];

        for k in &scalars {
            let expected = g.operate_with_self(*k);
            let result = g.glv_mul(k);
            assert_eq!(result.to_affine(), expected.to_affine());
        }
    }
}
