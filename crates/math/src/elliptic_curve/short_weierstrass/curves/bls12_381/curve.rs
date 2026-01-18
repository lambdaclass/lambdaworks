use super::{
    field_extension::{BLS12381PrimeField, Degree2ExtensionField},
    twist::BLS12381TwistCurve,
};
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::element::U256;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

pub const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

pub const CURVE_COFACTOR: U256 = U256::from_hex_unchecked("0x396c8c005555e1568c00aaab0000aaab");

pub type BLS12381FieldElement = FieldElement<BLS12381PrimeField>;
pub type BLS12381TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12381Curve;

impl IsEllipticCurve for BLS12381Curve {
    type BaseField = BLS12381PrimeField;
    // Use Jacobian coordinates for optimized point doubling (2M + 5S vs ~8M + 5S for projective)
    type PointRepresentation = ShortWeierstrassJacobianPoint<Self>;

    /// Returns the generator point of the BLS12-381 curve.
    ///
    /// # Safety
    ///
    /// - The generator point is mathematically verified to be a valid point on the curve.
    /// - `unwrap()` is safe because the provided coordinates satisfy the curve equation.
    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - These values are mathematically verified and known to be valid points on BLS12-381.
        // - `unwrap()` is safe because we **ensure** the input values satisfy the curve equation.
        let point= Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new_base("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"),
            FieldElement::<Self::BaseField>::new_base("8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1"),
            FieldElement::one()
        ]);
        point.unwrap()
    }
}

impl IsShortWeierstrass for BLS12381Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(4)
    }
}

/// This is equal to the frobenius trace of the BLS12 381 curve minus one or seed value z.
pub const MILLER_LOOP_CONSTANT: u64 = 0xd201000000010000;

/// 𝛽 : primitive cube root of unity of 𝐹ₚ that satisfies the minimal equation
/// 𝛽² + 𝛽 + 1 = 0 mod 𝑝
pub const CUBE_ROOT_OF_UNITY_G1: BLS12381FieldElement = FieldElement::from_hex_unchecked(
    "5f19672fdf76ce51ba69c6076a0f77eaddb3a93be6f89688de17d813620a00022e01fffffffefffe",
);

// GLV (Gallant-Lambert-Vanstone) Scalar Multiplication Constants
//
// The endomorphism φ(x, y) = (βx, y) satisfies φ(P) = [λ]P for all P in the r-torsion subgroup.
// GLV decomposition splits scalar k into k₁ + k₂·λ where |k₁|, |k₂| < √r.

/// The eigenvalue λ of the GLV endomorphism, satisfying λ² + λ + 1 ≡ 0 (mod r).
pub const GLV_LAMBDA: U256 =
    U256::from_hex_unchecked("73eda753299d7d483339d80809a1d804a7780001fffcb7fcfffffffe00000001");

/// The small cube root of unity ω in Fr (≈ 2^127), used for scalar decomposition.
const GLV_OMEGA: U256 =
    U256::from_hex_unchecked("00000000000000000000000000000000ac45a4010001a40200000000ffffffff");

/// ω + 1, used in the decomposition formula.
const GLV_OMEGA_PLUS_ONE: U256 =
    U256::from_hex_unchecked("00000000000000000000000000000000ac45a4010001a4020000000100000000");

/// x-coordinate of 𝜁 ∘ 𝜋_q ∘ 𝜁⁻¹, where 𝜁 is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E
pub const ENDO_U: BLS12381TwistCurveFieldElement =
BLS12381TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked("0"),
    FieldElement::from_hex_unchecked("1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaad")
]);

/// y-coordinate of 𝜁 ∘ 𝜋_q ∘ 𝜁⁻¹, where 𝜁 is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E
pub const ENDO_V: BLS12381TwistCurveFieldElement =
BLS12381TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked("135203e60180a68ee2e9c448d77a2cd91c3dedd930b1cf60ef396489f61eb45e304466cf3e67fa0af1ee7b04121bdea2"),
    FieldElement::from_hex_unchecked("6af0e0437ff400b6831e36d6bd17ffe48395dabc2d3435e77f76e17009241c5ee67992f72ec05f4c81084fbede3cc09")
]);

impl ShortWeierstrassJacobianPoint<BLS12381Curve> {
    /// Applies the GLV endomorphism: φ(x, y) = (βx, y) where β is the cube root of unity.
    /// Satisfies φ(P) = [λ]P where λ² + λ + 1 ≡ 0 (mod r).
    #[inline(always)]
    pub fn phi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        Self::new_unchecked([x * CUBE_ROOT_OF_UNITY_G1, y.clone(), z.clone()])
    }

    /// Checks subgroup membership using φ(P) = -u²P.
    pub fn is_in_subgroup(&self) -> bool {
        if self.is_neutral_element() {
            return true;
        }
        self.operate_with_self(MILLER_LOOP_CONSTANT)
            .operate_with_self(MILLER_LOOP_CONSTANT)
            .neg()
            == self.phi()
    }

    /// GLV scalar multiplication: computes [k]P using the endomorphism for ~2x speedup.
    ///
    /// Decomposes k = k1 + k2*ω with small k1, k2 (~128 bits each), then uses
    /// Shamir's trick for joint scalar multiplication.
    pub fn glv_mul(&self, k: &U256) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }

        let (k1_neg, k1, k2_neg, k2) = glv_decompose(k);
        let phi_p = self.phi();

        let p1 = if k1_neg { self.neg() } else { self.clone() };
        let p2 = if k2_neg { phi_p } else { phi_p.neg() };

        shamir_double_and_add(&p1, &k1, &p2, &k2)
    }
}

/// Decomposes scalar k into k₁ + k₂*ω (mod r) where |k₁|, |k₂| are approximately √r.
///
/// Returns (a_neg, |a|, b_neg, |b|) for the GLV formula: [k]P = [a]P + [b]φ(P).
fn glv_decompose(k: &U256) -> (bool, U256, bool, U256) {
    let zero = U256::from_u64(0);

    // Small scalars need no decomposition
    if *k < GLV_OMEGA {
        return (false, *k, false, zero);
    }

    // Compute k2 = k / (ω + 1)
    let (k2, _) = k.div_rem(&GLV_OMEGA_PLUS_ONE);

    // Compute k1 = k - k2*ω
    let (k2_omega_lo, k2_omega_hi) = U256::mul(&k2, &GLV_OMEGA);

    // Overflow check: fall back to direct computation if needed
    if k2_omega_hi != zero {
        return (false, *k, false, zero);
    }

    let (k1, k1_underflow) = if *k >= k2_omega_lo {
        (U256::sub(k, &k2_omega_lo).0, false)
    } else {
        (U256::sub(&k2_omega_lo, k).0, true)
    };

    // Compute a = k1 - k2
    let (a, a_neg) = if k1_underflow {
        let (sum, _) = U256::add(&k1, &k2);
        (sum, true)
    } else if k1 >= k2 {
        (U256::sub(&k1, &k2).0, false)
    } else {
        (U256::sub(&k2, &k1).0, true)
    };

    // Refine if a is still too large
    if a >= GLV_OMEGA && !a_neg {
        let (a_adj, _) = U256::sub(&a, &GLV_OMEGA);
        let (b_adj, _) = U256::add(&k2, &U256::from_u64(1));
        (false, a_adj, false, b_adj)
    } else {
        (a_neg, a, false, k2)
    }
}

/// Shamir's trick: computes [k1]P1 + [k2]P2 using joint double-and-add.
///
/// Shares doublings between both scalar multiplications for efficiency.
fn shamir_double_and_add(
    p1: &ShortWeierstrassJacobianPoint<BLS12381Curve>,
    k1: &U256,
    p2: &ShortWeierstrassJacobianPoint<BLS12381Curve>,
    k2: &U256,
) -> ShortWeierstrassJacobianPoint<BLS12381Curve> {
    let p1_plus_p2 = p1.operate_with(p2);
    let max_len = core::cmp::max(k1.bits_le(), k2.bits_le());

    if max_len == 0 {
        return ShortWeierstrassJacobianPoint::neutral_element();
    }

    let mut result = ShortWeierstrassJacobianPoint::neutral_element();

    for i in (0..max_len).rev() {
        result = result.double();

        match (get_bit(k1, i), get_bit(k2, i)) {
            (false, false) => {}
            (true, false) => result = result.operate_with(p1),
            (false, true) => result = result.operate_with(p2),
            (true, true) => result = result.operate_with(&p1_plus_p2),
        }
    }

    result
}

/// Gets bit at position `pos` from a U256 (little-endian bit indexing).
#[inline(always)]
fn get_bit(n: &U256, pos: usize) -> bool {
    if pos >= 256 {
        return false;
    }
    let limb_idx = 3 - pos / 64;
    let bit_idx = pos % 64;
    (n.limbs[limb_idx] >> bit_idx) & 1 == 1
}

/// Precomputed constants for ψ² endomorphism on G2.
/// ψ²(x, y, z) = (ENDO_U_2 * x, ENDO_V_2 * y, z)
/// Since ψ is the Frobenius, ψ² applies conjugate twice which is identity,
/// so we only need multiplication by the constants.
const ENDO_U_2: BLS12381TwistCurveFieldElement = BLS12381TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked(
        "1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaac",
    ),
    FieldElement::from_hex_unchecked("0"),
]);

const ENDO_V_2: BLS12381TwistCurveFieldElement = BLS12381TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaaa",
    ),
    FieldElement::from_hex_unchecked("0"),
]);

impl ShortWeierstrassJacobianPoint<BLS12381TwistCurve> {
    /// Computes 𝜓(P) = 𝜁 ∘ 𝜋ₚ ∘ 𝜁⁻¹, the Frobenius endomorphism on G2.
    /// 𝜁 is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E.
    /// 𝜋ₚ is the p-power Frobenius endomorphism.
    /// 𝜓 satisfies minimal equation 𝑋² + 𝑡𝑋 + 𝑞 = 𝑂
    /// https://eprint.iacr.org/2022/352.pdf 4.2 (7)
    ///
    /// Crucially: 𝜓(P) = [-x]P where x = MILLER_LOOP_CONSTANT (curve seed).
    pub fn psi(&self) -> Self {
        // The neutral element maps to itself
        if self.is_neutral_element() {
            return self.clone();
        }
        let [x, y, z] = self.coordinates();
        // SAFETY:
        // - `conjugate()` preserves the validity of the field element.
        // - `ENDO_U` and `ENDO_V` are precomputed constants that ensure the
        //   resulting point satisfies the curve equation.
        // - `unwrap()` is safe because the transformation follows
        //   **a known valid isomorphism** between the twist and E.
        let point = Self::new([
            x.conjugate() * ENDO_U,
            y.conjugate() * ENDO_V,
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

    /// 𝜓(P) = [-x]P, where x = SEED of the curve
    /// https://eprint.iacr.org/2022/352.pdf 4.2
    pub fn is_in_subgroup(&self) -> bool {
        // The neutral element is always in the subgroup
        if self.is_neutral_element() {
            return true;
        }
        self.psi() == self.operate_with_self(MILLER_LOOP_CONSTANT).neg()
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
        let (k0_neg, k0, k1_neg, k1, k2_neg, k2, k3_neg, k3) = gls_decompose_4d(k);

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
        shamir_4way(&p0, &k0, &p1, &k1, &p2, &k2, &p3, &k3)
    }
}

/// The curve seed x as U256 for division operations.
const GLS_X: U256 = U256::from_u64(MILLER_LOOP_CONSTANT);

/// GLS 4-dimensional scalar decomposition for BLS12-381 G2.
///
/// Decomposes scalar k into k₀ + k₁·λ + k₂·λ² + k₃·λ³ (mod r)
/// where λ = -x (curve seed negated in the scalar field).
///
/// Uses the minimal polynomial property of the endomorphism:
/// ψ²(P) + [t]ψ(P) + [q]P = O
///
/// For practical purposes, we use a simpler base-x decomposition:
/// k = k₀ + k₁·x + k₂·x² + k₃·x³
/// Then adjust signs based on ψ(P) = [-x]P property.
///
/// Returns (k0_neg, |k0|, k1_neg, |k1|, k2_neg, |k2|, k3_neg, |k3|)
fn gls_decompose_4d(k: &U256) -> (bool, U256, bool, U256, bool, U256, bool, U256) {
    let zero = U256::from_u64(0);

    // Small scalars: no decomposition needed
    if *k < GLS_X {
        return (false, *k, false, zero, false, zero, false, zero);
    }

    // Decompose in base x: k = k₀ + k₁·x + k₂·x² + k₃·x³
    let (q1, k0) = k.div_rem(&GLS_X);

    if q1 < GLS_X {
        // k fits in 2 limbs
        return (false, k0, true, q1, false, zero, false, zero);
    }

    let (q2, k1) = q1.div_rem(&GLS_X);

    if q2 < GLS_X {
        // k fits in 3 limbs
        return (false, k0, true, k1, false, q2, false, zero);
    }

    let (k3, k2) = q2.div_rem(&GLS_X);

    // Full 4-dimensional decomposition
    // Signs: k = k₀ - k₁·x + k₂·x² - k₃·x³ when using ψ(P) = [-x]P
    // Alternating signs because ψⁱ(P) = (-x)ⁱ P
    (false, k0, true, k1, false, k2, true, k3)
}

/// 4-way Shamir's trick for joint scalar multiplication.
/// Computes [k0]P0 + [k1]P1 + [k2]P2 + [k3]P3 sharing doublings.
fn shamir_4way(
    p0: &ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    k0: &U256,
    p1: &ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    k1: &U256,
    p2: &ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    k2: &U256,
    p3: &ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    k3: &U256,
) -> ShortWeierstrassJacobianPoint<BLS12381TwistCurve> {
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
        return ShortWeierstrassJacobianPoint::neutral_element();
    }

    let mut result = ShortWeierstrassJacobianPoint::neutral_element();

    for i in (0..max_len).rev() {
        result = result.double();

        let b0 = get_bit(k0, i);
        let b1 = get_bit(k1, i);
        let b2 = get_bit(k2, i);
        let b3 = get_bit(k3, i);

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
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::field_extension::BLS12381_PRIME_FIELD_ORDER,
            traits::EllipticCurveError,
        },
        field::element::FieldElement,
        unsigned_integer::element::U384,
    };

    // -15132376222941642751 = MILLER_LOOP_CONSTANT + 1 = -d20100000000ffff
    // we want the positive of this coordinate based on x^2 - tx + q
    pub const TRACE_OF_FROBENIUS: U256 = U256::from_u64(15132376222941642751);

    const ENDO_U_2: BLS12381TwistCurveFieldElement =
    BLS12381TwistCurveFieldElement::const_from_raw([
        FieldElement::from_hex_unchecked("1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaac"),
        FieldElement::from_hex_unchecked("0")
    ]);

    const ENDO_V_2: BLS12381TwistCurveFieldElement =
    BLS12381TwistCurveFieldElement::const_from_raw([
        FieldElement::from_hex_unchecked("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaaa"),
        FieldElement::from_hex_unchecked("0")
    ]);

    /// Computes the psi^2() 'Untwist Frobenius Endomorphism'
    ///  
    /// # Safety
    ///
    /// - This function assumes `p` is a valid point on the BLS12-381 twist curve.
    /// - The transformation involves multiplying the x and y coordinates by known constants.
    /// - `unwrap()` is used because the resulting point remains valid under the curve equations.
    fn psi_square(
        p: &ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    ) -> ShortWeierstrassJacobianPoint<BLS12381TwistCurve> {
        let [x, y, z] = p.coordinates();
        // Since power of frobenius map is 2 we apply once as applying twice is inverse

        // SAFETY:
        // - `ENDO_U_2` and `ENDO_V_2` are known valid constants.
        // - `unwrap()` is safe because the transformation preserves curve validity.
        ShortWeierstrassJacobianPoint::new([x * ENDO_U_2, y * ENDO_V_2, z.clone()]).unwrap()
    }

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BLS12381PrimeField>;
    #[allow(clippy::upper_case_acronyms)]
    type FTE = FieldElement<Degree2ExtensionField>;

    fn point_1() -> ShortWeierstrassJacobianPoint<BLS12381Curve> {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> ShortWeierstrassJacobianPoint<BLS12381Curve> {
        let x = FEE::new_base("32bcce7e71eb50384918e0c9809f73bde357027c6bf15092dd849aa0eac274d43af4c68a65fb2cda381734af5eecd5c");
        let y = FEE::new_base("11e48467b19458aabe7c8a42dc4b67d7390fdf1e150534caadddc7e6f729d8890b68a5ea6885a21b555186452b954d88");
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn adding_five_times_point_1_works() {
        let point_1 = point_1();
        let point_1_times_5 = point_1_times_5();
        assert_eq!(point_1.operate_with_self(5_u16), point_1_times_5);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(*p.x(), FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5"));
        assert_eq!(*p.y(), FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0"));
        assert_eq!(*p.z(), FEE::new_base("1"));
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            BLS12381Curve::create_point_from_affine(FEE::from(0), FEE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = BLS12381Curve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = BLS12381Curve::generator();
        let g2 = g.operate_with_self(2_u64);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        // calculate both sides of BLS12-381 equation
        let four = FieldElement::from(4);
        let y_sq_0 = x.pow(3_u16) + four;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = BLS12381Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }

    #[test]
    fn generator_g1_is_in_subgroup() {
        let g = BLS12381Curve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn arbitrary_g1_point_is_in_subgroup() {
        let g = BLS12381Curve::generator().operate_with_self(32u64);
        assert!(g.is_in_subgroup())
    }

    //TODO
    #[test]
    fn arbitrary_g1_point_not_in_subgroup() {
        let x = FEE::new_base("178212cbe4a3026c051d4f867364b3ea84af623f93233b347ffcd3d6b16f16e0a7aedbe1c78d33c6beca76b2b75c8486");
        let y = FEE::new_base("13a8b1347e5b43bc4051754b2a29928b5df78cf03ca3b1f73d0424b09fccdef116c9f0ecbec7420a99b2dd785209e9d");
        let p = BLS12381Curve::create_point_from_affine(x, y).unwrap();
        assert!(!p.is_in_subgroup())
    }

    #[test]
    fn generator_g2_is_in_subgroup() {
        let g = BLS12381TwistCurve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn arbitrary_g2_point_is_in_subgroup() {
        let g = BLS12381TwistCurve::generator().operate_with_self(32u64);
        assert!(g.is_in_subgroup())
    }

    //`TODO`
    #[test]
    fn arbitrary_g2_point_not_in_subgroup() {
        let x = FTE::new([
            FEE::new(U384::from_hex_unchecked("97798b4a61ac301bbee71e36b5174e2f4adfe3e1729bdae1fcc9965ae84181be373aa80414823eed694f1270014012d")),
            FEE::new(U384::from_hex_unchecked("c9852cc6e61868966249aec153b50b29b3c22409f4c7880fd13121981c103c8ef84d9ea29b552431360e82cf69219fa"))
        ]);
        let y = FTE::new([
            FEE::new(U384::from_hex_unchecked("16cb3a60f3fa52c8273aceeb94c4c7303e8074aa9eedec7355bbb1e8cceedd4ec1497f573f62822140377b8e339619ed")),
            FEE::new(U384::from_hex_unchecked("1cd919b08afe06bebe9adf6223a55868a6fd8b77efc5c67b60fff39be36e9b44b7f10db16827c83b43ad2dad1947778"))
        ]);

        let p = BLS12381TwistCurve::create_point_from_affine(x, y).unwrap();
        assert!(!p.is_in_subgroup())
    }

    #[test]
    fn g2_conjugate_works() {
        let a = FTE::zero();
        let mut expected = a.conjugate();
        expected = expected.conjugate();

        assert_eq!(a, expected);
    }

    #[test]
    fn untwist_morphism_has_minimal_poly() {
        // generator
        let p = BLS12381TwistCurve::generator();
        let psi_square = psi_square(&p);
        let tx = p.psi().operate_with_self(TRACE_OF_FROBENIUS).neg();
        let q = p.operate_with_self(BLS12381_PRIME_FIELD_ORDER);
        // Minimal Polynomial of Untwist Frobenius Endomorphism: X^2 + tX + q, where X = psh(P) -> psi(p)^2 - t * psi(p) + q * p = 0
        let min_poly = psi_square.operate_with(&tx.neg()).operate_with(&q);
        assert!(min_poly.is_neutral_element())
    }

    // GLV scalar multiplication tests

    #[test]
    fn glv_mul_small_scalar() {
        // Test with a small scalar (< λ)
        let g = BLS12381Curve::generator();
        let k = U256::from_u64(12345);
        let expected = g.operate_with_self(12345u64);
        let result = g.glv_mul(&k);
        assert_eq!(result, expected);
    }

    #[test]
    fn glv_mul_medium_scalar() {
        // Test with a medium scalar
        let g = BLS12381Curve::generator();
        let k = U256::from_hex_unchecked("123456789abcdef0123456789abcdef0");
        let expected = g.operate_with_self(k);
        let result = g.glv_mul(&k);
        assert_eq!(result, expected);
    }

    #[test]
    fn glv_mul_large_scalar() {
        // Test with a larger scalar that exercises the decomposition
        let g = BLS12381Curve::generator();
        let k = U256::from_hex_unchecked(
            "73eda753299d7d483339d80809a1d80553bda402fffe5bfefffffffe00000000",
        );
        let expected = g.operate_with_self(k);
        let result = g.glv_mul(&k);
        assert_eq!(result, expected);
    }

    #[test]
    fn glv_mul_neutral_element() {
        // GLV mul of neutral element should return neutral element
        let neutral = ShortWeierstrassJacobianPoint::<BLS12381Curve>::neutral_element();
        let k = U256::from_u64(12345);
        let result = neutral.glv_mul(&k);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn glv_mul_zero_scalar() {
        // [0]P should return neutral element
        let g = BLS12381Curve::generator();
        let k = U256::from_u64(0);
        let result = g.glv_mul(&k);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn phi_endomorphism_property() {
        // Verify φ(P) = [λ]P by comparing in affine coordinates
        let g = BLS12381Curve::generator();
        let phi_g = g.phi();
        let lambda_g = g.operate_with_self(GLV_LAMBDA);

        // Convert to affine for comparison to rule out Jacobian representation issues
        let phi_g_affine = phi_g.to_affine();
        let lambda_g_affine = lambda_g.to_affine();

        assert_eq!(
            phi_g_affine.x(),
            lambda_g_affine.x(),
            "φ(G).x should equal [λ]G.x"
        );
        assert_eq!(
            phi_g_affine.y(),
            lambda_g_affine.y(),
            "φ(G).y should equal [λ]G.y"
        );
    }

    #[test]
    fn phi_cube_is_identity() {
        // φ³ = identity, since φ has order 3
        let g = BLS12381Curve::generator();
        let phi_g = g.phi();
        let phi2_g = phi_g.phi();
        let phi3_g = phi2_g.phi();

        // Convert to affine for comparison
        let g_affine = g.to_affine();
        let phi3_g_affine = phi3_g.to_affine();

        assert_eq!(g_affine.x(), phi3_g_affine.x(), "φ³(G).x should equal G.x");
        assert_eq!(g_affine.y(), phi3_g_affine.y(), "φ³(G).y should equal G.y");
    }

    // G2 GLS scalar multiplication tests

    #[test]
    fn g2_gls_mul_small_scalar() {
        // Test with a small scalar (< curve seed x)
        let g = BLS12381TwistCurve::generator();
        let k = U256::from_u64(12345);
        let expected = g.operate_with_self(12345u64);
        let result = g.gls_mul(&k);
        assert_eq!(result, expected);
    }

    #[test]
    fn g2_gls_mul_medium_scalar() {
        // Test with a medium scalar that exercises decomposition
        let g = BLS12381TwistCurve::generator();
        let k = U256::from_hex_unchecked("123456789abcdef0123456789abcdef0");
        let expected = g.operate_with_self(k);
        let result = g.gls_mul(&k);
        assert_eq!(result, expected);
    }

    #[test]
    fn g2_gls_mul_large_scalar() {
        // Test with a larger scalar that uses full 4D decomposition
        let g = BLS12381TwistCurve::generator();
        let k = U256::from_hex_unchecked(
            "73eda753299d7d483339d80809a1d80553bda402fffe5bfefffffffe00000000",
        );
        let expected = g.operate_with_self(k);
        let result = g.gls_mul(&k);
        assert_eq!(result, expected);
    }

    #[test]
    fn g2_gls_mul_neutral_element() {
        // GLS mul of neutral element should return neutral element
        let neutral = ShortWeierstrassJacobianPoint::<BLS12381TwistCurve>::neutral_element();
        let k = U256::from_u64(12345);
        let result = neutral.gls_mul(&k);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn g2_gls_mul_zero_scalar() {
        // [0]P should return neutral element
        let g = BLS12381TwistCurve::generator();
        let k = U256::from_u64(0);
        let result = g.gls_mul(&k);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn g2_psi_endomorphism_property() {
        // Verify ψ(P) = [-x]P where x is the curve seed
        let g = BLS12381TwistCurve::generator();
        let psi_g = g.psi();
        let neg_x_g = g.operate_with_self(MILLER_LOOP_CONSTANT).neg();

        // Convert to affine for comparison
        let psi_g_affine = psi_g.to_affine();
        let neg_x_g_affine = neg_x_g.to_affine();

        assert_eq!(
            psi_g_affine.x(),
            neg_x_g_affine.x(),
            "ψ(G).x should equal [-x]G.x"
        );
        assert_eq!(
            psi_g_affine.y(),
            neg_x_g_affine.y(),
            "ψ(G).y should equal [-x]G.y"
        );
    }

    #[test]
    fn g2_psi2_endomorphism() {
        // Verify ψ²(P) = [x²]P (since ψ(P) = [-x]P, ψ²(P) = [x²]P)
        let g = BLS12381TwistCurve::generator();
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
        // Test with several different scalars to ensure correctness
        let g = BLS12381TwistCurve::generator();
        let scalars = [
            U256::from_u64(1),
            U256::from_u64(2),
            U256::from_u64(1000),
            U256::from_hex_unchecked("d201000000010000"), // = x
            U256::from_hex_unchecked("d201000000010001"), // = x + 1
            U256::from_hex_unchecked("ffffffffffffffff"),
            U256::from_hex_unchecked("ffffffffffffffffffffffffffffffff"),
        ];

        for k in scalars.iter() {
            let expected = g.operate_with_self(*k);
            let result = g.gls_mul(k);
            assert_eq!(result, expected, "GLS mul failed for scalar {:?}", k);
        }
    }
}
