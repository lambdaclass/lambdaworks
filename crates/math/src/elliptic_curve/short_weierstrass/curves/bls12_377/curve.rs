use super::{
    field_extension::BLS12377PrimeField,
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
const MILLER_LOOP_CONSTANT_SQ: u128 = (MILLER_LOOP_CONSTANT as u128) * (MILLER_LOOP_CONSTANT as u128);

/// 𝛽 : primitive cube root of unity of 𝐹ₚ that §satisfies the minimal equation
/// 𝛽² + 𝛽 + 1 = 0 mod 𝑝
pub const CUBE_ROOT_OF_UNITY_G1: BLS12377FieldElement = FieldElement::from_hex_unchecked(
    "0x1ae3a4617c510eabc8756ba8f8c524eb8882a75cc9bc8e359064ee822fb5bffd1e945779fffffffffffffffffffffff",
);

impl ShortWeierstrassProjectivePoint<BLS12377Curve> {
    /// Returns 𝜙(P) = (𝑥, 𝑦) ⇒ (𝛽𝑥, 𝑦), where 𝛽 is the Cube Root of Unity in the base prime field.
    /// See <https://eprint.iacr.org/2022/352.pdf> Section 2 Preliminaries.
    fn phi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        let new_x = x * CUBE_ROOT_OF_UNITY_G1;
        // SAFETY: The value `x` is computed correctly, so the point is in the curve.
        Self::new_unchecked([new_x, y.clone(), z.clone()])
    }

    /// 𝜙(P) = −𝑢²P.
    /// See <https://eprint.iacr.org/2022/352.pdf> Section 4.3 Prop. 4.
    pub fn is_in_subgroup(&self) -> bool {
        if self.is_neutral_element() {
            return true;
        }
        self.operate_with_self(MILLER_LOOP_CONSTANT_SQ)
            .neg()
            == self.phi()
    }
}

impl ShortWeierstrassProjectivePoint<BLS12377TwistCurve> {
    /// Computes 𝜓(P) = 𝜁 ∘ 𝜋ₚ ∘ 𝜁⁻¹, where 𝜁 is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E, 𝜋ₚ is the p-power frobenius endomorphism
    /// and 𝜓 satisfies minimal equation 𝑋² + 𝑡𝑋 + 𝑞 = 𝑂.
    /// See <https://eprint.iacr.org/2022/352.pdf> Section 4.2 (7).
    /// ψ(P) = (ψ_x * conjugate(x), ψ_y * conjugate(y), conjugate(z))
    ///
    /// # Safety
    ///
    /// - This function assumes `self` is a valid point on the BLS12-377 **twist** curve.
    /// - The conjugation operation preserves validity.
    /// - `unwrap()` is used because `psi()` is defined to **always return a valid point**.
    fn psi(&self) -> Self {
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
}
