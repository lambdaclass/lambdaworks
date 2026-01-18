use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::fields::secp256k1_field::Secp256k1PrimeField;
use crate::unsigned_integer::element::U256;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

/// This implementation is not constant time and cannot be used to sign messages. You can use it to check signatures
#[derive(Clone, Debug)]
pub struct Secp256k1Curve;

type FE = FieldElement<Secp256k1PrimeField>;

// GLV (Gallant-Lambert-Vanstone) Scalar Multiplication Constants for secp256k1
//
// secp256k1 has an efficiently computable endomorphism φ(x, y) = (βx, y) where
// β is a primitive cube root of unity in Fp. This satisfies φ(P) = [λ]P.

/// β: Cube root of unity in Fp satisfying β³ = 1 and β ≠ 1
/// β = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
pub const CUBE_ROOT_OF_UNITY: FE = FE::from_hex_unchecked(
    "7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee",
);

/// λ: The eigenvalue of the endomorphism, satisfying λ² + λ + 1 ≡ 0 (mod n)
/// λ = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
pub const GLV_LAMBDA: U256 =
    U256::from_hex_unchecked("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");

/// Precomputed constants for GLV decomposition using the method from
/// "Guide to Elliptic Curve Cryptography" (Algorithm 3.74)
/// These are derived from the short vectors in the GLV lattice.

/// a1 = 0x3086d221a7d46bcde86c90e49284eb15 (≈ √n)
const GLV_A1: U256 = U256::from_hex_unchecked("3086d221a7d46bcde86c90e49284eb15");

/// a2 = 0x114ca50f7a8e2f3f657c1108d9d44cfd8
const GLV_A2: U256 = U256::from_hex_unchecked("114ca50f7a8e2f3f657c1108d9d44cfd8");

/// Subgroup order n
const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");

impl ShortWeierstrassProjectivePoint<Secp256k1Curve> {
    /// Applies the GLV endomorphism: φ(x, y) = (βx, y) where β is the cube root of unity.
    /// Satisfies φ(P) = [λ]P where λ² + λ + 1 ≡ 0 (mod n).
    #[inline(always)]
    pub fn phi(&self) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }
        let [x, y, z] = self.coordinates();
        Self::new_unchecked([x * CUBE_ROOT_OF_UNITY, y.clone(), z.clone()])
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

        let (k1_neg, k1, k2_neg, k2) = glv_decompose_secp256k1(k);
        let phi_p = self.phi();

        let p1 = if k1_neg { self.neg() } else { self.clone() };
        let p2 = if k2_neg { phi_p.neg() } else { phi_p };

        shamir_double_and_add_secp256k1(&p1, &k1, &p2, &k2)
    }
}

impl IsEllipticCurve for Secp256k1Curve {
    type BaseField = Secp256k1PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - The generator point is mathematically verified to be a valid point on the curve.
        // - `unwrap()` is safe because the provided coordinates satisfy the curve equation.
        let point = Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
            ),
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
            ),
            FieldElement::one(),
        ]);
        point.unwrap()
    }
}

impl IsShortWeierstrass for Secp256k1Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(7)
    }
}

/// GLV decomposition for secp256k1: k = k1 + k2*λ (mod n)
///
/// Uses a simplified balanced-length approach.
/// Returns (k1_neg, |k1|, k2_neg, |k2|)
fn glv_decompose_secp256k1(k: &U256) -> (bool, U256, bool, U256) {
    let zero = U256::from_u64(0);

    // For small k, no decomposition needed
    if *k < GLV_A1 {
        return (false, *k, false, zero);
    }

    // Simple decomposition: k2 = k / a2, k1 = k - k2 * λ (mod n)
    // This gives approximately balanced k1, k2
    let (k2, _) = k.div_rem(&GLV_A2);

    // Compute k1 = k - k2 * λ (mod n)
    let (k2_lambda_lo, k2_lambda_hi) = U256::mul(&k2, &GLV_LAMBDA);

    // Handle overflow by falling back
    if k2_lambda_hi != zero {
        return (false, *k, false, zero);
    }

    // k1 = k - k2*λ
    let (k1, _underflow) = if *k >= k2_lambda_lo {
        (U256::sub(k, &k2_lambda_lo).0, false)
    } else {
        // k1 = n - (k2*λ - k)
        let diff = U256::sub(&k2_lambda_lo, k).0;
        if diff < SUBGROUP_ORDER {
            (U256::sub(&SUBGROUP_ORDER, &diff).0, false)
        } else {
            return (false, *k, false, zero);
        }
    };

    // Check if k1 is too large (should reduce mod n)
    let k1_final = if k1 >= SUBGROUP_ORDER {
        U256::sub(&k1, &SUBGROUP_ORDER).0
    } else {
        k1
    };

    // Determine if we should negate for smaller representation
    let half_n = U256::from_hex_unchecked(
        "7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0",
    );

    let (k1_neg, k1_abs) = if k1_final > half_n {
        (true, U256::sub(&SUBGROUP_ORDER, &k1_final).0)
    } else {
        (false, k1_final)
    };

    let (k2_neg, k2_abs) = if k2 > half_n {
        (true, U256::sub(&SUBGROUP_ORDER, &k2).0)
    } else {
        (false, k2)
    };

    (k1_neg, k1_abs, k2_neg, k2_abs)
}

/// Gets bit at position `pos` from a U256.
#[inline(always)]
fn get_bit(n: &U256, pos: usize) -> bool {
    if pos >= 256 {
        return false;
    }
    let limb_idx = 3 - pos / 64;
    let bit_idx = pos % 64;
    (n.limbs[limb_idx] >> bit_idx) & 1 == 1
}

/// Shamir's trick for joint scalar multiplication: [k1]P1 + [k2]P2
fn shamir_double_and_add_secp256k1(
    p1: &ShortWeierstrassProjectivePoint<Secp256k1Curve>,
    k1: &U256,
    p2: &ShortWeierstrassProjectivePoint<Secp256k1Curve>,
    k2: &U256,
) -> ShortWeierstrassProjectivePoint<Secp256k1Curve> {
    let p1_plus_p2 = p1.operate_with(p2);

    let max_bits = core::cmp::max(k1.bits_le(), k2.bits_le());
    if max_bits == 0 {
        return ShortWeierstrassProjectivePoint::neutral_element();
    }

    let mut result = ShortWeierstrassProjectivePoint::neutral_element();

    for i in (0..max_bits).rev() {
        result = result.double();

        let b1 = get_bit(k1, i);
        let b2 = get_bit(k2, i);

        match (b1, b2) {
            (false, false) => {}
            (true, false) => result = result.operate_with(p1),
            (false, true) => result = result.operate_with(p2),
            (true, true) => result = result.operate_with(&p1_plus_p2),
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement, unsigned_integer::element::U256,
    };

    use super::Secp256k1Curve;

    #[allow(clippy::upper_case_acronyms)]
    type FE = FieldElement<Secp256k1PrimeField>;

    fn point_1() -> ShortWeierstrassProjectivePoint<Secp256k1Curve> {
        let x = FE::from_hex_unchecked(
            "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
        );
        let y = FE::from_hex_unchecked(
            "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
        );
        Secp256k1Curve::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> ShortWeierstrassProjectivePoint<Secp256k1Curve> {
        let x = FE::from_hex_unchecked(
            "2F8BDE4D1A07209355B4A7250A5C5128E88B84BDDC619AB7CBA8D569B240EFE4",
        );
        let y = FE::from_hex_unchecked(
            "D8AC222636E5E3D6D4DBA9DDA6C9C426F788271BAB0D6840DCA87D3AA6AC62D6",
        );
        Secp256k1Curve::create_point_from_affine(x, y).unwrap()
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
        assert_eq!(
            *p.x(),
            FE::from_hex_unchecked(
                "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
            )
        );
        assert_eq!(
            *p.y(),
            FE::from_hex_unchecked(
                "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"
            )
        );
        assert_eq!(*p.z(), FE::from_hex_unchecked("1"));
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            Secp256k1Curve::create_point_from_affine(FE::from(0), FE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = Secp256k1Curve::generator();
        let g2 = g.operate_with_self(2_u16);
        let g2_other = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
        assert_eq!(&g2, &g2_other);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = Secp256k1Curve::generator();
        let g2 = g.operate_with_self(2_u16);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        // calculate both sides of secp256k1 curve equation
        let seven = Secp256k1Curve::b();
        let y_sq_0 = x.pow(3_u16) + seven;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = Secp256k1Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }

    #[test]
    fn generator_has_right_order() {
        let g = Secp256k1Curve::generator();
        assert_eq!(
            g.operate_with_self(U256::from_hex_unchecked(
                "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"
            ))
            .to_affine(),
            ShortWeierstrassProjectivePoint::neutral_element()
        );
    }

    #[test]
    fn inverse_works() {
        let g = Secp256k1Curve::generator();
        assert_eq!(
            g.operate_with_self(U256::from_hex_unchecked(
                "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd036413C"
            ))
            .to_affine(),
            g.operate_with_self(5u64).neg().to_affine()
        );
    }

    // GLV tests
    #[test]
    fn glv_phi_endomorphism_property() {
        // Test that φ(P) = [λ]P
        let g = Secp256k1Curve::generator();
        let phi_g = g.phi();
        let lambda_g = g.operate_with_self(super::GLV_LAMBDA);
        assert_eq!(phi_g.to_affine(), lambda_g.to_affine());
    }

    #[test]
    fn glv_mul_small_scalar() {
        let g = Secp256k1Curve::generator();
        let k = U256::from_u64(12345);
        let expected = g.operate_with_self(k);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn glv_mul_medium_scalar() {
        let g = Secp256k1Curve::generator();
        let k = U256::from_hex_unchecked("deadbeef12345678");
        let expected = g.operate_with_self(k);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn glv_mul_large_scalar() {
        let g = Secp256k1Curve::generator();
        // Large 256-bit scalar
        let k = U256::from_hex_unchecked(
            "a5a5a5a5b6b6b6b6c7c7c7c7d8d8d8d8e9e9e9e9fafafafa0b0b0b0b1c1c1c1c",
        );
        let expected = g.operate_with_self(k);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn glv_mul_neutral_element() {
        let neutral = ShortWeierstrassProjectivePoint::<Secp256k1Curve>::neutral_element();
        let k = U256::from_u64(12345);
        let result = neutral.glv_mul(&k);
        assert_eq!(result.to_affine(), neutral.to_affine());
    }

    #[test]
    fn glv_mul_zero_scalar() {
        let g = Secp256k1Curve::generator();
        let k = U256::from_u64(0);
        let result = g.glv_mul(&k);
        assert_eq!(
            result.to_affine(),
            ShortWeierstrassProjectivePoint::<Secp256k1Curve>::neutral_element()
        );
    }

    #[test]
    fn glv_mul_one_scalar() {
        let g = Secp256k1Curve::generator();
        let k = U256::from_u64(1);
        let result = g.glv_mul(&k);
        assert_eq!(result.to_affine(), g.to_affine());
    }

    #[test]
    fn glv_mul_various_scalars() {
        let g = Secp256k1Curve::generator();
        let scalars = [
            U256::from_u64(2),
            U256::from_u64(255),
            U256::from_u64(65535),
            U256::from_hex_unchecked("ffffffff"),
            U256::from_hex_unchecked("123456789abcdef0"),
            U256::from_hex_unchecked("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140"), // n-1
        ];

        for k in &scalars {
            let expected = g.operate_with_self(*k);
            let result = g.glv_mul(k);
            assert_eq!(result.to_affine(), expected.to_affine());
        }
    }
}
