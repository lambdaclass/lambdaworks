use crate::field::{
    element::FieldElement,
    errors::FieldError,
    fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField,
    traits::{IsField, IsSubFieldOf},
};
use crate::unsigned_integer::element::U64;

// =====================================================
// QUADRATIC EXTENSION (Fp2)
// =====================================================
// The non-residue was taken from [Plonky3](https://github.com/Plonky3/Plonky3/blob/main/goldilocks/src/extension.rs)
// The quadratic extension is constructed using x^2 - 7,
// where 7 is a quadratic non-residue in the Goldilocks field.
// This means Fp2 = Fp[x] / (x^2 - 7)
// Elements are represented as a0 + a1*w where w^2 = 7

type FpE = FieldElement<U64GoldilocksPrimeField>;

/// The quadratic non-residue W = 7 for the quadratic extension.
/// Fp2 is constructed as Fp[x] / (x^2 - 7)
/// We use the Montgomery form: 7 * R mod p where R = 2^64 and p = 2^64 - 2^32 + 1
/// 7 * 2^64 mod p = 7 * (p + 2^32 - 1) mod p = 7 * (2^32 - 1) mod p = 30064771065
pub const QUADRATIC_NON_RESIDUE: FpE = FpE::const_from_raw(U64::from_u64(30064771065));

/// Degree 2 extension field of Goldilocks
#[derive(Copy, Clone, Debug)]
pub struct Degree2GoldilocksExtensionField;

impl IsField for Degree2GoldilocksExtensionField {
    type BaseType = [FpE; 2];

    /// Returns the component-wise addition of `a` and `b`
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    /// Returns the multiplication of `a` and `b`:
    /// (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + W*a1*b1) + (a0*b1 + a1*b0)*w
    /// where w^2 = W = 7
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = a[0] * b[0];
        let a1b1 = a[1] * b[1];
        let z = (a[0] + a[1]) * (b[0] + b[1]);
        [a0b0 + QUADRATIC_NON_RESIDUE * a1b1, z - a0b0 - a1b1]
    }

    /// Returns the square of `a`:
    /// (a0 + a1*w)^2 = (a0^2 + W*a1^2) + 2*a0*a1*w
    fn square(a: &Self::BaseType) -> Self::BaseType {
        let a0_sq = a[0].square();
        let a1_sq = a[1].square();
        let a0a1 = a[0] * a[1];
        [a0_sq + QUADRATIC_NON_RESIDUE * a1_sq, a0a1.double()]
    }

    /// Returns the component-wise subtraction of `a` and `b`
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1]]
    }

    /// Returns the component-wise negation of `a`
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-a[0], -a[1]]
    }

    /// Returns the multiplicative inverse of `a`:
    /// (a0 + a1*w)^-1 = (a0 - a1*w) / (a0^2 - W*a1^2)
    /// The norm is a0^2 - W*a1^2. This is never zero for non-zero elements
    /// because W=7 is a quadratic non-residue (no x exists where x^2 = 7).
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let norm = a[0].square() - QUADRATIC_NON_RESIDUE * a[1].square();
        let norm_inv = norm.inv()?;
        Ok([a[0] * norm_inv, -a[1] * norm_inv])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, &b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero()]
    }

    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        [a[0].double(), a[1].double()]
    }
}

impl IsSubFieldOf<Degree2GoldilocksExtensionField> for U64GoldilocksPrimeField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::mul(a, b[1].value()));
        [c0, c1]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::add(a, b[0].value()));
        [c0, b[1]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> Result<<Degree2GoldilocksExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree2GoldilocksExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Degree2GoldilocksExtensionField>>::mul(a, &b_inv))
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::neg(b[1].value()));
        [c0, c1]
    }

    fn embed(a: Self::BaseType) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        [FpE::from_raw(a), FpE::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

/// Field element type for the quadratic extension of Goldilocks
pub type Fp2E = FieldElement<Degree2GoldilocksExtensionField>;

impl Fp2E {
    /// Returns the conjugate of self: conjugate(a0 + a1*w) = a0 - a1*w
    pub fn conjugate(&self) -> Self {
        Self::new([self.value()[0], -self.value()[1]])
    }
}

// =====================================================
// CUBIC EXTENSION (Fp3)
// =====================================================
// The cubic extension is constructed using x^3 - 2,
// where 2 is a cubic non-residue in the Goldilocks field.
// This means Fp3 = Fp[x] / (x^3 - 2)
// Elements are represented as a0 + a1*w + a2*w^2 where w^3 = 2
//
// Since the non-residue is 2, we use .double() for efficiency
// instead of multiplying by the non-residue constant.

/// Degree 3 extension field of Goldilocks
#[derive(Copy, Clone, Debug)]
pub struct Degree3GoldilocksExtensionField;

impl IsField for Degree3GoldilocksExtensionField {
    type BaseType = [FpE; 3];

    /// Returns the component-wise addition of `a` and `b`
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    /// Returns the multiplication of `a` and `b`:
    /// (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2) mod (w^3 - 2)
    /// Using Karatsuba-like optimization with W=2 replaced by .double()
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let v0 = a[0] * b[0];
        let v1 = a[1] * b[1];
        let v2 = a[2] * b[2];

        // c0 = v0 + 2 * ((a1 + a2)(b1 + b2) - v1 - v2)
        // c1 = (a0 + a1)(b0 + b1) - v0 - v1 + 2 * v2
        // c2 = (a0 + a2)(b0 + b2) - v0 + v1 - v2
        [
            v0 + ((a[1] + a[2]) * (b[1] + b[2]) - v1 - v2).double(),
            (a[0] + a[1]) * (b[0] + b[1]) - v0 - v1 + v2.double(),
            (a[0] + a[2]) * (b[0] + b[2]) - v0 + v1 - v2,
        ]
    }

    /// Returns the square of `a`
    fn square(a: &Self::BaseType) -> Self::BaseType {
        let s0 = a[0].square();
        let s1 = a[1].square();
        let s2 = a[2].square();
        let a01 = a[0] * a[1];
        let a02 = a[0] * a[2];
        let a12 = a[1] * a[2];

        // c0 = s0 + 2 * 2 * a12 = s0 + 4 * a12
        // c1 = 2 * a01 + 2 * s2
        // c2 = 2 * a02 + s1
        [
            s0 + a12.double().double(),
            a01.double() + s2.double(),
            a02.double() + s1,
        ]
    }

    /// Returns the component-wise subtraction of `a` and `b`
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    /// Returns the component-wise negation of `a`
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-a[0], -a[1], -a[2]]
    }

    /// Returns the multiplicative inverse of `a`
    /// Using the formula for cubic extension inverse with W=2
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let a0_sq = a[0].square();
        let a1_sq = a[1].square();
        let a2_sq = a[2].square();

        // W = 2, W^2 = 4
        // Compute the norm: N = a0^3 + 2*a1^3 + 4*a2^3 - 6*a0*a1*a2
        let a0_cubed = a0_sq * a[0];
        let a1_cubed = a1_sq * a[1];
        let a2_cubed = a2_sq * a[2];
        let a0a1a2 = a[0] * a[1] * a[2];

        // N = a0^3 + 2*a1^3 + 4*a2^3 - 6*a0*a1*a2
        let norm = a0_cubed + a1_cubed.double() + a2_cubed.double().double()
            - (a0a1a2.double() + a0a1a2).double();

        let norm_inv = norm.inv()?;

        // Compute inverse components:
        // inv[0] = (a0^2 - 2*a1*a2) / N
        // inv[1] = (2*a2^2 - a0*a1) / N
        // inv[2] = (a1^2 - a0*a2) / N
        Ok([
            (a0_sq - (a[1] * a[2]).double()) * norm_inv,
            (a2_sq.double() - a[0] * a[1]) * norm_inv,
            (a1_sq - a[0] * a[2]) * norm_inv,
        ])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, &b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
    }

    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero(), FpE::zero()]
    }

    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero(), FpE::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero(), FpE::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        [a[0].double(), a[1].double(), a[2].double()]
    }
}

impl IsSubFieldOf<Degree3GoldilocksExtensionField> for U64GoldilocksPrimeField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::mul(a, b[1].value()));
        let c2 = FpE::from_raw(<Self as IsField>::mul(a, b[2].value()));
        [c0, c1, c2]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::add(a, b[0].value()));
        [c0, b[1], b[2]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> Result<<Degree3GoldilocksExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree3GoldilocksExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Degree3GoldilocksExtensionField>>::mul(a, &b_inv))
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::neg(b[1].value()));
        let c2 = FpE::from_raw(<Self as IsField>::neg(b[2].value()));
        [c0, c1, c2]
    }

    fn embed(a: Self::BaseType) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        [FpE::from_raw(a), FpE::zero(), FpE::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

/// Field element type for the cubic extension of Goldilocks
pub type Fp3E = FieldElement<Degree3GoldilocksExtensionField>;

#[cfg(test)]
mod tests {
    use super::*;

    // =====================================================
    // QUADRATIC EXTENSION TESTS
    // =====================================================

    #[test]
    fn test_fp2_add() {
        let a = Fp2E::new([FpE::from(0u64), FpE::from(3u64)]);
        let b = Fp2E::new([FpE::from(2u64), FpE::from(8u64)]);
        let expected = Fp2E::new([FpE::from(2u64), FpE::from(11u64)]);
        assert_eq!(a + b, expected);
    }

    #[test]
    fn test_fp2_sub() {
        let a = Fp2E::new([FpE::from(10u64), FpE::from(5u64)]);
        let b = Fp2E::new([FpE::from(3u64), FpE::from(2u64)]);
        let expected = Fp2E::new([FpE::from(7u64), FpE::from(3u64)]);
        assert_eq!(a - b, expected);
    }

    #[test]
    fn test_fp2_mul() {
        // (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + a1*b1*7) + (a0*b1 + a1*b0)*w
        let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp2E::new([FpE::from(4u64), FpE::from(5u64)]);
        // c0 = 2*4 + 3*5*7 = 8 + 105 = 113
        // c1 = 2*5 + 3*4 = 10 + 12 = 22
        let expected = Fp2E::new([FpE::from(113u64), FpE::from(22u64)]);
        assert_eq!(a * b, expected);
    }

    #[test]
    fn test_fp2_mul_by_one() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let one = Fp2E::one();
        assert_eq!(a * one, a);
    }

    #[test]
    fn test_fp2_mul_by_zero() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let zero = Fp2E::zero();
        assert_eq!(a * zero, zero);
    }

    #[test]
    fn test_fp2_inv() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let a_inv = a.inv().unwrap();
        assert_eq!(a * a_inv, Fp2E::one());
    }

    #[test]
    fn test_fp2_inv_one() {
        let one = Fp2E::one();
        assert_eq!(one.inv().unwrap(), one);
    }

    #[test]
    fn test_fp2_div() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let b = Fp2E::new([FpE::from(4u64), FpE::from(2u64)]);
        let result = (a / b).unwrap();
        assert_eq!(result * b, a);
    }

    #[test]
    fn test_fp2_pow() {
        let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
        let a_squared = a * a;
        let a_cubed = a_squared * a;
        assert_eq!(a.pow(2u64), a_squared);
        assert_eq!(a.pow(3u64), a_cubed);
    }

    #[test]
    fn test_fp2_conjugate() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let expected = Fp2E::new([FpE::from(12u64), -FpE::from(5u64)]);
        assert_eq!(a.conjugate(), expected);
    }

    #[test]
    fn test_fp2_neg() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let neg_a = -a;
        assert_eq!(a + neg_a, Fp2E::zero());
    }

    #[test]
    fn test_fp2_square_equals_mul() {
        let a = Fp2E::new([FpE::from(7u64), FpE::from(11u64)]);
        assert_eq!(a.square(), a * a);
    }

    #[test]
    fn test_fp2_double() {
        let a = Fp2E::new([FpE::from(7u64), FpE::from(11u64)]);
        assert_eq!(a.double(), a + a);
    }

    // =====================================================
    // CUBIC EXTENSION TESTS
    // =====================================================

    #[test]
    fn test_fp3_add() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
        let expected = Fp3E::new([FpE::from(5u64), FpE::from(7u64), FpE::from(9u64)]);
        assert_eq!(a + b, expected);
    }

    #[test]
    fn test_fp3_sub() {
        let a = Fp3E::new([FpE::from(10u64), FpE::from(8u64), FpE::from(6u64)]);
        let b = Fp3E::new([FpE::from(3u64), FpE::from(2u64), FpE::from(1u64)]);
        let expected = Fp3E::new([FpE::from(7u64), FpE::from(6u64), FpE::from(5u64)]);
        assert_eq!(a - b, expected);
    }

    #[test]
    fn test_fp3_mul_by_one() {
        let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
        let one = Fp3E::one();
        assert_eq!(a * one, a);
    }

    #[test]
    fn test_fp3_mul_by_zero() {
        let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
        let zero = Fp3E::zero();
        assert_eq!(a * zero, zero);
    }

    #[test]
    fn test_fp3_mul_commutativity() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_fp3_inv() {
        let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
        let a_inv = a.inv().unwrap();
        assert_eq!(a * a_inv, Fp3E::one());
    }

    #[test]
    fn test_fp3_inv_one() {
        let one = Fp3E::one();
        assert_eq!(one.inv().unwrap(), one);
    }

    #[test]
    fn test_fp3_div() {
        let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(2u64), FpE::from(3u64)]);
        let result = (a / b).unwrap();
        assert_eq!(result * b, a);
    }

    #[test]
    fn test_fp3_pow() {
        let a = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
        let a_squared = a * a;
        let a_cubed = a_squared * a;
        assert_eq!(a.pow(2u64), a_squared);
        assert_eq!(a.pow(3u64), a_cubed);
    }

    #[test]
    fn test_fp3_neg() {
        let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
        let neg_a = -a;
        assert_eq!(a + neg_a, Fp3E::zero());
    }

    #[test]
    fn test_fp3_square_equals_mul() {
        let a = Fp3E::new([FpE::from(7u64), FpE::from(11u64), FpE::from(13u64)]);
        assert_eq!(a.square(), a * a);
    }

    #[test]
    fn test_fp3_double() {
        let a = Fp3E::new([FpE::from(7u64), FpE::from(11u64), FpE::from(13u64)]);
        assert_eq!(a.double(), a + a);
    }

    // =====================================================
    // EMBEDDING TESTS (Base field into extension)
    // =====================================================

    #[test]
    fn test_fp2_from_base() {
        let base = FpE::from(42u64);
        let ext = Fp2E::from(42u64);
        assert_eq!(ext.value()[0], base);
        assert_eq!(ext.value()[1], FpE::zero());
    }

    #[test]
    fn test_fp3_from_base() {
        let base = FpE::from(42u64);
        let ext = Fp3E::from(42u64);
        assert_eq!(ext.value()[0], base);
        assert_eq!(ext.value()[1], FpE::zero());
        assert_eq!(ext.value()[2], FpE::zero());
    }

    #[test]
    fn test_fp2_base_mul() {
        let a = FpE::from(5u64);
        let b = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
        let result = a * b;
        let expected = Fp2E::new([FpE::from(10u64), FpE::from(15u64)]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_fp3_base_mul() {
        let a = FpE::from(5u64);
        let b = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
        let result = a * b;
        let expected = Fp3E::new([FpE::from(10u64), FpE::from(15u64), FpE::from(20u64)]);
        assert_eq!(result, expected);
    }

    // =====================================================
    // ASSOCIATIVITY AND DISTRIBUTIVITY TESTS
    // =====================================================

    #[test]
    fn test_fp2_mul_associativity() {
        let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp2E::new([FpE::from(4u64), FpE::from(5u64)]);
        let c = Fp2E::new([FpE::from(6u64), FpE::from(7u64)]);
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_fp3_mul_associativity() {
        let a = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
        let b = Fp3E::new([FpE::from(5u64), FpE::from(6u64), FpE::from(7u64)]);
        let c = Fp3E::new([FpE::from(8u64), FpE::from(9u64), FpE::from(10u64)]);
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_fp2_distributivity() {
        let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp2E::new([FpE::from(4u64), FpE::from(5u64)]);
        let c = Fp2E::new([FpE::from(6u64), FpE::from(7u64)]);
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn test_fp3_distributivity() {
        let a = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
        let b = Fp3E::new([FpE::from(5u64), FpE::from(6u64), FpE::from(7u64)]);
        let c = Fp3E::new([FpE::from(8u64), FpE::from(9u64), FpE::from(10u64)]);
        assert_eq!(a * (b + c), a * b + a * c);
    }

    // =====================================================
    // LARGE VALUE TESTS
    // =====================================================

    #[test]
    fn test_fp2_large_values() {
        let a = Fp2E::new([
            FpE::from(18446744069414584300u64),
            FpE::from(12345678901234567u64),
        ]);
        let b = Fp2E::new([
            FpE::from(9876543210987654u64),
            FpE::from(11111111111111111u64),
        ]);

        let a_inv = a.inv().unwrap();
        assert_eq!(a * a_inv, Fp2E::one());

        let result = (a / b).unwrap();
        assert_eq!(result * b, a);
    }

    #[test]
    fn test_fp3_large_values() {
        let a = Fp3E::new([
            FpE::from(18446744069414584300u64),
            FpE::from(12345678901234567u64),
            FpE::from(98765432109876543u64),
        ]);
        let b = Fp3E::new([
            FpE::from(9876543210987654u64),
            FpE::from(11111111111111111u64),
            FpE::from(22222222222222222u64),
        ]);

        let a_inv = a.inv().unwrap();
        assert_eq!(a * a_inv, Fp3E::one());

        let result = (a / b).unwrap();
        assert_eq!(result * b, a);
    }
}
