use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::{IsField, IsSubFieldOf};
use crate::traits::ByteConversion;
use core::fmt::Debug;
use core::marker::PhantomData;

/// A general cubic extension field over `F`
/// with cubic non residue `Q::residue()`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubicExtensionField<F, T> {
    field: PhantomData<F>,
    non_residue: PhantomData<T>,
}

pub type CubicExtensionFieldElement<F, T> = FieldElement<CubicExtensionField<F, T>>;

/// Trait to fix a cubic non residue.
/// Used to construct a cubic extension field by adding
/// a cube root of `residue()`.
pub trait HasCubicNonResidue<F: IsField> {
    /// This function must return an element that is not a cube in Fp,
    /// that is, a cubic non-residue.
    fn residue() -> FieldElement<F>;
}

impl<F> ByteConversion for [FieldElement<F>; 3]
where
    F: IsField,
{
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        let mut bytes = self[0].value().to_bytes_be();
        bytes.extend(self[1].value().to_bytes_be());
        bytes.extend(self[2].value().to_bytes_be());
        bytes
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        let mut bytes = self[0].value().to_bytes_le();
        bytes.extend(self[1].value().to_bytes_le());
        bytes.extend(self[2].value().to_bytes_le());
        bytes
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        let element_size = bytes.len() / 3;
        let x0 = FieldElement::new(F::BaseType::from_bytes_be(&bytes[..element_size])?);
        let x1 = FieldElement::new(F::BaseType::from_bytes_be(
            &bytes[element_size..element_size * 2],
        )?);
        let x2 = FieldElement::new(F::BaseType::from_bytes_be(&bytes[element_size * 2..])?);
        Ok([x0, x1, x2])
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        let element_size = bytes.len() / 3;
        let x0 = FieldElement::new(F::BaseType::from_bytes_le(&bytes[..element_size])?);
        let x1 = FieldElement::new(F::BaseType::from_bytes_le(
            &bytes[element_size..element_size * 2],
        )?);
        let x2 = FieldElement::new(F::BaseType::from_bytes_le(&bytes[element_size * 2..])?);
        Ok([x0, x1, x2])
    }
}

impl<F, Q> IsField for CubicExtensionField<F, Q>
where
    F: IsField,
    Q: Clone + Debug + HasCubicNonResidue<F>,
{
    type BaseType = [FieldElement<F>; 3];

    /// Returns the component wise addition of `a` and `b`
    #[inline]
    fn add(a: &[FieldElement<F>; 3], b: &[FieldElement<F>; 3]) -> [FieldElement<F>; 3] {
        [&a[0] + &b[0], &a[1] + &b[1], &a[2] + &b[2]]
    }

    /// Returns the multiplication of `a` and `b` using Karatsuba-style formula.
    /// (a0 + a1*v + a2*v²) * (b0 + b1*v + b2*v²) where v³ = Q::residue()
    #[inline]
    fn mul(a: &[FieldElement<F>; 3], b: &[FieldElement<F>; 3]) -> [FieldElement<F>; 3] {
        let v0 = &a[0] * &b[0];
        let v1 = &a[1] * &b[1];
        let v2 = &a[2] * &b[2];
        let q = Q::residue();

        [
            &v0 + &q * ((&a[1] + &a[2]) * (&b[1] + &b[2]) - &v1 - &v2),
            (&a[0] + &a[1]) * (&b[0] + &b[1]) - &v0 - &v1 + &q * &v2,
            (&a[0] + &a[2]) * (&b[0] + &b[2]) - v0 + v1 - v2,
        ]
    }

    /// Optimized squaring using Chung-Hasan SQR2 formula.
    /// (a0 + a1*v + a2*v²)² where v³ = Q::residue()
    /// This requires 3 base field squares and 3 base field multiplications
    /// instead of 6 multiplications from generic mul(a, a).
    #[inline]
    fn square(a: &[FieldElement<F>; 3]) -> [FieldElement<F>; 3] {
        let s0 = a[0].square();
        let s1 = a[1].square();
        let s2 = a[2].square();
        let ab = &a[0] * &a[1];
        let bc = &a[1] * &a[2];
        let ac = &a[0] * &a[2];
        let q = Q::residue();

        // c0 = s0 + q * 2bc
        let two_bc = &bc + &bc;
        let c0 = &s0 + &q * &two_bc;

        // c1 = 2ab + q * s2
        let two_ab = &ab + &ab;
        let c1 = &two_ab + &q * &s2;

        // c2 = 2ac + s1
        let two_ac = &ac + &ac;
        let c2 = two_ac + s1;

        [c0, c1, c2]
    }

    /// Returns the component wise subtraction of `a` and `b`
    #[inline]
    fn sub(a: &[FieldElement<F>; 3], b: &[FieldElement<F>; 3]) -> [FieldElement<F>; 3] {
        [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2]]
    }

    /// Returns the component wise negation of `a`
    #[inline]
    fn neg(a: &[FieldElement<F>; 3]) -> [FieldElement<F>; 3] {
        [-&a[0], -&a[1], -&a[2]]
    }

    /// Returns the multiplicative inverse of `a`
    #[inline]
    fn inv(a: &[FieldElement<F>; 3]) -> Result<[FieldElement<F>; 3], FieldError> {
        let three = FieldElement::<F>::from(3_u64);
        // Use square() instead of pow(2) and cube for pow(3)
        let a0_sq = a[0].square();
        let a1_sq = a[1].square();
        let a2_sq = a[2].square();
        let a0_cube = &a0_sq * &a[0];
        let a1_cube = &a1_sq * &a[1];
        let a2_cube = &a2_sq * &a[2];
        let residue = Q::residue();
        let residue_sq = residue.square();
        let d = a0_cube + a1_cube * &residue + a2_cube * &residue_sq
            - three * &a[0] * &a[1] * &a[2] * &residue;
        let inv = d.inv()?;
        Ok([
            (a0_sq - &a[1] * &a[2] * &residue) * &inv,
            (-&a[0] * &a[1] + &a2_sq * &residue) * &inv,
            (-&a[0] * &a[2] + a1_sq) * &inv,
        ])
    }

    /// Returns the division of `a` and `b`
    fn div(
        a: &[FieldElement<F>; 3],
        b: &[FieldElement<F>; 3],
    ) -> Result<[FieldElement<F>; 3], FieldError> {
        let b_inv = &Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, b_inv))
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &[FieldElement<F>; 3], b: &[FieldElement<F>; 3]) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> [FieldElement<F>; 3] {
        [
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> [FieldElement<F>; 3] {
        [
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType {
        [
            FieldElement::from(x),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: [FieldElement<F>; 3]) -> [FieldElement<F>; 3] {
        x
    }
}

impl<F, Q> IsSubFieldOf<CubicExtensionField<F, Q>> for F
where
    F: IsField,
    Q: Clone + Debug + HasCubicNonResidue<F>,
{
    fn mul(
        a: &Self::BaseType,
        b: &<CubicExtensionField<F, Q> as IsField>::BaseType,
    ) -> <CubicExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::mul(a, b[0].value()));
        let c1 = FieldElement::from_raw(F::mul(a, b[1].value()));
        let c2 = FieldElement::from_raw(F::mul(a, b[2].value()));
        [c0, c1, c2]
    }

    fn add(
        a: &Self::BaseType,
        b: &<CubicExtensionField<F, Q> as IsField>::BaseType,
    ) -> <CubicExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::add(a, b[0].value()));
        [c0, b[1].clone(), b[2].clone()]
    }

    fn div(
        a: &Self::BaseType,
        b: &<CubicExtensionField<F, Q> as IsField>::BaseType,
    ) -> Result<<CubicExtensionField<F, Q> as IsField>::BaseType, FieldError> {
        let b_inv = <CubicExtensionField<F, Q> as IsField>::inv(b)?;
        Ok(<Self as IsSubFieldOf<CubicExtensionField<F, Q>>>::mul(
            a, &b_inv,
        ))
    }

    fn sub(
        a: &Self::BaseType,
        b: &<CubicExtensionField<F, Q> as IsField>::BaseType,
    ) -> <CubicExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::sub(a, b[0].value()));
        let c1 = FieldElement::from_raw(F::neg(b[1].value()));
        let c2 = FieldElement::from_raw(F::neg(b[2].value()));
        [c0, c1, c2]
    }

    fn embed(a: Self::BaseType) -> <CubicExtensionField<F, Q> as IsField>::BaseType {
        [
            FieldElement::from_raw(a),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <CubicExtensionField<F, Q> as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::field::fields::u64_prime_field::{U64FieldElement, U64PrimeField};

    const ORDER_P: u64 = 13;

    use super::*;

    #[derive(Debug, Clone)]
    struct MyCubicNonResidue;
    impl HasCubicNonResidue<U64PrimeField<ORDER_P>> for MyCubicNonResidue {
        fn residue() -> FieldElement<U64PrimeField<ORDER_P>> {
            -FieldElement::from(11)
        }
    }

    type FE = U64FieldElement<ORDER_P>;
    type MyFieldExtensionBackend = CubicExtensionField<U64PrimeField<ORDER_P>, MyCubicNonResidue>;
    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<MyFieldExtensionBackend>;

    #[test]
    fn test_add_1() {
        let a = FEE::new([FE::new(0), FE::new(3), FE::new(5)]);
        let b = FEE::new([-FE::new(2), FE::new(8), FE::new(10)]);
        let expected_result = FEE::new([FE::new(11), FE::new(11), FE::new(15)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_add_2() {
        let a = FEE::new([FE::new(12), FE::new(5), FE::new(3)]);
        let b = FEE::new([-FE::new(4), FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(8), FE::new(7), FE::new(11)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_sub_1() {
        let a = FEE::new([FE::new(0), FE::new(3), FE::new(3)]);
        let b = FEE::new([-FE::new(2), FE::new(8), FE::new(2)]);
        let expected_result = FEE::new([FE::new(2), FE::new(8), FE::new(1)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_sub_2() {
        let a = FEE::new([FE::new(12), FE::new(5), FE::new(3)]);
        let b = FEE::new([-FE::new(4), FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(16), FE::new(3), FE::new(8)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_mul_1() {
        let a = FEE::new([FE::new(0), FE::new(3), FE::new(5)]);
        let b = FEE::new([-FE::new(2), FE::new(8), FE::new(6)]);
        let expected_result = FEE::new([FE::new(12), FE::new(2), FE::new(1)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_mul_2() {
        let a = FEE::new([FE::new(12), FE::new(5), FE::new(11)]);
        let b = FEE::new([-FE::new(4), FE::new(2), FE::new(15)]);
        let expected_result = FEE::new([FE::new(3), FE::new(9), FE::new(3)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_div_1() {
        let a = FEE::new([FE::new(0), FE::new(3), FE::new(2)]);
        let b = FEE::new([-FE::new(2), FE::new(8), FE::new(5)]);
        let expected_result = FEE::new([FE::new(12), FE::new(6), FE::new(1)]);
        assert_eq!((a / b).unwrap(), expected_result);
    }

    #[test]
    fn test_div_2() {
        let a = FEE::new([FE::new(12), FE::new(5), FE::new(4)]);
        let b = FEE::new([-FE::new(4), FE::new(2), FE::new(2)]);
        let expected_result = FEE::new([FE::new(3), FE::new(8), FE::new(11)]);
        assert_eq!((a / b).unwrap(), expected_result);
    }

    #[test]
    fn test_pow_1() {
        let a = FEE::new([FE::new(0), FE::new(3), FE::new(3)]);
        let b: u64 = 5;
        let expected_result = FEE::new([FE::new(7), FE::new(3), FE::new(1)]);
        assert_eq!(a.pow(b), expected_result);
    }

    #[test]
    fn test_pow_2() {
        let a = FEE::new([FE::new(12), FE::new(5), FE::new(3)]);
        let b: u64 = 8;
        let expected_result = FEE::new([FE::new(5), FE::new(5), FE::new(12)]);
        assert_eq!(a.pow(b), expected_result);
    }

    #[test]
    fn test_inv() {
        let a = FEE::new([FE::new(12), FE::new(5), FE::new(3)]);
        let expected_result = FEE::new([FE::new(2), FE::new(2), FE::new(3)]);
        assert_eq!(a.inv().unwrap(), expected_result);
    }

    #[test]
    fn test_inv_1() {
        let a = FEE::new([FE::new(1), FE::new(0), FE::new(1)]);
        let expected_result = FEE::new([FE::new(8), FE::new(3), FE::new(5)]);
        assert_eq!(a.inv().unwrap(), expected_result);
    }

    #[test]
    fn test_add_as_subfield_1() {
        let a = FE::new(5);
        let b = FEE::new([-FE::new(2), FE::new(8), FE::new(10)]);
        let expected_result = FEE::new([FE::new(3), FE::new(8), FE::new(10)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_add_as_subfield_2() {
        let a = FE::new(12);
        let b = FEE::new([-FE::new(4), FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(8), FE::new(2), FE::new(8)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_sub_as_subfield_1() {
        let a = FE::new(3);
        let b = FEE::new([-FE::new(2), FE::new(8), FE::new(2)]);
        let expected_result = FEE::new([FE::new(5), FE::new(5), FE::new(11)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_sub_as_subfield_2() {
        let a = FE::new(12);
        let b = FEE::new([-FE::new(4), FE::new(2), FE::new(3)]);
        let expected_result = FEE::new([FE::new(3), FE::new(11), FE::new(10)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_mul_as_subfield_1() {
        let a = FE::new(5);
        let b = FEE::new([-FE::new(2), FE::new(8), FE::new(6)]);
        let expected_result = FEE::new([FE::new(3), FE::new(1), FE::new(4)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_mul_as_subfield_2() {
        let a = FE::new(11);
        let b = FEE::new([-FE::new(4), FE::new(2), FE::new(15)]);
        let expected_result = FEE::new([FE::new(8), FE::new(9), FE::new(9)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_div_as_subfield_1() {
        let a = FE::new(2);
        let b = FEE::new([-FE::new(2), FE::new(8), FE::new(5)]);
        let expected_result = FEE::new([FE::new(8), FE::new(4), FE::new(10)]);
        assert_eq!((a / b).unwrap(), expected_result);
    }

    #[test]
    fn test_div_as_subfield_2() {
        let a = FE::new(4);
        let b = FEE::new([-FE::new(4), FE::new(2), FE::new(2)]);
        let expected_result = FEE::new([FE::new(3), FE::new(6), FE::new(11)]);
        assert_eq!((a / b).unwrap(), expected_result);
    }
}
