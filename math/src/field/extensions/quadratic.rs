use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::{IsField, IsSubFieldOf};
#[cfg(feature = "lambdaworks-serde-binary")]
use crate::traits::ByteConversion;
use core::fmt::Debug;
use core::marker::PhantomData;

/// A general quadratic extension field over `F`
/// with quadratic non residue `Q::residue()`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadraticExtensionField<F, T>
where
    F: IsField,
    T: HasQuadraticNonResidue<F>,
{
    field: PhantomData<F>,
    non_residue: PhantomData<T>,
}

pub type QuadraticExtensionFieldElement<F, T> = FieldElement<QuadraticExtensionField<F, T>>;

/// Trait to fix a quadratic non residue.
/// Used to construct a quadratic extension field by adding
/// a square root of `residue()`.
pub trait HasQuadraticNonResidue<F: IsField> {
    fn residue() -> FieldElement<F>;
}

impl<F, Q> FieldElement<QuadraticExtensionField<F, Q>>
where
    F: IsField,
    Q: Clone + Debug + HasQuadraticNonResidue<F>,
{
    pub fn conjugate(&self) -> Self {
        let [a, b] = self.value();
        Self::new([a.clone(), -b])
    }
}

#[cfg(feature = "lambdaworks-serde-binary")]
impl<F> ByteConversion for [FieldElement<F>; 2]
where
    F: IsField,
{
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    fn from_bytes_be(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }

    fn from_bytes_le(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}

impl<F, Q> IsField for QuadraticExtensionField<F, Q>
where
    F: IsField,
    Q: Clone + Debug + HasQuadraticNonResidue<F>,
{
    type BaseType = [FieldElement<F>; 2];

    /// Returns the component wise addition of `a` and `b`
    fn add(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        [&a[0] + &b[0], &a[1] + &b[1]]
    }

    /// Returns the multiplication of `a` and `b` using the following
    /// equation:
    /// (a0 + a1 * t) * (b0 + b1 * t) = a0 * b0 + a1 * b1 * Q::residue() + (a0 * b1 + a1 * b0) * t
    /// where `t.pow(2)` equals `Q::residue()`.
    fn mul(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        let q = Q::residue();
        let a0b0 = &a[0] * &b[0];
        let a1b1 = &a[1] * &b[1];
        let z = (&a[0] + &a[1]) * (&b[0] + &b[1]);
        [&a0b0 + &a1b1 * q, z - a0b0 - a1b1]
    }

    fn square(a: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        let [a0, a1] = a;
        let v0 = a0 * a1;
        let c0 = (a0 + a1) * (a0 + Q::residue() * a1) - &v0 - Q::residue() * &v0;
        let c1 = &v0 + &v0;
        [c0, c1]
    }

    /// Returns the component wise subtraction of `a` and `b`
    fn sub(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        [&a[0] - &b[0], &a[1] - &b[1]]
    }

    /// Returns the component wise negation of `a`
    fn neg(a: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        [-&a[0], -&a[1]]
    }

    /// Returns the multiplicative inverse of `a`
    /// This uses the equality `(a0 + a1 * t) * (a0 - a1 * t) = a0.pow(2) - a1.pow(2) * Q::residue()`
    fn inv(a: &[FieldElement<F>; 2]) -> Result<[FieldElement<F>; 2], FieldError> {
        let inv_norm = (a[0].pow(2_u64) - Q::residue() * a[1].pow(2_u64)).inv()?;
        Ok([&a[0] * &inv_norm, -&a[1] * inv_norm])
    }

    /// Returns the division of `a` and `b`
    fn div(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        <Self as IsField>::mul(a, &Self::inv(b).unwrap())
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> [FieldElement<F>; 2] {
        [FieldElement::zero(), FieldElement::zero()]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> [FieldElement<F>; 2] {
        [FieldElement::one(), FieldElement::zero()]
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType {
        [FieldElement::from(x), FieldElement::zero()]
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: [FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        x
    }
}

impl<F, Q> IsSubFieldOf<QuadraticExtensionField<F, Q>> for F
where
    F: IsField,
    Q: Clone + Debug + HasQuadraticNonResidue<F>,
{
    fn mul(
        a: &Self::BaseType,
        b: &<QuadraticExtensionField<F, Q> as IsField>::BaseType,
    ) -> <QuadraticExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::mul(a, b[0].value()));
        let c1 = FieldElement::from_raw(F::mul(a, b[1].value()));
        [c0, c1]
    }

    fn add(
        a: &Self::BaseType,
        b: &<QuadraticExtensionField<F, Q> as IsField>::BaseType,
    ) -> <QuadraticExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::add(a, b[0].value()));
        [c0, b[1].clone()]
    }

    fn div(
        a: &Self::BaseType,
        b: &<QuadraticExtensionField<F, Q> as IsField>::BaseType,
    ) -> <QuadraticExtensionField<F, Q> as IsField>::BaseType {
        let b_inv = <QuadraticExtensionField<F, Q> as IsField>::inv(b).unwrap();
        <Self as IsSubFieldOf<QuadraticExtensionField<F, Q>>>::mul(a, &b_inv)
    }

    fn sub(
        a: &Self::BaseType,
        b: &<QuadraticExtensionField<F, Q> as IsField>::BaseType,
    ) -> <QuadraticExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::sub(a, b[0].value()));
        let c1 = FieldElement::from_raw(F::neg(b[1].value()));
        [c0, c1]
    }

    fn embed(a: Self::BaseType) -> <QuadraticExtensionField<F, Q> as IsField>::BaseType {
        [FieldElement::from_raw(a), FieldElement::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <QuadraticExtensionField<F, Q> as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

impl<F: IsField, Q: Clone + Debug + HasQuadraticNonResidue<F>>
    FieldElement<QuadraticExtensionField<F, Q>>
{
}

#[cfg(test)]
mod tests {
    use crate::field::fields::u64_prime_field::{U64FieldElement, U64PrimeField};

    const ORDER_P: u64 = 59;

    use super::*;

    #[derive(Debug, Clone)]
    struct MyQuadraticNonResidue;
    impl HasQuadraticNonResidue<U64PrimeField<ORDER_P>> for MyQuadraticNonResidue {
        fn residue() -> FieldElement<U64PrimeField<ORDER_P>> {
            -FieldElement::one()
        }
    }

    type FE = U64FieldElement<ORDER_P>;
    type MyFieldExtensionBackend =
        QuadraticExtensionField<U64PrimeField<ORDER_P>, MyQuadraticNonResidue>;
    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<MyFieldExtensionBackend>;

    #[test]
    fn test_add_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let b = FEE::new([-FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(57), FE::new(11)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_add_2() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let b = FEE::new([-FE::new(4), FE::new(2)]);
        let expected_result = FEE::new([FE::new(8), FE::new(7)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_sub_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let b = FEE::new([-FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(2), FE::new(54)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_sub_2() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let b = FEE::new([-FE::new(4), FE::new(2)]);
        let expected_result = FEE::new([FE::new(16), FE::new(3)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_mul_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let b = FEE::new([-FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(35), FE::new(53)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_mul_2() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let b = FEE::new([-FE::new(4), FE::new(2)]);
        let expected_result = FEE::new([FE::new(1), FE::new(4)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_div_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let b = FEE::new([-FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(42), FE::new(19)]);
        assert_eq!(a / b, expected_result);
    }

    #[test]
    fn test_div_2() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let b = FEE::new([-FE::new(4), FE::new(2)]);
        let expected_result = FEE::new([FE::new(4), FE::new(45)]);
        assert_eq!(a / b, expected_result);
    }

    #[test]
    fn test_pow_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let b: u64 = 5;
        let expected_result = FEE::new([FE::new(0), FE::new(7)]);
        assert_eq!(a.pow(b), expected_result);
    }

    #[test]
    fn test_pow_2() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let b: u64 = 8;
        let expected_result = FEE::new([FE::new(52), FE::new(35)]);
        assert_eq!(a.pow(b), expected_result);
    }

    #[test]
    fn test_inv_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let expected_result = FEE::new([FE::new(0), FE::new(39)]);
        assert_eq!(a.inv().unwrap(), expected_result);
    }

    #[test]
    fn test_inv() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let expected_result = FEE::new([FE::new(28), FE::new(8)]);
        assert_eq!(a.inv().unwrap(), expected_result);
    }

    #[test]
    fn test_conjugate() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let expected_result = FEE::new([FE::new(12), -FE::new(5)]);
        assert_eq!(a.conjugate(), expected_result);
    }

    #[test]
    fn test_add_as_subfield_1() {
        let a = -FE::new(2);
        let b = FEE::new([FE::new(0), FE::new(3)]);
        let expected_result = FEE::new([FE::new(57), FE::new(3)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_add_as_subfield_2() {
        let a = -FE::new(4);
        let b = FEE::new([FE::new(12), FE::new(5)]);
        let expected_result = FEE::new([FE::new(8), FE::new(5)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_sub_as_subfield_1() {
        let a = FE::new(0);
        let b = FEE::new([-FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(2), FE::new(51)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_sub_a_subfield_2() {
        let a = FE::new(12);
        let b = FEE::new([-FE::new(4), -FE::new(2)]);
        let expected_result = FEE::new([FE::new(16), FE::new(2)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_mul_as_subfield_1() {
        let a = FE::new(2);
        let b = FEE::new([-FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(55), FE::new(16)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_mul_as_subfield_2() {
        let a = FE::new(12);
        let b = FEE::new([-FE::new(4), FE::new(2)]);
        let expected_result = FEE::new([FE::new(11), FE::new(24)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_div_as_subfield_1() {
        let a = FE::new(3);
        let b = FEE::new([-FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(19), FE::new(17)]);
        assert_eq!(a / b, expected_result);
    }

    #[test]
    fn test_div_as_subfield_2() {
        let a = FE::new(22);
        let b = FEE::new([FE::new(4), FE::new(2)]);
        let expected_result = FEE::new([FE::new(28), FE::new(45)]);
        assert_eq!(a / b, expected_result);
    }
}
