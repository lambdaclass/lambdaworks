use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::IsField;
use core::fmt::Debug;
use core::marker::PhantomData;

/// A general cubic extension field over `F`
/// with cubic non residue `Q::residue()`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubicExtensionField<T> {
    phantom: PhantomData<T>,
}

pub type CubicExtensionFieldElement<T> = FieldElement<CubicExtensionField<T>>;

/// Trait to fix a cubic non residue.
/// Used to construct a cubic extension field by adding
/// a square root of `residue()`.
pub trait HasCubicNonResidue {
    type BaseField: IsField;

    /// This function must return an element that is not a cube in Fp,
    /// that is, a cubic non-residue.
    fn residue() -> FieldElement<Self::BaseField>;
}

impl<Q> IsField for CubicExtensionField<Q>
where
    Q: Clone + Debug + HasCubicNonResidue,
{
    type BaseType = [FieldElement<Q::BaseField>; 3];

    /// Returns the component wise addition of `a` and `b`
    fn add(
        a: &[FieldElement<Q::BaseField>; 3],
        b: &[FieldElement<Q::BaseField>; 3],
    ) -> [FieldElement<Q::BaseField>; 3] {
        [&a[0] + &b[0], &a[1] + &b[1], &a[2] + &b[2]]
    }

    /// Returns the multiplication of `a` and `b` using the following
    /// equation:
    /// (a0 + a1 * t) * (b0 + b1 * t) = a0 * b0 + a1 * b1 * Q::residue() + (a0 * b1 + a1 * b0) * t
    /// where `t.pow(2)` equals `Q::residue()`.
    fn mul(
        a: &[FieldElement<Q::BaseField>; 3],
        b: &[FieldElement<Q::BaseField>; 3],
    ) -> [FieldElement<Q::BaseField>; 3] {
        let v0 = &a[0] * &b[0];
        let v1 = &a[1] * &b[1];
        let v2 = &a[2] * &b[2];

        [
            &v0 + Q::residue() * ((&a[1] + &a[2]) * (&b[1] + &b[2]) - &v1 - &v2),
            (&a[0] + &a[1]) * (&b[0] + &b[1]) - &v0 - &v1 + Q::residue() * &v2,
            (&a[0] + &a[2]) * (&b[0] + &b[2]) - v0 + v1 - v2,
        ]
    }

    /// Returns the component wise subtraction of `a` and `b`
    fn sub(
        a: &[FieldElement<Q::BaseField>; 3],
        b: &[FieldElement<Q::BaseField>; 3],
    ) -> [FieldElement<Q::BaseField>; 3] {
        [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2]]
    }

    /// Returns the component wise negation of `a`
    fn neg(a: &[FieldElement<Q::BaseField>; 3]) -> [FieldElement<Q::BaseField>; 3] {
        [-&a[0], -&a[1], -&a[2]]
    }

    /// Returns the multiplicative inverse of `a`
    fn inv(
        a: &[FieldElement<Q::BaseField>; 3],
    ) -> Result<[FieldElement<Q::BaseField>; 3], FieldError> {
        let three = FieldElement::from(3_u64);

        let d = a[0].pow(3_u64)
            + a[1].pow(3_u64) * Q::residue()
            + a[2].pow(3_u64) * Q::residue().pow(2_u64)
            - three * &a[0] * &a[1] * &a[2] * Q::residue();
        let inv = d.inv()?;
        Ok([
            (a[0].pow(2_u64) - &a[1] * &a[2] * Q::residue()) * &inv,
            (-&a[0] * &a[1] + a[2].pow(2_u64) * Q::residue()) * &inv,
            (-&a[0] * &a[2] + a[1].pow(2_u64)) * &inv,
        ])
    }

    /// Returns the division of `a` and `b`
    fn div(
        a: &[FieldElement<Q::BaseField>; 3],
        b: &[FieldElement<Q::BaseField>; 3],
    ) -> [FieldElement<Q::BaseField>; 3] {
        Self::mul(a, &Self::inv(b).unwrap())
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &[FieldElement<Q::BaseField>; 3], b: &[FieldElement<Q::BaseField>; 3]) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> [FieldElement<Q::BaseField>; 3] {
        [
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> [FieldElement<Q::BaseField>; 3] {
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
    fn from_base_type(x: [FieldElement<Q::BaseField>; 3]) -> [FieldElement<Q::BaseField>; 3] {
        x
    }
}

#[cfg(test)]
mod tests {
    use crate::field::fields::u64_prime_field::{U64FieldElement, U64PrimeField};

    const ORDER_P: u64 = 13;

    use super::*;

    #[derive(Debug, Clone)]
    struct MyCubicNonResidue;
    impl HasCubicNonResidue for MyCubicNonResidue {
        type BaseField = U64PrimeField<ORDER_P>;

        fn residue() -> FieldElement<U64PrimeField<ORDER_P>> {
            -FieldElement::from(11)
        }
    }

    type FE = U64FieldElement<ORDER_P>;
    type MyFieldExtensionBackend = CubicExtensionField<MyCubicNonResidue>;
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
        assert_eq!(a / b, expected_result);
    }

    #[test]
    fn test_div_2() {
        let a = FEE::new([FE::new(12), FE::new(5), FE::new(4)]);
        let b = FEE::new([-FE::new(4), FE::new(2), FE::new(2)]);
        let expected_result = FEE::new([FE::new(3), FE::new(8), FE::new(11)]);
        assert_eq!(a / b, expected_result);
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
}
