use std::fmt::Debug;
use std::marker::PhantomData;
use crate::field::element::FieldElement;
use crate::field::traits::HasFieldOperations;

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
    type BaseField: HasFieldOperations;

    fn residue() -> FieldElement<Self::BaseField>;
}


impl<Q> HasFieldOperations for CubicExtensionField<Q>
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
        [
            &a[0] * &b[0] + &a[1] * &b[2] * Q::residue() + &a[2] * &b[1] * Q::residue(),
            &a[0] * &b[1] + &a[1] * &b[0] + &a[2] * &b[2] * Q::residue(),
            &a[0] * &b[2] + &a[2] * &b[0] + &a[1] * &b[1],
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
    /// This uses the equality `(a0 + a1 * t) * (a0 - a1 * t) = a0.pow(2) - a1.pow(2) * Q::residue()`
    fn inv(a: &[FieldElement<Q::BaseField>; 3]) -> [FieldElement<Q::BaseField>; 3] {
        todo!()
    }

    /// Returns the division of `a` and `b`
    fn div(
        a: &[FieldElement<Q::BaseField>; 3],
        b: &[FieldElement<Q::BaseField>; 3],
    ) -> [FieldElement<Q::BaseField>; 3] {
        Self::mul(a, &Self::inv(b))
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &[FieldElement<Q::BaseField>; 3], b: &[FieldElement<Q::BaseField>; 3]) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> [FieldElement<Q::BaseField>; 3] {
        [FieldElement::zero(), FieldElement::zero(), FieldElement::zero()]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> [FieldElement<Q::BaseField>; 3] {
        [FieldElement::one(), FieldElement::zero(), FieldElement::zero()]
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType {
        [FieldElement::from(x), FieldElement::zero(), FieldElement::zero()]
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

    const ORDER_P: u64 = 59;

    use super::*;

    #[derive(Debug, Clone)]
    struct MyCubicNonResidue;
    impl HasCubicNonResidue for MyCubicNonResidue {
        type BaseField = U64PrimeField<ORDER_P>;

        fn residue() -> FieldElement<U64PrimeField<ORDER_P>> {
            -FieldElement::one()
        }
    }

    type FE = U64FieldElement<ORDER_P>;
    type MyFieldExtensionBackend = CubicExtensionField<MyCubicNonResidue>;
    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<MyFieldExtensionBackend>;
}
