use crate::unsigned_integer::IsUnsignedInteger;
use std::fmt::Debug;

/// Trait to add field behaviour to a struct.
pub trait HasFieldOperations: Debug + Clone {
    /// The underlying base type for representing elements from the field.
    type BaseType: Clone + Debug;

    /// Returns the sum of `a` and `b`.
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    /// Returns the multiplication of `a` and `b`.
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    /// Returns`a` raised to the power of `exponent`.
    fn pow<T>(a: &Self::BaseType, mut exponent: T) -> Self::BaseType
    where
        T: IsUnsignedInteger,
    {
        let mut result = Self::one();
        let mut base = a.clone();

        while exponent > T::from(0) {
            if exponent & T::from(1) == T::from(1) {
                result = Self::mul(&result, &base);
            }
            exponent = exponent >> 1;
            base = Self::mul(&base, &base);
        }
        result
    }

    /// Returns the subtraction of `a` and `b`.
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    /// Returns the additive inverse of `a`.
    fn neg(a: &Self::BaseType) -> Self::BaseType;

    /// Returns the multiplicative inverse of `a`.
    fn inv(a: &Self::BaseType) -> Self::BaseType;

    /// Returns the division of `a` and `b`.
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    /// Returns a boolean indicating whether `a` and `b` are equal or not.
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool;

    /// Returns the additive neutral element.
    fn zero() -> Self::BaseType;

    /// Returns the multiplicative neutral element.
    fn one() -> Self::BaseType;

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType;

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    fn from_base_type(x: Self::BaseType) -> Self::BaseType;
}
