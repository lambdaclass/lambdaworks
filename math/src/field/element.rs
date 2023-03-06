use crate::field::traits::IsField;
use crate::unsigned_integer::traits::IsUnsignedInteger;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
};

/// A field element with operations algorithms defined in `F`
#[derive(Debug, Clone)]
pub struct FieldElement<F: IsField> {
    value: F::BaseType,
}

/// From overloading for field elements
impl<F> From<&F::BaseType> for FieldElement<F>
where
    F::BaseType: Clone,
    F: IsField,
{
    fn from(value: &F::BaseType) -> Self {
        Self {
            value: F::from_base_type(value.clone()),
        }
    }
}

/// From overloading for U64
impl<F> From<u64> for FieldElement<F>
where
    F: IsField,
{
    fn from(value: u64) -> Self {
        Self {
            value: F::from_u64(value),
        }
    }
}

/// Equality operator overloading for field elements
impl<F> PartialEq<FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    fn eq(&self, other: &FieldElement<F>) -> bool {
        F::eq(&self.value, &other.value)
    }
}

impl<F> Eq for FieldElement<F> where F: IsField {}

impl<F> Hash for FieldElement<F>
where
    F: IsField,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

/// Addition operator overloading for field elements
impl<F> Add<&FieldElement<F>> for &FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn add(self, rhs: &FieldElement<F>) -> Self::Output {
        Self::Output {
            value: F::add(&self.value, &rhs.value),
        }
    }
}

impl<F> Add<FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn add(self, rhs: FieldElement<F>) -> Self::Output {
        &self + &rhs
    }
}

impl<F> Add<&FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn add(self, rhs: &FieldElement<F>) -> Self::Output {
        &self + rhs
    }
}

impl<F> Add<FieldElement<F>> for &FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn add(self, rhs: FieldElement<F>) -> Self::Output {
        self + &rhs
    }
}

/// AddAssign operator overloading for field elements
impl<F> AddAssign<FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    fn add_assign(&mut self, rhs: FieldElement<F>) {
        self.value = F::add(&self.value, &rhs.value);
    }
}

/// Sum operator for field elements
impl<F> Sum<FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |augend, addend| augend + addend)
    }
}

/// Subtraction operator overloading for field elements*/
impl<F> Sub<&FieldElement<F>> for &FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn sub(self, rhs: &FieldElement<F>) -> Self::Output {
        Self::Output {
            value: F::sub(&self.value, &rhs.value),
        }
    }
}

impl<F> Sub<FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn sub(self, rhs: FieldElement<F>) -> Self::Output {
        &self - &rhs
    }
}

impl<F> Sub<&FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn sub(self, rhs: &FieldElement<F>) -> Self::Output {
        &self - rhs
    }
}

impl<F> Sub<FieldElement<F>> for &FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn sub(self, rhs: FieldElement<F>) -> Self::Output {
        self - &rhs
    }
}

/// Multiplication operator overloading for field elements*/
impl<F> Mul<&FieldElement<F>> for &FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn mul(self, rhs: &FieldElement<F>) -> Self::Output {
        Self::Output {
            value: F::mul(&self.value, &rhs.value),
        }
    }
}

impl<F> Mul<FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn mul(self, rhs: FieldElement<F>) -> Self::Output {
        &self * &rhs
    }
}

impl<F> Mul<&FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn mul(self, rhs: &FieldElement<F>) -> Self::Output {
        &self * rhs
    }
}

impl<F> Mul<FieldElement<F>> for &FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn mul(self, rhs: FieldElement<F>) -> Self::Output {
        self * &rhs
    }
}

/// Division operator overloading for field elements*/
impl<F> Div<&FieldElement<F>> for &FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn div(self, rhs: &FieldElement<F>) -> Self::Output {
        Self::Output {
            value: F::div(&self.value, &rhs.value),
        }
    }
}

impl<F> Div<FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn div(self, rhs: FieldElement<F>) -> Self::Output {
        &self / &rhs
    }
}

impl<F> Div<&FieldElement<F>> for FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn div(self, rhs: &FieldElement<F>) -> Self::Output {
        &self / rhs
    }
}

impl<F> Div<FieldElement<F>> for &FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn div(self, rhs: FieldElement<F>) -> Self::Output {
        self / &rhs
    }
}

/// Negation operator overloading for field elements*/
impl<F> Neg for &FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn neg(self) -> Self::Output {
        Self::Output {
            value: F::neg(&self.value),
        }
    }
}

impl<F> Neg for FieldElement<F>
where
    F: IsField,
{
    type Output = FieldElement<F>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

/// FieldElement general implementation
/// Most of this is delegated to the trait `F` that
/// implements the field operations.
impl<F> FieldElement<F>
where
    F: IsField,
{
    /// Creates a field element from `value`
    pub fn new(value: F::BaseType) -> Self {
        Self {
            value: F::from_base_type(value),
        }
    }

    /// Returns the underlying `value`
    pub fn value(&self) -> &F::BaseType {
        &self.value
    }

    // Returns the representative of the value stored
    pub fn representative(&self) -> F::BaseType {
        F::representative(self.value.clone())
    }

    /// Returns the multiplicative inverse of `self`
    pub fn inv(&self) -> Self {
        Self {
            value: F::inv(&self.value),
        }
    }

    /// Returns `self` raised to the power of `exponent`
    pub fn pow<T>(&self, exponent: T) -> Self
    where
        T: IsUnsignedInteger,
    {
        Self {
            value: F::pow(&self.value, exponent),
        }
    }

    /// Returns the multiplicative neutral element of the field.
    pub fn one() -> Self {
        Self { value: F::one() }
    }

    /// Returns the additive neutral element of the field.
    pub fn zero() -> Self {
        Self { value: F::zero() }
    }
}
