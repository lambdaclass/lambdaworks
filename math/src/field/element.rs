use crate::field::traits::HasFieldOperations;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Debug, Clone)]
pub struct FieldElement<F: HasFieldOperations> {
    value: F::BaseType,
}

/* From overloading for Algebraic Elements */
impl<F> From<&F::BaseType> for FieldElement<F>
where
    F::BaseType: Clone,
    F: HasFieldOperations,
{
    fn from(value: &F::BaseType) -> Self {
        Self {
            value: value.clone(),
        }
    }
}

/* From overloading for U64 */
impl<F> From<u64> for FieldElement<F>
where
    F: HasFieldOperations,
{
    fn from(value: u64) -> Self {
        Self {
            value: F::from_u64(value),
        }
    }
}

/* Equality operator overloading for Algebraic Elements */
impl<F> PartialEq<FieldElement<F>> for FieldElement<F>
where
    F: HasFieldOperations,
{
    fn eq(&self, other: &FieldElement<F>) -> bool {
        F::eq(&self.value, &other.value)
    }
}

impl<F> Eq for FieldElement<F> where F: HasFieldOperations {}

/* Addition operator overloading for Algebraic Elements */
impl<F> Add<&FieldElement<F>> for &FieldElement<F>
where
    F: HasFieldOperations,
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
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn add(self, rhs: FieldElement<F>) -> Self::Output {
        &self + &rhs
    }
}

impl<F> Add<&FieldElement<F>> for FieldElement<F>
where
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn add(self, rhs: &FieldElement<F>) -> Self::Output {
        &self + rhs
    }
}

impl<F> Add<FieldElement<F>> for &FieldElement<F>
where
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn add(self, rhs: FieldElement<F>) -> Self::Output {
        self + &rhs
    }
}

/* AddAssign operator overloading for Algebraic Elements */
impl<F> AddAssign<FieldElement<F>> for FieldElement<F>
where
    F: HasFieldOperations,
{
    fn add_assign(&mut self, rhs: FieldElement<F>) {
        self.value = F::add(&self.value, &rhs.value);
    }
}

/* Subtraction operator overloading for Algebraic Elements*/
impl<F> Sub<&FieldElement<F>> for &FieldElement<F>
where
    F: HasFieldOperations,
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
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn sub(self, rhs: FieldElement<F>) -> Self::Output {
        &self - &rhs
    }
}

impl<F> Sub<&FieldElement<F>> for FieldElement<F>
where
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn sub(self, rhs: &FieldElement<F>) -> Self::Output {
        &self - rhs
    }
}

impl<F> Sub<FieldElement<F>> for &FieldElement<F>
where
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn sub(self, rhs: FieldElement<F>) -> Self::Output {
        self - &rhs
    }
}

/* Multiplication operator overloading for Algebraic Elements*/
impl<F> Mul<&FieldElement<F>> for &FieldElement<F>
where
    F: HasFieldOperations,
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
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn mul(self, rhs: FieldElement<F>) -> Self::Output {
        &self * &rhs
    }
}

impl<F> Mul<&FieldElement<F>> for FieldElement<F>
where
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn mul(self, rhs: &FieldElement<F>) -> Self::Output {
        &self * rhs
    }
}

impl<F> Mul<FieldElement<F>> for &FieldElement<F>
where
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn mul(self, rhs: FieldElement<F>) -> Self::Output {
        self * &rhs
    }
}

/* Division operator overloading for Algebraic Elements*/
impl<F> Div<&FieldElement<F>> for &FieldElement<F>
where
    F: HasFieldOperations,
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
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn div(self, rhs: FieldElement<F>) -> Self::Output {
        &self / &rhs
    }
}

impl<F> Div<&FieldElement<F>> for FieldElement<F>
where
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn div(self, rhs: &FieldElement<F>) -> Self::Output {
        &self / rhs
    }
}

impl<F> Div<FieldElement<F>> for &FieldElement<F>
where
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn div(self, rhs: FieldElement<F>) -> Self::Output {
        self / &rhs
    }
}

/* Negation operator overloading for Algebraic Elements*/
impl<F> Neg for &FieldElement<F>
where
    F: HasFieldOperations,
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
    F: HasFieldOperations,
{
    type Output = FieldElement<F>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

/* FieldElement general implementation */
impl<F> FieldElement<F>
where
    F: HasFieldOperations,
{
    pub fn new(value: F::BaseType) -> Self {
        Self { value }
    }

    pub fn value(&self) -> &F::BaseType {
        &self.value
    }

    pub fn inv(&self) -> Self {
        Self {
            value: F::inv(&self.value),
        }
    }

    pub fn pow(&self, exponent: u128) -> Self {
        Self {
            value: F::pow(&self.value, exponent),
        }
    }

    pub fn one() -> Self {
        Self { value: F::one() }
    }

    pub fn zero() -> Self {
        Self { value: F::zero() }
    }
}
