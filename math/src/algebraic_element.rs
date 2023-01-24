use std::{
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
};

#[derive(Debug, Clone)]
pub struct FieldElement<F: Field> {
    value: F::BaseType
}

pub trait Field {
    type BaseType: Clone;

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    fn pow(a: &Self::BaseType, exponent: u128) -> Self::BaseType;

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    fn neg(a: &Self::BaseType) -> Self::BaseType;

    fn inv(a: &Self::BaseType) -> Self::BaseType;

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool;

    fn zero() -> Self::BaseType;

    fn one() -> Self::BaseType;

    fn representative(a: &Self::BaseType) -> Self::BaseType;
}

/* From overloading for Algebraic Elements */
impl<F> From<&F::BaseType> for FieldElement<F>
where
    F::BaseType: Clone,
    F: Field,
{
    fn from(value: &F::BaseType) -> Self {
        Self {
            value: value.clone()
        }
    }
}


/* Equality operator overloading for Algebraic Elements */
impl<F> PartialEq<FieldElement<F>> for FieldElement<F>
where
    F: Field,
{
    fn eq(&self, other: &FieldElement<F>) -> bool {
        F::eq(&self.value, &other.value)
    }
}

impl<F> Eq for FieldElement<F> where F: Field {}

/* Addition operator overloading for Algebraic Elements */
impl<F> Add<&FieldElement<F>> for &FieldElement<F>
where
    F: Field,
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
    F: Field,
{
    type Output = FieldElement<F>;

    fn add(self, rhs: FieldElement<F>) -> Self::Output {
        &self + &rhs
    }
}

impl<F> Add<&FieldElement<F>> for FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn add(self, rhs: &FieldElement<F>) -> Self::Output {
        &self + rhs
    }
}

impl<F> Add<FieldElement<F>> for &FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn add(self, rhs: FieldElement<F>) -> Self::Output {
        self + &rhs
    }
}

/* AddAssign operator overloading for Algebraic Elements */
impl<F> AddAssign<FieldElement<F>>
    for FieldElement<F>
where
    F: Field,
{
    fn add_assign(&mut self, rhs: FieldElement<F>) {
        self.value = F::add(&self.value, &rhs.value);
    }
}

/* Subtraction operator overloading for Algebraic Elements*/
impl<F> Sub<&FieldElement<F>>
    for &FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn sub(self, rhs: &FieldElement<F>) -> Self::Output {
        Self::Output {
            value: F::sub(&self.value, &rhs.value),
        }
    }
}

impl<F> Sub<FieldElement<F>>
    for FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn sub(self, rhs: FieldElement<F>) -> Self::Output {
        &self - &rhs
    }
}

impl<F> Sub<&FieldElement<F>>
    for FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn sub(self, rhs: &FieldElement<F>) -> Self::Output {
        &self - rhs
    }
}

impl<F> Sub<FieldElement<F>>
    for &FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn sub(self, rhs: FieldElement<F>) -> Self::Output {
        self - &rhs
    }
}

/* Multiplication operator overloading for Algebraic Elements*/
impl<F> Mul<&FieldElement<F>>
    for &FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn mul(self, rhs: &FieldElement<F>) -> Self::Output {
        Self::Output {
            value: F::mul(&self.value, &rhs.value),
        }
    }
}

impl<F> Mul<FieldElement<F>>
    for FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn mul(self, rhs: FieldElement<F>) -> Self::Output {
        &self * &rhs
    }
}

impl<F> Mul<&FieldElement<F>>
    for FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn mul(self, rhs: &FieldElement<F>) -> Self::Output {
        &self * rhs
    }
}

impl<F> Mul<FieldElement<F>>
    for &FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn mul(self, rhs: FieldElement<F>) -> Self::Output {
        self * &rhs
    }
}

/* Division operator overloading for Algebraic Elements*/
impl<F> Div<&FieldElement<F>>
    for &FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn div(self, rhs: &FieldElement<F>) -> Self::Output {
        Self::Output {
            value: F::div(&self.value, &rhs.value),
        }
    }
}

impl<F> Div<FieldElement<F>>
    for FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn div(self, rhs: FieldElement<F>) -> Self::Output {
        &self / &rhs
    }
}

impl<F> Div<&FieldElement<F>>
    for FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn div(self, rhs: &FieldElement<F>) -> Self::Output {
        &self / rhs
    }
}

impl<F> Div<FieldElement<F>>
    for &FieldElement<F>
where
    F: Field,
{
    type Output = FieldElement<F>;

    fn div(self, rhs: FieldElement<F>) -> Self::Output {
        self / &rhs
    }
}

/* Negation operator overloading for Algebraic Elements*/
impl<F> Neg for &FieldElement<F>
where
    F: Field,
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
    F: Field,
{
    type Output = FieldElement<F>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

/* FieldElement general implementation */
impl<F> FieldElement<F> 
where
    F: Field
{
    pub fn new(value: F::BaseType) -> Self {
        Self {
            value
        }
    }
    
    pub fn value(&self) -> &F::BaseType {
        &self.value
    }

    pub fn inv(&self) -> Self {
        Self {
            value: F::inv(&self.value)
        }
    }

    pub fn pow(&self, exponent: u128) -> Self {
        Self {
            value: F::pow(&self.value, exponent),
        }
    }

    pub fn one() -> Self {
        Self {
            value: F::one(),
        }
    }

    pub fn zero() -> Self {
        Self {
            value: F::zero()
        }
    }
}
