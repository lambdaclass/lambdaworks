use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
};

#[derive(Debug, Clone)]
pub struct FieldElement<Set, OperationsBackend> {
    value: Set,
    phantom: PhantomData<OperationsBackend>,
}

pub trait Field<Set> {
    fn add(a: &Set, b: &Set) -> Set;

    fn mul(a: &Set, b: &Set) -> Set;

    fn pow(a: &Set, exponent: u128) -> Set;

    fn sub(a: &Set, b: &Set) -> Set;

    fn neg(a: &Set) -> Set;

    fn inv(a: &Set) -> Set;

    fn div(a: &Set, b: &Set) -> Set;

    fn eq(a: &Set, b: &Set) -> bool;

    fn zero() -> Set;

    fn one() -> Set;

    fn representative(a: &Set) -> Set;
}

/* From overloading for Algebraic Elements */
impl<Set, OperationsBackend> From<&Set> for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    fn from(value: &Set) -> Self {
        Self {
            value: OperationsBackend::representative(value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> From<Set> for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    fn from(value: Set) -> Self {
        Self::from(&value)
    }
}

/* Equality operator overloading for Algebraic Elements */
impl<Set, OperationsBackend> PartialEq<FieldElement<Set, OperationsBackend>>
    for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    fn eq(&self, other: &FieldElement<Set, OperationsBackend>) -> bool {
        OperationsBackend::eq(&self.value, &other.value)
    }
}

impl<Set, OperationsBackend> Eq for FieldElement<Set, OperationsBackend> where
    OperationsBackend: Field<Set>
{
}

/* Addition operator overloading for Algebraic Elements */
impl<Set, OperationsBackend> Add<&FieldElement<Set, OperationsBackend>>
    for &FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn add(self, rhs: &FieldElement<Set, OperationsBackend>) -> Self::Output {
        Self::Output {
            value: OperationsBackend::add(&self.value, &rhs.value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> Add<FieldElement<Set, OperationsBackend>>
    for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn add(self, rhs: FieldElement<Set, OperationsBackend>) -> Self::Output {
        &self + &rhs
    }
}

impl<Set, OperationsBackend> Add<&FieldElement<Set, OperationsBackend>>
    for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn add(self, rhs: &FieldElement<Set, OperationsBackend>) -> Self::Output {
        &self + rhs
    }
}

impl<Set, OperationsBackend> Add<FieldElement<Set, OperationsBackend>>
    for &FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn add(self, rhs: FieldElement<Set, OperationsBackend>) -> Self::Output {
        self + &rhs
    }
}

/* AddAssign operator overloading for Algebraic Elements */
impl<Set, OperationsBackend> AddAssign<FieldElement<Set, OperationsBackend>>
    for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    fn add_assign(&mut self, rhs: FieldElement<Set, OperationsBackend>) {
        self.value = OperationsBackend::add(&self.value, &rhs.value);
    }
}

/* Subtraction operator overloading for Algebraic Elements*/
impl<Set, OperationsBackend> Sub<&FieldElement<Set, OperationsBackend>>
    for &FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn sub(self, rhs: &FieldElement<Set, OperationsBackend>) -> Self::Output {
        Self::Output {
            value: OperationsBackend::sub(&self.value, &rhs.value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> Sub<FieldElement<Set, OperationsBackend>>
    for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn sub(self, rhs: FieldElement<Set, OperationsBackend>) -> Self::Output {
        &self - &rhs
    }
}

impl<Set, OperationsBackend> Sub<&FieldElement<Set, OperationsBackend>>
    for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn sub(self, rhs: &FieldElement<Set, OperationsBackend>) -> Self::Output {
        &self - rhs
    }
}

impl<Set, OperationsBackend> Sub<FieldElement<Set, OperationsBackend>>
    for &FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn sub(self, rhs: FieldElement<Set, OperationsBackend>) -> Self::Output {
        self - &rhs
    }
}

/* Multiplication operator overloading for Algebraic Elements*/
impl<Set, OperationsBackend> Mul<&FieldElement<Set, OperationsBackend>>
    for &FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn mul(self, rhs: &FieldElement<Set, OperationsBackend>) -> Self::Output {
        Self::Output {
            value: OperationsBackend::mul(&self.value, &rhs.value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> Mul<FieldElement<Set, OperationsBackend>>
    for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn mul(self, rhs: FieldElement<Set, OperationsBackend>) -> Self::Output {
        &self * &rhs
    }
}

impl<Set, OperationsBackend> Mul<&FieldElement<Set, OperationsBackend>>
    for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn mul(self, rhs: &FieldElement<Set, OperationsBackend>) -> Self::Output {
        &self * rhs
    }
}

impl<Set, OperationsBackend> Mul<FieldElement<Set, OperationsBackend>>
    for &FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn mul(self, rhs: FieldElement<Set, OperationsBackend>) -> Self::Output {
        self * &rhs
    }
}

/* Division operator overloading for Algebraic Elements*/
impl<Set, OperationsBackend> Div<&FieldElement<Set, OperationsBackend>>
    for &FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn div(self, rhs: &FieldElement<Set, OperationsBackend>) -> Self::Output {
        Self::Output {
            value: OperationsBackend::div(&self.value, &rhs.value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> Div<FieldElement<Set, OperationsBackend>>
    for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn div(self, rhs: FieldElement<Set, OperationsBackend>) -> Self::Output {
        &self / &rhs
    }
}

impl<Set, OperationsBackend> Div<&FieldElement<Set, OperationsBackend>>
    for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn div(self, rhs: &FieldElement<Set, OperationsBackend>) -> Self::Output {
        &self / rhs
    }
}

impl<Set, OperationsBackend> Div<FieldElement<Set, OperationsBackend>>
    for &FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn div(self, rhs: FieldElement<Set, OperationsBackend>) -> Self::Output {
        self / &rhs
    }
}

/* Negation operator overloading for Algebraic Elements*/
impl<Set, OperationsBackend> Neg for &FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn neg(self) -> Self::Output {
        Self::Output {
            value: OperationsBackend::neg(&self.value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> Neg for FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    type Output = FieldElement<Set, OperationsBackend>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

/* FieldElement general implementation */
impl<Set, OperationsBackend> FieldElement<Set, OperationsBackend> {
    pub fn value(&self) -> &Set {
        &self.value
    }
}

/* Inv implementation for Algebraic Elements */
impl<Set, OperationsBackend> FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    pub fn inv(&self) -> Self {
        Self {
            value: OperationsBackend::inv(&self.value),
            phantom: PhantomData,
        }
    }
}

/* Pow implementation for Algebraic Elements */
impl<Set, OperationsBackend> FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    pub fn pow(&self, exponent: u128) -> Self {
        Self {
            value: OperationsBackend::pow(&self.value, exponent),
            phantom: PhantomData,
        }
    }
}

/* One implementation for Algebraic Elements */
impl<Set, OperationsBackend> FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    pub fn one() -> Self {
        Self {
            value: OperationsBackend::one(),
            phantom: PhantomData,
        }
    }
}

/* Zero implementation for Algebraic Elements */
impl<Set, OperationsBackend> FieldElement<Set, OperationsBackend>
where
    OperationsBackend: Field<Set>,
{
    pub fn zero() -> Self {
        Self {
            value: OperationsBackend::zero(),
            phantom: PhantomData,
        }
    }
}
