use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
};

#[derive(Debug, Clone)]
pub struct AlgebraicElement<Set, OperationsBackend> {
    value: Set,
    phantom: PhantomData<OperationsBackend>,
}

pub trait AdditionLaw<Set> {
    fn add(a: &Set, b: &Set) -> Set;
}

pub trait MultiplicationLaw<Set> {
    fn mul(a: &Set, b: &Set) -> Set;
}

pub trait PowerLaw<Set> {
    fn pow(a: &Set, exponent: u128) -> Set;
}

pub trait SubtractionLaw<Set> {
    fn sub(a: &Set, b: &Set) -> Set;
}

pub trait NegationLaw<Set> {
    fn neg(a: &Set) -> Set;
}

pub trait InversionLaw<Set> {
    fn inv(a: &Set) -> Set;
}

pub trait DivisionLaw<Set> {
    fn div(a: &Set, b: &Set) -> Set;
}

pub trait EqualityLaw<Set> {
    fn eq(a: &Set, b: &Set) -> bool;
}

pub trait Zero<Set> {
    fn zero() -> Set;
}

pub trait One<Set> {
    fn one() -> Set;
}

pub trait Representative<Set> {
    fn representative(a: &Set) -> Set;
}

/* From overloading for Algebraic Elements */
impl<Set, OperationsBackend> From<&Set> for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: Representative<Set>,
{
    fn from(value: &Set) -> Self {
        Self {
            value: OperationsBackend::representative(value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> From<Set> for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: Representative<Set>,
{
    fn from(value: Set) -> Self {
        Self::from(&value)
    }
}

/* Equality operator overloading for Algebraic Elements */
impl<Set, OperationsBackend> PartialEq<AlgebraicElement<Set, OperationsBackend>>
    for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: EqualityLaw<Set>,
{
    fn eq(&self, other: &AlgebraicElement<Set, OperationsBackend>) -> bool {
        OperationsBackend::eq(&self.value, &other.value)
    }
}

impl<Set, OperationsBackend> Eq for AlgebraicElement<Set, OperationsBackend> where
    OperationsBackend: EqualityLaw<Set>
{
}

/* Addition operator overloading for Algebraic Elements */
impl<Set, OperationsBackend> Add<&AlgebraicElement<Set, OperationsBackend>>
    for &AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: AdditionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn add(self, rhs: &AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        Self::Output {
            value: OperationsBackend::add(&self.value, &rhs.value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> Add<AlgebraicElement<Set, OperationsBackend>>
    for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: AdditionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn add(self, rhs: AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        &self + &rhs
    }
}

impl<Set, OperationsBackend> Add<&AlgebraicElement<Set, OperationsBackend>>
    for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: AdditionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn add(self, rhs: &AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        &self + rhs
    }
}

impl<Set, OperationsBackend> Add<AlgebraicElement<Set, OperationsBackend>>
    for &AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: AdditionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn add(self, rhs: AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        self + &rhs
    }
}

/* AddAssign operator overloading for Algebraic Elements */
impl<Set, OperationsBackend> AddAssign<AlgebraicElement<Set, OperationsBackend>>
    for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: AdditionLaw<Set>,
{
    fn add_assign(&mut self, rhs: AlgebraicElement<Set, OperationsBackend>) {
        self.value = OperationsBackend::add(&self.value, &rhs.value);
    }
}

/* Subtraction operator overloading for Algebraic Elements*/
impl<Set, OperationsBackend> Sub<&AlgebraicElement<Set, OperationsBackend>>
    for &AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: SubtractionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn sub(self, rhs: &AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        Self::Output {
            value: OperationsBackend::sub(&self.value, &rhs.value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> Sub<AlgebraicElement<Set, OperationsBackend>>
    for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: SubtractionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn sub(self, rhs: AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        &self - &rhs
    }
}

impl<Set, OperationsBackend> Sub<&AlgebraicElement<Set, OperationsBackend>>
    for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: SubtractionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn sub(self, rhs: &AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        &self - rhs
    }
}

impl<Set, OperationsBackend> Sub<AlgebraicElement<Set, OperationsBackend>>
    for &AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: SubtractionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn sub(self, rhs: AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        self - &rhs
    }
}

/* Multiplication operator overloading for Algebraic Elements*/
impl<Set, OperationsBackend> Mul<&AlgebraicElement<Set, OperationsBackend>>
    for &AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: MultiplicationLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn mul(self, rhs: &AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        Self::Output {
            value: OperationsBackend::mul(&self.value, &rhs.value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> Mul<AlgebraicElement<Set, OperationsBackend>>
    for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: MultiplicationLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn mul(self, rhs: AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        &self * &rhs
    }
}

impl<Set, OperationsBackend> Mul<&AlgebraicElement<Set, OperationsBackend>>
    for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: MultiplicationLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn mul(self, rhs: &AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        &self * rhs
    }
}

impl<Set, OperationsBackend> Mul<AlgebraicElement<Set, OperationsBackend>>
    for &AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: MultiplicationLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn mul(self, rhs: AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        self * &rhs
    }
}

/* Division operator overloading for Algebraic Elements*/
impl<Set, OperationsBackend> Div<&AlgebraicElement<Set, OperationsBackend>>
    for &AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: DivisionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn div(self, rhs: &AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        Self::Output {
            value: OperationsBackend::div(&self.value, &rhs.value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> Div<AlgebraicElement<Set, OperationsBackend>>
    for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: DivisionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn div(self, rhs: AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        &self / &rhs
    }
}

impl<Set, OperationsBackend> Div<&AlgebraicElement<Set, OperationsBackend>>
    for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: DivisionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn div(self, rhs: &AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        &self / rhs
    }
}

impl<Set, OperationsBackend> Div<AlgebraicElement<Set, OperationsBackend>>
    for &AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: DivisionLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn div(self, rhs: AlgebraicElement<Set, OperationsBackend>) -> Self::Output {
        self / &rhs
    }
}

/* Negation operator overloading for Algebraic Elements*/
impl<Set, OperationsBackend> Neg for &AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: NegationLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn neg(self) -> Self::Output {
        Self::Output {
            value: OperationsBackend::neg(&self.value),
            phantom: PhantomData,
        }
    }
}

impl<Set, OperationsBackend> Neg for AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: NegationLaw<Set>,
{
    type Output = AlgebraicElement<Set, OperationsBackend>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

/* AlgebraicElement general implementation */
impl<Set, OperationsBackend> AlgebraicElement<Set, OperationsBackend> {
    pub fn value(&self) -> &Set {
        &self.value
    }
}

/* Inv implementation for Algebraic Elements */
impl<Set, OperationsBackend> AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: InversionLaw<Set>,
{
    pub fn inv(&self) -> Self {
        Self {
            value: OperationsBackend::inv(&self.value),
            phantom: PhantomData,
        }
    }
}

/* Pow implementation for Algebraic Elements */
impl<Set, OperationsBackend> AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: PowerLaw<Set>,
{
    pub fn pow(&self, exponent: u128) -> Self {
        Self {
            value: OperationsBackend::pow(&self.value, exponent),
            phantom: PhantomData,
        }
    }
}

/* One implementation for Algebraic Elements */
impl<Set, OperationsBackend> AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: One<Set>,
{
    pub fn one() -> Self {
        Self {
            value: OperationsBackend::one(),
            phantom: PhantomData,
        }
    }
}

/* Zero implementation for Algebraic Elements */
impl<Set, OperationsBackend> AlgebraicElement<Set, OperationsBackend>
where
    OperationsBackend: Zero<Set>,
{
    pub fn zero() -> Self {
        Self {
            value: OperationsBackend::zero(),
            phantom: PhantomData,
        }
    }
}
