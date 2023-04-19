use crate::field::traits::IsField;
use crate::unsigned_integer::element::UnsignedInteger;
use crate::unsigned_integer::montgomery::MontgomeryAlgorithms;
use crate::unsigned_integer::traits::IsUnsignedInteger;
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
};

use super::fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField};
use super::traits::IsPrimeField;

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

impl<F> FieldElement<F>
where
    F::BaseType: Clone,
    F: IsField,
{
    pub fn from_raw(value: &F::BaseType) -> Self {
        Self {
            value: value.clone(),
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

impl<F: IsPrimeField> FieldElement<F> {
    // Returns the representative of the value stored
    pub fn representative(&self) -> F::RepresentativeType {
        F::representative(self.value())
    }
}

impl<M, const NUM_LIMBS: usize> fmt::Display
    for FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value: UnsignedInteger<NUM_LIMBS> = self.representative();

        let first_non_zero_pos = value.limbs.iter().position(|&limb| limb != 0);

        if let Some(first_non_zero_pos) = first_non_zero_pos {
            write!(f, "0x")?;

            let mut is_most_significant_limb = true;
            for limb in value.limbs.iter().skip(first_non_zero_pos) {
                if is_most_significant_limb {
                    write!(f, "{:x}", limb)?;
                    is_most_significant_limb = false;
                } else {
                    write!(f, "{:016x}", limb)?;
                }
            }
        } else {
            write!(f, "0x0")?;
        }

        Ok(())
    }
}

impl<M, const NUM_LIMBS: usize> FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    #[allow(unused)]
    pub const fn from_hex(hex: &str) -> Self {
        let integer = UnsignedInteger::<NUM_LIMBS>::from(hex);
        Self {
            value: MontgomeryAlgorithms::cios(
                &integer,
                &MontgomeryBackendPrimeField::<M, NUM_LIMBS>::R2,
                &M::MODULUS,
                &MontgomeryBackendPrimeField::<M, NUM_LIMBS>::MU,
            ),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::field::element::FieldElement;
    use crate::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
    use crate::field::test_fields::u64_test_field::U64TestField;
    use crate::unsigned_integer::element::UnsignedInteger;

    #[test]
    fn test_std_iter_sum_field_element() {
        let n = 164;
        const MODULUS: u64 = 18446744069414584321;
        assert_eq!(
            (0..n)
                .map(|x| { FieldElement::<U64TestField>::from(x) })
                .sum::<FieldElement<U64TestField>>()
                .value,
            ((n - 1) as f64 / 2. * ((n - 1) as f64 + 1.)) as u64 % MODULUS
        );
    }

    #[test]
    fn test_std_iter_sum_field_element_zero_length() {
        let n = 0;
        assert_eq!(
            (0..n)
                .map(|x| { FieldElement::<U64TestField>::from(x) })
                .sum::<FieldElement<U64TestField>>()
                .value,
            0
        );
    }

    #[test]
    fn test_display_montgomery_field() {
        let zero_field_element = FieldElement::<Stark252PrimeField>::from(0);
        assert_eq!(format!("{}", zero_field_element), "0x0");

        let some_field_element =
            FieldElement::<Stark252PrimeField>::from(&UnsignedInteger::from_limbs([
                0x0, 0x1, 0x0, 0x1,
            ]));

        // it should start with the first non-zero digit. Each limb has 16 digits in hex.
        assert_eq!(
            format!("{}", some_field_element),
            format!("0x{}{}{}{}", "1", "0".repeat(16), "0".repeat(15), "1")
        );
    }
}
