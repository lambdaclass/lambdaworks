use crate::errors::CreationError;
use crate::field::errors::FieldError;
use crate::field::traits::IsField;
use crate::unsigned_integer::element::UnsignedInteger;
use crate::unsigned_integer::montgomery::MontgomeryAlgorithms;
use crate::unsigned_integer::traits::IsUnsignedInteger;
use core::fmt;
use core::fmt::Debug;
use core::iter::Sum;
#[cfg(feature = "lambdaworks-serde")]
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
#[cfg(feature = "lambdaworks-serde")]
use serde::de::{self, Deserializer, MapAccess, Visitor};
#[cfg(feature = "lambdaworks-serde")]
use serde::ser::{Serialize, SerializeStruct, Serializer};
#[cfg(feature = "lambdaworks-serde")]
use serde::Deserialize;

use super::fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField};
use super::traits::{IsPrimeField, LegendreSymbol};

/// A field element with operations algorithms defined in `F`
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone, Hash, Copy)]
pub struct FieldElement<F: IsField> {
    value: F::BaseType,
}

#[cfg(feature = "std")]
impl<F: IsField> FieldElement<F> {
    // Source: https://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Multiple_inverses
    pub fn inplace_batch_inverse(numbers: &mut [Self]) -> Result<(), FieldError> {
        if numbers.is_empty() {
            return Ok(());
        }
        let count = numbers.len();
        let mut prod_prefix = Vec::with_capacity(count);
        prod_prefix.push(numbers[0].clone());
        for i in 1..count {
            prod_prefix.push(&prod_prefix[i - 1] * &numbers[i]);
        }
        let mut bi_inv = prod_prefix[count - 1].inv()?;
        for i in (1..count).rev() {
            let ai_inv = &bi_inv * &prod_prefix[i - 1];
            bi_inv = &bi_inv * &numbers[i];
            numbers[i] = ai_inv;
        }
        numbers[0] = bi_inv;
        Ok(())
    }
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

    pub const fn const_from_raw(value: F::BaseType) -> Self {
        Self { value }
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

impl<F> Default for FieldElement<F>
where
    F: IsField,
{
    fn default() -> Self {
        Self { value: F::zero() }
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
    #[inline(always)]
    pub fn new(value: F::BaseType) -> Self {
        Self {
            value: F::from_base_type(value),
        }
    }

    /// Returns the underlying `value`
    #[inline(always)]
    pub fn value(&self) -> &F::BaseType {
        &self.value
    }

    /// Returns the multiplicative inverse of `self`
    #[inline(always)]
    pub fn inv(&self) -> Result<Self, FieldError> {
        let value = F::inv(&self.value)?;
        Ok(Self { value })
    }

    /// Returns the square of `self`
    #[inline(always)]
    pub fn square(&self) -> Self {
        Self {
            value: F::square(&self.value),
        }
    }

    /// Returns `self` raised to the power of `exponent`
    #[inline(always)]
    pub fn pow<T>(&self, exponent: T) -> Self
    where
        T: IsUnsignedInteger,
    {
        Self {
            value: F::pow(&self.value, exponent),
        }
    }

    /// Returns the multiplicative neutral element of the field.
    #[inline(always)]
    pub fn one() -> Self {
        Self { value: F::one() }
    }

    /// Returns the additive neutral element of the field.
    #[inline(always)]
    pub fn zero() -> Self {
        Self { value: F::zero() }
    }
}

impl<F: IsPrimeField> FieldElement<F> {
    // Returns the representative of the value stored
    pub fn representative(&self) -> F::RepresentativeType {
        F::representative(self.value())
    }

    pub fn sqrt(&self) -> Option<(Self, Self)> {
        let sqrts = F::sqrt(&self.value);
        sqrts.map(|(sqrt1, sqrt2)| (Self { value: sqrt1 }, Self { value: sqrt2 }))
    }

    pub fn legendre_symbol(&self) -> LegendreSymbol {
        F::legendre_symbol(&self.value)
    }

    /// Creates a `FieldElement` from a hexstring. It can contain `0x` or not.
    /// Returns an `CreationError::InvalidHexString`if the value is not a hexstring.
    /// Returns a `CreationError::EmptyString` if the input string is empty.
    pub fn from_hex(hex_string: &str) -> Result<Self, CreationError> {
        if hex_string.is_empty() {
            return Err(CreationError::EmptyString)?;
        }

        Ok(Self {
            value: F::from_hex(hex_string)?,
        })
    }
}

#[cfg(feature = "lambdaworks-serde")]
impl<F: IsPrimeField> Serialize for FieldElement<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("FieldElement", 1)?;
        state.serialize_field("value", &F::representative(self.value()).to_string())?;
        state.end()
    }
}

#[cfg(feature = "lambdaworks-serde")]
impl<'de, F: IsPrimeField> Deserialize<'de> for FieldElement<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Value,
        }

        struct FieldElementVisitor<F>(PhantomData<fn() -> F>);

        impl<'de, F: IsPrimeField> Visitor<'de> for FieldElementVisitor<F> {
            type Value = FieldElement<F>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct FieldElement")
            }

            fn visit_map<M>(self, mut map: M) -> Result<FieldElement<F>, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut value = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Value => {
                            if value.is_some() {
                                return Err(de::Error::duplicate_field("value"));
                            }
                            value = Some(map.next_value()?);
                        }
                    }
                }
                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(FieldElement::from_hex(value).unwrap())
            }
        }

        const FIELDS: &[&str] = &["value"];
        deserializer.deserialize_struct("FieldElement", FIELDS, FieldElementVisitor(PhantomData))
    }
}

impl<M, const NUM_LIMBS: usize> fmt::Display
    for FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value: UnsignedInteger<NUM_LIMBS> = self.representative();
        write!(f, "{}", value)
    }
}

impl<M, const NUM_LIMBS: usize> FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    /// Creates a `FieldElement` from a hexstring. It can contain `0x` or not.
    /// # Panics
    /// Panics if value is not a hexstring
    pub const fn from_hex_unchecked(hex: &str) -> Self {
        let integer = UnsignedInteger::<NUM_LIMBS>::from_hex_unchecked(hex);
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
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::field::test_fields::u64_test_field::U64TestField;
    #[cfg(feature = "std")]
    use crate::unsigned_integer::element::UnsignedInteger;
    #[cfg(feature = "std")]
    use proptest::collection;
    use proptest::{prelude::*, prop_compose, proptest, strategy::Strategy};

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

    #[cfg(feature = "std")]
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

    #[test]
    fn one_of_sqrt_roots_for_4_is_2() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;

        let input = FrElement::from(4);
        let sqrt = input.sqrt().unwrap();
        let result = FrElement::from(2);
        assert_eq!(sqrt.0, result);
    }

    #[test]
    fn one_of_sqrt_roots_for_5_is_28_mod_41() {
        let input = FieldElement::<U64PrimeField<41>>::from(5);
        let sqrt = input.sqrt().unwrap();
        let result = FieldElement::from(28);
        assert_eq!(sqrt.0, result);
        assert_eq!(sqrt.1, -result);
    }

    #[test]
    fn one_of_sqrt_roots_for_25_is_5() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;
        let input = FrElement::from(25);
        let sqrt = input.sqrt().unwrap();
        let five = FrElement::from(5);
        assert!(sqrt.1 == five || sqrt.0 == five);
    }

    #[test]
    fn sqrt_works_for_prime_minus_one() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;

        let input = -FrElement::from(1);
        let sqrt = input.sqrt().unwrap();
        assert_eq!(sqrt.0.square(), input);
        assert_eq!(sqrt.1.square(), input);
        assert_ne!(sqrt.0, sqrt.1);
    }

    #[test]
    fn one_of_sqrt_roots_for_25_is_5_in_stark_field() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;

        let input = FrElement::from(25);
        let sqrt = input.sqrt().unwrap();
        let result = FrElement::from(5);
        assert_eq!(sqrt.0, result);
        assert_eq!(sqrt.1, -result);
    }

    #[test]
    fn sqrt_roots_for_0_are_0_in_stark_field() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;

        let input = FrElement::from(0);
        let sqrt = input.sqrt().unwrap();
        let result = FrElement::from(0);
        assert_eq!(sqrt.0, result);
        assert_eq!(sqrt.1, result);
    }

    #[test]
    fn sqrt_of_27_for_stark_field_does_not_exist() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;

        let input = FrElement::from(27);
        let sqrt = input.sqrt();
        assert!(sqrt.is_none());
    }

    #[test]
    fn from_hex_1a_is_26_for_stark252_prime_field_element() {
        type F = Stark252PrimeField;
        type FE = FieldElement<F>;
        assert_eq!(FE::from_hex("1a").unwrap(), FE::from(26))
    }

    #[test]
    fn from_hex_unchecked_zero_x_1a_is_26_for_stark252_prime_field_element() {
        type F = Stark252PrimeField;
        type FE = FieldElement<F>;
        assert_eq!(FE::from_hex_unchecked("0x1a"), FE::from(26))
    }

    #[test]
    fn construct_new_field_element_from_empty_string_errs() {
        type F = Stark252PrimeField;
        type FE = FieldElement<F>;
        assert!(FE::from_hex("").is_err());
    }

    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FieldElement::<Stark252PrimeField> {
            FieldElement::<Stark252PrimeField>::from(num)
        }
    }

    prop_compose! {
        #[cfg(feature = "std")]
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<FieldElement::<Stark252PrimeField>> {
            vec
        }
    }

    proptest! {
        #[cfg(feature = "std")]
        #[test]
        fn test_inplace_batch_inverse_returns_inverses(vec in field_vec(10)) {
            let input: Vec<_> = vec.into_iter().filter(|x| x != &FieldElement::<Stark252PrimeField>::zero()).collect();
            let mut inverses = input.clone();
            FieldElement::inplace_batch_inverse(&mut inverses).unwrap();

            for (i, x) in inverses.into_iter().enumerate() {
                prop_assert_eq!(x * input[i], FieldElement::<Stark252PrimeField>::one());
            }
        }
    }
}
