use crate::errors::{ByteConversionError, CreationError};
use crate::field::errors::FieldError;
use crate::field::traits::IsField;
//#[cfg(feature = "lambdaworks-serde-binary")]
use crate::traits::ByteConversion;
use crate::unsigned_integer::element::UnsignedInteger;
use crate::unsigned_integer::montgomery::MontgomeryAlgorithms;
use crate::unsigned_integer::traits::IsUnsignedInteger;
use core::fmt;
use core::fmt::Debug;
use core::iter::Sum;
#[cfg(any(
    feature = "lambdaworks-serde-binary",
    feature = "lambdaworks-serde-string"
))]
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};
use num_bigint::BigUint;
use num_traits::{CheckedAdd, Num};
#[cfg(any(
    feature = "lambdaworks-serde-binary",
    feature = "lambdaworks-serde-string"
))]
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
#[cfg(any(
    feature = "lambdaworks-serde-binary",
    feature = "lambdaworks-serde-string"
))]
use serde::ser::{Serialize, SerializeStruct, Serializer};
#[cfg(any(
    feature = "lambdaworks-serde-binary",
    feature = "lambdaworks-serde-string"
))]
use serde::Deserialize;

use super::fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField};
use super::traits::{IsPrimeField, IsSubFieldOf, LegendreSymbol};

/// A field element with operations algorithms defined in `F`
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone, Hash, Copy)]
pub struct FieldElement<F: IsField> {
    value: F::BaseType,
}

#[cfg(feature = "alloc")]
impl<F: IsField> FieldElement<F> {
    // Source: https://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Multiple_inverses
    pub fn inplace_batch_inverse(numbers: &mut [Self]) -> Result<(), FieldError> {
        if numbers.is_empty() {
            return Ok(());
        }
        let count = numbers.len();
        let mut prod_prefix = alloc::vec::Vec::with_capacity(count);
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

    #[inline(always)]
    pub fn to_subfield_vec<S>(self) -> alloc::vec::Vec<FieldElement<S>>
    where
        S: IsSubFieldOf<F>,
    {
        S::to_subfield_vec(self.value)
            .into_iter()
            .map(|x| FieldElement::from_raw(x))
            .collect()
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

/// From overloading for BigUint
impl<F> TryFrom<BigUint> for FieldElement<F>
where
    Self: ByteConversion,
    F: IsPrimeField,
{
    type Error = ByteConversionError;
    fn try_from(value: BigUint) -> Result<Self, ByteConversionError> {
        FieldElement::<F>::from_reduced_big_uint(&value)
    }
}

impl<F> FieldElement<F>
where
    F::BaseType: Clone,
    F: IsField,
{
    pub fn from_raw(value: F::BaseType) -> Self {
        Self { value }
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
impl<F, L> Add<&FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn add(self, rhs: &FieldElement<L>) -> Self::Output {
        Self::Output {
            value: <F as IsSubFieldOf<L>>::add(&self.value, &rhs.value),
        }
    }
}

impl<F, L> Add<FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn add(self, rhs: FieldElement<L>) -> Self::Output {
        &self + &rhs
    }
}

impl<F, L> Add<&FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn add(self, rhs: &FieldElement<L>) -> Self::Output {
        &self + rhs
    }
}

impl<F, L> Add<FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn add(self, rhs: FieldElement<L>) -> Self::Output {
        self + &rhs
    }
}

/// AddAssign operator overloading for field elements
impl<F, L> AddAssign<FieldElement<F>> for FieldElement<L>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    fn add_assign(&mut self, rhs: FieldElement<F>) {
        self.value = <F as IsSubFieldOf<L>>::add(&rhs.value, &self.value);
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
impl<F, L> Sub<&FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn sub(self, rhs: &FieldElement<L>) -> Self::Output {
        Self::Output {
            value: <F as IsSubFieldOf<L>>::sub(&self.value, &rhs.value),
        }
    }
}

impl<F, L> Sub<FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn sub(self, rhs: FieldElement<L>) -> Self::Output {
        &self - &rhs
    }
}

impl<F, L> Sub<&FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn sub(self, rhs: &FieldElement<L>) -> Self::Output {
        &self - rhs
    }
}

impl<F, L> Sub<FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn sub(self, rhs: FieldElement<L>) -> Self::Output {
        self - &rhs
    }
}

/// Multiplication operator overloading for field elements*/
impl<F, L> Mul<&FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn mul(self, rhs: &FieldElement<L>) -> Self::Output {
        Self::Output {
            value: <F as IsSubFieldOf<L>>::mul(&self.value, &rhs.value),
        }
    }
}

impl<F, L> Mul<FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn mul(self, rhs: FieldElement<L>) -> Self::Output {
        &self * &rhs
    }
}

impl<F, L> Mul<&FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn mul(self, rhs: &FieldElement<L>) -> Self::Output {
        &self * rhs
    }
}

impl<F, L> Mul<FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = FieldElement<L>;

    fn mul(self, rhs: FieldElement<L>) -> Self::Output {
        self * &rhs
    }
}

/// MulAssign operator overloading for field elements
impl<F, L> MulAssign<FieldElement<F>> for FieldElement<L>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    fn mul_assign(&mut self, rhs: FieldElement<F>) {
        self.value = <F as IsSubFieldOf<L>>::mul(&rhs.value, &self.value);
    }
}

/// MulAssign operator overloading for field elements
impl<F, L> MulAssign<&FieldElement<F>> for FieldElement<L>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    fn mul_assign(&mut self, rhs: &FieldElement<F>) {
        self.value = <F as IsSubFieldOf<L>>::mul(&rhs.value, &self.value);
    }
}

/// Division operator overloading for field elements*/
impl<F, L> Div<&FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = Result<FieldElement<L>, FieldError>;

    fn div(self, rhs: &FieldElement<L>) -> Self::Output {
        let value = <F as IsSubFieldOf<L>>::div(&self.value, &rhs.value)?;
        Ok(FieldElement::<L> { value })
    }
}

impl<F, L> Div<FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = Result<FieldElement<L>, FieldError>;

    fn div(self, rhs: FieldElement<L>) -> Self::Output {
        &self / &rhs
    }
}

impl<F, L> Div<&FieldElement<L>> for FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = Result<FieldElement<L>, FieldError>;

    fn div(self, rhs: &FieldElement<L>) -> Self::Output {
        &self / rhs
    }
}

impl<F, L> Div<FieldElement<L>> for &FieldElement<F>
where
    F: IsSubFieldOf<L>,
    L: IsField,
{
    type Output = Result<FieldElement<L>, FieldError>;

    fn div(self, rhs: FieldElement<L>) -> Self::Output {
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

    /// Returns the double of `self`
    #[inline(always)]
    pub fn double(&self) -> Self {
        Self {
            value: F::double(&self.value),
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

    /// Returns the raw base type
    pub fn to_raw(self) -> F::BaseType {
        self.value
    }

    #[inline(always)]
    pub fn to_extension<L: IsField>(self) -> FieldElement<L>
    where
        F: IsSubFieldOf<L>,
    {
        FieldElement {
            value: <F as IsSubFieldOf<L>>::embed(self.value),
        }
    }

    // Converts from a BigUint that is smaller than the modulus.
    // Returns error if the value is bigger than the modulus.
    // pub fn from_reduced_big_uint(
    //     value: &BigUint,
    // ) -> Result<Self, crate::field::element::ByteConversionError>
    // where
    //     Self: ByteConversion,
    //     F: IsPrimeField,
    // {
    //     let modulus_str = &F::modulus_minus_one().to_string();
    //     // Remove "0x" prefix if present (case-insensitive)
    //     let cleaned_str = modulus_str
    //         .trim_start_matches("0x")
    //         .trim_start_matches("0X");
    //     let modulus = BigUint::from_str_radix(cleaned_str, 16)
    //         .ok()
    //         .unwrap()
    //         .checked_add(&BigUint::from(1u32))
    //         .unwrap();

    //     if value >= &modulus {
    //         Err(ByteConversionError::ValueNotReduced)
    //     } else {
    //         let mut bytes = value.to_bytes_be();
    //         // We pad with zeros so that bytes has the length needed to do `from_bytes_le`.
    //         // Since the biggest montgomery field uses 384 bits (6 limbs of 8 bytes), at most we'll need 48 bytes.
    //         let pad = vec![0u8; 48];
    //         bytes.extend_from_slice(&pad);

    //         Ok(Self::from_bytes_le(&bytes).unwrap())
    //     }
    // }
    pub fn from_reduced_big_uint(value: &BigUint) -> Result<Self, ByteConversionError>
    where
        Self: ByteConversion,
        F: IsPrimeField,
    {
        // `modulus_minus_one` viene como string (p.e. "0xFFFFFFFF" ó "16").
        let m1_str = F::modulus_minus_one().to_string();

        // Detectamos la base una sola vez:
        let (digits, radix) = if let Some(hex) = m1_str
            .strip_prefix("0x")
            .or_else(|| m1_str.strip_prefix("0X"))
        {
            (hex, 16)
        } else {
            (m1_str.as_str(), 10)
        };

        let modulus =
            BigUint::from_str_radix(digits, radix).expect("invalid modulus representation") + 1u32; // modulus = (modulus-1) + 1

        if value >= &modulus {
            Err(ByteConversionError::ValueNotReduced)
        } else {
            // … el resto del cuerpo permanece igual …
            let mut bytes = value.to_bytes_be();
            bytes.resize(48, 0); // padding
            Ok(Self::from_bytes_le(&bytes).unwrap())
        }
    }

    /// Converts a field element into a BigUint.
    pub fn to_big_uint(&self) -> BigUint
    where
        Self: ByteConversion,
    {
        BigUint::from_bytes_be(&self.to_bytes_be())
    }

    /// Converts a hex string into a field element.
    pub fn from_hex_str(hex: &str) -> Result<Self, CreationError>
    where
        Self: ByteConversion,
        F: IsPrimeField,
    {
        let hex_str = hex.strip_prefix("0x").unwrap_or(hex);
        if hex_str.is_empty() {
            return Err(CreationError::EmptyString);
        }

        let value =
            BigUint::from_str_radix(hex_str, 16).map_err(|_| CreationError::InvalidHexString)?;

        Ok(Self::from_reduced_big_uint(&value).map_err(|_| CreationError::InvalidHexString)?)
    }

    /// Converts a field element into a hex string.
    pub fn to_hex_str(&self) -> String
    where
        Self: ByteConversion,
    {
        format!("0x{:02X}", self.to_big_uint())
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
    /// Returns a `CreationError::HexStringIsTooBig` if the the input hex string is bigger
    /// than the maximum amount of characters for this element.
    pub fn from_hex(hex_string: &str) -> Result<Self, CreationError> {
        if hex_string.is_empty() {
            return Err(CreationError::EmptyString);
        }
        let value = F::from_hex(hex_string)?;
        Ok(Self { value })
    }

    #[cfg(feature = "std")]
    /// Creates a hexstring from a `FieldElement` without `0x`.
    pub fn to_hex(&self) -> String {
        F::to_hex(&self.value)
    }
}

#[cfg(feature = "lambdaworks-serde-binary")]
impl<F> Serialize for FieldElement<F>
where
    F: IsField,
    F::BaseType: ByteConversion,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("FieldElement", 1)?;
        let data = self.value().to_bytes_be();
        state.serialize_field("value", &data)?;
        state.end()
    }
}

#[cfg(all(
    feature = "lambdaworks-serde-string",
    not(feature = "lambdaworks-serde-binary")
))]
impl<F: IsPrimeField> Serialize for FieldElement<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use crate::alloc::string::ToString;
        let mut state = serializer.serialize_struct("FieldElement", 1)?;
        state.serialize_field("value", &F::representative(self.value()).to_string())?;
        state.end()
    }
}

#[cfg(feature = "lambdaworks-serde-binary")]
impl<'de, F> Deserialize<'de> for FieldElement<F>
where
    F: IsField,
    F::BaseType: ByteConversion,
{
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

        impl<'de, F: IsField> Visitor<'de> for FieldElementVisitor<F> {
            type Value = FieldElement<F>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct FieldElement")
            }

            fn visit_map<M>(self, mut map: M) -> Result<FieldElement<F>, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut value: Option<alloc::vec::Vec<u8>> = None;
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
                let val = F::BaseType::from_bytes_be(&value).unwrap();
                Ok(FieldElement::from_raw(val))
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<FieldElement<F>, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut value: Option<alloc::vec::Vec<u8>> = None;
                while let Some(val) = seq.next_element()? {
                    if value.is_some() {
                        return Err(de::Error::duplicate_field("value"));
                    }
                    value = Some(val);
                }
                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                let val = F::BaseType::from_bytes_be(&value).unwrap();
                Ok(FieldElement::from_raw(val))
            }
        }

        const FIELDS: &[&str] = &["value"];
        deserializer.deserialize_struct("FieldElement", FIELDS, FieldElementVisitor(PhantomData))
    }
}

#[cfg(all(
    feature = "lambdaworks-serde-string",
    not(feature = "lambdaworks-serde-binary")
))]
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
                let mut value: Option<&str> = None;
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
                FieldElement::from_hex(&value).map_err(|_| de::Error::custom("invalid hex"))
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<FieldElement<F>, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut value: Option<&str> = None;
                while let Some(val) = seq.next_element()? {
                    if value.is_some() {
                        return Err(de::Error::duplicate_field("value"));
                    }
                    value = Some(val);
                }
                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                FieldElement::from_hex(&value).map_err(|_| de::Error::custom("invalid hex"))
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
    use super::*;
    use crate::field::fields::fft_friendly::babybear::Babybear31PrimeField;
    use crate::field::fields::fft_friendly::{
        stark_101_prime_field, stark_252_prime_field::Stark252PrimeField,
    };
    use crate::field::fields::montgomery_backed_prime_fields::U384PrimeField;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::field::test_fields::u64_test_field::U64TestField;
    //#[cfg(feature = "lambdaworks-serde-binary")]
    use crate::traits::ByteConversion;
    #[cfg(feature = "alloc")]
    use crate::unsigned_integer::element::UnsignedInteger;
    use crate::unsigned_integer::element::U384;
    #[cfg(feature = "alloc")]
    use alloc::vec::Vec;
    use num_bigint::BigUint;
    #[cfg(feature = "alloc")]
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

    #[cfg(feature = "alloc")]
    #[test]
    fn test_display_montgomery_field() {
        use alloc::format;

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
        #[cfg(feature = "alloc")]
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<FieldElement::<Stark252PrimeField>> {
            vec
        }
    }

    proptest! {
        #[cfg(feature = "alloc")]
        #[test]
        fn test_inplace_batch_inverse_returns_inverses(vec in field_vec(10)) {
            use alloc::format;

            let input: Vec<_> = vec.into_iter().filter(|x| x != &FieldElement::<Stark252PrimeField>::zero()).collect();
            let mut inverses = input.clone();
            FieldElement::inplace_batch_inverse(&mut inverses).unwrap();

            for (i, x) in inverses.into_iter().enumerate() {
                prop_assert_eq!(x * input[i], FieldElement::<Stark252PrimeField>::one());
            }
        }
    }

    // Tests for BigUint conversion with reduction.
    type TestField = U64PrimeField<17>;
    type FE = FieldElement<TestField>;

    type BabyBear = Babybear31PrimeField;
    type BabyBearFE = FieldElement<BabyBear>;

    #[derive(Clone, Debug)]
    struct U384Modulus23;
    impl IsModulus<U384> for U384Modulus23 {
        const MODULUS: U384 = UnsignedInteger::from_u64(23);
    }

    type U384F23 = U384PrimeField<U384Modulus23>;
    type U384F23Element = FieldElement<U384F23>;

    #[test]
    fn test_reduced_biguint_conversion_small_number() {
        let value = BigUint::from(10u32);
        let fe = FE::from_reduced_big_uint(&value).unwrap();
        let back_to_biguint = fe.to_big_uint();
        assert_eq!(value, back_to_biguint);
    }

    #[test]
    fn test_reduced_biguint_conversion_small_number_stark() {
        let value = BigUint::from(10u32);
        let fe = BabyBearFE::from_reduced_big_uint(&value).unwrap();
        let back_to_biguint = fe.to_big_uint();
        assert_eq!(value, back_to_biguint);
    }

    #[test]
    fn test_reduced_biguint_conversion_small_number_u384() {
        let value = BigUint::from(10u32);
        let fe = U384F23Element::from_reduced_big_uint(&value).unwrap();
        let back_to_biguint = fe.to_big_uint();
        assert_eq!(value, back_to_biguint);
    }

    #[test]
    fn test_reduced_biguint_conversion_errors() {
        let value = BigUint::from(17u32);
        let result = FE::from_reduced_big_uint(&value);
        assert_eq!(result, Err(ByteConversionError::ValueNotReduced));
    }

    #[test]
    fn test_hex_string_conversion() {
        let hex_str = "0x0a";
        let fe = FE::from_hex_str(hex_str).unwrap();
        assert_eq!(fe, FE::from(10));
        assert_eq!(fe.to_hex_str(), "0x0A");
    }

    #[test]
    fn test_invalid_hex_string() {
        let hex_str = "0xzz";
        let result = FE::from_hex_str(hex_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_hex_string() {
        let hex_str = "";
        let result = FE::from_hex_str(hex_str);
        assert!(result.is_err());
    }

    // #[test]
    // fn test_byte_conversion_with_reduction() {
    //     let value = BigUint::from(20u32);
    //     let fe = FE::from_big_uint(&value);
    //     let bytes = fe.to_bytes_be();
    //     let back_to_fe = FE::from_bytes_be(&bytes).unwrap();
    //     assert_eq!(fe, back_to_fe);
    //     let back_to_biguint = back_to_fe.to_big_uint();
    //     assert_eq!(back_to_biguint, BigUint::from(3u32)); // 20 mod 17 = 3
    // }
}
