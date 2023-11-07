use lambdaworks_math::{
    field::{
        element::FieldElement, fields::{fft_friendly::stark_252_prime_field::{Stark252PrimeField, MontgomeryConfigStark252PrimeField}, montgomery_backed_prime_fields::IsModulus}, traits::IsField,
    },
    traits::ByteConversion,
};
use core::{
    mem,
    ops::{DivAssign, MulAssign, SubAssign},
    slice,
};
use std::{ops::{Neg, AddAssign, Div, Mul, Sub, Add}};
use winter_utils::{AsBytes, DeserializationError, Randomizable};
use winterfell::math::ExtensibleField;
use winterfell::{
    math::{FieldElement as IsWinterfellFieldElement, StarkField},
    Deserializable, Serializable,
};
use core::fmt;

use crate::field_element::positive_integer::AdapterPositiveInteger;


#[derive(Debug, Copy, Clone)]
pub struct AdapterFieldElement(pub FieldElement<Stark252PrimeField>);


impl AdapterFieldElement {
    pub const fn from_hex_unchecked(hex: &str) -> AdapterFieldElement {
        AdapterFieldElement(FieldElement::from_hex_unchecked(hex))
    }
}

impl IsWinterfellFieldElement for AdapterFieldElement {
    type PositiveInteger = AdapterPositiveInteger;
    type BaseField = Self;
    const EXTENSION_DEGREE: usize = 1;
    const ELEMENT_BYTES: usize = 32;
    const IS_CANONICAL: bool = false; // Check if related to montgomery
    const ZERO: Self = Self::from_hex_unchecked("0");
    const ONE: Self = Self::from_hex_unchecked("1");

    fn inv(self) -> Self {
        AdapterFieldElement(FieldElement::<Stark252PrimeField>::inv(&self.0).unwrap())
    }

    fn conjugate(&self) -> Self {
        *self
    }

    fn base_element(&self, i: usize) -> Self::BaseField {
        match i {
            0 => *self,
            _ => panic!("element index must be 0, but was {i}"),
        }
    }

    fn slice_as_base_elements(elements: &[Self]) -> &[Self::BaseField] {
        elements
    }

    fn slice_from_base_elements(elements: &[Self::BaseField]) -> &[Self] {
        elements
    }

    fn elements_as_bytes(elements: &[Self]) -> &[u8] {
        let p = elements.as_ptr();
        let len = elements.len() * Self::ELEMENT_BYTES;
        unsafe { slice::from_raw_parts(p as *const u8, len) }
    }

    unsafe fn bytes_as_elements(bytes: &[u8]) -> Result<&[Self], winterfell::DeserializationError> {
        if bytes.len() % Self::ELEMENT_BYTES != 0 {
            return Err(DeserializationError::InvalidValue(format!(
                "number of bytes ({}) does not divide into whole number of field elements",
                bytes.len(),
            )));
        }

        let p = bytes.as_ptr();
        let len = bytes.len() / Self::ELEMENT_BYTES;

        if (p as usize) % mem::align_of::<u64>() != 0 {
            return Err(DeserializationError::InvalidValue(
                "slice memory alignment is not valid for this field element type".to_string(),
            ));
        }

        Ok(slice::from_raw_parts(p as *const Self, len))
    }
}

impl AddAssign<AdapterFieldElement> for AdapterFieldElement
{
    fn add_assign(&mut self, rhs: AdapterFieldElement) {
        *self = *self + rhs; 
    }
}

impl Div<&AdapterFieldElement> for &AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn div(self, rhs: &AdapterFieldElement) -> Self::Output {
        AdapterFieldElement(self.0.div(&rhs.0))
    }
}

impl Div<AdapterFieldElement> for AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn div(self, rhs: AdapterFieldElement) -> Self::Output {
        &self / &rhs
    }
}

impl Div<&AdapterFieldElement> for AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn div(self, rhs: &AdapterFieldElement) -> Self::Output {
        &self / rhs
    }
}

impl Div<AdapterFieldElement> for &AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn div(self, rhs: AdapterFieldElement) -> Self::Output {
        self / &rhs
    }
}

impl Mul<&AdapterFieldElement> for &AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn mul(self, rhs: &AdapterFieldElement) -> Self::Output {
        AdapterFieldElement(self.0.mul(&rhs.0))
    }
}

impl Mul<AdapterFieldElement> for AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn mul(self, rhs: AdapterFieldElement) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<&AdapterFieldElement> for AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn mul(self, rhs: &AdapterFieldElement) -> Self::Output {
        &self * rhs
    }
}

impl Mul<AdapterFieldElement> for &AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn mul(self, rhs: AdapterFieldElement) -> Self::Output {
        self * &rhs
    }
}

impl Sub<&AdapterFieldElement> for &AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn sub(self, rhs: &AdapterFieldElement) -> Self::Output {
        AdapterFieldElement(self.0.sub(&rhs.0))
    }
}

impl Sub<AdapterFieldElement> for AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn sub(self, rhs: AdapterFieldElement) -> Self::Output {
        &self - &rhs
    }
}

impl Sub<&AdapterFieldElement> for AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn sub(self, rhs: &AdapterFieldElement) -> Self::Output {
        &self - rhs
    }
}

impl Sub<AdapterFieldElement> for &AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn sub(self, rhs: AdapterFieldElement) -> Self::Output {
        self - &rhs
    }
}

impl Add<&AdapterFieldElement> for &AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn add(self, rhs: &AdapterFieldElement) -> Self::Output {
        AdapterFieldElement(self.0.add(&self.0))
    }
}

impl Add<AdapterFieldElement> for AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn add(self, rhs: AdapterFieldElement) -> Self::Output {
        &self + &rhs
    }
}

impl Add<&AdapterFieldElement> for AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn add(self, rhs: &AdapterFieldElement) -> Self::Output {
        &self + rhs
    }
}

impl Add<AdapterFieldElement> for &AdapterFieldElement
{
    type Output = AdapterFieldElement;

    fn add(self, rhs: AdapterFieldElement) -> Self::Output {
        self + &rhs
    }
}

impl PartialEq<AdapterFieldElement> for AdapterFieldElement
{
    fn eq(&self, other: &AdapterFieldElement) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for AdapterFieldElement {}

impl Default for AdapterFieldElement
{
    fn default() -> Self {
        AdapterFieldElement(FieldElement::<Stark252PrimeField>::default())
    }
}

impl fmt::Display for AdapterFieldElement
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl StarkField for AdapterFieldElement {
    const MODULUS: Self::PositiveInteger = AdapterPositiveInteger::from_hex_unchecked("800000000000011000000000000000000000000000000000000000000000001");

    const MODULUS_BITS: u32 = 252;

    const GENERATOR: Self = Self::from_hex_unchecked("3");

    const TWO_ADICITY: u32 = 192;

    const TWO_ADIC_ROOT_OF_UNITY: Self =
        Self::from_hex_unchecked("5282db87529cfa3f0464519c8b0fa5ad187148e11a61616070024f42f8ef94");

    fn get_modulus_le_bytes() -> Vec<u8> {
        Self::MODULUS.to_bytes_le()
    }

    fn as_int(&self) -> Self::PositiveInteger {
        AdapterPositiveInteger(FieldElement::representative(&self.0))
    }
}

impl From<u8> for AdapterFieldElement {
    fn from(value: u8) -> Self {
        Self::from(value as u64)
    }
}

impl From<u16> for AdapterFieldElement {
    fn from(value: u16) -> Self {
        Self::from(value as u64)
    }
}

impl From<u32> for AdapterFieldElement {
    fn from(value: u32) -> Self {
        Self::from(value as u64)
    }
}

impl From<u64> for AdapterFieldElement {
    fn from(value: u64) -> Self {
        AdapterFieldElement(FieldElement::new(Stark252PrimeField::from_u64(value)))
    }
}

impl From<u128> for AdapterFieldElement {
    fn from(_value: u128) -> Self {
        todo!()
    }
}

impl DivAssign<AdapterFieldElement> for AdapterFieldElement {
    fn div_assign(&mut self, rhs: AdapterFieldElement) {
        *self *= rhs.inv();
    }
}

impl MulAssign<AdapterFieldElement> for AdapterFieldElement {
    fn mul_assign(&mut self, rhs: AdapterFieldElement) {
        *self = *self * rhs;
    }
}
impl SubAssign<AdapterFieldElement> for AdapterFieldElement {
    fn sub_assign(&mut self, rhs: AdapterFieldElement) {
        *self = *self - rhs;
    }
}

impl Neg for AdapterFieldElement {
    type Output = AdapterFieldElement;

    fn neg(self) -> Self::Output {
        AdapterFieldElement(self.0.neg())
    }
}

impl Deserializable for AdapterFieldElement {
    fn read_from<R: winter_utils::ByteReader>(
        _source: &mut R,
    ) -> Result<Self, winter_utils::DeserializationError> {
        todo!()
    }
}

impl ExtensibleField<2> for AdapterFieldElement {
    #[inline(always)]
    fn mul(a: [Self; 2], b: [Self; 2]) -> [Self; 2] {
        let z = a[0] * b[0];
        [z + a[1] * b[1], (a[0] + a[1]) * (b[0] + b[1]) - z]
    }

    #[inline(always)]
    fn mul_base(a: [Self; 2], b: Self) -> [Self; 2] {
        [a[0] * b, a[1] * b]
    }

    #[inline(always)]
    fn frobenius(x: [Self; 2]) -> [Self; 2] {
        [x[0] + x[1], -x[1]]
    }
}

// CUBIC EXTENSION
// ================================================================================================

/// Defines a cubic extension of the base field over an irreducible polynomial x<sup>3</sup> +
/// 2x + 2. Thus, an extension element is defined as α + β * φ + γ * φ^2, where φ is a root of this
/// polynomial, and α, β and γ are base field elements.
impl ExtensibleField<3> for AdapterFieldElement {
    #[inline(always)]
    fn mul(a: [Self; 3], b: [Self; 3]) -> [Self; 3] {
        // performs multiplication in the extension field using 6 multiplications, 8 additions,
        // and 9 subtractions in the base field. overall, a single multiplication in the extension
        // field is roughly equal to 12 multiplications in the base field.
        let a0b0 = a[0] * b[0];
        let a1b1 = a[1] * b[1];
        let a2b2 = a[2] * b[2];

        let a0b0_a0b1_a1b0_a1b1 = (a[0] + a[1]) * (b[0] + b[1]);
        let minus_a0b0_a0b2_a2b0_minus_a2b2 = (a[0] - a[2]) * (b[2] - b[0]);
        let a1b1_minus_a1b2_minus_a2b1_a2b2 = (a[1] - a[2]) * (b[1] - b[2]);
        let a0b0_a1b1 = a0b0 + a1b1;

        let minus_2a1b2_minus_2a2b1 = (a1b1_minus_a1b2_minus_a2b1_a2b2 - a1b1 - a2b2).double();

        let a0b0_minus_2a1b2_minus_2a2b1 = a0b0 + minus_2a1b2_minus_2a2b1;
        let a0b1_a1b0_minus_2a1b2_minus_2a2b1_minus_2a2b2 =
            a0b0_a0b1_a1b0_a1b1 + minus_2a1b2_minus_2a2b1 - a2b2.double() - a0b0_a1b1;
        let a0b2_a1b1_a2b0_minus_2a2b2 = minus_a0b0_a0b2_a2b0_minus_a2b2 + a0b0_a1b1 - a2b2;
        [
            a0b0_minus_2a1b2_minus_2a2b1,
            a0b1_a1b0_minus_2a1b2_minus_2a2b1_minus_2a2b2,
            a0b2_a1b1_a2b0_minus_2a2b2,
        ]
    }

    #[inline(always)]
    fn mul_base(a: [Self; 3], b: Self) -> [Self; 3] {
        [a[0] * b, a[1] * b, a[2] * b]
    }

    #[inline(always)]
    fn frobenius(x: [Self; 3]) -> [Self; 3] {
        // coefficients were computed using SageMath
        [
            x[0] + AdapterFieldElement::from(2061766055618274781u64) * x[1]
                + AdapterFieldElement::from(786836585661389001u64) * x[2],
            AdapterFieldElement::from(2868591307402993000u64) * x[1]
                + AdapterFieldElement::from(3336695525575160559u64) * x[2],
            AdapterFieldElement::from(2699230790596717670u64) * x[1]
                + AdapterFieldElement::from(1743033688129053336u64) * x[2],
        ]
    }
}

/*
    Many of the following traits are required by Winterfell, but these
    are not needed for the adapter to work. E.g.: the AIR adapter only needs
    to compute the main and auxiliary transitions, and be able to construct
    the auxiliary RAP trace. Serializing or sampling random elements is not
    needed, because these are already covered by the lambdaworks field element.
*/
impl Serializable for AdapterFieldElement {
    fn write_into<W: winter_utils::ByteWriter>(&self, _target: &mut W) {
        todo!()
    }
}

impl Randomizable for AdapterFieldElement {
    const VALUE_SIZE: usize = 8;

    fn from_random_bytes(_source: &[u8]) -> Option<Self> {
        todo!()
    }
}

impl AsBytes for AdapterFieldElement {
    fn as_bytes(&self) -> &[u8] {
        todo!()
    }
}

impl<'a> TryFrom<&'a [u8]> for AdapterFieldElement {
    type Error = DeserializationError;

    fn try_from(_value: &'a [u8]) -> Result<Self, Self::Error> {
        todo!()
    }
}