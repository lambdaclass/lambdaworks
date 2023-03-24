use crate::{
    fft::{bit_reversing::reverse_index, errors::FFTError},
    unsigned_integer::traits::IsUnsignedInteger,
};
use std::{fmt::Debug, hash::Hash};

use super::element::FieldElement;

/// Represents different configurations that powers of roots of unity can be in. Some of these may
/// be necessary for FFT (as twiddle factors).
#[derive(Clone, Copy)]
pub enum RootsConfig {
    Natural,            // w^0, w^1, w^2...
    NaturalInversed,    // w^0, w^-1, w^-2...
    BitReverse,         // same as first but exponents are bit-reversed.
    BitReverseInversed, // same as above but exponents are negated.
}

/// Trait to define necessary parameters for FFT-friendly Fields.
/// Two-Adic fields are ones whose order is of the form  $2^n k + 1$.
/// Here $n$ is usually called the *two-adicity* of the field. The
/// reason we care about it is that in an $n$-adic field there are $2^j$-roots
/// of unity for every `j` between 1 and n, which is needed to do Fast Fourier.
/// A two-adic primitive root of unity is a number w that satisfies w^(2^n) = 1
/// and w^(j) != 1 for every j below 2^n. With this primitive root we can generate
/// any other root of unity we need to perform FFT.
pub trait IsTwoAdicField: IsField {
    const TWO_ADICITY: u64;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType;

    /// Returns a primitive root of unity of order $2^{order}$.
    fn get_primitive_root_of_unity(order: u64) -> Result<FieldElement<Self>, FFTError> {
        let two_adic_primitive_root_of_unity =
            FieldElement::new(Self::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY);
        if order == 0 {
            return Err(FFTError::RootOfUnityError(
                "Cannot get root of unity for order = 0".to_string(),
                order,
            ));
        }
        if order > Self::TWO_ADICITY {
            return Err(FFTError::RootOfUnityError(
                "Order cannot exceed 2^{Self::TWO_ADICITY}".to_string(),
                order,
            ));
        }
        let power = 1u64 << (Self::TWO_ADICITY - order);
        Ok(two_adic_primitive_root_of_unity.pow(power))
    }

    /// Returns a `Vec` of the powers of a `2^n`th primitive root of unity in some configuration
    /// `config`. For example, in a `Natural` config this would yield: w^0, w^1, w^2...
    fn get_powers_of_primitive_root(
        n: u64,
        count: usize,
        config: RootsConfig,
    ) -> Result<Vec<FieldElement<Self>>, FFTError> {
        let root = Self::get_primitive_root_of_unity(n)?;

        let calc = |i| match config {
            RootsConfig::Natural => root.pow(i),
            RootsConfig::NaturalInversed => root.pow(i).inv(),
            RootsConfig::BitReverse => root.pow(reverse_index(&i, count as u64)),
            RootsConfig::BitReverseInversed => root.pow(reverse_index(&i, count as u64)).inv(),
        };

        let results = (0..count).map(calc);
        Ok(results.collect())
    }

    /// Returns a `Vec` of the powers of a `2^n`th primitive root of unity, scaled `offset` times,
    /// in a Natural configuration.
    fn get_powers_of_primitive_root_coset(
        n: u64,
        count: usize,
        offset: &FieldElement<Self>,
    ) -> Result<Vec<FieldElement<Self>>, FFTError> {
        let root = Self::get_primitive_root_of_unity(n)?;
        let results = (0..count).map(|i| root.pow(i) * offset);

        Ok(results.collect())
    }

    /// Returns 2^`order` / 2 twiddle factors for FFT in some configuration `config`.
    /// Twiddle factors are powers of a primitive root of unity of the field, used for FFT
    /// computations. FFT only requires the first half of all the powers
    fn get_twiddles(order: u64, config: RootsConfig) -> Result<Vec<FieldElement<Self>>, FFTError> {
        Self::get_powers_of_primitive_root(order, (1 << order) / 2, config)
    }
}

/// Trait to add field behaviour to a struct.
pub trait IsField: Debug + Clone {
    /// The underlying base type for representing elements from the field.
    type BaseType: Clone + Debug + Hash;

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
        let zero = T::from(0);
        let one = T::from(1);
        while exponent > zero {
            if exponent & one == one {
                result = Self::mul(&result, &base);
            }
            base = Self::mul(&base, &base);
            exponent = exponent >> 1;
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

pub trait IsPrimeField: IsField {
    type RepresentativeType: IsUnsignedInteger;

    // Returns the representative of the value stored
    fn representative(a: &Self::BaseType) -> Self::RepresentativeType;
}

#[cfg(test)]
mod tests {
    use crate::{
        fft::bit_reversing::in_place_bit_reverse_permute,
        field::test_fields::u64_test_field::U64TestField,
    };
    use proptest::prelude::*;

    use super::*;

    const MODULUS: u64 = 0xFFFFFFFF00000001;
    type F = U64TestField<MODULUS>;

    proptest! {
        #[test]
        fn test_gen_twiddles_bit_reversed_validity(n in 1..8_u64) {
            let twiddles = F::get_twiddles(n, RootsConfig::Natural).unwrap();
            let mut twiddles_to_reorder = F::get_twiddles(n, RootsConfig::BitReverse).unwrap();
            in_place_bit_reverse_permute(&mut twiddles_to_reorder); // so now should be naturally ordered

            prop_assert_eq!(twiddles, twiddles_to_reorder);
        }
    }
}
