use crate::unsigned_integer::traits::IsUnsignedInteger;
use std::{fmt::Debug, hash::Hash};

use super::{element::FieldElement, errors::FieldError};

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
pub trait IsFFTField: IsField {
    const TWO_ADICITY: u64;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType;

    /// Used for searching this field's implementation in other languages, e.g in MSL
    /// for executing parallel operations with the Metal API.
    fn field_name() -> &'static str {
        ""
    }

    /// Returns a primitive root of unity of order $2^{order}$.
    fn get_primitive_root_of_unity<F: IsFFTField>(
        order: u64,
    ) -> Result<FieldElement<F>, FieldError> {
        let two_adic_primitive_root_of_unity =
            FieldElement::new(F::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY);
        if order == 0 {
            return Ok(FieldElement::one());
        }
        if order > F::TWO_ADICITY {
            return Err(FieldError::RootOfUnityError(
                "Order cannot exceed 2^{F::TWO_ADICITY}".to_string(),
                order,
            ));
        }
        let power = 1u64 << (F::TWO_ADICITY - order);
        Ok(two_adic_primitive_root_of_unity.pow(power))
    }
}

/// Trait to add field behaviour to a struct.
pub trait IsField: Debug + Clone {
    /// The underlying base type for representing elements from the field.
    // TODO: Relax Unpin for non cuda usage
    type BaseType: Clone + Debug + Hash + Unpin;

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

#[derive(PartialEq)]
pub enum LegendreSymbol {
    MinusOne,
    Zero,
    One,
}

pub trait IsPrimeField: IsField<BaseType = Self::RepresentativeType> {
    type RepresentativeType: IsUnsignedInteger;

    // Returns the representative of the value stored
    fn representative(a: &Self::BaseType) -> Self::BaseType;

    fn is_even(a: &Self::BaseType) -> bool {
        Self::representative(a) & 1.into() == 0.into()
    }

    fn modulus_minus_one() -> Self::BaseType {
        Self::representative(&Self::neg(&Self::one()))
    }

    fn legendre_symbol(a: &Self::BaseType) -> LegendreSymbol {
        let symbol = Self::pow(a, Self::modulus_minus_one() >> 1);

        match symbol {
            x if Self::eq(&x, &Self::zero()) => LegendreSymbol::Zero,
            x if Self::eq(&x, &Self::one()) => LegendreSymbol::One,
            _ => LegendreSymbol::MinusOne,
        }
    }

    /// Returns the two square roots of `self` if it exists
    /// `None` if it doesn't
    fn sqrt(a: &Self::BaseType) -> Option<(Self::BaseType, Self::BaseType)> {
        match Self::legendre_symbol(a) {
            LegendreSymbol::Zero => return Some((Self::zero(), Self::zero())),
            LegendreSymbol::MinusOne => return None,
            LegendreSymbol::One => (),
        };

        let (zero, one, two) = (Self::zero(), Self::one(), Self::from_u64(2));

        let mut q = Self::neg(&Self::one());
        let mut s = Self::zero();

        while Self::is_even(&q) {
            s = Self::add(&s, &one);
            q = Self::div(&q, &two);
        }

        let mut c = {
            // Calculate a non residue:
            let mut non_qr = one.clone();
            while Self::legendre_symbol(&non_qr) != LegendreSymbol::MinusOne {
                non_qr = Self::add(&non_qr, &Self::one());
            }

            Self::pow(&non_qr, Self::representative(&q))
        };

        let mut x = Self::pow(
            a,
            Self::representative(&Self::div(&Self::add(&q, &one), &two)),
        );

        let mut t = Self::pow(a, Self::representative(&q));
        let mut m = s;

        while !Self::eq(&t, &one) {
            let mut i = zero.clone();
            let mut e = Self::from_u64(2);
            while Self::representative(&i) < Self::representative(&m) {
                i = Self::add(&i, &one);
                if Self::eq(&Self::pow(&t, Self::representative(&e)), &one) {
                    break;
                }
                e = Self::mul(&e, &two);
            }

            let b = Self::pow(
                &c,
                Self::representative(&Self::pow(
                    &two,
                    Self::representative(&Self::sub(&Self::sub(&m, &i), &one)),
                )),
            );

            x = Self::mul(&x, &b);
            t = Self::mul(&Self::mul(&t, &b), &b);
            c = Self::mul(&b, &b);
            m = i;
        }

        let neg_x = Self::neg(&x);
        Some((x, neg_x))
    }
}
