use super::{element::FieldElement, errors::FieldError};
#[cfg(feature = "lambdaworks-serde-binary")]
use crate::traits::ByteConversion;
use crate::{errors::CreationError, unsigned_integer::traits::IsUnsignedInteger};
use core::fmt::Debug;

/// Represents different configurations that powers of roots of unity can be in. Some of these may
/// be necessary for FFT (as twiddle factors).
#[derive(Clone, Copy)]
pub enum RootsConfig {
    Natural,            // w^0, w^1, w^2...
    NaturalInversed,    // w^0, w^-1, w^-2...
    BitReverse,         // same as first but exponents are bit-reversed.
    BitReverseInversed, // same as above but exponents are negated.
}

/// Represents the subfield relation between two fields.
pub trait IsSubFieldOf<F: IsField>: IsField {
    fn mul(a: &Self::BaseType, b: &F::BaseType) -> F::BaseType;
    fn add(a: &Self::BaseType, b: &F::BaseType) -> F::BaseType;
    fn div(a: &Self::BaseType, b: &F::BaseType) -> Result<F::BaseType, FieldError>;
    fn sub(a: &Self::BaseType, b: &F::BaseType) -> F::BaseType;
    fn embed(a: Self::BaseType) -> F::BaseType;
    #[cfg(feature = "alloc")]
    fn to_subfield_vec(b: F::BaseType) -> alloc::vec::Vec<Self::BaseType>;
}

impl<F> IsSubFieldOf<F> for F
where
    F: IsField,
{
    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &F::BaseType) -> F::BaseType {
        F::mul(a, b)
    }

    #[inline(always)]
    fn add(a: &Self::BaseType, b: &F::BaseType) -> F::BaseType {
        F::add(a, b)
    }

    #[inline(always)]
    fn sub(a: &Self::BaseType, b: &F::BaseType) -> F::BaseType {
        F::sub(a, b)
    }

    #[inline(always)]
    fn div(a: &Self::BaseType, b: &F::BaseType) -> Result<Self::BaseType, FieldError> {
        F::div(a, b)
    }

    #[inline(always)]
    fn embed(a: Self::BaseType) -> F::BaseType {
        a
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(b: F::BaseType) -> alloc::vec::Vec<Self::BaseType> {
        alloc::vec![b]
    }
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
    fn get_primitive_root_of_unity(order: u64) -> Result<FieldElement<Self>, FieldError> {
        let two_adic_primitive_root_of_unity =
            FieldElement::new(Self::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY);
        if order == 0 {
            return Ok(FieldElement::one());
        }
        if order > Self::TWO_ADICITY {
            return Err(FieldError::RootOfUnityError(order));
        }
        let log_power = Self::TWO_ADICITY - order;
        let root = (0..log_power).fold(two_adic_primitive_root_of_unity, |acc, _| acc.square());
        Ok(root)
    }
}

/// Trait to add field behaviour to a struct.
pub trait IsField: Debug + Clone {
    /// The underlying base type for representing elements from the field.
    // TODO: Relax Unpin for non cuda usage
    #[cfg(feature = "lambdaworks-serde-binary")]
    type BaseType: Clone + Debug + Unpin + ByteConversion + Default;
    #[cfg(not(feature = "lambdaworks-serde-binary"))]
    type BaseType: Clone + Debug + Unpin + Default;

    /// Returns the sum of `a` and `b`.
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    /// Returns the double of `a`.
    fn double(a: &Self::BaseType) -> Self::BaseType {
        Self::add(a, a)
    }

    /// Returns the multiplication of `a` and `b`.
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    /// Returns the multiplication of `a` and `a`.
    fn square(a: &Self::BaseType) -> Self::BaseType {
        Self::mul(a, a)
    }

    fn pow<T>(a: &Self::BaseType, mut exponent: T) -> Self::BaseType
    where
        T: IsUnsignedInteger,
    {
        let zero = T::from(0);
        let one = T::from(1);

        if exponent == zero {
            Self::one()
        } else if exponent == one {
            a.clone()
        } else {
            let mut result = a.clone();

            while exponent & one == zero {
                result = Self::square(&result);
                exponent >>= 1;
            }

            if exponent == zero {
                result
            } else {
                let mut base = result.clone();
                exponent >>= 1;

                while exponent != zero {
                    base = Self::square(&base);
                    if exponent & one == one {
                        result = Self::mul(&result, &base);
                    }
                    exponent >>= 1;
                }

                result
            }
        }
    }

    /// Returns the subtraction of `a` and `b`.
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    /// Returns the additive inverse of `a`.
    fn neg(a: &Self::BaseType) -> Self::BaseType;

    /// Returns the multiplicative inverse of `a`.
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError>;

    /// Returns the division of `a` and `b`.
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError>;

    /// Returns a boolean indicating whether `a` and `b` are equal or not.
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool;

    /// Returns the additive neutral element.
    fn zero() -> Self::BaseType {
        Self::BaseType::default()
    }

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

pub trait IsPrimeField: IsField {
    type RepresentativeType: IsUnsignedInteger;

    /// Returns the integer representative in
    /// the range [0, p-1], where p the modulus
    fn representative(a: &Self::BaseType) -> Self::RepresentativeType;

    fn modulus_minus_one() -> Self::RepresentativeType {
        Self::representative(&Self::neg(&Self::one()))
    }

    /// Creates a BaseType from a Hex String
    /// 0x is optional
    /// Returns an `CreationError::InvalidHexString`if the value is not a hexstring
    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError>;

    #[cfg(feature = "std")]
    /// Creates a hexstring from a `FieldElement` without `0x`.
    fn to_hex(a: &Self::BaseType) -> String;

    /// Returns the number of bits of the max element of the field, as per field documentation, not internal representation.
    /// This is `log2(max FE)` rounded up
    fn field_bit_size() -> usize;

    fn legendre_symbol(a: &Self::BaseType) -> LegendreSymbol {
        let symbol = Self::pow(a, Self::modulus_minus_one() >> 1);

        match symbol {
            x if Self::eq(&x, &Self::zero()) => LegendreSymbol::Zero,
            x if Self::eq(&x, &Self::one()) => LegendreSymbol::One,
            _ => LegendreSymbol::MinusOne,
        }
    }

    /// Returns the two square roots of `self` if they exist and
    /// `None` otherwise
    fn sqrt(a: &Self::BaseType) -> Option<(Self::BaseType, Self::BaseType)> {
        match Self::legendre_symbol(a) {
            LegendreSymbol::Zero => return Some((Self::zero(), Self::zero())),
            LegendreSymbol::MinusOne => return None,
            LegendreSymbol::One => (),
        };

        let integer_one = Self::RepresentativeType::from(1_u16);
        let mut s: usize = 0;
        let mut q = Self::modulus_minus_one();

        while q & integer_one != integer_one {
            s += 1;
            q >>= 1;
        }

        let mut c = {
            // Calculate a non residue:
            let mut non_qr = Self::from_u64(2);
            while Self::legendre_symbol(&non_qr) != LegendreSymbol::MinusOne {
                non_qr = Self::add(&non_qr, &Self::one());
            }

            Self::pow(&non_qr, q)
        };

        let mut x = Self::pow(a, (q + integer_one) >> 1);
        let mut t = Self::pow(a, q);
        let mut m = s;

        let one = Self::one();
        while !Self::eq(&t, &one) {
            let i = {
                let mut i = 0;
                let mut t = t.clone();
                let minus_one = Self::neg(&Self::one());
                while !Self::eq(&t, &minus_one) {
                    i += 1;
                    t = Self::mul(&t, &t);
                }
                i + 1
            };

            let b = (0..(m - i - 1)).fold(c, |acc, _| Self::square(&acc));

            c = Self::mul(&b, &b);
            x = Self::mul(&x, &b);
            t = Self::mul(&t, &c);
            m = i;
        }

        let neg_x = Self::neg(&x);
        Some((x, neg_x))
    }
}
