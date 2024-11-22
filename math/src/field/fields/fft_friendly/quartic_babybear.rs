use crate::{
    field::{
        element::FieldElement, fields::fft_friendly::babybear::Babybear31PrimeField,
        traits::IsField,
    },
    traits::ByteConversion,
};

pub const BETA: FieldElement<Babybear31PrimeField> = FieldElement::<Babybear31PrimeField>::from(11);

#[derive(Clone, Debug)]
pub struct Degree4BabyBearExtensionField;

impl IsField for Degree4BabyBearExtensionField {
    type BaseType = [FieldElement<Babybear31PrimeField>; 4];

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] + &b[0], &a[1] + &b[1], &a[2] + &b[2], &a[3] + &b[3]]
    }

    // Result of multiplying two polynomials a = a0 + a1 * x + a2 * x^2 + a3 * x^3 and
    // b = b0 + b1 * x + b2 * x^2 + b3 * x^3 by applying distribution and taking
    // the remainder of the division by x^4 + 11.
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [
            a[0] * b[0] - BETA * (a[1] * b[3] + a[3] * b[1] + a[2] * b[2]),
            a[0] * b[1] + a[1] * b[0] - BETA * (a[2] * b[3] + a[3] * b[2]),
            a[0] * b[2] + a[2] * b[0] + a[1] * b[1] - BETA * (a[3] * b[3]),
            a[0] * b[3] + a[3] * b[0] + a[1] * b[2] + a[2] * b[1],
        ]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        [
            a[0].square() - BETA * (2 * (a[1] * a[3]) + a[2].square()),
            2 * (a[0] * a[1] - BETA * (a[2] * a[3])),
            2 * (a[0] * a[2]) + a[1].square() - BETA * (a[3].square()),
            2 * (a[0] * a[3] + a[1] * a[2]),
        ]
    }
    /// Returns the component wise subtraction of `a` and `b`
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2], &a[3] - &b[3]]
    }

    /// Returns the component wise negation of `a`
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-&a[0], -&a[1], &a[2], -&a[3]]
    }

    /// Returns the multiplicative inverse of `a`
    /// This uses the equality `(a0 + a1 * u) * (a0 - a1 * u) = a0.pow(2) - a1.pow(2) * residue()`
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let mut b0 = a[0] * a[0] + BETA * (a[1] * (a[3] + a[3]) - a[2] * a[2]);
        let mut b2 = a[0] * (a[2] + a[2]) - a[1] * a[1] + BETA * (a[3] * a[3]);
        let c = b0 * b0 + BETA * b2 * b2;
        let c_inv = c.inv();

        b0 *= c_inv;
        b2 *= c_inv;

        ExtField([
            a[0] * b0 + BETA * a[2] * b2,
            -a[1] * b0 + NBETA * a[3] * b2,
            -a[0] * b2 + a[2] * b0,
            a[1] * b2 - a[3] * b0,
        ])
        //let inv_norm = (a[0].square() + a[1].square()).inv()?;
        //Ok([&a[0] * &inv_norm, -&a[1] * inv_norm])
    }

    /// Returns the division of `a` and `b`
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        <Self as IsField>::mul(a, &Self::inv(b).unwrap())
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> Self::BaseType {
        [
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> Self::BaseType {
        [
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType {
        [
            FieldElement::from(x),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        Self::add(a, a)
    }

    fn pow<T>(a: &Self::BaseType, mut exponent: T) -> Self::BaseType
    where
        T: crate::unsigned_integer::traits::IsUnsignedInteger,
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
}

impl IsSubFieldOf<Degree4BabyBearExtensionField> for Babybear31PrimeField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FieldElement::from_raw(<Self as IsField>::mul(a, b[1].value()));
        [c0, c1]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::add(a, b[0].value()));
        let c1 = FieldElement::from_raw(*b[1].value());
        [c0, c1]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        let b_inv = Degree2ExtensionField::inv(b).unwrap();
        <Self as IsSubFieldOf<Degree2ExtensionField>>::mul(a, &b_inv)
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FieldElement::from_raw(<Self as IsField>::neg(b[1].value()));
        [c0, c1]
    }

    fn embed(a: Self::BaseType) -> <Degree2ExtensionField as IsField>::BaseType {
        [FieldElement::from_raw(a), FieldElement::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree2ExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

#[cfg(feature = "lambdaworks-serde-binary")]
impl ByteConversion for [FieldElement<Babybear31PrimeField>; 4] {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    fn from_bytes_be(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }

    fn from_bytes_le(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}
