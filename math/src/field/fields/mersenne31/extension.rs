use crate::field::{
    element::FieldElement,
    errors::FieldError,
    extensions::{
        cubic::{CubicExtensionField, HasCubicNonResidue},
        quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
    traits::IsField,
};

use super::field::Mersenne31Field;

//Note: The inverse calculation in mersenne31/plonky3 differs from the default quadratic extension so I implemented the complex extension.
//////////////////
#[derive(Clone, Debug)]
pub struct Mersenne31Complex;

impl IsField for Mersenne31Complex {
    //Elements represents a[0] = real, a[1] = imaginary
    type BaseType = [FieldElement<Mersenne31Field>; 2];

    /// Returns the component wise addition of `a` and `b`
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    //NOTE: THIS uses Gauss algorithm. Bench this against plonky 3 implementation to see what is faster.
    /// Returns the multiplication of `a` and `b` using the following
    /// equation:
    /// (a0 + a1 * t) * (b0 + b1 * t) = a0 * b0 + a1 * b1 * Self::residue() + (a0 * b1 + a1 * b0) * t
    /// where `t.pow(2)` equals `Q::residue()`.
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = a[0] * b[0];
        let a1b1 = a[1] * b[1];
        let z = (a[0] + a[1]) * (b[0] + b[1]);
        [a0b0 - a1b1, z - a0b0 - a1b1]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        let [a0, a1] = a;
        let v0 = a0 * a1;
        let c0 = (a0 + a1) * (a0 - a1);
        let c1 = v0 + v0;
        [c0, c1]
    }
    /// Returns the component wise subtraction of `a` and `b`
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1]]
    }

    /// Returns the component wise negation of `a`
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-a[0], -a[1]]
    }

    /// Returns the multiplicative inverse of `a`
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let inv_norm = (a[0].pow(2_u64) + a[1].pow(2_u64)).inv()?;
        Ok([a[0] * inv_norm, -a[1] * inv_norm])
    }

    /// Returns the division of `a` and `b`
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        Self::mul(a, &Self::inv(b).unwrap())
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> Self::BaseType {
        [FieldElement::zero(), FieldElement::zero()]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> Self::BaseType {
        [FieldElement::one(), FieldElement::zero()]
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType {
        [FieldElement::from(x), FieldElement::zero()]
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}

pub type Mersenne31ComplexQuadraticExtensionField = QuadraticExtensionField<Mersenne31Complex>;

//TODO: Check this should be for complex and not base field
impl HasQuadraticNonResidue for Mersenne31Complex {
    type BaseField = Mersenne31Complex;

    // Verifiable in Sage with
    // ```sage
    // p = 2**31 - 1  # Mersenne31
    // F = GF(p)  # The base field GF(p)
    // R.<x> = F[]  # The polynomial ring over F
    // K.<i> = F.extension(x^2 + 1)  # The complex extension field
    // R2.<y> = K[]
    // f2 = y^2 - i - 2
    // assert f2.is_irreducible()
    // ```
    fn residue() -> FieldElement<Mersenne31Complex> {
        FieldElement::from(&Mersenne31Complex::from_base_type([
            FieldElement::<Mersenne31Field>::from(2),
            FieldElement::<Mersenne31Field>::one(),
        ]))
    }
}

pub type Mersenne31ComplexCubicExtensionField = CubicExtensionField<Mersenne31Complex>;

impl HasCubicNonResidue for Mersenne31Complex {
    type BaseField = Mersenne31Complex;

    // Verifiable in Sage with
    // ```sage
    // p = 2**31 - 1  # Mersenne31
    // F = GF(p)  # The base field GF(p)
    // R.<x> = F[]  # The polynomial ring over F
    // K.<i> = F.extension(x^2 + 1)  # The complex extension field
    // R2.<y> = K[]
    // f2 = y^3 - 5*i
    // assert f2.is_irreducible()
    // ```
    fn residue() -> FieldElement<Mersenne31Complex> {
        FieldElement::from(&Mersenne31Complex::from_base_type([
            FieldElement::<Mersenne31Field>::zero(),
            FieldElement::<Mersenne31Field>::from(5),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use crate::field::fields::mersenne31::field::MERSENNE_31_PRIME_FIELD_ORDER;

    use super::*;

    type Fi = Mersenne31Complex;
    type F = FieldElement<Mersenne31Field>;

    //NOTE: from_u64 reflects from_real
    //NOTE: for imag use from_base_type

    #[test]
    fn add_real_one_plus_one_is_two() {
        assert_eq!(Fi::add(&Fi::one(), &Fi::one()), Fi::from_u64(2))
    }

    #[test]
    fn add_real_neg_one_plus_one_is_zero() {
        assert_eq!(Fi::add(&Fi::neg(&Fi::one()), &Fi::one()), Fi::zero())
    }

    #[test]
    fn add_real_neg_one_plus_two_is_one() {
        assert_eq!(Fi::add(&Fi::neg(&Fi::one()), &Fi::from_u64(2)), Fi::one())
    }

    #[test]
    fn add_real_neg_one_plus_neg_one_is_order_sub_two() {
        assert_eq!(
            Fi::add(&Fi::neg(&Fi::one()), &Fi::neg(&Fi::one())),
            Fi::from_u64((MERSENNE_31_PRIME_FIELD_ORDER - 2).into())
        )
    }

    #[test]
    fn add_complex_one_plus_one_two() {
        //Manually declare the complex part to one
        let one = Fi::from_base_type([F::zero(), F::one()]);
        let two = Fi::from_base_type([F::zero(), F::from(2)]);
        assert_eq!(Fi::add(&one, &one), two)
    }

    #[test]
    fn add_complex_neg_one_plus_one_is_zero() {
        //Manually declare the complex part to one
        let neg_one = Fi::from_base_type([F::zero(), -F::one()]);
        let one = Fi::from_base_type([F::zero(), F::one()]);
        assert_eq!(Fi::add(&neg_one, &one), Fi::zero())
    }

    #[test]
    fn add_complex_neg_one_plus_two_is_one() {
        let neg_one = Fi::from_base_type([F::zero(), -F::one()]);
        let two = Fi::from_base_type([F::zero(), F::from(2)]);
        let one = Fi::from_base_type([F::zero(), F::one()]);
        assert_eq!(Fi::add(&neg_one, &two), one)
    }

    #[test]
    fn add_complex_neg_one_plus_neg_one_imag_is_order_sub_two() {
        let neg_one = Fi::from_base_type([F::zero(), -F::one()]);
        assert_eq!(
            Fi::add(&neg_one, &neg_one)[1],
            F::new(MERSENNE_31_PRIME_FIELD_ORDER - 2)
        )
    }

    #[test]
    fn add_order() {
        let a = Fi::from_base_type([-F::one(), F::one()]);
        let b = Fi::from_base_type([F::from(2), F::new(MERSENNE_31_PRIME_FIELD_ORDER - 2)]);
        let c = Fi::from_base_type([F::one(), -F::one()]);
        assert_eq!(Fi::add(&a, &b), c)
    }

    #[test]
    fn add_equal_zero() {
        let a = Fi::from_base_type([-F::one(), -F::one()]);
        let b = Fi::from_base_type([F::one(), F::one()]);
        assert_eq!(Fi::add(&a, &b), Fi::zero())
    }

    #[test]
    fn add_plus_one() {
        let a = Fi::from_base_type([F::one(), F::from(2)]);
        let b = Fi::from_base_type([F::one(), F::one()]);
        let c = Fi::from_base_type([F::from(2), F::from(3)]);
        assert_eq!(Fi::add(&a, &b), c)
    }

    #[test]
    fn sub_real_one_sub_one_is_zero() {
        assert_eq!(Fi::sub(&Fi::one(), &Fi::one()), Fi::zero())
    }

    #[test]
    fn sub_real_two_sub_two_is_zero() {
        assert_eq!(
            Fi::sub(&Fi::from_u64(2u64), &Fi::from_u64(2u64)),
            Fi::zero()
        )
    }

    #[test]
    fn sub_real_neg_one_sub_neg_one_is_zero() {
        assert_eq!(
            Fi::sub(&Fi::neg(&Fi::one()), &Fi::neg(&Fi::one())),
            Fi::zero()
        )
    }

    #[test]
    fn sub_real_two_sub_one_is_one() {
        assert_eq!(Fi::sub(&Fi::from_u64(2), &Fi::one()), Fi::one())
    }

    #[test]
    fn sub_real_neg_one_sub_zero_is_neg_one() {
        assert_eq!(
            Fi::sub(&Fi::neg(&Fi::one()), &Fi::zero()),
            Fi::neg(&Fi::one())
        )
    }

    #[test]
    fn sub_complex_one_sub_one_is_zero() {
        let one = Fi::from_base_type([F::zero(), F::one()]);
        assert_eq!(Fi::sub(&one, &one), Fi::zero())
    }

    #[test]
    fn sub_complex_two_sub_two_is_zero() {
        let two = Fi::from_base_type([F::zero(), F::from(2)]);
        assert_eq!(Fi::sub(&two, &two), Fi::zero())
    }

    #[test]
    fn sub_complex_neg_one_sub_neg_one_is_zero() {
        let neg_one = Fi::from_base_type([F::zero(), -F::one()]);
        assert_eq!(Fi::sub(&neg_one, &neg_one), Fi::zero())
    }

    #[test]
    fn sub_complex_two_sub_one_is_one() {
        let two = Fi::from_base_type([F::zero(), F::from(2)]);
        let one = Fi::from_base_type([F::zero(), F::one()]);
        assert_eq!(Fi::sub(&two, &one), one)
    }

    #[test]
    fn sub_complex_neg_one_sub_zero_is_neg_one() {
        let neg_one = Fi::from_base_type([F::zero(), -F::one()]);
        assert_eq!(Fi::sub(&neg_one, &Fi::zero()), neg_one)
    }

    #[test]
    fn mul() {
        let a = Fi::from_base_type([F::from(2), F::from(2)]);
        let b = Fi::from_base_type([F::from(4), F::from(5)]);
        let c = Fi::from_base_type([-F::from(2), F::from(18)]);
        assert_eq!(Fi::mul(&a, &b), c)
    }
}
