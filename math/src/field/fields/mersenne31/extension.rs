use crate::field::{
    element::FieldElement,
    errors::FieldError,
    traits::{IsField, IsSubFieldOf},
};

use super::field::Mersenne31Field;

type FpE = FieldElement<Mersenne31Field>;

//Note: The inverse calculation in mersenne31/plonky3 differs from the default quadratic extension so I implemented the complex extension.
//////////////////
#[derive(Clone, Debug)]
pub struct Degree2ExtensionField;

impl IsField for Degree2ExtensionField {
    //Elements represents a[0] = real, a[1] = imaginary
    type BaseType = [FpE; 2];

    /// Returns the component wise addition of `a` and `b`
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    /// Returns the multiplication of `a` and `b` using the following
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
        let c1 = v0.double();
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
        let inv_norm = (a[0].square() + a[1].square()).inv()?;
        Ok([a[0] * inv_norm, -a[1] * inv_norm])
    }

    /// Returns the division of `a` and `b`
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        <Self as IsField>::mul(a, &Self::inv(b).unwrap())
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero()]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero()]
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero()]
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}

impl IsSubFieldOf<Degree2ExtensionField> for Mersenne31Field {
    fn add(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        [FpE::from(a) + b[0], FpE::from(a) + b[1]]
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        [FpE::from(a) - b[0], FpE::from(a) - b[1]]
    }

    fn mul(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        [FpE::from(a) * b[0], FpE::from(a) * b[1]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        let b_inv = Degree2ExtensionField::inv(b).unwrap();
        <Self as IsSubFieldOf<Degree2ExtensionField>>::mul(a, &b_inv)
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

#[cfg(test)]
mod tests {
    use core::ops::Neg;

    use crate::field::fields::mersenne31::field::MERSENNE_31_PRIME_FIELD_ORDER;

    use super::*;

    type Fp2E = FieldElement<Degree2ExtensionField>;

    #[test]
    fn add_real_one_plus_one_is_two() {
        println!("{:?}", Fp2E::from(2));
        assert_eq!(Fp2E::one() + Fp2E::one(), Fp2E::from(2))
    }

    #[test]
    fn add_real_neg_one_plus_one_is_zero() {
        assert_eq!(Fp2E::one() + Fp2E::one().neg(), Fp2E::zero())
    }

    #[test]
    fn add_real_neg_one_plus_two_is_one() {
        assert_eq!(Fp2E::one().neg() + Fp2E::from(2), Fp2E::one())
    }

    #[test]
    fn add_real_neg_one_plus_neg_one_is_order_sub_two() {
        assert_eq!(
            Fp2E::one().neg() + Fp2E::one().neg(),
            Fp2E::new([FpE::from(&(MERSENNE_31_PRIME_FIELD_ORDER - 2)), FpE::zero()])
        )
    }

    #[test]
    fn add_complex_one_plus_one_two() {
        let one_i = Fp2E::new([FpE::zero(), FpE::one()]);
        let two_i = Fp2E::new([FpE::zero(), FpE::from(2)]);
        assert_eq!(&one_i + &one_i, two_i)
    }

    #[test]
    fn add_complex_neg_one_plus_one_is_zero() {
        //Manually declare the complex part to one
        let neg_one_i = Fp2E::new([FpE::zero(), -FpE::one()]);
        let one_i = Fp2E::new([FpE::zero(), FpE::one()]);
        assert_eq!(neg_one_i + one_i, Fp2E::zero())
    }

    #[test]
    fn add_complex_neg_one_plus_two_is_one() {
        let neg_one_i = Fp2E::new([FpE::zero(), -FpE::one()]);
        let two_i = Fp2E::new([FpE::zero(), FpE::from(2)]);
        let one_i = Fp2E::new([FpE::zero(), FpE::one()]);
        assert_eq!(&neg_one_i + &two_i, one_i)
    }

    #[test]
    fn add_complex_neg_one_plus_neg_one_imag_is_order_sub_two() {
        let neg_one_i = Fp2E::new([FpE::zero(), -FpE::one()]);
        assert_eq!(
            (&neg_one_i + &neg_one_i).value()[1],
            FpE::from(&(MERSENNE_31_PRIME_FIELD_ORDER - 2))
        )
    }

    #[test]
    fn add_order() {
        let a = Fp2E::new([-FpE::one(), FpE::one()]);
        let b = Fp2E::new([
            FpE::from(2),
            FpE::from(&(MERSENNE_31_PRIME_FIELD_ORDER - 2)),
        ]);
        let c = Fp2E::new([FpE::one(), -FpE::one()]);
        assert_eq!(&a + &b, c)
    }

    #[test]
    fn add_equal_zero() {
        let a = Fp2E::new([-FpE::one(), -FpE::one()]);
        let b = Fp2E::new([FpE::one(), FpE::one()]);
        assert_eq!(&a + &b, Fp2E::zero())
    }

    #[test]
    fn add_plus_one() {
        let a = Fp2E::new([FpE::one(), FpE::from(2)]);
        let b = Fp2E::new([FpE::one(), FpE::one()]);
        let c = Fp2E::new([FpE::from(2), FpE::from(3)]);
        assert_eq!(&a + &b, c)
    }

    #[test]
    fn sub_real_one_sub_one_is_zero() {
        assert_eq!(&Fp2E::one() - &Fp2E::one(), Fp2E::zero())
    }

    #[test]
    fn sub_real_two_sub_two_is_zero() {
        assert_eq!(&Fp2E::from(2) - &Fp2E::from(2), Fp2E::zero())
    }

    #[test]
    fn sub_real_neg_one_sub_neg_one_is_zero() {
        assert_eq!(Fp2E::one().neg() - Fp2E::one().neg(), Fp2E::zero())
    }

    #[test]
    fn sub_real_two_sub_one_is_one() {
        assert_eq!(Fp2E::from(2) - Fp2E::one(), Fp2E::one())
    }

    #[test]
    fn sub_real_neg_one_sub_zero_is_neg_one() {
        assert_eq!(Fp2E::one().neg() - Fp2E::zero(), Fp2E::one().neg())
    }

    #[test]
    fn sub_complex_one_sub_one_is_zero() {
        let one = Fp2E::new([FpE::zero(), FpE::one()]);
        assert_eq!(&one - &one, Fp2E::zero())
    }

    #[test]
    fn sub_complex_two_sub_two_is_zero() {
        let two = Fp2E::new([FpE::zero(), FpE::from(2)]);
        assert_eq!(&two - &two, Fp2E::zero())
    }

    #[test]
    fn sub_complex_neg_one_sub_neg_one_is_zero() {
        let neg_one = Fp2E::new([FpE::zero(), -FpE::one()]);
        assert_eq!(&neg_one - &neg_one, Fp2E::zero())
    }

    #[test]
    fn sub_complex_two_sub_one_is_one() {
        let two = Fp2E::new([FpE::zero(), FpE::from(2)]);
        let one = Fp2E::new([FpE::zero(), FpE::one()]);
        assert_eq!(&two - &one, one)
    }

    #[test]
    fn sub_complex_neg_one_sub_zero_is_neg_one() {
        let neg_one = Fp2E::new([FpE::zero(), -FpE::one()]);
        assert_eq!(&neg_one - &Fp2E::zero(), neg_one)
    }

    #[test]
    fn mul() {
        let a = Fp2E::new([FpE::from(2), FpE::from(2)]);
        let b = Fp2E::new([FpE::from(4), FpE::from(5)]);
        let c = Fp2E::new([-FpE::from(2), FpE::from(18)]);
        assert_eq!(&a * &b, c)
    }

    #[test]
    fn square_equals_mul_by_itself() {
        let a = Fp2E::new([FpE::from(2), FpE::from(3)]);
        assert_eq!(a.square(), &a * &a)
    }

    #[test]
    fn test_base_field_2_extension_add() {
        let a = Fee::new([FE::from(0), FE::from(3)]);
        let b = Fee::new([-FE::from(2), FE::from(8)]);
        let expected_result = Fee::new([FE::from(0) - FE::from(2), FE::from(3) + FE::from(8)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_base_field_2_extension_sub() {
        let a = Fee::new([FE::from(0), FE::from(3)]);
        let b = Fee::new([-FE::from(2), FE::from(8)]);
        let expected_result = Fee::new([FE::from(0) + FE::from(2), FE::from(3) - FE::from(8)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_degree_2_extension_mul() {
        let a = Fee::new([FE::from(12), FE::from(5)]);
        let b = Fee::new([-FE::from(4), FE::from(2)]);
        let expected_result = Fee::new([
            FE::from(12) * (-FE::from(4))
                + FE::from(5) * FE::from(2) * Babybear31PrimeField::residue(),
            FE::from(12) * FE::from(2) + FE::from(5) * (-FE::from(4)),
        ]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_degree_2_extension_inv() {
        let a = Fee::new([FE::from(12), FE::from(5)]);
        let inv_norm = (FE::from(12).pow(2_u64)
            - Babybear31PrimeField::residue() * FE::from(5).pow(2_u64))
        .inv()
        .unwrap();
        let expected_result = Fee::new([FE::from(12) * &inv_norm, -&FE::from(5) * inv_norm]);
        assert_eq!(a.inv().unwrap(), expected_result);
    }

    #[test]
    fn test_degree_2_extension_div() {
        let a = Fee::new([FE::from(12), FE::from(5)]);
        let b = Fee::new([-FE::from(4), FE::from(2)]);
        let expected_result = &a * b.inv().unwrap();
        assert_eq!(a / b, expected_result);
    }
}
