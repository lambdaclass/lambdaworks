use super::field::Mersenne31Field;
use crate::field::{
    element::FieldElement,
    errors::FieldError,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

type FpE = FieldElement<Mersenne31Field>;
type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp4E = FieldElement<Degree4ExtensionField>;

#[derive(Clone, Debug)]
pub struct Degree2ExtensionField;

impl Degree2ExtensionField {
    pub fn mul_fp2_by_nonresidue(a: &Fp2E) -> Fp2E {
        Fp2E::new([
            a.value()[0].double() - a.value()[1],
            a.value()[1].double() + a.value()[0],
        ])
    }
}

impl IsField for Degree2ExtensionField {
    //Element representation: a[0] = real part, a[1] = imaginary part
    type BaseType = [FpE; 2];

    /// Returns the component wise addition of `a` and `b`
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    /// Returns the multiplication of `a` and `b`.
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
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = &Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(<Self as IsField>::mul(a, b_inv))
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
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

impl IsFFTField for Degree2ExtensionField {
    // Values taken from stwo
    // https://github.com/starkware-libs/stwo/blob/dev/crates/prover/src/core/circle.rs#L203-L209
    const TWO_ADICITY: u64 = 31;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType =
        [FpE::const_from_raw(2), FpE::const_from_raw(1268011823)];
}

impl IsSubFieldOf<Degree2ExtensionField> for Mersenne31Field {
    fn add(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        [FpE::from(a) + b[0], b[1]]
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        [FpE::from(a) - b[0], -b[1]]
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
    ) -> Result<<Degree2ExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree2ExtensionField::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(<Self as IsSubFieldOf<Degree2ExtensionField>>::mul(
            a, &b_inv,
        ))
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

#[derive(Clone, Debug)]
pub struct Degree4ExtensionField;

impl Degree4ExtensionField {
    pub const fn const_from_coefficients(a: u32, b: u32, c: u32, d: u32) -> Fp4E {
        Fp4E::const_from_raw([
            Fp2E::const_from_raw([FpE::const_from_raw(a), FpE::const_from_raw(b)]),
            Fp2E::const_from_raw([FpE::const_from_raw(c), FpE::const_from_raw(d)]),
        ])
    }
}

impl IsField for Degree4ExtensionField {
    type BaseType = [Fp2E; 2];

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] + &b[0], &a[1] + &b[1]]
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] - &b[0], &a[1] - &b[1]]
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-&a[0], -&a[1]]
    }

    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        // Algorithm from: https://github.com/ingonyama-zk/papers/blob/main/Mersenne31_polynomial_arithmetic.pdf (page 5):
        let a0b0 = &a[0] * &b[0];
        let a1b1 = &a[1] * &b[1];
        [
            &a0b0 + Degree2ExtensionField::mul_fp2_by_nonresidue(&a1b1),
            (&a[0] + &a[1]) * (&b[0] + &b[1]) - a0b0 - a1b1,
        ]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        let a0_square = &a[0].square();
        let a1_square = &a[1].square();
        [
            a0_square + Degree2ExtensionField::mul_fp2_by_nonresidue(a1_square),
            (&a[0] + &a[1]).square() - a0_square - a1_square,
        ]
    }

    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let inv_norm =
            (a[0].square() - Degree2ExtensionField::mul_fp2_by_nonresidue(&a[1].square())).inv()?;
        Ok([&a[0] * &inv_norm, -&a[1] * &inv_norm])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = &Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(<Self as IsField>::mul(a, b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    fn zero() -> Self::BaseType {
        [Fp2E::zero(), Fp2E::zero()]
    }

    fn one() -> Self::BaseType {
        [Fp2E::one(), Fp2E::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [Fp2E::from(x), Fp2E::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}

impl IsSubFieldOf<Degree4ExtensionField> for Mersenne31Field {
    fn add(
        a: &Self::BaseType,
        b: &<Degree4ExtensionField as IsField>::BaseType,
    ) -> <Degree4ExtensionField as IsField>::BaseType {
        [FpE::from(a) + &b[0], b[1].clone()]
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree4ExtensionField as IsField>::BaseType,
    ) -> <Degree4ExtensionField as IsField>::BaseType {
        [FpE::from(a) - &b[0], -&b[1]]
    }

    fn mul(
        a: &Self::BaseType,
        b: &<Degree4ExtensionField as IsField>::BaseType,
    ) -> <Degree4ExtensionField as IsField>::BaseType {
        let c0 = FpE::from(a) * &b[0];
        let c1 = FpE::from(a) * &b[1];
        [c0, c1]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree4ExtensionField as IsField>::BaseType,
    ) -> Result<<Degree4ExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree4ExtensionField::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(<Self as IsSubFieldOf<Degree4ExtensionField>>::mul(
            a, &b_inv,
        ))
    }

    fn embed(a: Self::BaseType) -> <Degree4ExtensionField as IsField>::BaseType {
        [
            Fp2E::from_raw(<Self as IsSubFieldOf<Degree2ExtensionField>>::embed(a)),
            Fp2E::zero(),
        ]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree4ExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        // TODO: Repace this for with a map similarly to this:
        // b.into_iter().map(|x| x.to_raw()).collect()
        let mut result = Vec::new();
        for fp2e in b {
            result.push(fp2e.value()[0].to_raw());
            result.push(fp2e.value()[1].to_raw());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use core::ops::Neg;

    use crate::field::fields::mersenne31::field::MERSENNE_31_PRIME_FIELD_ORDER;

    use super::*;

    type FpE = FieldElement<Mersenne31Field>;
    type Fp2E = FieldElement<Degree2ExtensionField>;
    type Fp4E = FieldElement<Degree4ExtensionField>;

    #[test]
    fn add_real_one_plus_one_is_two() {
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
    fn mul_fp2_is_correct() {
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
    fn test_fp2_add() {
        let a = Fp2E::new([FpE::from(0), FpE::from(3)]);
        let b = Fp2E::new([-FpE::from(2), FpE::from(8)]);
        let expected_result = Fp2E::new([FpE::from(0) - FpE::from(2), FpE::from(3) + FpE::from(8)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_fp2_add_2() {
        let a = Fp2E::new([FpE::from(2), FpE::from(4)]);
        let b = Fp2E::new([-FpE::from(2), -FpE::from(4)]);
        let expected_result = Fp2E::new([FpE::from(2) - FpE::from(2), FpE::from(4) - FpE::from(4)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_fp2_add_3() {
        let a = Fp2E::new([FpE::from(&MERSENNE_31_PRIME_FIELD_ORDER), FpE::from(1)]);
        let b = Fp2E::new([FpE::from(1), FpE::from(&MERSENNE_31_PRIME_FIELD_ORDER)]);
        let expected_result = Fp2E::new([FpE::from(1), FpE::from(1)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_fp2_sub() {
        let a = Fp2E::new([FpE::from(0), FpE::from(3)]);
        let b = Fp2E::new([-FpE::from(2), FpE::from(8)]);
        let expected_result = Fp2E::new([FpE::from(0) + FpE::from(2), FpE::from(3) - FpE::from(8)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_fp2_sub_2() {
        let a = Fp2E::new([FpE::zero(), FpE::from(&MERSENNE_31_PRIME_FIELD_ORDER)]);
        let b = Fp2E::new([FpE::one(), -FpE::one()]);
        let expected_result =
            Fp2E::new([FpE::from(&(MERSENNE_31_PRIME_FIELD_ORDER - 1)), FpE::one()]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_fp2_sub_3() {
        let a = Fp2E::new([FpE::from(5), FpE::from(&MERSENNE_31_PRIME_FIELD_ORDER)]);
        let b = Fp2E::new([FpE::from(5), FpE::from(&MERSENNE_31_PRIME_FIELD_ORDER)]);
        let expected_result = Fp2E::new([FpE::zero(), FpE::zero()]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_fp2_mul() {
        let a = Fp2E::new([FpE::from(12), FpE::from(5)]);
        let b = Fp2E::new([-FpE::from(4), FpE::from(2)]);
        let expected_result = Fp2E::new([-FpE::from(58), FpE::new(4)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_fp2_mul_2() {
        let a = Fp2E::new([FpE::one(), FpE::zero()]);
        let b = Fp2E::new([FpE::from(12), -FpE::from(8)]);
        let expected_result = Fp2E::new([FpE::from(12), -FpE::new(8)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_fp2_mul_3() {
        let a = Fp2E::new([FpE::zero(), FpE::zero()]);
        let b = Fp2E::new([FpE::from(2), FpE::from(7)]);
        let expected_result = Fp2E::new([FpE::zero(), FpE::zero()]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_fp2_mul_4() {
        let a = Fp2E::new([FpE::from(2), FpE::from(7)]);
        let b = Fp2E::new([FpE::zero(), FpE::zero()]);
        let expected_result = Fp2E::new([FpE::zero(), FpE::zero()]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_fp2_mul_5() {
        let a = Fp2E::new([FpE::from(&MERSENNE_31_PRIME_FIELD_ORDER), FpE::one()]);
        let b = Fp2E::new([FpE::from(2), FpE::from(&MERSENNE_31_PRIME_FIELD_ORDER)]);
        let expected_result = Fp2E::new([FpE::zero(), FpE::from(2)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_fp2_inv() {
        let a = Fp2E::new([FpE::one(), FpE::zero()]);
        let expected_result = Fp2E::new([FpE::one(), FpE::zero()]);
        assert_eq!(a.inv().unwrap(), expected_result);
    }

    #[test]
    fn test_fp2_inv_2() {
        let a = Fp2E::new([FpE::from(&(MERSENNE_31_PRIME_FIELD_ORDER - 1)), FpE::one()]);
        let expected_result = Fp2E::new([FpE::from(1073741823), FpE::from(1073741823)]);
        assert_eq!(a.inv().unwrap(), expected_result);
    }

    #[test]
    fn test_fp2_inv_3() {
        let a = Fp2E::new([FpE::from(2063384121), FpE::from(1232183486)]);
        let expected_result = Fp2E::new([FpE::from(1244288232), FpE::from(1321511038)]);
        assert_eq!(a.inv().unwrap(), expected_result);
    }

    #[test]
    fn test_fp2_mul_inv() {
        let a = Fp2E::new([FpE::from(12), FpE::from(5)]);
        let b = a.inv().unwrap();
        let expected_result = Fp2E::new([FpE::one(), FpE::zero()]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_fp2_div() {
        let a = Fp2E::new([FpE::from(12), FpE::from(5)]);
        let b = Fp2E::new([FpE::from(4), FpE::from(2)]);
        let expected_result = Fp2E::new([FpE::from(644245097), FpE::from(1288490188)]);
        assert_eq!((a / b).unwrap(), expected_result);
    }

    #[test]
    fn test_fp2_div_2() {
        let a = Fp2E::new([FpE::from(4), FpE::from(7)]);
        let b = Fp2E::new([FpE::one(), FpE::zero()]);
        let expected_result = Fp2E::new([FpE::from(4), FpE::from(7)]);
        assert_eq!((a / b).unwrap(), expected_result);
    }

    #[test]
    fn test_fp2_div_3() {
        let a = Fp2E::new([FpE::zero(), FpE::zero()]);
        let b = Fp2E::new([FpE::from(3), FpE::from(12)]);
        let expected_result = Fp2E::new([FpE::zero(), FpE::zero()]);
        assert_eq!((a / b).unwrap(), expected_result);
    }

    #[test]
    fn mul_fp4_by_zero_is_zero() {
        let a = Fp4E::new([
            Fp2E::new([FpE::from(2), FpE::from(3)]),
            Fp2E::new([FpE::from(4), FpE::from(5)]),
        ]);
        assert_eq!(Fp4E::zero(), a * Fp4E::zero())
    }

    #[test]
    fn mul_fp4_by_one_is_identity() {
        let a = Fp4E::new([
            Fp2E::new([FpE::from(2), FpE::from(3)]),
            Fp2E::new([FpE::from(4), FpE::from(5)]),
        ]);
        assert_eq!(a, a.clone() * Fp4E::one())
    }

    #[test]
    fn square_fp4_equals_mul_two_times() {
        let a = Fp4E::new([
            Fp2E::new([FpE::from(3), FpE::from(4)]),
            Fp2E::new([FpE::from(5), FpE::from(6)]),
        ]);

        assert_eq!(a.square(), &a * &a)
    }

    #[test]
    fn fp4_mul_by_inv_is_one() {
        let a = Fp4E::new([
            Fp2E::new([FpE::from(2147483647), FpE::from(2147483648)]),
            Fp2E::new([FpE::from(2147483649), FpE::from(2147483650)]),
        ]);

        assert_eq!(&a * a.inv().unwrap(), Fp4E::one())
    }

    #[test]
    fn embed_fp_with_fp4() {
        let a = FpE::from(3);
        let a_extension = Fp4E::from(3);
        assert_eq!(a.to_extension::<Degree4ExtensionField>(), a_extension);
    }

    #[test]
    fn add_fp_and_fp4() {
        let a = FpE::from(3);
        let a_extension = Fp4E::from(3);
        let b = Fp4E::from(2);
        assert_eq!(a + &b, a_extension + b);
    }

    #[test]
    fn mul_fp_by_fp4() {
        let a = FpE::from(30000000000);
        let a_extension = a.to_extension::<Degree4ExtensionField>();
        let b = Fp4E::new([
            Fp2E::new([FpE::from(1), FpE::from(2)]),
            Fp2E::new([FpE::from(3), FpE::from(4)]),
        ]);
        assert_eq!(a * &b, a_extension * b);
    }
}
