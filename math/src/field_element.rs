use super::cyclic_group::CyclicBilinearGroup;
use rand::prelude::*;
use std::ops;

#[derive(Debug, PartialEq, Eq)]
pub enum FieldElementError {
    OutOfRangeValue,
    DivisionByZero,
}

/// Represents an element in Fp. (E.g: 0, 1, 2 are the elements of F3)
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FieldElement<const ORDER: u128> {
    value: u128,
}

impl<const ORDER: u128> FieldElement<ORDER> {
    /// Creates a new field element with `value` modulo order of the field
    pub fn new(value: u128) -> Self {
        Self {
            value: value % ORDER,
        }
    }

    /// Returns a representative for a field element.
    /// E.g.: 8, 5 and 29 in F3 are all represented by 2.
    pub fn representative(&self) -> u128 {
        self.value
    }

    /// Returns a random element from the field.
    pub fn random() -> Self {
        let value: u128 = rand::thread_rng().gen_range(1..ORDER);
        FieldElement { value }
    }

    /// Returns `self` to the power of `exponent` using
    /// right-to-left binary method for modular exponentiation.
    pub fn pow(self, mut exponent: u128) -> Self {
        let mut result = Self::new(1);
        let mut base = self;

        while exponent > 0 {
            // exponent % 2 == 1
            if exponent & 1 == 1 {
                result = result * base;
            }
            // exponent = exponent / 2
            exponent >>= 1;
            base = base * base;
        }
        result
    }

    /// Computes the inverse of the element `self`.
    /// Based on Fermat's little theorem.
    pub fn inv(self) -> Result<Self, FieldElementError> {
        if self.value != 0 {
            Ok(self.pow(ORDER - 2))
        } else {
            Err(FieldElementError::DivisionByZero)
        }
    }
}

impl<const ORDER: u128> ops::Add<FieldElement<ORDER>> for FieldElement<ORDER> {
    type Output = FieldElement<ORDER>;

    fn add(self, a_field_element: FieldElement<ORDER>) -> FieldElement<ORDER> {
        FieldElement::new(self.value + a_field_element.value)
    }
}

impl<const ORDER: u128> ops::AddAssign for FieldElement<ORDER> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<const ORDER: u128> ops::Neg for FieldElement<ORDER> {
    type Output = FieldElement<ORDER>;

    fn neg(self) -> FieldElement<ORDER> {
        FieldElement::new(ORDER - self.value)
    }
}

impl<const ORDER: u128> ops::Sub<FieldElement<ORDER>> for FieldElement<ORDER> {
    type Output = FieldElement<ORDER>;

    fn sub(self, substrahend: FieldElement<ORDER>) -> FieldElement<ORDER> {
        let neg_substrahend = -substrahend;
        self + neg_substrahend
    }
}

impl<const ORDER: u128> ops::Mul for FieldElement<ORDER> {
    type Output = FieldElement<ORDER>;

    fn mul(self, a_field_element: Self) -> Self {
        FieldElement::new(self.value * a_field_element.value)
    }
}

impl<const ORDER: u128> ops::Div for FieldElement<ORDER> {
    type Output = FieldElement<ORDER>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, dividend: Self) -> Self {
        self * dividend.inv().unwrap()
    }
}

impl<const ORDER: u128> CyclicBilinearGroup for FieldElement<ORDER> {
    type PairingOutput = Self;

    fn generator() -> FieldElement<ORDER> {
        FieldElement::new(1)
    }

    fn neutral_element() -> FieldElement<ORDER> {
        FieldElement::new(0)
    }

    fn operate_with_self(&self, times: u128) -> Self {
        FieldElement::new(times) * *self
    }

    fn pairing(&self, other: &Self) -> Self {
        *self * *other
    }

    fn operate_with(&self, other: &Self) -> Self {
        *self + *other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const ORDER: u128 = 13;
    type FE = FieldElement<ORDER>;

    #[test]
    fn order_must_small_as_to_not_allow_overflows() {
        // ORDER*ORDER < u128::MAX
        assert!(ORDER <= u64::MAX.into());
    }

    #[test]
    fn two_plus_one_is_three() {
        assert_eq!(FE::new(2) + FE::new(1), FE::new(3));
    }

    #[test]
    fn max_order_plus_1_is_0() {
        assert_eq!(FE::new(ORDER - 1) + FE::new(1), FE::new(0));
    }

    #[test]
    fn when_comparing_13_and_13_they_are_equal() {
        let a: FE = FE::new(13);
        let b: FE = FE::new(13);
        assert_eq!(a, b);
    }

    #[test]
    fn when_comparing_13_and_8_they_are_different() {
        let a: FE = FE::new(13);
        let b: FE = FE::new(8);
        assert_ne!(a, b);
    }

    #[test]
    fn mul_neutral_element() {
        let a: FE = FE::new(1);
        let b: FE = FE::new(2);
        assert_eq!(a * b, FE::new(2));
    }

    #[test]
    fn mul_2_3_is_6() {
        let a: FE = FE::new(2);
        let b: FE = FE::new(3);
        assert_eq!(a * b, FE::new(6));
    }

    #[test]
    fn mul_order_minus_1() {
        let a: FE = FE::new(ORDER - 1);
        let b: FE = FE::new(ORDER - 1);
        assert_eq!(a * b, FE::new(1));
    }

    #[test]
    fn inv_0_error() {
        let a: FE = FE::new(0);
        assert_eq!(a.inv().unwrap_err(), FieldElementError::DivisionByZero);
    }

    #[test]
    fn inv_2() {
        let a: FE = FE::new(2);
        assert_eq!(a * a.inv().unwrap(), FE::new(1));
    }

    #[test]
    fn pow_2_3() {
        assert_eq!(FE::new(2).pow(3), FE::new(8))
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(FE::new(2).pow(ORDER - 1), FE::new(1))
    }

    #[test]
    fn div_1() {
        assert_eq!(FE::new(2) / FE::new(1), FE::new(2))
    }

    #[test]
    fn div_4_2() {
        assert_eq!(FE::new(4) / FE::new(2), FE::new(2))
    }

    #[test]
    fn div_4_3() {
        assert_eq!(FE::new(4) / FE::new(3) * FE::new(3), FE::new(4))
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = FE::new(2);

        assert_eq!(two + (-two), FE::new(0))
    }

    #[test]
    fn four_minus_three_is_1() {
        let four = FE::new(4);
        let three = FE::new(3);

        assert_eq!(four - three, FE::new(1))
    }

    #[test]
    fn zero_minus_1_is_order_minus_1() {
        let zero = FE::new(0);
        let one = FE::new(1);

        assert_eq!(zero - one, FE::new(ORDER - 1))
    }

    #[test]
    fn neg_zero_is_zero() {
        let zero = FE::new(0);

        assert_eq!(-zero, zero);
    }

    #[test]
    fn zero_constructor_returns_zero() {
        assert_eq!(FE::new(0), FE::new(0));
    }

    #[test]
    fn field_element_as_group_element_generator_returns_one() {
        assert_eq!(FE::generator(), FE::new(1));
    }

    #[test]
    fn field_element_as_group_element_multiplication_by_scalar_works_as_multiplication_in_finite_fields(
    ) {
        let a = FE::new(3);
        let b = FE::new(12);
        assert_eq!(a * b, a.operate_with_self(12));
    }

    #[test]
    fn field_element_as_group_element_pairing_works_as_multiplication_in_finite_fields() {
        let a = FE::new(3);
        let b = FE::new(12);
        assert_eq!(a * b, a.pairing(&b));
    }
}
