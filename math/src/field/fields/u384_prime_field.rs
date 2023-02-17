use crate::unsigned_integer::element::U384;
use crate::{
    field::traits::IsField, unsigned_integer::element::UnsignedInteger,
    unsigned_integer::montgomery::MontgomeryAlgorithms,
};
use std::fmt::Debug;
use std::marker::PhantomData;
use crate::field::errors::FieldError;

/// This trait is necessary for us to be able to use unsigned integer types bigger than
/// `u128` (the biggest native `unit`) as constant generics.
/// This trait should be removed when Rust supports this feature.
pub trait IsMontgomeryConfiguration {
    const MODULUS: U384;
    const R2: U384;
    const MP: u64;
}

#[derive(Clone, Debug)]
pub struct MontgomeryBackendPrimeField<C> {
    phantom: PhantomData<C>,
}

impl<C> MontgomeryBackendPrimeField<C>
where
    C: IsMontgomeryConfiguration,
{
    const ZERO: U384 = UnsignedInteger::from_u64(0);
}

impl<C> IsField for MontgomeryBackendPrimeField<C>
where
    C: IsMontgomeryConfiguration + Clone + Debug,
{
    type BaseType = U384;

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let (sum, overflow) = UnsignedInteger::add(a, b);
        if !overflow {
            if sum < C::MODULUS {
                sum
            } else {
                sum - C::MODULUS
            }
        } else {
            let (diff, _) = UnsignedInteger::sub(&sum, &C::MODULUS);
            diff
        }
    }

    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        MontgomeryAlgorithms::cios(a, b, &C::MODULUS, &C::MP)
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        if b <= a {
            a - b
        } else {
            C::MODULUS - (b - a)
        }
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        if a == &Self::ZERO {
            *a
        } else {
            C::MODULUS - a
        }
    }

    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        if a == &Self::ZERO {
            return Err(FieldError::DivisionByZero);
        }
        Ok(Self::pow(a, C::MODULUS - Self::BaseType::from_u64(2)))
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let inv_b = Self::inv(b).expect("Division by zero!");
        Self::mul(a, &inv_b)
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a == b
    }

    fn zero() -> Self::BaseType {
        Self::ZERO
    }

    fn one() -> Self::BaseType {
        Self::from_u64(1)
    }

    fn from_u64(x: u64) -> Self::BaseType {
        MontgomeryAlgorithms::cios(&UnsignedInteger::from_u64(x), &C::R2, &C::MODULUS, &C::MP)
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        MontgomeryAlgorithms::cios(&x, &C::R2, &C::MODULUS, &C::MP)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        field::element::FieldElement,
        unsigned_integer::element::{UnsignedInteger, U384},
        field::errors::FieldError,
    };

    use super::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField};

    // F23
    #[derive(Clone, Debug)]
    struct MontgomeryConfig23;
    impl IsMontgomeryConfiguration for MontgomeryConfig23 {
        const MODULUS: U384 = UnsignedInteger::from_u64(23);
        const MP: u64 = 3208129404123400281;
        const R2: U384 = UnsignedInteger::from_u64(6);
    }

    type F23 = MontgomeryBackendPrimeField<MontgomeryConfig23>;
    type F23Element = FieldElement<F23>;

    #[test]
    fn montgomery_backend_multiplication_works_0() {
        let x = F23Element::from(11_u64);
        let y = F23Element::from(10_u64);
        let c = F23Element::from(110_u64);
        assert_eq!(x * y, c);
    }

    const ORDER: usize = 23;
    #[test]
    fn two_plus_one_is_three() {
        assert_eq!(
            F23Element::from(2) + F23Element::from(1),
            F23Element::from(3)
        );
    }

    #[test]
    fn max_order_plus_1_is_0() {
        assert_eq!(
            F23Element::from((ORDER - 1) as u64) + F23Element::from(1),
            F23Element::from(0)
        );
    }

    #[test]
    fn when_comparing_13_and_13_they_are_equal() {
        let a: F23Element = F23Element::from(13);
        let b: F23Element = F23Element::from(13);
        assert_eq!(a, b);
    }

    #[test]
    fn when_comparing_13_and_8_they_are_different() {
        let a: F23Element = F23Element::from(13);
        let b: F23Element = F23Element::from(8);
        assert_ne!(a, b);
    }

    #[test]
    fn mul_neutral_element() {
        let a: F23Element = F23Element::from(1);
        let b: F23Element = F23Element::from(2);
        assert_eq!(a * b, F23Element::from(2));
    }

    #[test]
    fn mul_2_3_is_6() {
        let a: F23Element = F23Element::from(2);
        let b: F23Element = F23Element::from(3);
        assert_eq!(a * b, F23Element::from(6));
    }

    #[test]
    fn mul_order_minus_1() {
        let a: F23Element = F23Element::from((ORDER - 1) as u64);
        let b: F23Element = F23Element::from((ORDER - 1) as u64);
        assert_eq!(a * b, F23Element::from(1));
    }

    #[test]
    fn inv_0_error() {
        let result = F23Element::from(0).inv();
        assert!(matches!(result, Err(FieldError::DivisionByZero)));
    }

    #[test]
    fn inv_2() {
        let a: F23Element = F23Element::from(2);
        assert_eq!(&a * a.inv().unwrap(), F23Element::from(1));
    }

    #[test]
    fn pow_2_3() {
        assert_eq!(F23Element::from(2).pow(3_u64), F23Element::from(8))
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(F23Element::from(2).pow(ORDER - 1), F23Element::from(1))
    }

    #[test]
    fn div_1() {
        let expected = F23Element::from(2);
        let actual = F23Element::from(2) / F23Element::from(1);

        assert_eq!(actual, expected);
    }

    #[test]
    #[should_panic]
    fn div_2_0() {
        let _actual = F23Element::from(2) / F23Element::from(0);
    }


    #[test]
    fn div_4_2() {
        let expected = F23Element::from(2);
        let actual = F23Element::from(4) / F23Element::from(2);

        assert_eq!(actual, expected);
    }

    #[test]
    fn div_4_3() {
        let a = F23Element::from(4);
        let b = F23Element::from(3);
        let c = F23Element::from(3);
        let actual_result = a / b;
        assert_eq!(
            actual_result * c,
            F23Element::from(4)
        )
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = F23Element::from(2);

        assert_eq!(&two + (-&two), F23Element::from(0))
    }

    #[test]
    fn four_minus_three_is_1() {
        let four = F23Element::from(4);
        let three = F23Element::from(3);

        assert_eq!(four - three, F23Element::from(1))
    }

    #[test]
    fn zero_minus_1_is_order_minus_1() {
        let zero = F23Element::from(0);
        let one = F23Element::from(1);

        assert_eq!(zero - one, F23Element::from((ORDER - 1) as u64))
    }

    #[test]
    fn neg_zero_is_zero() {
        let zero = F23Element::from(0);

        assert_eq!(-&zero, zero);
    }

    // FP1
    #[derive(Clone, Debug)]
    struct MontgomeryConfigP1;
    impl IsMontgomeryConfiguration for MontgomeryConfigP1 {
        const MODULUS: U384 = UnsignedInteger {
            limbs: [
                0,
                0,
                0,
                3450888597,
                5754816256417943771,
                15923941673896418529,
            ],
        };
        const MP: u64 = 16085280245840369887;
        const R2: U384 = UnsignedInteger {
            limbs: [0, 0, 0, 362264696, 173086217205162856, 7848132598488868435],
        };
    }

    #[test]
    fn montgomery_prime_field_addition_works_0() {
        let x = FP1Element::new(UnsignedInteger::from(
            "05ed176deb0e80b4deb7718cdaa075165f149c",
        ));
        let y = FP1Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        let c = FP1Element::new(UnsignedInteger::from(
            "64fd5279bf47fe02d4185ce279d8aa55e00352",
        ));
        assert_eq!(x + y, c);
    }

    type FP1 = MontgomeryBackendPrimeField<MontgomeryConfigP1>;
    type FP1Element = FieldElement<FP1>;
    #[test]
    fn montgomery_prime_field_multiplication_works_0() {
        let x = FP1Element::new(UnsignedInteger::from(
            "05ed176deb0e80b4deb7718cdaa075165f149c",
        ));
        let y = FP1Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        let c = FP1Element::new(UnsignedInteger::from(
            "73d23e8d462060dc23d5c15c00fc432d95621a3c",
        ));
        assert_eq!(x * y, c);
    }

    // FP2
    #[derive(Clone, Debug)]
    struct MontgomeryConfigP2;
    impl IsMontgomeryConfiguration for MontgomeryConfigP2 {
        const MODULUS: U384 = UnsignedInteger {
            limbs: [
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551275,
            ],
        };
        const MP: u64 = 14984598558409225213;
        const R2: U384 = UnsignedInteger {
            limbs: [0, 0, 0, 0, 0, 116281],
        };
    }

    type FP2 = MontgomeryBackendPrimeField<MontgomeryConfigP2>;
    type FP2Element = FieldElement<FP2>;

    #[test]
    fn montgomery_prime_field_addition_works_1() {
        let x = FP2Element::new(UnsignedInteger::from(
            "05ed176deb0e80b4deb7718cdaa075165f149c",
        ));
        let y = FP2Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        let c = FP2Element::new(UnsignedInteger::from(
            "64fd5279bf47fe02d4185ce279d8aa55e00352",
        ));
        assert_eq!(x + y, c);
    }

    #[test]
    fn montgomery_prime_field_multiplication_works_1() {
        let x = FP2Element::one();
        let y = FP2Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        assert_eq!(&y * x, y);
    }
}
