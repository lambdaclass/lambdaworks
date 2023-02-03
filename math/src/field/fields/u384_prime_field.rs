use std::fmt::Debug;
use std::marker::PhantomData;

use crate::field::element::FieldElement;
use crate::field::traits::IsField;
use crate::unsigned_integer::UnsignedInteger384 as U384;

pub trait HasU384Constant: Debug + Clone + Eq + PartialEq {
    const VALUE: U384;
}

/// Type representing prime fields over unsigned 64-bit integers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct U384PrimeField<MODULO: HasU384Constant> {
    phantom: PhantomData<MODULO>,
}

pub type U384FieldElement<ORDER> = FieldElement<U384PrimeField<ORDER>>;

impl<MODULO: HasU384Constant> IsField for U384PrimeField<MODULO> {
    type BaseType = U384;

    fn add(a: &U384, b: &U384) -> U384 {
        U384::add_mod(a, b, &MODULO::VALUE)
    }

    fn sub(a: &U384, b: &U384) -> U384 {
        U384::sub_mod(a, b, &MODULO::VALUE)
    }

    fn neg(a: &U384) -> U384 {
        U384::neg_mod(a, &MODULO::VALUE)
    }

    fn mul(a: &U384, b: &U384) -> U384 {
        U384::mul_mod(a, b, &MODULO::VALUE)
    }

    fn div(a: &U384, b: &U384) -> U384 {
        Self::mul(a, &Self::inv(b))
    }

    fn inv(a: &U384) -> U384 {
        assert_ne!(*a, U384::from(0_u8), "Cannot invert zero element");
        let exponent = MODULO::VALUE - U384::from(2_u8);
        Self::pow(a, exponent)
    }

    fn eq(a: &U384, b: &U384) -> bool {
        Self::from_base_type(*a) == Self::from_base_type(*b)
    }

    fn zero() -> U384 {
        U384::from(0_u8)
    }

    fn one() -> U384 {
        U384::from(1_u8)
    }

    fn from_u64(x: u64) -> U384 {
        U384::from(x) % MODULO::VALUE
    }

    fn from_base_type(x: U384) -> U384 {
        x % MODULO::VALUE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORDER: u16 = 13;

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct Mod13;
    impl HasU384Constant for Mod13 {
        const VALUE: U384 = U384::from_const("00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000d");
    }

    type FE = FieldElement<U384PrimeField<Mod13>>;

    #[test]
    fn two_plus_one_is_three() {
        assert_eq!(FE::from(2) + FE::from(1), FE::from(3));
    }

    #[test]
    fn max_order_plus_1_is_0() {
        assert_eq!(FE::from((ORDER - 1) as u64) + FE::from(1), FE::from(0));
    }

    #[test]
    fn when_comparing_13_and_13_they_are_equal() {
        let a: FE = FE::from(13);
        let b: FE = FE::from(13);
        assert_eq!(a, b);
    }

    #[test]
    fn when_comparing_13_and_8_they_are_different() {
        let a: FE = FE::from(13);
        let b: FE = FE::from(8);
        assert_ne!(a, b);
    }

    #[test]
    fn mul_neutral_element() {
        let a: FE = FE::from(1);
        let b: FE = FE::from(2);
        assert_eq!(a * b, FE::from(2));
    }

    #[test]
    fn mul_2_3_is_6() {
        let a: FE = FE::from(2);
        let b: FE = FE::from(3);
        assert_eq!(a * b, FE::from(6));
    }

    #[test]
    fn mul_order_minus_1() {
        let a: FE = FE::from((ORDER - 1) as u64);
        let b: FE = FE::from((ORDER - 1) as u64);
        assert_eq!(a * b, FE::from(1));
    }

    #[test]
    #[should_panic]
    fn inv_0_error() {
        FE::from(0).inv();
    }

    #[test]
    fn inv_2() {
        let a: FE = FE::from(2);
        assert_eq!(&a * a.inv(), FE::from(1));
    }

    #[test]
    fn pow_2_3() {
        assert_eq!(FE::from(2).pow(3_u64), FE::from(8))
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(FE::from(2).pow(ORDER - 1), FE::from(1))
    }

    #[test]
    fn div_1() {
        assert_eq!(FE::from(2) / FE::from(1), FE::from(2))
    }

    #[test]
    fn div_4_2() {
        assert_eq!(FE::from(4) / FE::from(2), FE::from(2))
    }

    #[test]
    fn div_4_3() {
        assert_eq!(FE::from(4) / FE::from(3) * FE::from(3), FE::from(4))
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = FE::from(2);

        assert_eq!(&two + (-&two), FE::from(0))
    }

    #[test]
    fn four_minus_three_is_1() {
        let four = FE::from(4);
        let three = FE::from(3);

        assert_eq!(four - three, FE::from(1))
    }

    #[test]
    fn zero_minus_1_is_order_minus_1() {
        let zero = FE::from(0);
        let one = FE::from(1);

        assert_eq!(zero - one, FE::from((ORDER - 1) as u64))
    }

    #[test]
    fn neg_zero_is_zero() {
        let zero = FE::from(0);

        assert_eq!(-&zero, zero);
    }

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct ModP;
    impl HasU384Constant for ModP {
        const VALUE: U384 = U384::from_const("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");
    }

    #[test]
    fn test_big_numbers() {
        let y = FieldElement::<U384PrimeField<ModP>>::new(U384::from("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0"));
        let y2 = FieldElement::<U384PrimeField<ModP>>::new(U384::from("955318e3b9b4e806ba0ac178662fc6879f48785d418ae2bbe861c23d09f0e60701061e29fbecfa8d8780679f44dadda"));
        assert_eq!(y.pow(2_u64), y2);
    }
}
