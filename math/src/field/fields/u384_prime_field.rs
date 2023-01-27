use crypto_bigint::{NonZero, Wrapping, U384};
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::field::element::FieldElement;
use crate::field::traits::HasFieldOperations;

pub trait HasU384Constant: Debug + Clone + Eq + PartialEq {
    const VALUE: U384;
}

/// Type representing prime fields over unsigned 64-bit integers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct U384PrimeField<MODULO: HasU384Constant> {
    phantom: PhantomData<MODULO>,
}

pub type U384FieldElement<ORDER> = FieldElement<U384PrimeField<ORDER>>;

impl<MODULO: HasU384Constant> HasFieldOperations for U384PrimeField<MODULO> {
    type BaseType = U384;

    fn add(a: &U384, b: &U384) -> U384 {
        a.add_mod(b, &MODULO::VALUE)
    }

    fn sub(a: &U384, b: &U384) -> U384 {
        a.sub_mod(b, &MODULO::VALUE)
    }

    fn neg(a: &U384) -> U384 {
        a.neg_mod(&MODULO::VALUE)
    }

    fn mul(a: &U384, b: &U384) -> U384 {
        (Wrapping(*a) * Wrapping(*b)).0 % NonZero::new(MODULO::VALUE).unwrap()
    }

    fn div(a: &U384, b: &U384) -> U384 {
        Self::mul(a, &Self::inv(b))
    }

    fn inv(a: &U384) -> U384 {
        assert_ne!(*a, U384::from_u16(0), "Cannot invert zero element");
        let exponent = Wrapping(MODULO::VALUE) - Wrapping(U384::from_u16(2));
        Self::pow(a, exponent.0)
    }

    fn eq(a: &U384, b: &U384) -> bool {
        Self::from_base_type(*a) == Self::from_base_type(*b)
    }

    fn zero() -> U384 {
        U384::from_u16(0)
    }

    fn one() -> U384 {
        U384::from_u16(1)
    }

    fn from_u64(x: u64) -> U384 {
        U384::from_u64(x) % NonZero::new(MODULO::VALUE).unwrap()
    }

    fn from_base_type(x: U384) -> U384 {
        x % NonZero::new(MODULO::VALUE).unwrap()
    }
}

impl<ORDER: HasU384Constant> Copy for U384FieldElement<ORDER> {}

#[cfg(test)]
mod tests {
    use super::*;

    const ORDER: u16 = 13;

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct Mod13;
    impl HasU384Constant for Mod13 {
        const VALUE: U384 = U384::from_u16(ORDER);
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
        assert_eq!(a * a.inv(), FE::from(1));
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

        assert_eq!(two + (-two), FE::from(0))
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

        assert_eq!(-zero, zero);
    }
}
