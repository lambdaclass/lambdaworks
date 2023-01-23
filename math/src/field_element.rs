use super::algebraic_element::*;
use super::cyclic_group::CyclicBilinearGroup;
use rand::prelude::*;
use std::ops;

#[derive(Debug, Clone)]
pub struct NativeU64Modulus<const MODULO: u64>;

impl<const MODULO: u64> AdditionLaw<u64> for NativeU64Modulus<MODULO> {
    fn add(a: &u64, b: &u64) -> u64 {
        ((*a as u128 + *b as u128) % MODULO as u128) as u64
    }
}

impl<const MODULO: u64> SubtractionLaw<u64> for NativeU64Modulus<MODULO> {
    fn sub(a: &u64, b: &u64) -> u64 {
        (((*a as u128 + MODULO as u128) - *b as u128) % MODULO as u128) as u64
    }
}

impl<const MODULO: u64> NegationLaw<u64> for NativeU64Modulus<MODULO> {
    fn neg(a: &u64) -> u64 {
        MODULO - a
    }
}

impl<const MODULO: u64> MultiplicationLaw<u64> for NativeU64Modulus<MODULO> {
    fn mul(a: &u64, b: &u64) -> u64 {
        ((*a as u128 * *b as u128) % MODULO as u128) as u64
    }
}

impl<const MODULO: u64> DivisionLaw<u64> for NativeU64Modulus<MODULO>
where
    NativeU64Modulus<MODULO>: InversionLaw<u64> + MultiplicationLaw<u64>,
{
    fn div(a: &u64, b: &u64) -> u64 {
        Self::mul(a, &Self::inv(b))
    }
}

impl<const MODULO: u64> PowerLaw<u64> for NativeU64Modulus<MODULO>
where
    NativeU64Modulus<MODULO>: MultiplicationLaw<u64>,
{
    fn pow(a: &u64, mut exponent: u128) -> u64 {
        let mut result = 1;
        let mut base = a.clone();

        while exponent > 0 {
            if exponent & 1 == 1 {
                result = Self::mul(&result, &base);
            }
            exponent >>= 1;
            base = Self::mul(&base, &base);
        }
        result
    }
}

impl<const MODULO: u64> InversionLaw<u64> for NativeU64Modulus<MODULO>
where
    NativeU64Modulus<MODULO>: PowerLaw<u64>,
{
    fn inv(a: &u64) -> u64 {
        assert_ne!(*a, 0, "Cannot invert zero element");
        Self::pow(a, (MODULO - 2) as u128)
    }
}

impl<const MODULO: u64> EqualityLaw<u64> for NativeU64Modulus<MODULO>
where
    NativeU64Modulus<MODULO>: Representative<u64>,
{
    fn eq(a: &u64, b: &u64) -> bool {
        Self::representative(a) == Self::representative(b)
    }
}

impl<const MODULO: u64> Zero<u64> for NativeU64Modulus<MODULO> {
    fn zero() -> u64 {
        0
    }
}

impl<const MODULO: u64> One<u64> for NativeU64Modulus<MODULO> {
    fn one() -> u64 {
        1
    }
}

impl<const MODULO: u64> Representative<u64> for NativeU64Modulus<MODULO> {
    fn representative(a: &u64) -> u64 {
        a % MODULO
    }
}

pub type FieldElement<const ORDER: u64> = AlgebraicElement<u64, NativeU64Modulus<ORDER>>;
impl<const ORDER: u64> Copy for FieldElement<ORDER> {}

/// Represents an element in Fp. (E.g: 0, 1, 2 are the elements of F3)
impl<const ORDER: u64> CyclicBilinearGroup for FieldElement<ORDER> {
    type PairingOutput = Self;

    fn generator() -> FieldElement<ORDER> {
        FieldElement::one()
    }

    fn neutral_element() -> FieldElement<ORDER> {
        FieldElement::zero()
    }

    fn operate_with_self(&self, times: u128) -> Self {
        FieldElement::from(times as u64) * *self
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
    const ORDER: u64 = 13;
    type FE = AlgebraicElement<u64, NativeU64Modulus<ORDER>>;

    #[test]
    fn order_must_small_as_to_not_allow_overflows() {
        // ORDER*ORDER < u128::MAX
        assert!(ORDER <= u64::MAX.into());
    }

    #[test]
    fn two_plus_one_is_three() {
        assert_eq!(FE::from(2) + FE::from(1), FE::from(3));
    }

    #[test]
    fn max_order_plus_1_is_0() {
        assert_eq!(FE::from(ORDER - 1) + FE::from(1), FE::from(0));
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
        let a: FE = FE::from(ORDER - 1);
        let b: FE = FE::from(ORDER - 1);
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
        assert_eq!(FE::from(2).pow(3), FE::from(8))
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(FE::from(2).pow((ORDER - 1) as u128), FE::from(1))
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

        assert_eq!(zero - one, FE::from(ORDER - 1))
    }

    #[test]
    fn neg_zero_is_zero() {
        let zero = FE::from(0);

        assert_eq!(-zero, zero);
    }

    #[test]
    fn zero_constructor_returns_zero() {
        assert_eq!(FE::from(0), FE::from(0));
    }

    #[test]
    fn field_element_as_group_element_generator_returns_one() {
        assert_eq!(FE::generator(), FE::from(1));
    }

    #[test]
    fn field_element_as_group_element_multiplication_by_scalar_works_as_multiplication_in_finite_fields(
    ) {
        let a = FE::from(3);
        let b = FE::from(12);
        assert_eq!(a * b, a.operate_with_self(12));
    }

    #[test]
    fn field_element_as_group_element_pairing_works_as_multiplication_in_finite_fields() {
        let a = FE::from(3);
        let b = FE::from(12);
        assert_eq!(a * b, a.pairing(&b));
    }
}
