use crate::cyclic_group::IsCyclicBilinearGroup;
use crate::field::element::FieldElement;
use crate::field::traits::{HasFieldOperations, IsLinearField};

/// Type representing prime fields over unsigned 64-bit integers.
#[derive(Debug, Clone)]
pub struct U64PrimeField<const MODULO: u64>;
pub type U64FieldElement<const ORDER: u64> = FieldElement<U64PrimeField<ORDER>>;

impl<const MODULO: u64> HasFieldOperations for U64PrimeField<MODULO> {
    type BaseType = u64;

    fn add(a: &u64, b: &u64) -> u64 {
        ((*a as u128 + *b as u128) % MODULO as u128) as u64
    }

    fn sub(a: &u64, b: &u64) -> u64 {
        (((*a as u128 + MODULO as u128) - *b as u128) % MODULO as u128) as u64
    }

    fn neg(a: &u64) -> u64 {
        MODULO - a
    }

    fn mul(a: &u64, b: &u64) -> u64 {
        ((*a as u128 * *b as u128) % MODULO as u128) as u64
    }

    fn div(a: &u64, b: &u64) -> u64 {
        Self::mul(a, &Self::inv(b))
    }

    fn inv(a: &u64) -> u64 {
        assert_ne!(*a, 0, "Cannot invert zero element");
        Self::pow(a, (MODULO - 2) as u128)
    }

    fn eq(a: &u64, b: &u64) -> bool {
        Self::from_u64(*a) == Self::from_u64(*b)
    }

    fn zero() -> u64 {
        0
    }

    fn one() -> u64 {
        1
    }

    fn from_u64(x: u64) -> u64 {
        x % MODULO
    }

    fn from_base_type(x: u64) -> u64 {
        Self::from_u64(x)
    }
}

impl<const MODULO: u64> IsLinearField for U64PrimeField<MODULO> {
    fn bit_size() -> usize {
        (Self::BaseType::BITS - (MODULO - 1).leading_zeros()) as usize
    }

    fn shr(a: &Self::BaseType, n: usize) -> Self::BaseType {
        a >> n
    }

    fn shl(a: &Self::BaseType, n: usize) -> Self::BaseType {
        a << n
    }

    fn and(a: &Self::BaseType, mask: usize) -> usize {
        (*a as usize) & mask
    }
}

impl<const ORDER: u64> Copy for U64FieldElement<ORDER> {}

/// Represents an element in Fp. (E.g: 0, 1, 2 are the elements of F3)
impl<const ORDER: u64> IsCyclicBilinearGroup for U64FieldElement<ORDER> {
    type PairingOutput = Self;

    fn generator() -> U64FieldElement<ORDER> {
        U64FieldElement::one()
    }

    fn neutral_element() -> U64FieldElement<ORDER> {
        U64FieldElement::zero()
    }

    fn operate_with_self(&self, times: u128) -> Self {
        U64FieldElement::from(&(times as u64)) * *self
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
    use test_case::test_case;
    const ORDER: u64 = 13;
    type FE = FieldElement<U64PrimeField<ORDER>>;

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
    #[should_panic]
    fn inv_0_error() {
        FE::new(0).inv();
    }

    #[test]
    fn inv_2() {
        let a: FE = FE::new(2);
        assert_eq!(a * a.inv(), FE::new(1));
    }

    #[test]
    fn pow_2_3() {
        assert_eq!(FE::new(2).pow(3), FE::new(8))
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(FE::new(2).pow((ORDER - 1) as u128), FE::new(1))
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

    #[test_case(8.into(), 1, 4.into())]
    #[test_case(8.into(), 2, 2.into())]
    #[test_case(8.into(), 3, 1.into())]
    #[test_case(8.into(), 4, 0.into())]
    #[test_case(17.into(), 1, 8.into())]
    fn shr(
        f: FieldElement<U64PrimeField<32>>,
        shift: usize,
        expected: FieldElement<U64PrimeField<32>>,
    ) {
        assert_eq!(f >> shift, expected)
    }

    #[test_case(8.into(), 1, 16.into())]
    #[test_case(8.into(), 2, 32.into())]
    #[test_case(8.into(), 3, 64.into())]
    #[test_case(8.into(), 4, 128.into())]
    #[test_case(17.into(), 1, 34.into())]
    fn shl(
        f: FieldElement<U64PrimeField<32>>,
        shift: usize,
        expected: FieldElement<U64PrimeField<32>>,
    ) {
        assert_eq!(f << shift, expected)
    }

    #[test_case(0b000010.into(), 0b000011, 0b000010)]
    #[test_case(0b000011.into(), 0b000001, 0b000001)]
    #[test_case(0b111111.into(), 0b101010, 0b101010)]
    #[test_case(0b010101.into(), 0b101010, 0b000000)]
    #[test_case(0b010101.into(), 0b010101, 0b010101)]
    fn and(f: FieldElement<U64PrimeField<64>>, mask: usize, expected: usize) {
        assert_eq!(f & mask, expected)
    }

    #[test]
    fn bit_size() {
        assert_eq!(FieldElement::<U64PrimeField<2>>::bit_size(), 1);
        assert_eq!(FieldElement::<U64PrimeField<3>>::bit_size(), 2);
        assert_eq!(FieldElement::<U64PrimeField<4>>::bit_size(), 2);
        assert_eq!(FieldElement::<U64PrimeField<5>>::bit_size(), 3);
        assert_eq!(FieldElement::<U64PrimeField<42>>::bit_size(), 6);
    }
}
