use std::marker::PhantomData;
use std::fmt::Debug;
use crate::{unsigned_integer::unsigned_integer::{UnsignedInteger, MontgomeryAlgorithms}, field::traits::IsField};

pub trait IsMontgomeryConfiguration<const NUM_LIMBS: usize> {
    const MODULUS: UnsignedInteger<NUM_LIMBS>;
    const R: UnsignedInteger<NUM_LIMBS>;
    const R2: UnsignedInteger<NUM_LIMBS>;
    const MP: u64;
}

#[derive(Clone, Debug)]
pub struct MontgomeryBackendPrimeField<const NUM_LIMBS: usize, C> {
    phantom: PhantomData<C>,
}

impl<const NUM_LIMBS: usize, C> MontgomeryBackendPrimeField<NUM_LIMBS, C>
where
    C: IsMontgomeryConfiguration<NUM_LIMBS>,
{
    const ZERO: UnsignedInteger<NUM_LIMBS> = UnsignedInteger::from_u64(0);
}

impl<const NUM_LIMBS: usize, C> IsField for MontgomeryBackendPrimeField<NUM_LIMBS, C>
where
    C: IsMontgomeryConfiguration<NUM_LIMBS> + Clone + Debug,
{
    type BaseType = UnsignedInteger<NUM_LIMBS>;

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
        C::MODULUS - a
    }

    fn inv(a: &Self::BaseType) -> Self::BaseType {
        Self::pow(a, C::MODULUS - Self::BaseType::from_u64(2))
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        Self::mul(a, &Self::inv(b))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a == b
    }

    fn zero() -> Self::BaseType {
        Self::ZERO
    }

    fn one() -> Self::BaseType {
        C::R
    }

    fn from_u64(x: u64) -> Self::BaseType {
        MontgomeryAlgorithms::cios(
            &UnsignedInteger::from_u64(x),
            &C::R2,
            &C::MODULUS,
            &C::MP,
        )
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        MontgomeryAlgorithms::cios(&x, &C::R2, &C::MODULUS, &C::MP)
    }
}


#[cfg(test)]
mod tests {
    use crate::{unsigned_integer::unsigned_integer::UnsignedInteger, field::element::FieldElement};

    use super::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField};
    const NUM_LIMBS: usize = 6;
    // F23
    #[derive(Clone, Debug)]
    struct MontgomeryConfig23;
    impl IsMontgomeryConfiguration<6> for MontgomeryConfig23 {
        const MODULUS: UnsignedInteger<NUM_LIMBS> = UnsignedInteger::from_u64(23);
        const MP: u64 = 3208129404123400281;
        const R: UnsignedInteger<NUM_LIMBS> = UnsignedInteger::from_u64(12);
        const R2: UnsignedInteger<NUM_LIMBS> = UnsignedInteger::from_u64(6);
    }

    type F23 = MontgomeryBackendPrimeField<6, MontgomeryConfig23>;
    type F23Element = FieldElement<F23>;

    #[test]
    fn montgomery_backend_multiplication_works_0() {
        let x = F23Element::from(11_u64);
        let y = F23Element::from(10_u64);
        let c = F23Element::from(110_u64);
        assert_eq!(x * y, c);
    }

    // FP1
    #[derive(Clone, Debug)]
    struct MontgomeryConfigP1;
    impl IsMontgomeryConfiguration<6> for MontgomeryConfigP1 {
        const MODULUS: UnsignedInteger<NUM_LIMBS> = UnsignedInteger {
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
        const R: UnsignedInteger<NUM_LIMBS> = UnsignedInteger {
            limbs: [0, 0, 0, 1491054817, 12960619100389563983, 4822041506656656691],
        };
        const R2: UnsignedInteger<NUM_LIMBS> = UnsignedInteger {
            limbs: [0, 0, 0, 362264696, 173086217205162856, 7848132598488868435],
        };
    }

    type FP1 = MontgomeryBackendPrimeField<6, MontgomeryConfigP1>;
    type FP1Element = FieldElement<FP1>;
    #[test]
    fn montgomery_backend_multiplication_works_1() {
        let x = FP1Element::new(UnsignedInteger::from("05ed176deb0e80b4deb7718cdaa075165f149c"));
        let y = FP1Element::new(UnsignedInteger::from("5f103b0bd4397d4df560eb559f38353f80eeb6"));
        let c = FP1Element::new(UnsignedInteger::from("73d23e8d462060dc23d5c15c00fc432d95621a3c"));
        assert_eq!(x * y, c);
    }

    #[test]
    fn montgomery_backend_multiplication_works_2() {
        let x = FP1Element::new(UnsignedInteger::from("05ed176deb0e80b4deb7718cdaa075165f149c"));
        let y = FP1Element::new(UnsignedInteger::from("5f103b0bd4397d4df560eb559f38353f80eeb6"));
        let c = FP1Element::new(UnsignedInteger::from("64fd5279bf47fe02d4185ce279d8aa55e00352"));
        assert_eq!(x + y, c);
    }

    // FP2
    #[derive(Clone, Debug)]
    struct MontgomeryConfigP2;
    impl IsMontgomeryConfiguration<6> for MontgomeryConfigP2 {
        const MODULUS: UnsignedInteger<NUM_LIMBS> = UnsignedInteger {
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
        const R: UnsignedInteger<NUM_LIMBS> = UnsignedInteger {
            limbs: [0, 0, 0, 0, 0, 341],
        };
        const R2: UnsignedInteger<NUM_LIMBS> = UnsignedInteger {
            limbs: [0, 0, 0, 0, 0, 116281],
        };
    }


    type FP2 = MontgomeryBackendPrimeField<6, MontgomeryConfigP2>;
    type FP2Element = FieldElement<FP2>;
    #[test]
    fn montgomery_backend_addition_works_1() {
        let x = FP2Element::new(UnsignedInteger::from("05ed176deb0e80b4deb7718cdaa075165f149c"));
        let y = FP2Element::new(UnsignedInteger::from("5f103b0bd4397d4df560eb559f38353f80eeb6"));
        let c = FP2Element::new(UnsignedInteger::from("64fd5279bf47fe02d4185ce279d8aa55e00352"));
        assert_eq!(x + y, c);
    }

    #[test]
    fn montgomery_backend_multiplication_works_4() {
        let x = FP2Element::one();
        let y = FP2Element::new(UnsignedInteger::from("5f103b0bd4397d4df560eb559f38353f80eeb6"));
        assert_eq!(&y * x, y);
    }
}

/*
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
*/