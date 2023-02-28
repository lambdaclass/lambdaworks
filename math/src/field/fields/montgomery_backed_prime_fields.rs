use crate::field::element::FieldElement;
use crate::traits::ByteConversion;
use crate::{
    field::traits::IsField, unsigned_integer::element::UnsignedInteger,
    unsigned_integer::montgomery::MontgomeryAlgorithms,
};
use std::fmt::Debug;
use std::marker::PhantomData;

pub type U384PrimeField<C> = MontgomeryBackendPrimeField<C, 6>;
pub type U256PrimeField<C> = MontgomeryBackendPrimeField<C, 4>;

/// This trait is necessary for us to be able to use unsigned integer types bigger than
/// `u128` (the biggest native `unit`) as constant generics.
/// This trait should be removed when Rust supports this feature.

pub trait IsMontgomeryConfiguration<const NUM_LIMBS: usize> {
    const MODULUS: UnsignedInteger<NUM_LIMBS>;
    const R2: UnsignedInteger<NUM_LIMBS>;
    const MP: u64;
}

#[derive(Clone, Debug)]
pub struct MontgomeryBackendPrimeField<C, const NUM_LIMBS: usize> {
    phantom: PhantomData<C>,
}

impl<C, const NUM_LIMBS: usize> MontgomeryBackendPrimeField<C, NUM_LIMBS>
where
    C: IsMontgomeryConfiguration<NUM_LIMBS>,
{
    const ZERO: UnsignedInteger<NUM_LIMBS> = UnsignedInteger::from_u64(0);
}

impl<C, const NUM_LIMBS: usize> IsField for MontgomeryBackendPrimeField<C, NUM_LIMBS>
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
        if a == &Self::ZERO {
            *a
        } else {
            C::MODULUS - a
        }
    }

    fn inv(a: &Self::BaseType) -> Self::BaseType {
        if a == &Self::ZERO {
            panic!("Division by zero error.")
        }
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
        Self::from_u64(1)
    }

    fn from_u64(x: u64) -> Self::BaseType {
        MontgomeryAlgorithms::cios(&UnsignedInteger::from_u64(x), &C::R2, &C::MODULUS, &C::MP)
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        MontgomeryAlgorithms::cios(&x, &C::R2, &C::MODULUS, &C::MP)
    }
}

impl<C, const NUM_LIMBS: usize> ByteConversion
    for FieldElement<MontgomeryBackendPrimeField<C, NUM_LIMBS>>
where
    C: IsMontgomeryConfiguration<NUM_LIMBS> + Clone + Debug,
{
    fn to_bytes_be(&self) -> Vec<u8> {
        MontgomeryAlgorithms::cios(
            self.value(),
            &UnsignedInteger::from_u64(1),
            &C::MODULUS,
            &C::MP,
        )
        .to_bytes_be()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        MontgomeryAlgorithms::cios(
            self.value(),
            &UnsignedInteger::from_u64(1),
            &C::MODULUS,
            &C::MP,
        )
        .to_bytes_le()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let value = UnsignedInteger::from_bytes_be(bytes)?;
        Ok(Self::new(value))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let value = UnsignedInteger::from_bytes_le(bytes)?;
        Ok(Self::new(value))
    }
}

#[cfg(test)]
mod tests_u384_prime_fields {
    use crate::field::element::FieldElement;
    use crate::field::fields::montgomery_backed_prime_fields::{
        IsMontgomeryConfiguration, U384PrimeField,
    };
    use crate::traits::ByteConversion;
    use crate::unsigned_integer::element::UnsignedInteger;
    use crate::unsigned_integer::element::U384;

    #[derive(Clone, Debug)]
    struct U384MontgomeryConfiguration23;
    impl IsMontgomeryConfiguration<6> for U384MontgomeryConfiguration23 {
        const MODULUS: U384 = UnsignedInteger::from_u64(23);
        const MP: u64 = 3208129404123400281;
        const R2: U384 = UnsignedInteger::from_u64(6);
    }

    type U384F23 = U384PrimeField<U384MontgomeryConfiguration23>;
    type U384F23Element = FieldElement<U384F23>;

    #[test]
    fn montgomery_backend_multiplication_works_0() {
        let x = U384F23Element::from(11_u64);
        let y = U384F23Element::from(10_u64);
        let c = U384F23Element::from(110_u64);
        assert_eq!(x * y, c);
    }

    const ORDER: usize = 23;
    #[test]
    fn two_plus_one_is_three() {
        assert_eq!(
            U384F23Element::from(2) + U384F23Element::from(1),
            U384F23Element::from(3)
        );
    }

    #[test]
    fn max_order_plus_1_is_0() {
        assert_eq!(
            U384F23Element::from((ORDER - 1) as u64) + U384F23Element::from(1),
            U384F23Element::from(0)
        );
    }

    #[test]
    fn when_comparing_13_and_13_they_are_equal() {
        let a: U384F23Element = U384F23Element::from(13);
        let b: U384F23Element = U384F23Element::from(13);
        assert_eq!(a, b);
    }

    #[test]
    fn when_comparing_13_and_8_they_are_different() {
        let a: U384F23Element = U384F23Element::from(13);
        let b: U384F23Element = U384F23Element::from(8);
        assert_ne!(a, b);
    }

    #[test]
    fn mul_neutral_element() {
        let a: U384F23Element = U384F23Element::from(1);
        let b: U384F23Element = U384F23Element::from(2);
        assert_eq!(a * b, U384F23Element::from(2));
    }

    #[test]
    fn mul_2_3_is_6() {
        let a: U384F23Element = U384F23Element::from(2);
        let b: U384F23Element = U384F23Element::from(3);
        assert_eq!(a * b, U384F23Element::from(6));
    }

    #[test]
    fn mul_order_minus_1() {
        let a: U384F23Element = U384F23Element::from((ORDER - 1) as u64);
        let b: U384F23Element = U384F23Element::from((ORDER - 1) as u64);
        assert_eq!(a * b, U384F23Element::from(1));
    }

    #[test]
    #[should_panic]
    fn inv_0_error() {
        U384F23Element::from(0).inv();
    }

    #[test]
    fn inv_2() {
        let a: U384F23Element = U384F23Element::from(2);
        assert_eq!(&a * a.inv(), U384F23Element::from(1));
    }

    #[test]
    fn pow_2_3() {
        assert_eq!(U384F23Element::from(2).pow(3_u64), U384F23Element::from(8))
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(
            U384F23Element::from(2).pow(ORDER - 1),
            U384F23Element::from(1)
        )
    }

    #[test]
    fn div_1() {
        assert_eq!(
            U384F23Element::from(2) / U384F23Element::from(1),
            U384F23Element::from(2)
        )
    }

    #[test]
    fn div_4_2() {
        assert_eq!(
            U384F23Element::from(4) / U384F23Element::from(2),
            U384F23Element::from(2)
        )
    }

    #[test]
    fn div_4_3() {
        assert_eq!(
            U384F23Element::from(4) / U384F23Element::from(3) * U384F23Element::from(3),
            U384F23Element::from(4)
        )
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = U384F23Element::from(2);

        assert_eq!(&two + (-&two), U384F23Element::from(0))
    }

    #[test]
    fn four_minus_three_is_1() {
        let four = U384F23Element::from(4);
        let three = U384F23Element::from(3);

        assert_eq!(four - three, U384F23Element::from(1))
    }

    #[test]
    fn zero_minus_1_is_order_minus_1() {
        let zero = U384F23Element::from(0);
        let one = U384F23Element::from(1);

        assert_eq!(zero - one, U384F23Element::from((ORDER - 1) as u64))
    }

    #[test]
    fn neg_zero_is_zero() {
        let zero = U384F23Element::from(0);

        assert_eq!(-&zero, zero);
    }

    // FP1
    #[derive(Clone, Debug)]
    struct U384MontgomeryConfigP1;
    impl IsMontgomeryConfiguration<6> for U384MontgomeryConfigP1 {
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

    type U384FP1 = U384PrimeField<U384MontgomeryConfigP1>;
    type U384FP1Element = FieldElement<U384FP1>;

    #[test]
    fn montgomery_prime_field_addition_works_0() {
        let x = U384FP1Element::new(UnsignedInteger::from(
            "05ed176deb0e80b4deb7718cdaa075165f149c",
        ));
        let y = U384FP1Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        let c = U384FP1Element::new(UnsignedInteger::from(
            "64fd5279bf47fe02d4185ce279d8aa55e00352",
        ));
        assert_eq!(x + y, c);
    }

    #[test]
    fn montgomery_prime_field_multiplication_works_0() {
        let x = U384FP1Element::new(UnsignedInteger::from(
            "05ed176deb0e80b4deb7718cdaa075165f149c",
        ));
        let y = U384FP1Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        let c = U384FP1Element::new(UnsignedInteger::from(
            "73d23e8d462060dc23d5c15c00fc432d95621a3c",
        ));
        assert_eq!(x * y, c);
    }

    // FP2
    #[derive(Clone, Debug)]
    struct U384MontgomeryConfigP2;
    impl IsMontgomeryConfiguration<6> for U384MontgomeryConfigP2 {
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

    type U384FP2 = U384PrimeField<U384MontgomeryConfigP2>;
    type U384FP2Element = FieldElement<U384FP2>;

    #[test]
    fn montgomery_prime_field_addition_works_1() {
        let x = U384FP2Element::new(UnsignedInteger::from(
            "05ed176deb0e80b4deb7718cdaa075165f149c",
        ));
        let y = U384FP2Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        let c = U384FP2Element::new(UnsignedInteger::from(
            "64fd5279bf47fe02d4185ce279d8aa55e00352",
        ));
        assert_eq!(x + y, c);
    }

    #[test]
    fn montgomery_prime_field_multiplication_works_1() {
        let x = U384FP2Element::one();
        let y = U384FP2Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        assert_eq!(&y * x, y);
    }

    #[test]
    fn to_bytes_from_bytes_be_is_the_identity() {
        let x = U384FP2Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        assert_eq!(U384FP2Element::from_bytes_be(&x.to_bytes_be()).unwrap(), x);
    }

    #[test]
    fn from_bytes_to_bytes_be_is_the_identity_for_one() {
        let bytes = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        assert_eq!(
            U384FP2Element::from_bytes_be(&bytes).unwrap().to_bytes_be(),
            bytes
        );
    }

    #[test]
    fn to_bytes_from_bytes_le_is_the_identity() {
        let x = U384FP2Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        assert_eq!(U384FP2Element::from_bytes_le(&x.to_bytes_le()).unwrap(), x);
    }

    #[test]
    fn from_bytes_to_bytes_le_is_the_identity_for_one() {
        let bytes = vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(
            U384FP2Element::from_bytes_le(&bytes).unwrap().to_bytes_le(),
            bytes
        );
    }
}

#[cfg(test)]
mod tests_u256_prime_fields {
    use crate::field::element::FieldElement;
    use crate::field::fields::montgomery_backed_prime_fields::{
        IsMontgomeryConfiguration, U256PrimeField,
    };
    use crate::traits::ByteConversion;
    use crate::unsigned_integer::element::UnsignedInteger;
    use crate::unsigned_integer::element::U256;

    #[derive(Clone, Debug)]
    struct U256MontgomeryConfiguration29;
    impl IsMontgomeryConfiguration<4> for U256MontgomeryConfiguration29 {
        const MODULUS: U256 = UnsignedInteger::from_u64(29);
        const MP: u64 = 14630176334321368523;
        const R2: U256 = UnsignedInteger::from_u64(24);
    }

    type U256F29 = U256PrimeField<U256MontgomeryConfiguration29>;
    type U256F29Element = FieldElement<U256F29>;

    #[test]
    fn montgomery_backend_multiplication_works_0() {
        let x = U256F29Element::from(11_u64);
        let y = U256F29Element::from(10_u64);
        let c = U256F29Element::from(110_u64);
        assert_eq!(x * y, c);
    }

    const ORDER: usize = 29;
    #[test]
    fn two_plus_one_is_three() {
        assert_eq!(
            U256F29Element::from(2) + U256F29Element::from(1),
            U256F29Element::from(3)
        );
    }

    #[test]
    fn max_order_plus_1_is_0() {
        assert_eq!(
            U256F29Element::from((ORDER - 1) as u64) + U256F29Element::from(1),
            U256F29Element::from(0)
        );
    }

    #[test]
    fn when_comparing_13_and_13_they_are_equal() {
        let a: U256F29Element = U256F29Element::from(13);
        let b: U256F29Element = U256F29Element::from(13);
        assert_eq!(a, b);
    }

    #[test]
    fn when_comparing_13_and_8_they_are_different() {
        let a: U256F29Element = U256F29Element::from(13);
        let b: U256F29Element = U256F29Element::from(8);
        assert_ne!(a, b);
    }

    #[test]
    fn mul_neutral_element() {
        let a: U256F29Element = U256F29Element::from(1);
        let b: U256F29Element = U256F29Element::from(2);
        assert_eq!(a * b, U256F29Element::from(2));
    }

    #[test]
    fn mul_2_3_is_6() {
        let a: U256F29Element = U256F29Element::from(2);
        let b: U256F29Element = U256F29Element::from(3);
        assert_eq!(a * b, U256F29Element::from(6));
    }

    #[test]
    fn mul_order_minus_1() {
        let a: U256F29Element = U256F29Element::from((ORDER - 1) as u64);
        let b: U256F29Element = U256F29Element::from((ORDER - 1) as u64);
        assert_eq!(a * b, U256F29Element::from(1));
    }

    #[test]
    #[should_panic]
    fn inv_0_error() {
        U256F29Element::from(0).inv();
    }

    #[test]
    fn inv_2() {
        let a: U256F29Element = U256F29Element::from(2);
        assert_eq!(&a * a.inv(), U256F29Element::from(1));
    }

    #[test]
    fn pow_2_3() {
        assert_eq!(U256F29Element::from(2).pow(3_u64), U256F29Element::from(8))
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(
            U256F29Element::from(2).pow(ORDER - 1),
            U256F29Element::from(1)
        )
    }

    #[test]
    fn div_1() {
        assert_eq!(
            U256F29Element::from(2) / U256F29Element::from(1),
            U256F29Element::from(2)
        )
    }

    #[test]
    fn div_4_2() {
        assert_eq!(
            U256F29Element::from(4) / U256F29Element::from(2),
            U256F29Element::from(2)
        )
    }

    #[test]
    fn div_4_3() {
        assert_eq!(
            U256F29Element::from(4) / U256F29Element::from(3) * U256F29Element::from(3),
            U256F29Element::from(4)
        )
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = U256F29Element::from(2);

        assert_eq!(&two + (-&two), U256F29Element::from(0))
    }

    #[test]
    fn four_minus_three_is_1() {
        let four = U256F29Element::from(4);
        let three = U256F29Element::from(3);

        assert_eq!(four - three, U256F29Element::from(1))
    }

    #[test]
    fn zero_minus_1_is_order_minus_1() {
        let zero = U256F29Element::from(0);
        let one = U256F29Element::from(1);

        assert_eq!(zero - one, U256F29Element::from((ORDER - 1) as u64))
    }

    #[test]
    fn neg_zero_is_zero() {
        let zero = U256F29Element::from(0);

        assert_eq!(-&zero, zero);
    }

    // FP1
    //TODO, buil a test using SAGE

    /*Python helper to work with limbs

    def from_limbs(limbs):
        uint=0
        for i in range(len(limbs)):
            index = len(limbs) - i -1
            uint = uint + (limbs[index] << (64*i))
        return uint

    def from_uint(uint,n_limbs):
        limbs = []
        for i in range(n_limbs):
            index = n_limbs - i -1
            mask = (pow(2,64) -1) << (64*index)
            limbs.append((uint & mask) >> 64*index)
        return limbs
     */
    #[derive(Clone, Debug)]
    struct U256MontgomeryConfigP1;
    impl IsMontgomeryConfiguration<4> for U256MontgomeryConfigP1 {
        const MODULUS: U256 = UnsignedInteger {
            limbs: [
                63,
                3026140964809723730,
                7835788046214001203,
                16962966595042587443,
            ],
            //396487151739502753262805464559750051512474188475143212544819
        };
        const MP: u64 = 8665272125093150725;
        const R2: U256 = UnsignedInteger {
            limbs: [
                7,
                9694576771155704591,
                15518527791430854876,
                15195315649472397251,
            ],
            //[7, 9694576771155704591, 15518527791430854876, 15195315649472397251]
        };
    }

    type U256FP1 = U256PrimeField<U256MontgomeryConfigP1>;
    type U256FP1Element = FieldElement<U256FP1>;

    #[test]
    fn montgomery_prime_field_addition_works_0() {
        let x = U256FP1Element::new(UnsignedInteger::from(
            "05ed176deb0e80b4deb7718cdaa075165f149c",
        ));
        let y = U256FP1Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        let c = U256FP1Element::new(UnsignedInteger::from(
            "64fd5279bf47fe02d4185ce279d8aa55e00352",
        ));
        assert_eq!(x + y, c);
    }

    /*
    MAKE TEST PASS, USE SAGE MATH
    #[test]
    fn montgomery_prime_field_multiplication_works_0() {
        let x = U256FP1Element::new(UnsignedInteger::from(
            "05ed176deb0e80b4deb7718cdaa075165f149c",
        ));
        let y = U256FP1Element::new(UnsignedInteger::from(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        let c = U256FP1Element::new(UnsignedInteger::from(
            "73d23e8d462060dc23d5c15c00fc432d95621a3c",
        ));
        assert_eq!(x * y, c);
    }
    */
    /*
       // FP2
       #[derive(Clone, Debug)]
       struct MontgomeryConfigP2;
       impl IsMontgomeryConfiguration<4> for MontgomeryConfigP2 {
           const MODULUS: U256 = UnsignedInteger {
               limbs: [
                   18446744073709551615,
                   18446744073709551615,
                   18446744073709551615,
                   18446744073709551275,
               ],
           };
           const MP: u64 = 14984598558409225213;
           const R2: U256 = UnsignedInteger {
               limbs: [0, 0, 0, 116281],
           };
       }

       type FP2 = U256PrimeField<MontgomeryConfigP2>;
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

       #[test]
       fn to_bytes_from_bytes_be_is_the_identity() {
           let x = FP2Element::new(UnsignedInteger::from(
               "5f103b0bd4397d4df560eb559f38353f80eeb6",
           ));
           assert_eq!(FP2Element::from_bytes_be(&x.to_bytes_be()).unwrap(), x);
       }

       #[test]
       fn from_bytes_to_bytes_be_is_the_identity_for_one() {
           let bytes = vec![
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           ];
           assert_eq!(
               FP2Element::from_bytes_be(&bytes).unwrap().to_bytes_be(),
               bytes
           );
       }

       #[test]
       fn to_bytes_from_bytes_le_is_the_identity() {
           let x = FP2Element::new(UnsignedInteger::from(
               "5f103b0bd4397d4df560eb559f38353f80eeb6",
           ));
           assert_eq!(FP2Element::from_bytes_le(&x.to_bytes_le()).unwrap(), x);
       }

       #[test]
       fn from_bytes_to_bytes_le_is_the_identity_for_one() {
           let bytes = vec![
               1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           ];
           assert_eq!(
               FP2Element::from_bytes_le(&bytes).unwrap().to_bytes_le(),
               bytes
           );
       }
    */
}
