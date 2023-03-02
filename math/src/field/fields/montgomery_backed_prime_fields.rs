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

/// Computes `- modulus^{-1} mod 2^{64}`
/// This algorithm is given  by Dussé and Kaliski Jr. in
/// "S. R. Dussé and B. S. Kaliski Jr. A cryptographic library for the Motorola
/// DSP56000. In I. Damgård, editor, Advances in Cryptology – EUROCRYPT’90,
/// volume 473 of Lecture Notes in Computer Science, pages 230–244. Springer,
/// Heidelberg, May 1991."
const fn compute_mu_parameter<const NUM_LIMBS: usize>(modulus: &UnsignedInteger<NUM_LIMBS>) -> u64 {
    let mut y = 1;
    let word_size = 64;
    let mut i: usize = 2;
    while i <= word_size {
        let (_, lo) = UnsignedInteger::mul(modulus, &UnsignedInteger::from_u64(y));
        let least_significant_limb = lo.limbs[NUM_LIMBS - 1];
        if (least_significant_limb << (word_size - i)) >> (word_size - i) != 1 {
            y += 1 << (i - 1);
        }
        i += 1;
    }
    y.wrapping_neg()
}

/// Computes 2^{384 * 2} modulo `modulus`
const fn compute_r2_parameter<const NUM_LIMBS: usize>(
    modulus: &UnsignedInteger<NUM_LIMBS>,
) -> UnsignedInteger<NUM_LIMBS> {
    let word_size = 64;
    let mut l: usize = 0;
    let zero = UnsignedInteger::from_u64(0);
    // Define `c` as the largest power of 2 smaller than `modulus`
    while l < NUM_LIMBS * word_size {
        if UnsignedInteger::const_ne(&modulus.const_shr(l), &zero) {
            break;
        }
        l += 1;
    }
    let mut c = UnsignedInteger::from_u64(1).const_shl(l);

    // Double `c` and reduce modulo `modulus` until getting
    // `2^{2 * number_limbs * word_size}` mod `modulus`
    let mut i: usize = 1;
    while i <= 2 * NUM_LIMBS * word_size - l {
        let (double_c, overflow) = UnsignedInteger::add(&c, &c);
        c = if UnsignedInteger::const_le(modulus, &double_c) || overflow {
            UnsignedInteger::sub(&double_c, modulus).0
        } else {
            double_c
        };
        i += 1;
    }
    c
}

/// This trait is necessary for us to be able to use unsigned integer types bigger than
/// `u128` (the biggest native `unit`) as constant generics.
/// This trait should be removed when Rust supports this feature.

pub trait IsMontgomeryConfiguration<const NUM_LIMBS: usize> {
    const MODULUS: UnsignedInteger<NUM_LIMBS>;
    const R2: UnsignedInteger<NUM_LIMBS> = compute_r2_parameter(&Self::MODULUS);
    const MU: u64 = compute_mu_parameter(&Self::MODULUS);
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
        MontgomeryAlgorithms::cios(a, b, &C::MODULUS, &C::MU)
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
        MontgomeryAlgorithms::cios(&UnsignedInteger::from_u64(x), &C::R2, &C::MODULUS, &C::MU)
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        MontgomeryAlgorithms::cios(&x, &C::R2, &C::MODULUS, &C::MU)
    }

    // TO DO: Add tests for representatives
    fn representative(x: Self::BaseType) -> Self::BaseType {
        MontgomeryAlgorithms::cios(&x, &UnsignedInteger::from_u64(1), &C::MODULUS, &C::MU)
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
            &C::MU,
        )
        .to_bytes_be()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        MontgomeryAlgorithms::cios(
            self.value(),
            &UnsignedInteger::from_u64(1),
            &C::MODULUS,
            &C::MU,
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
    #[derive(Clone, Debug)]
    struct U256MontgomeryConfigP1;
    impl IsMontgomeryConfiguration<4> for U256MontgomeryConfigP1 {
        const MODULUS: U256 = UnsignedInteger {
            limbs: [
                8366,
                8155137382671976874,
                227688614771682406,
                15723111795979912613,
            ],
        };
    }

    type U256FP1 = U256PrimeField<U256MontgomeryConfigP1>;
    type U256FP1Element = FieldElement<U256FP1>;

    #[test]
    fn montgomery_prime_field_addition_works_0() {
        let x = U256FP1Element::new(UnsignedInteger::from(
            "93e712950bf3fe589aa030562a44b1cec66b09192c4bcf705a5",
        ));
        let y = U256FP1Element::new(UnsignedInteger::from(
            "10a712235c1f6b4172a1e35da6aef1a7ec6b09192c4bb88cfa5",
        ));
        let c = U256FP1Element::new(UnsignedInteger::from(
            "a48e24b86813699a0d4213b3d0f3a376b2d61232589787fd54a",
        ));
        assert_eq!(x + y, c);
    }

    #[test]
    fn montgomery_prime_field_multiplication_works_0() {
        let x = U256FP1Element::new(UnsignedInteger::from(
            "93e712950bf3fe589aa030562a44b1cec66b09192c4bcf705a5",
        ));
        let y = U256FP1Element::new(UnsignedInteger::from(
            "10a712235c1f6b4172a1e35da6aef1a7ec6b09192c4bb88cfa5",
        ));
        let c = U256FP1Element::new(UnsignedInteger::from(
            "7808e74c3208d9a66791ef9cc15a46acc9951ee312102684021",
        ));
        assert_eq!(x * y, c);
    }

    // FP2
    #[derive(Clone, Debug)]
    struct MontgomeryConfigP2;
    impl IsMontgomeryConfiguration<4> for MontgomeryConfigP2 {
        const MODULUS: U256 = UnsignedInteger {
            limbs: [
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551427,
            ],
        };
    }

    type FP2 = U256PrimeField<MontgomeryConfigP2>;
    type FP2Element = FieldElement<FP2>;

    #[test]
    fn montgomery_prime_field_addition_works_1() {
        let x = FP2Element::new(UnsignedInteger::from(
            "acbbb7ca01c65cfffffc72815b397fff9ab130ad53a5ffffffb8f21b207dfedf",
        ));
        let y = FP2Element::new(UnsignedInteger::from(
            "d65ddbe509d3fffff21f494c588cbdbfe43e929b0543e3ffffffffffffffff43",
        ));
        let c = FP2Element::new(UnsignedInteger::from(
            "831993af0b9a5cfff21bbbcdb3c63dbf7eefc34858e9e3ffffb8f21b207dfedf",
        ));
        assert_eq!(x + y, c);
    }

    #[test]
    fn montgomery_prime_field_multiplication_works_1() {
        let x = FP2Element::new(UnsignedInteger::from(
            "acbbb7ca01c65cfffffc72815b397fff9ab130ad53a5ffffffb8f21b207dfedf",
        ));
        let y = FP2Element::new(UnsignedInteger::from(
            "d65ddbe509d3fffff21f494c588cbdbfe43e929b0543e3ffffffffffffffff43",
        ));
        let c = FP2Element::new(UnsignedInteger::from(
            "2b1e80d553ecab2e4d41eb53c4c8ad89ebacac6cf6b91dcf2213f311093aa05d",
        ));
        assert_eq!(&y * x, c);
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
            0, 0, 1,
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
            0, 0, 0,
        ];
        assert_eq!(
            FP2Element::from_bytes_le(&bytes).unwrap().to_bytes_le(),
            bytes
        );
    }
}
