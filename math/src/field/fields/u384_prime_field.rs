use crate::field::element::FieldElement;
use crate::traits::ByteConversion;
use crate::unsigned_integer::element::U384;
use crate::{
    field::traits::IsField, unsigned_integer::element::UnsignedInteger,
    unsigned_integer::montgomery::MontgomeryAlgorithms,
};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::fmt::Debug;
use std::marker::PhantomData;

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

impl<C> ByteConversion for FieldElement<MontgomeryBackendPrimeField<C>>
where
    C: IsMontgomeryConfiguration + Clone + Debug,
{
    fn to_bytes_be(&self) -> Vec<u8> {
        MontgomeryAlgorithms::cios(self.value(), &U384::from_u64(1), &C::MODULUS, &C::MP)
            .to_bytes_be()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        MontgomeryAlgorithms::cios(self.value(), &U384::from_u64(1), &C::MODULUS, &C::MP)
            .to_bytes_le()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let value = U384::from_bytes_be(bytes)?;
        Ok(Self::new(value))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let value = U384::from_bytes_le(bytes)?;
        Ok(Self::new(value))
    }
}

// Implements the Distribution trait on type FieldElement<MontgomeryBackendPrimeField<C>> for Standard in order to allow random generation.

impl<C> Distribution<FieldElement<MontgomeryBackendPrimeField<C>>> for Standard
where
    C: IsMontgomeryConfiguration + Clone + Debug,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> FieldElement<MontgomeryBackendPrimeField<C>> {
        let mut rand_limbs: [u64; 6] = [0; 6];
        let mut is_msl: bool = true; //is Most Significative Limb of MODULUS

        for i in 0..rand_limbs.len() {
            match C::MODULUS.limbs[i] {
                mod_limb if (mod_limb == 0) && is_msl => rand_limbs[i] = 0,
                mod_limb if (i == 5) && is_msl => rand_limbs[i] = rng.gen_range(1..mod_limb),
                mod_limb if is_msl => {
                    rand_limbs[i] = rng.gen_range(0..=mod_limb);
                    is_msl = false;
                }
                _ if (i == 5) => {
                    if rand_limbs == [0; 6] {
                        //to avoid 0 in the generation set
                        rand_limbs[i] = rng.gen_range(1..=u64::MAX);
                    } else {
                        rand_limbs[i] = rng.gen_range(0..=u64::MAX);
                    }
                }
                _ => rand_limbs[i] = rng.gen_range(0..=u64::MAX),
            }
        }

        let mut rand_hex: String = "".to_string();

        for i in rand_limbs {
            rand_hex.push_str(&format!("{:0>16X}", i));
        }

        let integer: U384 = U384::from(&rand_hex);

        FieldElement::new(integer)
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::ThreadRng, Rng};

    use crate::{
        field::element::FieldElement,
        traits::ByteConversion,
        unsigned_integer::element::{UnsignedInteger, U384},
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
    #[should_panic]
    fn inv_0_error() {
        F23Element::from(0).inv();
    }

    #[test]
    fn inv_2() {
        let a: F23Element = F23Element::from(2);
        assert_eq!(&a * a.inv(), F23Element::from(1));
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
        assert_eq!(
            F23Element::from(2) / F23Element::from(1),
            F23Element::from(2)
        )
    }

    #[test]
    fn div_4_2() {
        assert_eq!(
            F23Element::from(4) / F23Element::from(2),
            F23Element::from(2)
        )
    }

    #[test]
    fn div_4_3() {
        assert_eq!(
            F23Element::from(4) / F23Element::from(3) * F23Element::from(3),
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

    #[test]
    fn gen_random_for_f23() {
        for _ in 0..1000 {
            let rand: F23Element = rand::random();
            assert!(
                (&MontgomeryConfig23::MODULUS > rand.value())
                    && (rand.value() > &U384::from_u64(0))
            );
        }
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

    #[test]
    fn gen_random_for_fp1() {
        for _ in 0..1000 {
            let rand: FP1Element = rand::random();
            let mut rng:ThreadRng = rand::thread_rng();
            let rand: FP1Element = rng.gen();
            assert!(
                (&MontgomeryConfigP1::MODULUS > rand.value())
                    && (rand.value() > &U384::from_u64(0))
            );
        }
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

    #[test]
    fn gen_random_for_fp2() {
        for _ in 0..10000 {
            let rand: FP2Element = rand::random();
            assert!(
                (&MontgomeryConfigP2::MODULUS > rand.value())
                    && (rand.value() > &U384::from_u64(0))
            );
        }
    }
}
