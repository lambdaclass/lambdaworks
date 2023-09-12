use crate::field::element::FieldElement;
use crate::field::traits::IsPrimeField;
use crate::traits::ByteConversion;
use crate::{
    field::traits::IsField, unsigned_integer::u32_word::element::UnsignedInteger,
    unsigned_integer::u32_word::montgomery::MontgomeryAlgorithms,
};

use core::fmt::Debug;
use core::marker::PhantomData;

pub type U384PrimeField32<M> = MontgomeryBackendPrimeField32<M, 12>;
pub type U256PrimeField32<M> = MontgomeryBackendPrimeField32<M, 8>;

pub trait IsModulus<U>: Debug {
    const MODULUS: U;
}

#[cfg_attr(
    feature = "lambdaworks-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Clone, Debug, Hash, Copy)]
pub struct MontgomeryBackendPrimeField32<M, const NUM_LIMBS: usize> {
    phantom: PhantomData<M>,
}

impl<M, const NUM_LIMBS: usize> MontgomeryBackendPrimeField32<M, NUM_LIMBS>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>>,
{
    pub const R2: UnsignedInteger<NUM_LIMBS> = Self::compute_r2_parameter(&M::MODULUS);
    pub const MU: u32 = Self::compute_mu_parameter(&M::MODULUS);
    pub const ZERO: UnsignedInteger<NUM_LIMBS> = UnsignedInteger::from_u32(0);
    pub const ONE: UnsignedInteger<NUM_LIMBS> = MontgomeryAlgorithms::cios(
        &UnsignedInteger::from_u32(1),
        &Self::R2,
        &M::MODULUS,
        &Self::MU,
    );
    const MODULUS_HAS_ONE_SPARE_BIT: bool = Self::modulus_has_one_spare_bit();

    /// Computes `- modulus^{-1} mod 2^{32}`
    /// This algorithm is given  by Dussé and Kaliski Jr. in
    /// "S. R. Dussé and B. S. Kaliski Jr. A cryptographic library for the Motorola
    /// DSP56000. In I. Damgård, editor, Advances in Cryptology – EUROCRYPT’90,
    /// volume 473 of Lecture Notes in Computer Science, pages 230–244. Springer,
    /// Heidelberg, May 1991."
    const fn compute_mu_parameter(modulus: &UnsignedInteger<NUM_LIMBS>) -> u32 {
        let mut y = 1;
        let word_size = 32;
        let mut i: usize = 2;
        while i <= word_size {
            let (_, lo) = UnsignedInteger::mul(modulus, &UnsignedInteger::from_u32(y));
            let least_significant_limb = lo.limbs[NUM_LIMBS - 1];
            if (least_significant_limb << (word_size - i)) >> (word_size - i) != 1 {
                y += 1 << (i - 1);
            }
            i += 1;
        }
        y.wrapping_neg()
    }

    /// Computes 2^{384 * 2} modulo `modulus`
    const fn compute_r2_parameter(
        modulus: &UnsignedInteger<NUM_LIMBS>,
    ) -> UnsignedInteger<NUM_LIMBS> {
        let word_size = 32;
        let mut l: usize = 0;
        let zero = UnsignedInteger::from_u32(0);
        // Define `c` as the largest power of 2 smaller than `modulus`
        while l < NUM_LIMBS * word_size {
            if UnsignedInteger::const_ne(&modulus.const_shr(l), &zero) {
                break;
            }
            l += 1;
        }
        let mut c = UnsignedInteger::from_u32(1).const_shl(l);

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

    /// Checks whether the most significant limb of the modulus is at
    /// most `0x7FFFFFFFFFFFFFFE`. This check is useful since special
    /// optimizations exist for this kind of moduli.
    #[inline(always)]
    const fn modulus_has_one_spare_bit() -> bool {
        M::MODULUS.limbs[0] < (1u32 << 31) - 1
    }
}

impl<M, const NUM_LIMBS: usize> IsField for MontgomeryBackendPrimeField32<M, NUM_LIMBS>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    type BaseType = UnsignedInteger<NUM_LIMBS>;

    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let (sum, overflow) = UnsignedInteger::add(a, b);
        if Self::MODULUS_HAS_ONE_SPARE_BIT {
            if sum >= M::MODULUS {
                sum - M::MODULUS
            } else {
                sum
            }
        } else if overflow || sum >= M::MODULUS {
            let (diff, _) = UnsignedInteger::sub(&sum, &M::MODULUS);
            diff
        } else {
            sum
        }
    }

    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        if Self::MODULUS_HAS_ONE_SPARE_BIT {
            MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(
                a,
                b,
                &M::MODULUS,
                &Self::MU,
            )
        } else {
            MontgomeryAlgorithms::cios(a, b, &M::MODULUS, &Self::MU)
        }
    }

    #[inline(always)]
    fn square(a: &UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        MontgomeryAlgorithms::sos_square(a, &M::MODULUS, &Self::MU)
    }

    #[inline(always)]
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        if b <= a {
            a - b
        } else {
            M::MODULUS - (b - a)
        }
    }

    #[inline(always)]
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        if a == &Self::ZERO {
            *a
        } else {
            M::MODULUS - a
        }
    }

    #[inline(always)]
    fn inv(a: &Self::BaseType) -> Self::BaseType {
        if a == &Self::ZERO {
            panic!("Division by zero error.")
        } else {
            // Guajardo Kumar Paar Pelzl
            // Efficient Software-Implementation of Finite Fields with Applications to
            // Cryptography
            // Algorithm 16 (BEA for Inversion in Fp)

            //These can be done with const  functions
            let one: UnsignedInteger<NUM_LIMBS> = UnsignedInteger::from_u32(1);
            let modulus: UnsignedInteger<NUM_LIMBS> = M::MODULUS;
            let modulus_has_spare_bits = M::MODULUS.limbs[0] >> 31 == 0;

            let mut u: UnsignedInteger<NUM_LIMBS> = *a;
            let mut v = M::MODULUS;
            let mut b = Self::R2; // Avoids unnecessary reduction step.
            let mut c = Self::zero();

            while u != one && v != one {
                while u.limbs[NUM_LIMBS - 1] & 1 == 0 {
                    u >>= 1;
                    if b.limbs[NUM_LIMBS - 1] & 1 == 0 {
                        b >>= 1;
                    } else {
                        let carry;
                        (b, carry) = UnsignedInteger::<NUM_LIMBS>::add(&b, &modulus);
                        b >>= 1;
                        if !modulus_has_spare_bits && carry {
                            b.limbs[0] |= 1 << 31;
                        }
                    }
                }

                while v.limbs[NUM_LIMBS - 1] & 1 == 0 {
                    v >>= 1;

                    if c.limbs[NUM_LIMBS - 1] & 1 == 0 {
                        c >>= 1;
                    } else {
                        let carry;
                        (c, carry) = UnsignedInteger::<NUM_LIMBS>::add(&c, &modulus);
                        c >>= 1;
                        if !modulus_has_spare_bits && carry {
                            c.limbs[0] |= 1 << 31;
                        }
                    }
                }

                if v <= u {
                    u = u - v;
                    if b < c {
                        b = b + modulus;
                    }
                    b = b - c;
                } else {
                    v = v - u;
                    if c < b {
                        c = c + modulus;
                    }
                    c = c - b;
                }
            }

            if u == one {
                b
            } else {
                c
            }
        }
    }

    #[inline(always)]
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        Self::mul(a, &Self::inv(b))
    }

    #[inline(always)]
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a == b
    }

    #[inline(always)]
    fn zero() -> Self::BaseType {
        Self::ZERO
    }

    #[inline(always)]
    fn one() -> Self::BaseType {
        Self::ONE
    }

    #[inline(always)]
    fn from_u64(x: u64) -> Self::BaseType {
        MontgomeryAlgorithms::cios(
            &UnsignedInteger::from_u64(x),
            &Self::R2,
            &M::MODULUS,
            &Self::MU,
        )
    }

    #[inline(always)]
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        MontgomeryAlgorithms::cios(&x, &Self::R2, &M::MODULUS, &Self::MU)
    }
}

impl<M, const NUM_LIMBS: usize> IsPrimeField for MontgomeryBackendPrimeField32<M, NUM_LIMBS>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    type RepresentativeType = Self::BaseType;

    fn representative(x: &Self::BaseType) -> Self::RepresentativeType {
        MontgomeryAlgorithms::cios(x, &UnsignedInteger::from_u32(1), &M::MODULUS, &Self::MU)
    }

    fn field_bit_size() -> usize {
        let mut evaluated_bit = NUM_LIMBS * 32 - 1;
        let max_element = M::MODULUS - UnsignedInteger::<NUM_LIMBS>::from_u32(1);
        let one = UnsignedInteger::from_u64(1);

        while ((max_element >> evaluated_bit) & one) != one {
            evaluated_bit -= 1;
        }

        evaluated_bit + 1
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, crate::errors::CreationError> {
        let integer = Self::BaseType::from_hex(hex_string)?;
        Ok(MontgomeryAlgorithms::cios(
            &integer,
            &MontgomeryBackendPrimeField32::<M, NUM_LIMBS>::R2,
            &M::MODULUS,
            &MontgomeryBackendPrimeField32::<M, NUM_LIMBS>::MU,
        ))
    }
}

impl<M, const NUM_LIMBS: usize> FieldElement<MontgomeryBackendPrimeField32<M, NUM_LIMBS>> where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug
{
}

impl<M, const NUM_LIMBS: usize> ByteConversion
    for FieldElement<MontgomeryBackendPrimeField32<M, NUM_LIMBS>>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    #[cfg(feature = "std")]
    fn to_bytes_be(&self) -> Vec<u8> {
        MontgomeryAlgorithms::cios(
            self.value(),
            &UnsignedInteger::from_u32(1),
            &M::MODULUS,
            &MontgomeryBackendPrimeField32::<M, NUM_LIMBS>::MU,
        )
        .to_bytes_be()
    }

    #[cfg(feature = "std")]
    fn to_bytes_le(&self) -> Vec<u8> {
        MontgomeryAlgorithms::cios(
            self.value(),
            &UnsignedInteger::from_u32(1),
            &M::MODULUS,
            &MontgomeryBackendPrimeField32::<M, NUM_LIMBS>::MU,
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
mod tests_u384_prime_fields_u32_limb {
    use crate::field::element::FieldElement;
    use crate::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
    use crate::field::fields::montgomery_backed_prime_fields::{
        IsModulus, U256PrimeField, U384PrimeField,
    };
    use crate::field::traits::IsField;
    use crate::field::traits::IsPrimeField;
    #[cfg(feature = "std")]
    use crate::traits::ByteConversion;
    use crate::unsigned_integer::u32_word::element::U384;
    use crate::unsigned_integer::u32_word::element::{UnsignedInteger, U256};

    #[derive(Clone, Debug)]
    struct U384Modulus23;
    impl IsModulus<U384> for U384Modulus23 {
        const MODULUS: U384 = UnsignedInteger::from_u32(23);
    }

    type U384F23 = U384PrimeField<U384Modulus23>;
    type U384F23Element = FieldElement<U384F23>;

    #[test]
    fn u384_mod_23_uses_5_bits() {
        assert_eq!(U384F23::field_bit_size(), 5);
    }

    #[test]
    fn stark_252_prime_field_uses_252_bits() {
        assert_eq!(Stark252PrimeField::field_bit_size(), 252);
    }

    #[test]
    fn u256_mod_2_uses_1_bit() {
        #[derive(Clone, Debug)]
        struct U256Modulus1;
        impl IsModulus<U256> for U256Modulus1 {
            const MODULUS: U256 = UnsignedInteger::from_u32(2);
        }
        type U256OneField = U256PrimeField<U256Modulus1>;
        assert_eq!(U256OneField::field_bit_size(), 1);
    }

    #[test]
    fn u256_with_first_bit_set_uses_256_bit() {
        #[derive(Clone, Debug)]
        struct U256ModulusBig;
        impl IsModulus<U256> for U256ModulusBig {
            const MODULUS: U256 = UnsignedInteger::from_hex_unchecked(
                "F0000000F0000000F0000000F0000000F0000000F0000000F0000000F0000000",
            );
        }
        type U256OneField = U256PrimeField<U256ModulusBig>;
        assert_eq!(U256OneField::field_bit_size(), 256);
    }

    #[test]
    fn montgomery_backend_primefield_compute_mu_parameter() {
        assert_eq!(U384F23::MU, 3208129404123400281);
    }

    type FP2 = U256PrimeField<ModulusP2>;
    type FP2Element = FieldElement<FP2>;
}