use crate::errors::ByteConversionError;
use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::{HasDefaultTranscript, IsField};
use crate::traits::ByteConversion;

use super::field::TowerFieldElement;

/// Implementation of `ByteConversion` for `u128`, needed as the `BaseType` for
/// `BinaryTowerField128`. Binary tower field elements at level 7 are 128-bit values
/// where every bit pattern is a valid field element.
impl ByteConversion for u128 {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        let needed = bytes
            .get(0..16)
            .ok_or(ByteConversionError::FromBEBytesError)?;
        Ok(u128::from_be_bytes(
            needed
                .try_into()
                .map_err(|_| ByteConversionError::FromBEBytesError)?,
        ))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        let needed = bytes
            .get(0..16)
            .ok_or(ByteConversionError::FromLEBytesError)?;
        Ok(u128::from_le_bytes(
            needed
                .try_into()
                .map_err(|_| ByteConversionError::FromLEBytesError)?,
        ))
    }
}

/// A zero-sized type representing GF(2^128), the binary tower field at level 7.
///
/// This implements `IsField` with `BaseType = u128`, delegating arithmetic
/// to `TowerFieldElement` at level 7. In binary fields:
/// - Addition and subtraction are XOR
/// - Negation is identity
/// - Multiplication uses the recursive Karatsuba tower construction
///
/// Every 128-bit value is a valid field element (no reduction needed).
#[derive(Clone, Copy, Debug)]
pub struct BinaryTowerField128;

impl IsField for BinaryTowerField128 {
    type BaseType = u128;

    #[inline(always)]
    fn add(a: &u128, b: &u128) -> u128 {
        a ^ b
    }

    #[inline(always)]
    fn double(a: &u128) -> u128 {
        // In characteristic 2, a + a = 0
        let _ = a;
        0
    }

    #[inline(always)]
    fn sub(a: &u128, b: &u128) -> u128 {
        // In characteristic 2, subtraction is the same as addition
        a ^ b
    }

    #[inline(always)]
    fn neg(a: &u128) -> u128 {
        // In characteristic 2, -a = a
        *a
    }

    fn mul(a: &u128, b: &u128) -> u128 {
        let ta = TowerFieldElement::new(*a, 7);
        let tb = TowerFieldElement::new(*b, 7);
        (ta * tb).value()
    }

    fn square(a: &u128) -> u128 {
        let ta = TowerFieldElement::new(*a, 7);
        (ta * ta).value()
    }

    fn inv(a: &u128) -> Result<u128, FieldError> {
        let ta = TowerFieldElement::new(*a, 7);
        ta.inv()
            .map(|r| r.value())
            .map_err(|_| FieldError::InvZeroError)
    }

    fn div(a: &u128, b: &u128) -> Result<u128, FieldError> {
        let inv_b = Self::inv(b)?;
        Ok(Self::mul(a, &inv_b))
    }

    #[inline(always)]
    fn eq(a: &u128, b: &u128) -> bool {
        a == b
    }

    #[inline(always)]
    fn zero() -> u128 {
        0
    }

    #[inline(always)]
    fn one() -> u128 {
        1
    }

    #[inline(always)]
    fn from_u64(x: u64) -> u128 {
        x as u128
    }

    #[inline(always)]
    fn from_base_type(x: u128) -> u128 {
        x
    }
}

impl ByteConversion for FieldElement<BinaryTowerField128> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        self.value().to_be_bytes().to_vec()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        self.value().to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        let val = u128::from_bytes_be(bytes)?;
        Ok(FieldElement::new(val))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        let val = u128::from_bytes_le(bytes)?;
        Ok(FieldElement::new(val))
    }
}

impl HasDefaultTranscript for BinaryTowerField128 {
    fn get_random_field_element_from_rng(rng: &mut impl rand::Rng) -> FieldElement<Self> {
        // Every 128-bit value is a valid element of GF(2^128), so no rejection sampling needed.
        let val: u128 = rng.gen();
        FieldElement::new(val)
    }
}

/// Type alias for convenience.
pub type BinaryFieldElement = FieldElement<BinaryTowerField128>;

/// Convert a `TowerFieldElement` to a `FieldElement<BinaryTowerField128>`.
/// The tower element is treated at level 7 (GF(2^128)) regardless of its original level.
pub fn from_tower(t: &TowerFieldElement) -> BinaryFieldElement {
    FieldElement::new(t.value())
}

/// Convert a `FieldElement<BinaryTowerField128>` to a `TowerFieldElement` at level 7.
pub fn to_tower(fe: &BinaryFieldElement) -> TowerFieldElement {
    TowerFieldElement::new(*fe.value(), 7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_is_xor() {
        let a = FieldElement::<BinaryTowerField128>::new(0b1010u128);
        let b = FieldElement::<BinaryTowerField128>::new(0b1100u128);
        let c = a + b;
        assert_eq!(c.to_raw(), 0b0110u128);
    }

    #[test]
    fn test_sub_equals_add() {
        let a = FieldElement::<BinaryTowerField128>::new(42u128);
        let b = FieldElement::<BinaryTowerField128>::new(17u128);
        assert_eq!(a - b, a + b);
    }

    #[test]
    fn test_neg_is_identity() {
        let a = FieldElement::<BinaryTowerField128>::new(12345u128);
        assert_eq!(-a, a);
    }

    #[test]
    fn test_double_is_zero() {
        let a = FieldElement::<BinaryTowerField128>::new(999u128);
        assert_eq!(a + a, FieldElement::zero());
    }

    #[test]
    fn test_zero_and_one() {
        let zero = FieldElement::<BinaryTowerField128>::zero();
        let one = FieldElement::<BinaryTowerField128>::one();
        assert_eq!(zero.to_raw(), 0u128);
        assert_eq!(one.to_raw(), 1u128);
    }

    #[test]
    fn test_mul_by_zero() {
        let a = FieldElement::<BinaryTowerField128>::new(0xDEADBEEFu128);
        let zero = FieldElement::<BinaryTowerField128>::zero();
        assert_eq!(a * zero, zero);
    }

    #[test]
    fn test_mul_by_one() {
        let a = FieldElement::<BinaryTowerField128>::new(0xDEADBEEFu128);
        let one = FieldElement::<BinaryTowerField128>::one();
        assert_eq!(a * one, a);
    }

    #[test]
    fn test_mul_consistency_with_tower() {
        // Verify that BinaryTowerField128::mul matches TowerFieldElement at level 7
        let vals = [
            (3u128, 5u128),
            (0xFF, 0xFF),
            (0xABCD, 0x1234),
            (u128::MAX, 1),
            (u128::MAX, u128::MAX),
        ];
        for (a, b) in vals {
            let fe_result = FieldElement::<BinaryTowerField128>::new(a)
                * FieldElement::<BinaryTowerField128>::new(b);
            let tower_result = TowerFieldElement::new(a, 7) * TowerFieldElement::new(b, 7);
            assert_eq!(
                fe_result.to_raw(),
                tower_result.value(),
                "Mismatch for a={a}, b={b}"
            );
        }
    }

    #[test]
    fn test_inv_and_mul_inverse() {
        let vals = [1u128, 2, 3, 0xFF, 0xABCDEF, u128::MAX];
        for v in vals {
            let a = FieldElement::<BinaryTowerField128>::new(v);
            let a_inv = a.inv().unwrap();
            let product = a * a_inv;
            assert_eq!(product, FieldElement::one(), "a * a^-1 != 1 for a={v}");
        }
    }

    #[test]
    fn test_inv_zero_fails() {
        let zero = FieldElement::<BinaryTowerField128>::zero();
        assert!(zero.inv().is_err());
    }

    #[test]
    fn test_mul_commutativity() {
        let a = FieldElement::<BinaryTowerField128>::new(0x12345678u128);
        let b = FieldElement::<BinaryTowerField128>::new(0x9ABCDEF0u128);
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_mul_associativity() {
        let a = FieldElement::<BinaryTowerField128>::new(83u128);
        let b = FieldElement::<BinaryTowerField128>::new(31u128);
        let c = FieldElement::<BinaryTowerField128>::new(7u128);
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_distributivity() {
        let a = FieldElement::<BinaryTowerField128>::new(0xABu128);
        let b = FieldElement::<BinaryTowerField128>::new(0xCDu128);
        let c = FieldElement::<BinaryTowerField128>::new(0xEFu128);
        // a * (b + c) = a*b + a*c
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn test_from_u64() {
        let fe = FieldElement::<BinaryTowerField128>::from(42u64);
        assert_eq!(fe.to_raw(), 42u128);
    }

    #[test]
    fn test_byte_conversion_roundtrip_be() {
        let a = FieldElement::<BinaryTowerField128>::new(0x0123456789ABCDEFu128);
        let bytes = a.to_bytes_be();
        let recovered = FieldElement::<BinaryTowerField128>::from_bytes_be(&bytes).unwrap();
        assert_eq!(a, recovered);
    }

    #[test]
    fn test_byte_conversion_roundtrip_le() {
        let a = FieldElement::<BinaryTowerField128>::new(0xFEDCBA9876543210u128);
        let bytes = a.to_bytes_le();
        let recovered = FieldElement::<BinaryTowerField128>::from_bytes_le(&bytes).unwrap();
        assert_eq!(a, recovered);
    }

    #[test]
    fn test_byte_conversion_max_value() {
        let a = FieldElement::<BinaryTowerField128>::new(u128::MAX);
        let bytes_be = a.to_bytes_be();
        let bytes_le = a.to_bytes_le();
        assert_eq!(bytes_be.len(), 16);
        assert_eq!(bytes_le.len(), 16);
        let recovered_be = FieldElement::<BinaryTowerField128>::from_bytes_be(&bytes_be).unwrap();
        let recovered_le = FieldElement::<BinaryTowerField128>::from_bytes_le(&bytes_le).unwrap();
        assert_eq!(a, recovered_be);
        assert_eq!(a, recovered_le);
    }

    #[test]
    fn test_tower_conversion_roundtrip() {
        let vals = [0u128, 1, 42, 0xDEADBEEF, u128::MAX];
        for v in vals {
            let fe = FieldElement::<BinaryTowerField128>::new(v);
            let tower = to_tower(&fe);
            let back = from_tower(&tower);
            assert_eq!(fe, back, "Roundtrip failed for v={v}");
        }
    }

    #[test]
    fn test_transcript_produces_elements() {
        use crate::field::traits::HasDefaultTranscript;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let e1 = BinaryTowerField128::get_random_field_element_from_rng(&mut rng);
        let e2 = BinaryTowerField128::get_random_field_element_from_rng(&mut rng);
        // Two samples from same seed should be different (overwhelmingly likely)
        assert_ne!(e1, e2);
        // And non-trivial (not zero)
        assert_ne!(e1, FieldElement::zero());
    }

    #[test]
    fn test_transcript_deterministic() {
        use crate::field::traits::HasDefaultTranscript;
        use rand::SeedableRng;

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(123);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(123);

        let e1 = BinaryTowerField128::get_random_field_element_from_rng(&mut rng1);
        let e2 = BinaryTowerField128::get_random_field_element_from_rng(&mut rng2);
        assert_eq!(e1, e2);
    }

    #[cfg(feature = "std")]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        fn arb_field_element() -> impl Strategy<Value = FieldElement<BinaryTowerField128>> {
            any::<u128>().prop_map(FieldElement::new)
        }

        proptest! {
            #[test]
            fn prop_add_commutative(a in arb_field_element(), b in arb_field_element()) {
                prop_assert_eq!(a + b, b + a);
            }

            #[test]
            fn prop_add_associative(
                a in arb_field_element(),
                b in arb_field_element(),
                c in arb_field_element()
            ) {
                prop_assert_eq!((a + b) + c, a + (b + c));
            }

            #[test]
            fn prop_mul_commutative(a in arb_field_element(), b in arb_field_element()) {
                prop_assert_eq!(a * b, b * a);
            }

            #[test]
            fn prop_mul_associative(
                a in arb_field_element(),
                b in arb_field_element(),
                c in arb_field_element()
            ) {
                prop_assert_eq!((a * b) * c, a * (b * c));
            }

            #[test]
            fn prop_distributive(
                a in arb_field_element(),
                b in arb_field_element(),
                c in arb_field_element()
            ) {
                prop_assert_eq!(a * (b + c), a * b + a * c);
            }

            #[test]
            fn prop_add_identity(a in arb_field_element()) {
                let zero = FieldElement::<BinaryTowerField128>::zero();
                prop_assert_eq!(a + zero, a);
            }

            #[test]
            fn prop_mul_identity(a in arb_field_element()) {
                let one = FieldElement::<BinaryTowerField128>::one();
                prop_assert_eq!(a * one, a);
            }

            #[test]
            fn prop_add_inverse(a in arb_field_element()) {
                // In char 2, a + a = 0
                prop_assert_eq!(a + a, FieldElement::<BinaryTowerField128>::zero());
            }

            #[test]
            fn prop_mul_consistency_with_tower(a_val in any::<u128>(), b_val in any::<u128>()) {
                let fe_result = FieldElement::<BinaryTowerField128>::new(a_val)
                    * FieldElement::<BinaryTowerField128>::new(b_val);
                let tower_result =
                    TowerFieldElement::new(a_val, 7) * TowerFieldElement::new(b_val, 7);
                prop_assert_eq!(fe_result.to_raw(), tower_result.value());
            }

            #[test]
            fn prop_byte_conversion_roundtrip(val in any::<u128>()) {
                let fe = FieldElement::<BinaryTowerField128>::new(val);
                let bytes_be = fe.to_bytes_be();
                let bytes_le = fe.to_bytes_le();
                let recovered_be = FieldElement::<BinaryTowerField128>::from_bytes_be(&bytes_be).unwrap();
                let recovered_le = FieldElement::<BinaryTowerField128>::from_bytes_le(&bytes_le).unwrap();
                prop_assert_eq!(fe, recovered_be);
                prop_assert_eq!(fe, recovered_le);
            }
        }
    }
}
