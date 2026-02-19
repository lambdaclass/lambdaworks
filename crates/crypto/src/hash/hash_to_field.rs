use alloc::vec::Vec;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::UnsignedInteger,
};

#[derive(Debug)]
pub struct HashToFieldError;

/// Converts a hash, represented by an array of bytes, into a vector of field elements
/// For more info, see
/// https://www.ietf.org/id/draft-irtf-cfrg-hash-to-curve-16.html#name-hashing-to-a-finite-field
///
/// # Errors
/// Returns an error if `pseudo_random_bytes` doesn't contain enough bytes for `count` field elements.
pub fn hash_to_field<M: IsModulus<UnsignedInteger<N>> + Clone, const N: usize>(
    pseudo_random_bytes: &[u8],
    count: usize,
) -> Result<Vec<FieldElement<MontgomeryBackendPrimeField<M, N>>>, HashToFieldError> {
    let order = M::MODULUS;
    let l = compute_length(order);

    // Bounds check: ensure we have enough bytes
    let required_bytes = l * count;
    if pseudo_random_bytes.len() < required_bytes {
        return Err(HashToFieldError);
    }

    let mut u = vec![FieldElement::zero(); count];
    for (i, item) in u.iter_mut().enumerate() {
        let elm_offset = l * i;
        let tv = &pseudo_random_bytes[elm_offset..elm_offset + l];

        *item = os2ip::<M, N>(tv);
    }
    Ok(u)
}

fn compute_length<const N: usize>(order: UnsignedInteger<N>) -> usize {
    // L = ceil((ceil(log2(p)) + k) / 8), where k is the security parameter of the cryptosystem
    // (e.g. k = ceil(log2(p) / 2) for 128-bit security)
    //
    // We compute the actual bit length of the modulus, not just limb count * 64,
    // to handle moduli with leading zeros correctly.
    let log2_p = compute_bit_length(&order);
    let k = log2_p.div_ceil(2);
    (log2_p + k).div_ceil(8)
}

/// Computes the bit length of an unsigned integer (position of highest set bit + 1).
fn compute_bit_length<const N: usize>(value: &UnsignedInteger<N>) -> usize {
    for (i, &limb) in value.limbs.iter().enumerate() {
        if limb != 0 {
            // Found the first non-zero limb (most significant)
            let limb_bits = 64 - limb.leading_zeros() as usize;
            return (N - i) * 64 - (64 - limb_bits);
        }
    }
    0 // Value is zero
}

/// Converts an octet string to a nonnegative integer.
/// For more info, see https://www.rfc-editor.org/rfc/pdfrfc/rfc8017.txt.pdf
fn os2ip<M: IsModulus<UnsignedInteger<N>> + Clone, const N: usize>(
    x: &[u8],
) -> FieldElement<MontgomeryBackendPrimeField<M, N>> {
    type F<M, const N: usize> = MontgomeryBackendPrimeField<M, N>;
    // Use Horner's method: result = sum(byte[i] * 256^(n-1-i))
    // Process bytes from most significant to least significant
    let mut result: FieldElement<F<M, N>> = FieldElement::zero();
    let base: FieldElement<F<M, N>> = FieldElement::from(256u64);
    for byte in x.iter() {
        result = result * &base + FieldElement::from(*byte as u64);
    }
    result
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use lambdaworks_math::{
        field::{
            element::FieldElement,
            fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        },
        unsigned_integer::element::UnsignedInteger,
    };

    use crate::hash::{hash_to_field::hash_to_field, sha3::Sha3Hasher};

    type F = MontgomeryBackendPrimeField<U64, 1>;

    #[derive(Clone, Debug)]
    struct U64;
    impl IsModulus<UnsignedInteger<1>> for U64 {
        const MODULUS: UnsignedInteger<1> = UnsignedInteger::from_u64(18446744069414584321_u64);
    }

    #[test]
    fn test_same_message_produce_same_field_elements() {
        let input = Sha3Hasher::expand_message(b"helloworld", b"dsttest", 500).unwrap();
        let field_elements: Vec<FieldElement<F>> = hash_to_field(&input, 40).unwrap();
        let other_field_elements = hash_to_field(&input, 40).unwrap();
        assert_eq!(field_elements, other_field_elements);
    }

    #[test]
    fn test_os2ip_big_endian_bytes() {
        let input = [0x01_u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let expected = FieldElement::<F>::from_hex_unchecked("0102030405060708");
        let got = super::os2ip::<U64, 1>(&input);
        assert_eq!(got, expected);
    }

    // RFC 9380 compliance tests for expand_message
    #[test]
    fn test_expand_message_output_length() {
        // Verify expand_message produces exactly the requested length
        for len in [32, 48, 64, 128, 256] {
            let result = Sha3Hasher::expand_message(b"test", b"QUUX-V01-CS02", len).unwrap();
            assert_eq!(
                result.len(),
                len as usize,
                "Output length must match requested length"
            );
        }
    }

    #[test]
    fn test_expand_message_dst_separation() {
        // Different DSTs must produce different outputs (domain separation)
        let msg = b"test message";
        let output1 = Sha3Hasher::expand_message(msg, b"DST1", 32).unwrap();
        let output2 = Sha3Hasher::expand_message(msg, b"DST2", 32).unwrap();
        assert_ne!(
            output1, output2,
            "Different DSTs must produce different outputs"
        );
    }

    #[test]
    fn test_expand_message_deterministic() {
        // Same inputs must always produce same output (determinism)
        let msg = b"deterministic test";
        let dst = b"QUUX-V01-CS02-with-expander-SHA3-256";
        let output1 = Sha3Hasher::expand_message(msg, dst, 128).unwrap();
        let output2 = Sha3Hasher::expand_message(msg, dst, 128).unwrap();
        assert_eq!(output1, output2, "expand_message must be deterministic");
    }

    #[test]
    fn test_expand_message_empty_input() {
        // Empty message should work (RFC 9380 test case)
        let result = Sha3Hasher::expand_message(b"", b"QUUX-V01-CS02", 32);
        assert!(result.is_ok(), "expand_message should handle empty input");
        assert_eq!(result.unwrap().len(), 32);
    }

    #[test]
    fn test_expand_message_long_output() {
        // Test with output length requiring multiple blocks (exercises the loop)
        // With b=32 (SHA3-256 output), requesting 128 bytes needs ell=4 blocks
        let result = Sha3Hasher::expand_message(b"long output test", b"DST", 128).unwrap();
        assert_eq!(result.len(), 128);

        // Verify different parts of output are different (not just repeated)
        let first_32 = &result[0..32];
        let second_32 = &result[32..64];
        assert_ne!(
            first_32, second_32,
            "Different blocks should produce different output"
        );
    }

    #[test]
    fn test_expand_message_msg_independence() {
        // Different messages must produce different outputs
        let dst = b"QUUX-V01-CS02";
        let output1 = Sha3Hasher::expand_message(b"message1", dst, 64).unwrap();
        let output2 = Sha3Hasher::expand_message(b"message2", dst, 64).unwrap();
        assert_ne!(
            output1, output2,
            "Different messages must produce different outputs"
        );
    }
}
