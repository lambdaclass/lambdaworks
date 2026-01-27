use alloc::vec::Vec;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::UnsignedInteger,
};

/// Converts a hash, represented by an array of bytes, into a vector of field elements
/// For more info, see
/// https://www.ietf.org/id/draft-irtf-cfrg-hash-to-curve-16.html#name-hashing-to-a-finite-field
///
/// # Panics
/// Panics if `pseudo_random_bytes` doesn't contain enough bytes for `count` field elements.
pub fn hash_to_field<M: IsModulus<UnsignedInteger<N>> + Clone, const N: usize>(
    pseudo_random_bytes: &[u8],
    count: usize,
) -> Vec<FieldElement<MontgomeryBackendPrimeField<M, N>>> {
    let order = M::MODULUS;
    let l = compute_length(order);

    // Bounds check: ensure we have enough bytes
    let required_bytes = l * count;
    assert!(
        pseudo_random_bytes.len() >= required_bytes,
        "pseudo_random_bytes has {} bytes, but {} bytes are required for {} field elements (L={})",
        pseudo_random_bytes.len(),
        required_bytes,
        count,
        l
    );

    let mut u = vec![FieldElement::zero(); count];
    for (i, item) in u.iter_mut().enumerate() {
        let elm_offset = l * i;
        let tv = &pseudo_random_bytes[elm_offset..elm_offset + l];

        *item = os2ip::<M, N>(tv);
    }
    u
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
        let field_elements: Vec<FieldElement<F>> = hash_to_field(&input, 40);
        let other_field_elements = hash_to_field(&input, 40);
        assert_eq!(field_elements, other_field_elements);
    }

    #[test]
    fn test_os2ip_big_endian_bytes() {
        let input = [0x01_u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let expected = FieldElement::<F>::from_hex_unchecked("0102030405060708");
        let got = super::os2ip::<U64, 1>(&input);
        assert_eq!(got, expected);
    }
}
