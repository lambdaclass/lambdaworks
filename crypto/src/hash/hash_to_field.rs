use std::fmt::Debug;

use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::UnsignedInteger,
};

use super::helpers;

pub fn hash_to_field<M: IsModulus<UnsignedInteger<N>> + Clone + Debug, const N: usize>(
    msg: &[u8],
    count: u64,
    dst: &[u8],
) -> Vec<FieldElement<MontgomeryBackendPrimeField<M, N>>> {
    let order = M::MODULUS;
    let mut u = vec![FieldElement::zero(); count as usize];
    //L = ceil((ceil(log2(p)) + k) / 8), where k is the security parameter of the cryptosystem (e.g. k = ceil(log2(p) / 2))
    let log2_p = (order.limbs.len() * 8) as f64;
    let k = (log2_p / 2.0).ceil() * 8.0;
    let l = (((log2_p * 8.0) + k) / 8.0).ceil() as u64;
    let len_in_bytes = count * l;
    let pseudo_random_bytes = helpers::expand_message(msg, dst, len_in_bytes).unwrap();
    for i in 0..count {
        let elm_offset = l * i;
        let tv = &pseudo_random_bytes[elm_offset as usize..elm_offset as usize + l as usize];

        u[i as usize] = helpers::os2ip::<M, N>(tv);
    }
    u
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        field::{
            element::FieldElement,
            fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        },
        unsigned_integer::element::UnsignedInteger,
    };

    use crate::hash::hash_to_field::hash_to_field;

    type F = MontgomeryBackendPrimeField<U64, 1>;

    #[derive(Clone, Debug)]
    struct U64;
    impl IsModulus<UnsignedInteger<1>> for U64 {
        const MODULUS: UnsignedInteger<1> = UnsignedInteger::from_u64(18446744069414584321_u64);
    }

    #[test]
    fn test_same_message_produce_same_field_elements() {
        let field_elements: Vec<FieldElement<F>> = hash_to_field(b"testing", 40, b"dsttest");
        let other_field_elements = hash_to_field(b"testing", 40, b"dsttest");
        assert_eq!(field_elements, other_field_elements);
    }
}
