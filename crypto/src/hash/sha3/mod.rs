use std::{collections::btree_map::Range, hash, ops::Rem, vec};

use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
    unsigned_integer::element::U256,
};
use sha3::{digest::typenum::Log2, Digest, Sha3_256};

use super::traits::IsCryptoHash;
pub struct Sha3Hasher;

/// Sha3 Hasher used over fields
/// Notice while it's generic over F, it's only generates enough randomness for fields of at most 256 bits
impl Sha3Hasher {
    pub const fn new() -> Self {
        Self
    }

    pub fn hash_to_field<F: IsField>(&self, msg: &[u8], count: u64) -> Vec<FieldElement<F>> {
        let p: f32 = 2.0;
        // L = ceil((ceil(log2(p)) + k) / 8)
        let L = ((((p.log2().ceil() as u64) + 128) / 8) as f64).ceil() as u64;
        let len_in_bytes = count * L;
        let DST = b"a";
        let pseudo_random_bytes = expand_message(msg, DST, len_in_bytes);
    }
}
impl<F: IsField> IsCryptoHash<F> for Sha3Hasher {
    fn hash_one(&self, input: &FieldElement<F>) -> FieldElement<F>
    where
        FieldElement<F>: ByteConversion,
    {
        let mut hasher = Sha3_256::new();
        hasher.update(input.to_bytes_be());
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&hasher.finalize());
        FieldElement::<F>::from_bytes_le(&result_hash).unwrap()
    }

    fn hash_two(&self, left: &FieldElement<F>, right: &FieldElement<F>) -> FieldElement<F>
    where
        FieldElement<F>: ByteConversion,
    {
        let mut hasher = Sha3_256::new();
        hasher.update(left.to_bytes_be());
        hasher.update(right.to_bytes_be());
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&hasher.finalize());
        FieldElement::<F>::from_bytes_le(&result_hash).unwrap()
    }
}

fn expand_message(msg: &[u8], DST: &[u8], len_in_bytes: u64) {
    let mut hasher = Sha3_256::new();
    let mut b = Vec::new();
    let ell = ((len_in_bytes / (256 / 8)) as f64).ceil() as u64;
    if ell > 255 {
        return;
    }
    let DST_prime: Vec<u8> = I2OSP(DST.len() as u64, 1)
        .iter()
        .zip(DST)
        .map(|(a, b)| a | b)
        .collect();
    let Z_pad = I2OSP(0, 64);
    let l_i_b_str = I2OSP(len_in_bytes, 2);

    let b_0_mid: Vec<u8> = Z_pad
        .iter()
        .zip(msg)
        .zip(l_i_b_str)
        .zip(I2OSP(0, 1))
        .zip(DST_prime)
        .map(|((((a, b), c), d), e)| a | b | c | d | e)
        .collect();
    let b_0 = hasher.update(b_0_mid);
    b[0] = &hasher.finalize()[..];

    let b_1_mid: Vec<u8> = b[0]
        .iter()
        .zip(I2OSP(1, 1))
        .zip(DST_prime)
        .map(|((a, b), c)| a | b | c)
        .collect();
    let b_1 = hasher.update(b_1_mid);
    b[1] = &hasher.finalize();

    for i in 2..ell {
        let b_i_mid: Vec<u8> = strxor(b[0], b[(i - 1) as usize])
            .iter()
            .zip(I2OSP(i, 1))
            .zip(DST_prime)
            .map(|((a, b), c)| a | b | c)
            .collect();
        let b_i = hasher.update(b_i_mid);
        let b_i = &hasher.finalize();
        b[i as usize] = b_i;
    }
    let pseudo_random_bytes = b
        .iter()
        .reduce(|acc, b| &&stror(acc, b)[..])
        .unwrap()
        .to_vec();

    // FIXME this line does not compile
    pseudo_random_bytes[0..len_in_bytes]
}

fn I2OSP(x: u64, length: u64) -> Vec<u8> {
    let mut digits = Vec::new();
    while x != 0 {
        digits.push((x % 256) as u8);
        x /= 256;
    }
    for i in 0..(length - digits.len() as u64) {
        digits.push(0);
    }
    digits.reverse();
    digits
}

fn strxor(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(a, b)| a ^ b).collect()
}

fn stror(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(a, b)| a | b).collect()
}
