use std::fmt::Debug;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::montgomery_backed_prime_fields::IsModulus;
use lambdaworks_math::field::fields::montgomery_backed_prime_fields::MontgomeryBackendPrimeField;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use sha3::Digest;
use sha3::Sha3_256;

pub fn expand_message(msg: &[u8], dst: &[u8], len_in_bytes: u64) -> Result<Vec<u8>, String> {
    let b_in_bytes = Sha3_256::output_size() as u64;

    let ell = (len_in_bytes + b_in_bytes - 1) / b_in_bytes;
    if ell > 255 {
        return Err("Abort".to_string());
    }

    let dst_prime: Vec<u8> = [dst, &i2osp(dst.len() as u64, 1)].concat();
    let z_pad = i2osp(0, 64);
    let l_i_b_str = i2osp(len_in_bytes, 2);
    let msg_prime = [
        z_pad,
        msg.to_vec(),
        l_i_b_str,
        i2osp(0, 1),
        dst_prime.clone(),
    ]
    .concat();
    let b_0: Vec<u8> = Sha3_256::digest(msg_prime).to_vec();
    let a = [b_0.clone(), i2osp(1, 1), dst_prime.clone()].concat();
    let b_1 = Sha3_256::digest(a).to_vec();

    let mut b_vals = Vec::<Vec<u8>>::with_capacity(ell as usize * b_in_bytes as usize);
    b_vals.push(b_1);
    for idx in 1..ell {
        let aux = strxor(&b_0, &b_vals[idx as usize - 1]);
        let b_i = [aux, i2osp(idx, 1), dst_prime.clone()].concat();
        b_vals.push(Sha3_256::digest(b_i).to_vec());
    }

    let mut b_vals = b_vals.concat();
    b_vals.truncate(len_in_bytes as usize);

    Ok(b_vals)
}

pub fn i2osp(x: u64, length: u64) -> Vec<u8> {
    let mut x_aux = x;
    let mut digits = Vec::new();
    while x_aux != 0 {
        digits.push((x_aux % 256) as u8);
        x_aux /= 256;
    }
    digits.resize(digits.len() + (length - digits.len() as u64) as usize, 0);
    digits.reverse();
    digits
}

pub fn os2ip<M: IsModulus<UnsignedInteger<N>> + Clone + Debug, const N: usize>(
    x: &[u8],
) -> FieldElement<MontgomeryBackendPrimeField<M, N>> {
    let mut aux_x = x.to_vec();
    aux_x.reverse();
    let two_to_the_nth = build_two_to_the_nth();
    let mut j = 0_u32;
    let mut item_hex = String::with_capacity(N * 16);
    let mut result = FieldElement::zero();
    for item_u8 in aux_x.iter() {
        item_hex += &format!("{:x}", item_u8);
        if item_hex.len() == item_hex.capacity() {
            result += FieldElement::from_hex(&item_hex) * two_to_the_nth.pow(j);
            item_hex.clear();
            j += 1;
        }
    }
    result
}

pub fn strxor(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(a, b)| a ^ b).collect()
}

/// Builds a `FieldElement` for `2^(N*16)`, where `N` is the number of limbs of the `UnsignedInteger`
/// used for the prime field.
fn build_two_to_the_nth<M: IsModulus<UnsignedInteger<N>> + Clone + Debug, const N: usize>(
) -> FieldElement<MontgomeryBackendPrimeField<M, N>> {
    // The hex used to build the FieldElement is a 1 followed by N * 16 zeros
    let mut two_to_the_nth = String::with_capacity(N * 16);
    for _ in 0..two_to_the_nth.capacity() - 1 {
        two_to_the_nth.push('1');
    }
    FieldElement::from_hex(&two_to_the_nth) + FieldElement::one()
}
