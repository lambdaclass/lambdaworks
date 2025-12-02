use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use sha3::{Digest, Sha3_256};

pub struct Sha3Hasher;

/// Sha3 Hasher used over fields
/// Notice while it's generic over F, it's only generates enough randomness for fields of at most 256 bits
impl Sha3Hasher {
    pub const fn new() -> Self {
        Self
    }

    pub fn expand_message(msg: &[u8], dst: &[u8], len_in_bytes: u64) -> Result<Vec<u8>, String> {
        let b_in_bytes = u64::try_from(Sha3_256::output_size())
            .expect("Sha3_256::output_size() <= u64::MAX this should never fail");

        let ell = len_in_bytes.div_ceil(b_in_bytes);
        if ell > 255 {
            return Err("Abort".to_string());
        }
        let dst_64 = u64::try_from(dst.len()).map_err(|e| format!("size conversion error: {e}"))?;
        let dst_prime: Vec<u8> = [dst, &Self::i2osp(dst_64, 1)?].concat();
        let z_pad = Self::i2osp(0, 64)?;
        let l_i_b_str = Self::i2osp(len_in_bytes, 2)?;
        let msg_prime = [
            z_pad,
            msg.to_vec(),
            l_i_b_str,
            Self::i2osp(0, 1)?,
            dst_prime.clone(),
        ]
        .concat();
        let b_0: Vec<u8> = Sha3_256::digest(msg_prime).to_vec();
        let a = [b_0.clone(), Self::i2osp(1, 1)?, dst_prime.clone()].concat();
        let b_1 = Sha3_256::digest(a).to_vec();

        let mut b_vals = Vec::<Vec<u8>>::with_capacity(
            usize::try_from(ell).expect("as 0 <=ell<= 255 this should never fail"),
        );
        b_vals.push(b_1);
        for idx in 1..ell {
            let aux = Self::strxor(
                &b_0,
                &b_vals
                    [usize::try_from(idx).expect("as idx < ell<= 256 this should never fail") - 1],
            );
            let b_i = [aux, Self::i2osp(idx, 1)?, dst_prime.clone()].concat();
            b_vals.push(Sha3_256::digest(b_i).to_vec());
        }

        let mut b_vals = b_vals.concat();
        b_vals.truncate(
            usize::try_from(len_in_bytes).map_err(|e| format!("size conversion error: {e}"))?,
        );

        Ok(b_vals)
    }

    fn i2osp(x: u64, length: u64) -> Result<Vec<u8>, String> {
        let mut x_aux = x;
        let mut digits = Vec::new();
        while x_aux != 0 {
            digits.push(
                u8::try_from(x_aux % 256)
                    .expect("as (x_aux % 256) <= u8::MAX this should never fail"),
            );
            x_aux /= 256;
        }
        let length = usize::try_from(length).map_err(|e| format!("size conversion error: {e}"))?;
        digits.resize(length, 0);
        digits.reverse();
        Ok(digits)
    }

    fn strxor(a: &[u8], b: &[u8]) -> Vec<u8> {
        a.iter().zip(b).map(|(a, b)| a ^ b).collect()
    }
}

impl Default for Sha3Hasher {
    fn default() -> Self {
        Self::new()
    }
}
