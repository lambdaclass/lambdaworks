use alloc::vec::Vec;
use core::fmt;
use sha2::{Digest, Sha256};

/// Error type for SHA-256 message expansion
#[derive(Debug)]
pub enum Sha2Error {
    /// Message expansion length too large
    ExpansionLengthTooLarge(u64),
}

impl fmt::Display for Sha2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sha2Error::ExpansionLengthTooLarge(ell) => {
                write!(
                    f,
                    "Message expansion length too large: ell value {} exceeds maximum of 255",
                    ell
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Sha2Error {}

/// SHA-256 based hasher implementing expand_message_xmd from RFC 9380.
pub struct Sha2Hasher;

impl Sha2Hasher {
    /// expand_message_xmd using SHA-256 (RFC 9380 Section 5.3.1).
    pub fn expand_message(msg: &[u8], dst: &[u8], len_in_bytes: u64) -> Result<Vec<u8>, Sha2Error> {
        let b_in_bytes = <Sha256 as Digest>::output_size() as u64; // 32

        let ell = len_in_bytes.div_ceil(b_in_bytes);
        if ell > 255 {
            return Err(Sha2Error::ExpansionLengthTooLarge(ell));
        }

        let dst_prime: Vec<u8> = [dst, &i2osp(dst.len() as u64, 1)].concat();
        let z_pad = i2osp(0, 64);
        let l_i_b_str = i2osp(len_in_bytes, 2);
        let msg_prime = [&z_pad[..], msg, &l_i_b_str, &i2osp(0, 1), &dst_prime].concat();
        let b_0: Vec<u8> = Sha256::digest(&msg_prime).to_vec();
        let b_1 = Sha256::digest([&b_0[..], &i2osp(1, 1), &dst_prime].concat()).to_vec();

        let mut b_vals = Vec::<Vec<u8>>::with_capacity(ell as usize);
        b_vals.push(b_1);
        for idx in 1..ell {
            let aux = strxor(&b_0, &b_vals[idx as usize - 1]);
            let b_i = [&aux[..], &i2osp(idx + 1, 1), &dst_prime].concat();
            b_vals.push(Sha256::digest(&b_i).to_vec());
        }

        let mut result = b_vals.concat();
        result.truncate(len_in_bytes as usize);

        Ok(result)
    }
}

fn i2osp(x: u64, length: u64) -> Vec<u8> {
    let bytes = x.to_be_bytes();
    let len = length as usize;
    if len >= 8 {
        let mut result = vec![0u8; len - 8];
        result.extend_from_slice(&bytes);
        result
    } else {
        bytes[8 - len..].to_vec()
    }
}

fn strxor(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(a, b)| a ^ b).collect()
}
