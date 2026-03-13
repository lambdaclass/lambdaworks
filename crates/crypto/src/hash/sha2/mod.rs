use alloc::vec::Vec;
use core::fmt;
use sha2::{Digest, Sha256};

/// Error type for SHA-256 message expansion
#[derive(Debug)]
pub enum Sha2Error {
    /// Message expansion length too large
    ExpansionLengthTooLarge(u64),
    /// DST longer than 255 bytes (RFC 9380 §5.3.1)
    DstTooLong(usize),
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
            Sha2Error::DstTooLong(len) => {
                write!(
                    f,
                    "DST length {} exceeds maximum of 255 bytes (RFC 9380 §5.3.1)",
                    len
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
    ///
    /// Uses incremental hashing to avoid intermediate allocations.
    pub fn expand_message(msg: &[u8], dst: &[u8], len_in_bytes: u64) -> Result<Vec<u8>, Sha2Error> {
        let ell = len_in_bytes.div_ceil(32);
        if ell > 255 {
            return Err(Sha2Error::ExpansionLengthTooLarge(ell));
        }
        if dst.len() > 255 {
            return Err(Sha2Error::DstTooLong(dst.len()));
        }

        let dst_len_byte = [dst.len() as u8];
        let lib_str = (len_in_bytes as u16).to_be_bytes();

        // b_0 = H(Z_pad || msg || l_i_b_str || I2OSP(0,1) || DST_prime)
        let mut h = Sha256::new();
        h.update([0u8; 64]);
        h.update(msg);
        h.update(lib_str);
        h.update([0u8]);
        h.update(dst);
        h.update(dst_len_byte);
        let b_0: [u8; 32] = h.finalize().into();

        // b_1 = H(b_0 || I2OSP(1,1) || DST_prime)
        let mut h = Sha256::new();
        h.update(b_0);
        h.update([1u8]);
        h.update(dst);
        h.update(dst_len_byte);
        let mut prev: [u8; 32] = h.finalize().into();

        let mut result = Vec::with_capacity(len_in_bytes as usize);
        result.extend_from_slice(&prev);

        for idx in 1..ell {
            // b_i = H(strxor(b_0, b_{i-1}) || I2OSP(i+1,1) || DST_prime)
            let mut xored = [0u8; 32];
            for j in 0..32 {
                xored[j] = b_0[j] ^ prev[j];
            }
            let mut h = Sha256::new();
            h.update(xored);
            h.update([(idx + 1) as u8]);
            h.update(dst);
            h.update(dst_len_byte);
            prev = h.finalize().into();
            result.extend_from_slice(&prev);
        }

        result.truncate(len_in_bytes as usize);
        Ok(result)
    }
}
