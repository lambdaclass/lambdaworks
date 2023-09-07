use sha3::{Digest, Keccak256};

/// Build data with the concatenation of transcript hash and value.
/// Computes the hash of this element and returns the number of
/// leading zeros in the resulting value (in the big-endian representation).
///
/// # Parameters
///
/// * `transcript_challenge` - the hash value obtained from the transcript
/// * `value` - the value to be concatenated with the transcript hash
/// (i.e. a candidate nonce).
///
/// # Returns
///
/// The number of leading zeros in the resulting hash value.
#[inline(always)]
pub fn hash_transcript_with_int_and_get_leading_zeros(
    transcript_challenge: &[u8; 32],
    value: u64,
) -> u8 {
    let mut data = [0; 40];
    data[..32].copy_from_slice(transcript_challenge);
    data[32..].copy_from_slice(&value.to_le_bytes());

    let digest = Keccak256::digest(data);

    let seed_head = u64::from_be_bytes(digest[..8].try_into().unwrap());
    seed_head.trailing_zeros() as u8
}

/// Performs grinding, generating a new nonce for the proof.
/// The nonce generated is such that:
/// Hash(transcript_hash || nonce) has a number of leading zeros
/// greater or equal than `grinding_factor`.
///
/// # Parameters
///
/// * `transcript` - the hash of the transcript
/// * `grinding_factor` - the number of leading zeros needed
pub fn generate_nonce_with_grinding(
    transcript_challenge: &[u8; 32],
    grinding_factor: u8,
) -> Option<u64> {
    (0..u64::MAX).find(|&candidate_nonce| {
        hash_transcript_with_int_and_get_leading_zeros(transcript_challenge, candidate_nonce)
            >= grinding_factor
    })
}

#[cfg(test)]
mod test {
    use sha3::{Digest, Keccak256};

    #[test]
    fn hash_transcript_with_int_and_get_leading_zeros_works() {
        let transcript_challenge = [
            226_u8, 27, 133, 168, 62, 203, 20, 59, 122, 230, 227, 33, 76, 44, 53, 150, 200, 45,
            136, 162, 249, 239, 142, 90, 204, 191, 45, 4, 53, 22, 103, 240,
        ];
        let grinding_factor = 10;

        let nonce =
            super::generate_nonce_with_grinding(&transcript_challenge, grinding_factor).unwrap();
        assert_eq!(nonce, 33);

        // check generated hash has more trailing_zeros than grinding_factor
        let mut data = [0; 40];
        data[..32].copy_from_slice(&transcript_challenge);
        data[32..].copy_from_slice(&nonce.to_le_bytes());

        let digest = Keccak256::digest(data);

        let seed_head = u64::from_be_bytes(digest[..8].try_into().unwrap());
        let trailing_zeors = seed_head.trailing_zeros() as u8;

        assert!(trailing_zeors >= grinding_factor);
    }
}
