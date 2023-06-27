use super::transcript::Transcript;
use sha3::{Digest, Keccak256};

pub struct DefaultTranscript {
    hasher: Keccak256,
}

impl Transcript for DefaultTranscript {
    fn append(&mut self, new_data: &[u8]) {
        self.hasher.update(&mut new_data.to_owned());
    }

    fn challenge(&mut self) -> [u8; 32] {
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&self.hasher.finalize_reset());
        result_hash.reverse();
        self.hasher.update(result_hash);
        result_hash
    }
}

impl Default for DefaultTranscript {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultTranscript {
    pub fn new() -> Self {
        Self {
            hasher: Keccak256::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_challenge() {
        let mut transcript = DefaultTranscript::new();

        let point_a: Vec<u8> = vec![0xFF, 0xAB];
        let point_b: Vec<u8> = vec![0xDD, 0x8C, 0x9D];

        transcript.append(&point_a); // point_a
        transcript.append(&point_b); // point_a + point_b

        let challenge1 = transcript.challenge(); // Hash(point_a  + point_b)

        assert_eq!(
            challenge1,
            [
                0x0c, 0x2b, 0xd8, 0xcf, 0x2d, 0x71, 0xe0, 0x0a, 0xce, 0xa3, 0xbd, 0x5d, 0xc7, 0x9f,
                0x4f, 0x93, 0xed, 0x57, 0x42, 0xd0, 0x23, 0xbd, 0x47, 0xc9, 0x04, 0xc2, 0x67, 0x9d,
                0xbc, 0xfa, 0x7c, 0xa7
            ]
        );

        let point_c: Vec<u8> = vec![0xFF, 0xAB];
        let point_d: Vec<u8> = vec![0xDD, 0x8C, 0x9D];

        transcript.append(&point_c); // Hash(point_a  + point_b) + point_c
        transcript.append(&point_d); // Hash(point_a  + point_b) + point_c + point_d

        let challenge2 = transcript.challenge(); // Hash(Hash(point_a  + point_b) + point_c + point_d)
        assert_eq!(
            challenge2,
            [
                0x81, 0x61, 0x51, 0xc5, 0x7e, 0xcb, 0x45, 0xd5, 0x17, 0x1a, 0x3c, 0x2e, 0x38, 0x04,
                0x5d, 0xfb, 0x3a, 0x3d, 0x33, 0x8a, 0x22, 0xaf, 0xf8, 0x60, 0x85, 0xb9, 0x54, 0x3f,
                0xf8, 0x32, 0x32, 0xbc
            ]
        );
    }
}
