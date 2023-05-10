use sha3::{Digest, Sha3_256};

use super::transcript::Transcript;

pub struct DefaultTranscript {
    hasher: Sha3_256,
}

impl Transcript for DefaultTranscript {
    fn append(&mut self, new_data: &[u8]) {
        self.hasher.update(&mut new_data.to_owned());
    }

    fn challenge(&mut self) -> [u8; 32] {
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&self.hasher.finalize_reset());
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
            hasher: Sha3_256::new(),
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
                0x30, 0xfc, 0x05, 0x69, 0x13, 0x60, 0xe0, 0xbd, 0xa0, 0x6a, 0x0b, 0x5f, 0x5f, 0xf5,
                0xa6, 0x3a, 0x67, 0x48, 0x62, 0x9c, 0x90, 0x8f, 0xef, 0x25, 0x2c, 0x8f, 0x8b, 0x9e,
                0xb8, 0xd2, 0xce, 0x43
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
                0x51, 0x50, 0x15, 0xd7, 0xe3, 0x59, 0x5a, 0x19, 0x58, 0xc9, 0x8e, 0xf4, 0x36, 0xb5,
                0xf7, 0xea, 0x36, 0x62, 0x16, 0xfa, 0x6b, 0xfb, 0x7a, 0xe6, 0x38, 0xa8, 0xc4, 0x59,
                0x79, 0x1b, 0x54, 0xe3
            ]
        );
    }
}
