use sha2::{Digest, Sha256, Sha224};

use super::transcript::Transcript;

pub struct EmbeddedTranscript {
    hasher: Sha224,
}

impl Transcript for EmbeddedTranscript {
    fn append(&mut self, new_data: &[u8]) {
        self.hasher.update(&mut new_data.to_owned());
    }

    fn challenge(&mut self) -> [u8; 28] {
        let mut result_hash = [0_u8; 28];
        result_hash.copy_from_slice(&self.hasher.finalize_reset());
        self.hasher.update(result_hash);
        result_hash
    }
}

impl Default for EmbeddedTranscript {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddedTranscript {
    pub fn new() -> Self {
        Self {
            hasher: Sha224::new(),
        }
    }
}
