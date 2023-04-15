use super::transcript::Transcript;

/// This transcript will ALWAYS return the exact same value every time it's called.
/// It is meant for testing only, never use this in production.
pub struct TestTranscript;

impl Transcript for TestTranscript {
    fn append(&mut self, _new_data: &[u8]) {}

    fn challenge(&mut self) -> [u8; 32] {
        [1; 32]
    }
}

impl Default for TestTranscript {
    fn default() -> Self {
        Self::new()
    }
}

impl TestTranscript {
    pub fn new() -> Self {
        Self {}
    }
}
