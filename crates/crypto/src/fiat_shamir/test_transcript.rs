/// This transcript will ALWAYS return the exact same value every time it's called.
/// It is meant for testing only, never use this in production.
///
/// Note: This is a simple test transcript that provides deterministic outputs.
/// It does not implement the full IsTranscript trait as it's designed for basic testing.
pub struct TestTranscript;

impl TestTranscript {
    /// Append data (no-op for test transcript)
    pub fn append(&mut self, _new_data: &[u8]) {}

    /// Returns a deterministic challenge (all 1s)
    pub fn challenge(&mut self) -> [u8; 32] {
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
