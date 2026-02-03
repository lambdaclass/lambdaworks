//! Transcript trait for Fiat-Shamir transformation in PCS.
//!
//! The transcript provides a way to generate verifier challenges
//! deterministically from the prover's messages, converting an
//! interactive protocol into a non-interactive one.

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Transcript for Fiat-Shamir transformation.
///
/// A transcript accumulates messages from the prover and generates
/// challenges that are deterministically derived from all previous
/// messages. This converts interactive proofs into non-interactive ones.
///
/// # Security
///
/// The transcript must be:
/// - **Binding**: Once a message is appended, it cannot be changed.
/// - **Unpredictable**: Challenges cannot be predicted before appending
///   the messages that determine them.
///
/// # Example
///
/// ```ignore
/// // Prover side
/// transcript.append_bytes(b"commitment", &commitment_bytes);
/// let challenge = transcript.challenge_field_element::<F>(b"challenge");
///
/// // Verifier side (same operations, same challenge)
/// transcript.append_bytes(b"commitment", &commitment_bytes);
/// let challenge = transcript.challenge_field_element::<F>(b"challenge");
/// ```
pub trait PCSTranscript {
    /// Append raw bytes to the transcript.
    ///
    /// # Arguments
    ///
    /// * `label` - Domain separation label for this message.
    /// * `data` - The bytes to append.
    fn append_bytes(&mut self, label: &'static [u8], data: &[u8]);

    /// Append a field element to the transcript.
    ///
    /// # Arguments
    ///
    /// * `label` - Domain separation label for this element.
    /// * `element` - The field element to append.
    fn append_field_element<F: IsField>(&mut self, label: &'static [u8], element: &FieldElement<F>)
    where
        FieldElement<F>: AsBytes;

    /// Get a challenge as a field element.
    ///
    /// The challenge is deterministically derived from all messages
    /// appended so far.
    ///
    /// # Arguments
    ///
    /// * `label` - Domain separation label for this challenge.
    fn challenge_field_element<F: IsField>(&mut self, label: &'static [u8]) -> FieldElement<F>
    where
        FieldElement<F>: FromBytes;

    /// Get multiple challenge field elements.
    ///
    /// # Arguments
    ///
    /// * `label` - Domain separation label for these challenges.
    /// * `count` - Number of challenges to generate.
    #[cfg(feature = "alloc")]
    fn challenge_field_elements<F: IsField>(
        &mut self,
        label: &'static [u8],
        count: usize,
    ) -> Vec<FieldElement<F>>
    where
        F: IsField,
        FieldElement<F>: FromBytes,
    {
        (0..count)
            .map(|_| self.challenge_field_element::<F>(label))
            .collect()
    }

    /// Get a challenge as raw bytes.
    ///
    /// # Arguments
    ///
    /// * `label` - Domain separation label for this challenge.
    /// * `dest` - Buffer to write the challenge bytes into.
    fn challenge_bytes(&mut self, label: &'static [u8], dest: &mut [u8]);
}

/// Trait for types that can be serialized to bytes for transcript.
pub trait AsBytes {
    /// Convert to bytes for transcript.
    fn as_bytes(&self) -> &[u8];
}

/// Trait for types that can be deserialized from bytes.
pub trait FromBytes: Sized {
    /// Create from bytes (for challenge generation).
    ///
    /// This should reduce the bytes modulo the field order.
    fn from_bytes(bytes: &[u8]) -> Self;
}

/// Extension trait for appending PCS-specific types to transcripts.
pub trait PCSTranscriptExt: PCSTranscript {
    /// Append a commitment to the transcript.
    ///
    /// # Arguments
    ///
    /// * `label` - Domain separation label.
    /// * `commitment` - The commitment to append.
    fn append_commitment<C: AsBytes>(&mut self, label: &'static [u8], commitment: &C) {
        self.append_bytes(label, commitment.as_bytes());
    }

    /// Append multiple commitments to the transcript.
    #[cfg(feature = "alloc")]
    fn append_commitments<C: AsBytes>(&mut self, label: &'static [u8], commitments: &[C]) {
        for commitment in commitments {
            self.append_bytes(label, commitment.as_bytes());
        }
    }

    /// Append a proof to the transcript.
    fn append_proof<P: AsBytes>(&mut self, label: &'static [u8], proof: &P) {
        self.append_bytes(label, proof.as_bytes());
    }
}

// Blanket implementation
impl<T: PCSTranscript> PCSTranscriptExt for T {}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock transcript for testing
    struct MockTranscript {
        data: Vec<u8>,
    }

    impl MockTranscript {
        fn new() -> Self {
            Self { data: Vec::new() }
        }
    }

    impl PCSTranscript for MockTranscript {
        fn append_bytes(&mut self, label: &'static [u8], data: &[u8]) {
            self.data.extend_from_slice(label);
            self.data.extend_from_slice(data);
        }

        fn append_field_element<F: IsField>(
            &mut self,
            label: &'static [u8],
            _element: &FieldElement<F>,
        ) where
            FieldElement<F>: AsBytes,
        {
            self.data.extend_from_slice(label);
            // In a real implementation, serialize the field element
        }

        fn challenge_field_element<F: IsField>(&mut self, label: &'static [u8]) -> FieldElement<F>
        where
            F: IsField,
            FieldElement<F>: FromBytes,
        {
            self.data.extend_from_slice(label);
            // In a real implementation, hash and reduce
            FieldElement::zero()
        }

        fn challenge_bytes(&mut self, label: &'static [u8], dest: &mut [u8]) {
            self.data.extend_from_slice(label);
            // In a real implementation, hash to fill dest
            dest.fill(0);
        }
    }

    #[test]
    fn test_mock_transcript() {
        let mut transcript = MockTranscript::new();
        transcript.append_bytes(b"test", &[1, 2, 3]);
        assert!(!transcript.data.is_empty());
    }
}
