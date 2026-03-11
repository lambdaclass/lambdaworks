use alloc::vec::Vec;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

/// A linear error-correcting code used as the encoding backend for the
/// Ligero/Brakedown polynomial commitment scheme.
///
/// Implementors provide:
/// - **Reed-Solomon encoding** (Ligero): O(n log n) via FFT, requires an FFT-friendly field.
/// - **Expander code encoding** (Brakedown): O(n) via sparse matrix multiplication, field-agnostic.
pub trait LinearCodeEncoding<F: IsField>: Clone {
    /// Encode a message of length `message_len()` into a codeword of length `codeword_len()`.
    fn encode(&self, msg: &[FieldElement<F>]) -> Vec<FieldElement<F>>;

    /// Length of the codeword produced by `encode`.
    fn codeword_len(&self) -> usize;

    /// Length of the message accepted by `encode`.
    fn message_len(&self) -> usize;

    /// Relative minimum distance of the code as `(numerator, denominator)`.
    /// For RS with rate rho, this is `(1 - rho)`, e.g. `(1, 2)` for rate 1/2.
    fn distance(&self) -> (usize, usize);
}
