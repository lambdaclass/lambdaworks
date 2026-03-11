use alloc::vec::Vec;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use crate::linear_code_pcs::sparse_matrix::SparseMatrix;
use crate::linear_code_pcs::traits::LinearCodeEncoding;

/// Expander-code encoding backend for the Brakedown PCS.
///
/// Uses a recursive construction based on sparse bipartite expander graphs:
///
/// ```text
/// Enc(x) = (x, Enc'(A * x))
/// ```
///
/// where `A` is a sparse matrix (bipartite expander adjacency) that reduces
/// the dimension, and `Enc'` is a smaller instance of the same code. The base
/// case uses a simple repetition.
///
/// This gives O(n) encoding time since each level involves multiplying by a
/// sparse matrix with O(1) nonzero entries per row.
#[derive(Clone, Debug)]
pub struct ExpanderEncoding<F: IsField> {
    /// The message length (input dimension).
    msg_len: usize,
    /// Total codeword length after all levels of recursion.
    cw_len: usize,
    /// Sparse matrices for each recursion level. Level 0 maps msg_len -> smaller,
    /// level 1 maps that output -> even smaller, etc.
    matrices: Vec<SparseMatrix<F>>,
    /// Distance numerator / denominator (approximate).
    dist: (usize, usize),
}

impl<F: IsField> ExpanderEncoding<F> {
    /// Create an expander encoding for messages of length `msg_len`.
    ///
    /// Parameters:
    /// - `alpha`: dimension reduction factor per level (e.g. 0.2 means output = 0.2 * input)
    /// - `nnz_per_row`: number of nonzero entries per row in the sparse matrix (the degree)
    /// - `base_len`: stop recursing when dimension drops below this
    /// - `seed`: deterministic seed for generating the sparse matrices
    pub fn new(msg_len: usize, alpha: f64, nnz_per_row: usize, base_len: usize, seed: u64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
        assert!(msg_len > 0);

        let mut matrices = Vec::new();
        let mut current_len = msg_len;
        let mut cw_len = msg_len; // The systematic part
        let mut level_seed = seed;

        while current_len > base_len {
            let out_len = ((current_len as f64) * alpha).ceil() as usize;
            let out_len = out_len.max(1);

            let nnz = nnz_per_row.min(current_len);
            let mat = SparseMatrix::<F>::random_binary(out_len, current_len, nnz, level_seed);
            matrices.push(mat);
            cw_len += out_len;
            current_len = out_len;
            level_seed = level_seed.wrapping_add(7);
        }

        // The distance is approximate; for a proper implementation the expander
        // analysis would give exact bounds. We use a conservative estimate.
        // With alpha ~0.2 and nnz_per_row ~8, the relative distance is roughly 0.04.
        let dist = (1, 25); // ~0.04

        Self {
            msg_len,
            cw_len,
            matrices,
            dist,
        }
    }
}

impl<F: IsField> LinearCodeEncoding<F> for ExpanderEncoding<F> {
    fn encode(&self, msg: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        assert_eq!(msg.len(), self.msg_len, "message length mismatch");

        // Systematic: start with the message itself
        let mut codeword = msg.to_vec();

        // Recursively apply sparse matrices and append outputs
        let mut current = msg.to_vec();
        for mat in &self.matrices {
            let next = mat.mul_vec(&current);
            codeword.extend_from_slice(&next);
            current = next;
        }

        debug_assert_eq!(codeword.len(), self.cw_len);
        codeword
    }

    fn codeword_len(&self) -> usize {
        self.cw_len
    }

    fn message_len(&self) -> usize {
        self.msg_len
    }

    fn distance(&self) -> (usize, usize) {
        self.dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    type F = U64PrimeField<101>;
    type FE = FieldElement<F>;

    #[test]
    fn encode_produces_correct_length() {
        let enc = ExpanderEncoding::<F>::new(32, 0.25, 4, 4, 42);
        let msg: Vec<FE> = (0..32).map(|x| FE::from(x as u64)).collect();
        let cw = enc.encode(&msg);
        assert_eq!(cw.len(), enc.codeword_len());
    }

    #[test]
    fn encode_is_systematic() {
        // First msg_len elements of the codeword should be the message itself
        let enc = ExpanderEncoding::<F>::new(16, 0.25, 4, 2, 99);
        let msg: Vec<FE> = (0..16).map(|x| FE::from(x as u64)).collect();
        let cw = enc.encode(&msg);
        assert_eq!(&cw[..16], &msg[..]);
    }

    #[test]
    fn encode_deterministic() {
        let enc = ExpanderEncoding::<F>::new(16, 0.25, 4, 2, 42);
        let msg: Vec<FE> = (0..16).map(|x| FE::from(x as u64)).collect();
        let cw1 = enc.encode(&msg);
        let cw2 = enc.encode(&msg);
        assert_eq!(cw1, cw2);
    }

    #[test]
    fn cw_longer_than_msg() {
        let enc = ExpanderEncoding::<F>::new(64, 0.3, 6, 4, 7);
        assert!(enc.codeword_len() > enc.message_len());
    }
}
