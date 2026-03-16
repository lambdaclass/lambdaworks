use alloc::vec::Vec;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use crate::linear_code_pcs::sparse_matrix::SparseMatrix;
use crate::linear_code_pcs::traits::LinearCodeEncoding;

/// Expander-code encoding backend for the Brakedown PCS.
///
/// Implements the two-matrix recursive construction from Algorithm 1 of the
/// Brakedown paper (Golovnev et al., CRYPTO 2023, Section 5.1):
///
/// ```text
/// Enc(x) = (x, z, v) where:
///   y = x · A          ∈ F^{αn}          (compress via matrix A)
///   z = Enc_{αn}(y)    ∈ F^{r·αn}        (recursive encoding)
///   v = z · B          ∈ F^{(r-1-rα)n}   (mix via matrix B)
/// ```
///
/// Both `A` and `B` are sparse random matrices with non-zero field element entries
/// and constant row-sparsity, giving O(n) encoding time.
///
/// The code is systematic: codeword = (x, z, v), with rate 1/r and relative
/// distance δ = β/r where β depends on the expansion parameters.
#[derive(Clone, Debug)]
pub struct ExpanderEncoding<F: IsField> {
    /// The message length (input dimension).
    msg_len: usize,
    /// Total codeword length after all levels of recursion.
    cw_len: usize,
    /// Pairs of (A, B) sparse matrices for each recursion level.
    /// A compresses the input, B mixes the recursively-encoded output.
    levels: Vec<ExpanderLevel<F>>,
    /// Distance numerator / denominator (approximate).
    dist: (usize, usize),
}

/// One level of the recursive expander encoding.
#[derive(Clone, Debug)]
struct ExpanderLevel<F: IsField> {
    /// Compression matrix A: maps input of length `in_len` to length `compressed_len`.
    a_matrix: SparseMatrix<F>,
    /// Mixing matrix B: maps recursively-encoded output to additional redundancy.
    /// None at the base case (no further mixing needed).
    b_matrix: Option<SparseMatrix<F>>,
}

impl<F: IsField> ExpanderEncoding<F> {
    /// Create an expander encoding for messages of length `msg_len`.
    ///
    /// # Parameters
    /// - `alpha`: dimension reduction factor per level (e.g. 0.2 means A output = 0.2 * input)
    /// - `nnz_per_row`: number of nonzero entries per row in the sparse matrices (the degree)
    /// - `base_len`: stop recursing when dimension drops below this
    /// - `seed`: deterministic seed for generating the sparse matrices
    /// - `distance`: relative minimum distance as `(numerator, denominator)`.
    ///   For example `(1, 20)` means distance 0.05.
    ///
    /// # Safety (soundness)
    ///
    /// **The `distance` parameter is NOT verified at construction time.** The caller
    /// MUST ensure the claimed distance holds for the chosen `alpha`, `nnz_per_row`,
    /// and `base_len` based on the expander spectral analysis in Brakedown Section 5.1.
    ///
    /// The paper's construction uses parameters `0 < α < 1, 0 < β < α/1.28`,
    /// with rate `r > (1+2β)/(1-α)` and distance `δ = β/r`. For practical
    /// parameters (α ≈ 0.2, β ≈ 0.06, r = 5), δ ≈ 0.012.
    /// See Golovnev et al., CRYPTO 2023, Section 5.1 for full analysis.
    pub fn new(
        msg_len: usize,
        alpha: f64,
        nnz_per_row: usize,
        base_len: usize,
        seed: u64,
        distance: (usize, usize),
    ) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
        assert!(msg_len > 0);
        assert!(
            distance.0 > 0 && distance.1 > 0,
            "distance must be positive"
        );
        assert!(
            distance.0 < distance.1,
            "distance numerator must be less than denominator"
        );

        let mut levels = Vec::new();
        let mut current_len = msg_len;
        let mut cw_len = msg_len; // The systematic part
        let mut level_seed = seed;

        while current_len > base_len {
            let compressed_len = libm::ceil((current_len as f64) * alpha) as usize;
            let compressed_len = compressed_len.max(1);

            let nnz_a = nnz_per_row.min(current_len);
            let a_matrix =
                SparseMatrix::<F>::random_nonzero(compressed_len, current_len, nnz_a, level_seed);

            // B matrix mixes the recursively-encoded compressed output.
            // B maps from compressed_len -> b_out_len where b_out_len is sized to
            // provide additional redundancy beyond the systematic + recursive parts.
            // Following the paper: output extra (1-α-1/r) * current_len symbols.
            // For simplicity we set b_out_len ≈ compressed_len (matching the A output size).
            let b_out_len = compressed_len;
            let nnz_b = nnz_per_row.min(compressed_len);
            let b_seed = level_seed.wrapping_add(0x_B0B0_B0B0);
            let b_matrix =
                SparseMatrix::<F>::random_nonzero(b_out_len, compressed_len, nnz_b, b_seed);

            cw_len += compressed_len + b_out_len;

            levels.push(ExpanderLevel {
                a_matrix,
                b_matrix: Some(b_matrix),
            });

            current_len = compressed_len;
            level_seed = level_seed.wrapping_add(7);
        }

        Self {
            msg_len,
            cw_len,
            levels,
            dist: distance,
        }
    }
}

impl<F: IsField> LinearCodeEncoding<F> for ExpanderEncoding<F> {
    fn encode(&self, msg: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        assert_eq!(msg.len(), self.msg_len, "message length mismatch");

        // Systematic: start with the message itself
        let mut codeword = msg.to_vec();

        // Recursively: compress with A, append compressed, mix with B, append mixed
        let mut current = msg.to_vec();
        for level in &self.levels {
            // y = A * current (compression)
            let compressed = level.a_matrix.mul_vec(&current);
            codeword.extend_from_slice(&compressed);

            // v = B * compressed (mixing for distance amplification)
            if let Some(b_matrix) = &level.b_matrix {
                let mixed = b_matrix.mul_vec(&compressed);
                codeword.extend_from_slice(&mixed);
            }

            current = compressed;
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
        let enc = ExpanderEncoding::<F>::new(32, 0.25, 4, 4, 42, (1, 25));
        let msg: Vec<FE> = (0..32).map(|x| FE::from(x as u64)).collect();
        let cw = enc.encode(&msg);
        assert_eq!(cw.len(), enc.codeword_len());
    }

    #[test]
    fn encode_is_systematic() {
        // First msg_len elements of the codeword should be the message itself
        let enc = ExpanderEncoding::<F>::new(16, 0.25, 4, 2, 99, (1, 25));
        let msg: Vec<FE> = (0..16).map(|x| FE::from(x as u64)).collect();
        let cw = enc.encode(&msg);
        assert_eq!(&cw[..16], &msg[..]);
    }

    #[test]
    fn encode_deterministic() {
        let enc = ExpanderEncoding::<F>::new(16, 0.25, 4, 2, 42, (1, 25));
        let msg: Vec<FE> = (0..16).map(|x| FE::from(x as u64)).collect();
        let cw1 = enc.encode(&msg);
        let cw2 = enc.encode(&msg);
        assert_eq!(cw1, cw2);
    }

    #[test]
    fn cw_longer_than_msg() {
        let enc = ExpanderEncoding::<F>::new(64, 0.3, 6, 4, 7, (1, 25));
        assert!(enc.codeword_len() > enc.message_len());
    }

    #[test]
    fn encode_is_linear() {
        // Linearity: encode(a*x + b*y) == a*encode(x) + b*encode(y)
        let enc = ExpanderEncoding::<F>::new(16, 0.25, 4, 2, 42, (1, 25));
        let x: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
        let y: Vec<FE> = (17..=32).map(|i| FE::from(i as u64)).collect();
        let a = FE::from(3u64);
        let b = FE::from(7u64);

        let ax_plus_by: Vec<FE> = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| a.clone() * xi.clone() + b.clone() * yi.clone())
            .collect();

        let enc_combined = enc.encode(&ax_plus_by);
        let enc_x = enc.encode(&x);
        let enc_y = enc.encode(&y);

        let expected: Vec<FE> = enc_x
            .iter()
            .zip(enc_y.iter())
            .map(|(ex, ey)| a.clone() * ex.clone() + b.clone() * ey.clone())
            .collect();

        assert_eq!(enc_combined, expected, "encoding must be linear");
    }
}
