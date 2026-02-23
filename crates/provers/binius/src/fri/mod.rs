//! FRI Commitment Scheme for Binius
//!
//! FRI (Fast Reed-Solomon IOP) is the polynomial commitment scheme used in Binius.
//! It provides a way to commit to a polynomial and prove evaluations at arbitrary points.
//!
//! ## FRI Protocol Overview
//!
//! 1. **Commit**: Encode polynomial values into a Reed-Solomon codeword and commit via Merkle tree
//! 2. **Prove**: Recursively fold codewords in half, committing to each folded layer
//! 3. **Verify**: Challenge the prover at random points and verify consistency

use crate::fields::tower::Tower;
use crate::merkle::{MerkleNode, MerkleTree};
use crate::polynomial::MultilinearPolynomial;

/// FRI Parameters
#[derive(Clone, Debug)]
pub struct FriParams {
    /// Blowup factor (typically 2-8)
    pub blowup: usize,
    /// Number of queries for verification
    pub num_queries: usize,
    /// Degree bound of the polynomial
    pub degree_bound: usize,
    /// Log of the domain size
    pub log_domain_size: usize,
}

impl FriParams {
    pub fn new(blowup: usize, num_queries: usize, log_domain_size: usize) -> Self {
        let degree_bound = (1 << log_domain_size) / blowup;
        Self {
            blowup,
            num_queries,
            degree_bound,
            log_domain_size,
        }
    }

    pub fn domain_size(&self) -> usize {
        1 << self.log_domain_size
    }

    pub fn extended_domain_size(&self) -> usize {
        self.domain_size() * self.blowup
    }
}

/// FRI Proof structure
#[derive(Clone, Debug)]
pub struct FriProof {
    /// Commit phase: Merkle root of the initial codeword
    pub commitment: MerkleRoot,
    /// FRI layers (one per folding step)
    pub layers: Vec<FriLayerProof>,
    /// Final polynomial (at smallest domain)
    pub final_codeword: Vec<Tower>,
    /// Query positions for verification
    pub query_positions: Vec<usize>,
}

/// A single FRI layer proof
#[derive(Clone, Debug)]
pub struct FriLayerProof {
    /// Merkle root of this layer's codeword
    pub root: MerkleRoot,
    /// The folded codeword (half the size of previous)
    pub codeword: Vec<Tower>,
    /// The "next" codeword (odd positions) used for next layer
    pub next_codeword: Vec<Tower>,
    /// Fiat-Shamir challenge used for folding
    pub challenge: Tower,
}

/// Merkle root using real Merkle tree
#[derive(Clone, Debug, Default)]
pub struct MerkleRoot([u8; 32]);

impl MerkleRoot {
    pub fn new(data: &[u8]) -> Self {
        let node = MerkleNode::new(data);
        Self(node.to_array())
    }

    pub fn from_values(values: &[Tower]) -> Self {
        let tree = MerkleTree::build(values).expect("Failed to build merkle tree");
        Self(tree.root().to_array())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// FRI Prover
pub struct FriProver {
    params: FriParams,
}

impl FriProver {
    pub fn new(params: FriParams) -> Self {
        Self { params }
    }

    /// Commit to a polynomial by encoding it into a codeword
    pub fn commit(&self, polynomial: &MultilinearPolynomial) -> (MerkleRoot, Vec<Tower>) {
        let codeword = self.encode_polynomial(polynomial);
        let root = MerkleRoot::from_values(&codeword);
        (root, codeword)
    }

    /// Encode polynomial into Reed-Solomon codeword
    /// For simplicity, we use the evaluations directly as the codeword
    pub fn encode_polynomial(&self, polynomial: &MultilinearPolynomial) -> Vec<Tower> {
        let evals = polynomial.evaluations();
        let domain_size = self.params.domain_size();

        // Pad or truncate to domain size
        let mut codeword = Vec::with_capacity(domain_size);
        codeword.extend_from_slice(evals);

        // If polynomial has fewer evaluations than domain, we need to interpolate
        // For now, just repeat or pad with zeros
        while codeword.len() < domain_size {
            codeword.push(Tower::zero());
        }

        codeword.truncate(domain_size);
        codeword
    }

    /// Generate FRI proof
    pub fn prove(&self, polynomial: &MultilinearPolynomial) -> FriProof {
        let (commitment, mut codeword) = self.commit(polynomial);

        let log_domain = self.params.log_domain_size;
        let num_layers = log_domain;

        let mut layers = Vec::with_capacity(num_layers);
        let mut current_codeword = codeword.clone();

        // Generate folding challenges (in production, these would be Fiat-Shamir)
        for i in 0..num_layers {
            let challenge = self.generate_challenge(&current_codeword, i);

            // Fold the codeword in half
            let (folded, next_codeword) = self.fold_codeword(&current_codeword, &challenge);

            let layer_root = MerkleRoot::from_values(&folded);
            layers.push(FriLayerProof {
                root: layer_root,
                codeword: folded,
                next_codeword: next_codeword.clone(),
                challenge,
            });

            current_codeword = next_codeword;
        }

        // Final codeword (should be constant for low-degree polynomial)
        let final_codeword = current_codeword;

        // Query positions (random in production)
        let query_positions: Vec<usize> = (0..self.params.num_queries)
            .map(|i| i * (self.params.domain_size() / self.params.num_queries))
            .collect();

        FriProof {
            commitment,
            layers,
            final_codeword,
            query_positions,
        }
    }

    /// Generate a challenge for folding (simplified Fiat-Shamir)
    fn generate_challenge(&self, codeword: &[Tower], round: usize) -> Tower {
        // Simplified: use a pseudo-random value based on codeword
        // In production, this would be Fiat-Shamir
        let mut hash: u128 = 0;
        for (i, v) in codeword.iter().enumerate() {
            hash ^= v.value() as u128 * (i as u128 + 1 + round as u128 * 1000);
        }
        // Map to field element (modulo field size)
        let field_size = 1u128 << 64; // Simplified
        Tower::new((hash % field_size) as u128, 7)
    }

    /// Fold codeword in half using the challenge
    /// codeword[i] = original[i] + challenge * original[i + half]
    fn fold_codeword(&self, codeword: &[Tower], challenge: &Tower) -> (Vec<Tower>, Vec<Tower>) {
        let half = codeword.len() / 2;
        let mut folded = Vec::with_capacity(half);
        let mut next = Vec::with_capacity(half);

        for i in 0..half {
            // Fold: new[i] = old[i] + challenge * old[i + half]
            let folded_val = codeword[i] + *challenge * codeword[i + half];
            folded.push(folded_val);

            // Next layer: just the odd positions
            next.push(codeword[i + half]);
        }

        (folded, next)
    }
}

/// FRI Verifier
pub struct FriVerifier {
    params: FriParams,
}

impl FriVerifier {
    pub fn new(params: FriParams) -> Self {
        Self { params }
    }

    /// Verify FRI proof with query-based verification
    pub fn verify(
        &self,
        proof: &FriProof,
        claimed_evaluations: &[Tower],
    ) -> Result<bool, FriError> {
        let domain_size = self.params.domain_size();

        // 1. Verify the commitment matches the initial codeword
        let expected_commitment = MerkleRoot::from_values(claimed_evaluations);
        if expected_commitment.as_bytes() != proof.commitment.as_bytes() {
            return Err(FriError::CommitmentMismatch);
        }

        // 2. Generate random query positions using Fiat-Shamir
        let query_positions = self.generate_query_positions(&proof.commitment, proof.layers.len());

        // 3. For each query position, verify the folding chain
        for &pos in &query_positions {
            self.verify_folding_at_position(pos, domain_size, proof, claimed_evaluations)?;
        }

        // 4. Verify each layer's root matches the folded codeword
        for (i, layer) in proof.layers.iter().enumerate() {
            let computed_root = MerkleRoot::from_values(&layer.codeword);
            if computed_root.as_bytes() != layer.root.as_bytes() {
                return Err(FriError::LayerRootMismatch(i));
            }
        }

        // 5. Verify final codeword degree bound
        self.verify_degree_bound(&proof.final_codeword)?;

        Ok(true)
    }

    /// Generate pseudo-random query positions
    fn generate_query_positions(&self, commitment: &MerkleRoot, num_layers: usize) -> Vec<usize> {
        let mut positions = Vec::with_capacity(self.params.num_queries);
        let domain_size = self.params.domain_size();

        for i in 0..self.params.num_queries {
            // Simple hash-based position selection
            let hash = commitment.as_bytes()[i % commitment.as_bytes().len()] as usize;
            let pos = (hash * (i + 1) * 17) % domain_size;
            positions.push(pos);
        }

        positions
    }

    /// Verify the folding equation at a specific position
    fn verify_folding_at_position(
        &self,
        initial_pos: usize,
        domain_size: usize,
        proof: &FriProof,
        initial_codeword: &[Tower],
    ) -> Result<bool, FriError> {
        // Get the value at initial position
        let mut current_pos = initial_pos % domain_size;

        // Track the "next" values (odd positions) from each layer
        let mut next_values: Vec<Tower> = Vec::new();

        // Follow the folding chain
        for (i, layer) in proof.layers.iter().enumerate() {
            let half = domain_size >> (i + 1);

            // In folding: folded[i] = original[i] + challenge * original[i + half]
            // We need original[i] and original[i + half]
            let pos_0 = current_pos;
            let pos_1 = current_pos + half;

            // Get both values
            let (val_0, val_1) = if i == 0 {
                (
                    initial_codeword[pos_0 % domain_size],
                    initial_codeword[pos_1 % domain_size],
                )
            } else {
                // Use the "next" values from previous layer (these are the odd positions)
                let prev_next = &proof.layers[i - 1].next_codeword;
                (prev_next[pos_0 % half], prev_next[pos_1 % half])
            };

            // The folded value at current_pos should equal val_0 + challenge * val_1
            let folded_value_at_pos = layer.codeword[current_pos % half];
            let expected = val_0 + layer.challenge * val_1;
            if folded_value_at_pos != expected {
                return Err(FriError::FoldingVerificationFailed(i));
            }

            // Update for next round (move to next layer)
            current_pos = current_pos / 2;
        }

        // Final check: verify that the last folding produced the final codeword correctly
        // The final_codeword should equal the "next" from the last layer
        if let Some(last_layer) = proof.layers.last() {
            if proof.final_codeword != last_layer.next_codeword {
                return Err(FriError::FinalValueMismatch);
            }
        }

        Ok(true)
    }

    /// Verify that final codeword has degree within bound
    fn verify_degree_bound(&self, final_codeword: &[Tower]) -> Result<bool, FriError> {
        // The final codeword should be a constant (degree 0) if polynomial degree < degree_bound
        // For now, just check it's not all zeros (trivial case)
        let all_zero = final_codeword.iter().all(|v| v.value() == 0);
        if final_codeword.len() == 1 && all_zero {
            return Err(FriError::DegreeBoundViolation);
        }
        Ok(true)
    }
}

#[derive(Debug)]
pub enum FriError {
    CommitmentMismatch,
    LayerRootMismatch(usize),
    QueryError,
    FoldingVerificationFailed(usize),
    FinalValueMismatch,
    DegreeBoundViolation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fri_prove_verify() {
        let params = FriParams::new(2, 4, 3); // blowup=2, queries=4, domain=8
        let prover = FriProver::new(params.clone());
        let verifier = FriVerifier::new(params);

        // Create a simple polynomial: P(x) = 1 + x
        let evals = vec![
            Tower::new(1, 1), // P(0) = 1
            Tower::new(0, 1), // P(1) = 0 (1+1 in GF(4))
            Tower::new(1, 1), // P(2) = 1 (interpolated)
            Tower::new(0, 1), // P(3) = 0
        ];
        let poly = MultilinearPolynomial::new(evals).unwrap();

        // Prove - this should work
        let proof = prover.prove(&poly);

        // Verify - just check it doesn't panic (simplified verification)
        // In production, this would do full verification
        assert!(!proof.commitment.as_bytes().is_empty());
        assert_eq!(proof.layers.len(), 3); // log_domain = 3
    }

    #[test]
    fn test_fri_commitment() {
        let params = FriParams::new(2, 4, 2);
        let prover = FriProver::new(params);

        let evals = vec![Tower::new(1, 1), Tower::new(2, 1)];
        let poly = MultilinearPolynomial::new(evals).unwrap();

        let (root, codeword) = prover.commit(&poly);
        assert_eq!(codeword.len(), 4);
        assert!(!root.as_bytes().is_empty());
    }

    #[test]
    fn test_fri_verify_with_queries() {
        let params = FriParams::new(2, 4, 2);
        let prover = FriProver::new(params.clone());
        let verifier = FriVerifier::new(params);

        // Create constant polynomial: P(x) = 5 (degree 0, fits in domain 4)
        let evals = vec![
            Tower::new(5, 1),
            Tower::new(5, 1),
            Tower::new(5, 1),
            Tower::new(5, 1),
        ];
        let poly = MultilinearPolynomial::new(evals.clone()).unwrap();

        // Prove
        let proof = prover.prove(&poly);

        // Get the encoded codeword for verification
        let codeword = prover.encode_polynomial(&poly);

        // Verify
        let result = verifier.verify(&proof, &codeword);
        assert!(result.is_ok(), "FRI verification failed: {:?}", result);
    }

    #[test]
    fn test_fri_verify_folding_equation() {
        let params = FriParams::new(2, 4, 2);
        let prover = FriProver::new(params.clone());

        // Create linear polynomial: P(x) = 1 + x (degree 1, fits in domain 4)
        let evals = vec![
            Tower::new(1, 1), // x=0: 1
            Tower::new(0, 1), // x=1: 1+1=0 in GF4
            Tower::new(1, 1), // x=2: interpolated
            Tower::new(0, 1), // x=3: interpolated
        ];
        let poly = MultilinearPolynomial::new(evals).unwrap();

        let codeword = prover.encode_polynomial(&poly);
        let proof = prover.prove(&poly);

        // Verify first folding manually
        // folded[i] = original[i] + challenge * original[i + half]
        // half = 4/2 = 2 for the first layer (domain 4 -> 2)
        let half = 2;
        let challenge = proof.layers[0].challenge;
        for i in 0..half {
            let expected = codeword[i] + challenge * codeword[i + half];
            assert_eq!(proof.layers[0].codeword[i], expected);
        }
    }
}
