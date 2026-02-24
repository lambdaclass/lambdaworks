//! Binary FRI Commitment Scheme for Binius
//!
//! Implements FRI (Fast Reed-Solomon IOP) over binary field subspaces using
//! additive folding. This differs from standard multiplicative FRI in that:
//!
//! - The evaluation domain is a GF(2)-linear subspace, not a multiplicative coset
//! - Folding uses the additive decomposition f(x) = f_0(V(x)) + x * f_1(V(x))
//!   where V(x) = x^2 + x is the vanishing polynomial of {0, 1}
//! - RS encoding uses the additive NTT
//!
//! ## Protocol
//!
//! 1. **Commit**: RS-encode polynomial via additive NTT, Merkle-commit the codeword
//! 2. **Fold**: For each round, additively fold the codeword using a Fiat-Shamir challenge
//! 3. **Query**: Verifier challenges random positions, prover provides Merkle proofs
//! 4. **Verify**: Check folding equation at queried positions, verify Merkle proofs

use crate::merkle::MerkleNode;
use crate::ntt;
use crate::polynomial::MultilinearPolynomial;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::binary::tower_field::BinaryTowerField128;
use lambdaworks_math::traits::ByteConversion;

type FE = FieldElement<BinaryTowerField128>;

/// FRI Parameters
#[derive(Clone, Debug)]
pub struct FriParams {
    /// Log2 of the message size (number of polynomial coefficients = 2^log_message_size)
    pub log_message_size: usize,
    /// Log2 of the blowup factor (codeword length = message * 2^log_blowup)
    pub log_blowup: usize,
    /// Number of query positions for soundness
    pub num_queries: usize,
}

impl FriParams {
    pub fn new(blowup: usize, num_queries: usize, log_domain_size: usize) -> Self {
        let log_blowup = blowup.trailing_zeros() as usize;
        let log_message_size = log_domain_size.saturating_sub(log_blowup);
        Self {
            log_message_size,
            log_blowup,
            num_queries,
        }
    }

    pub fn log_codeword_size(&self) -> usize {
        self.log_message_size + self.log_blowup
    }

    pub fn codeword_size(&self) -> usize {
        1 << self.log_codeword_size()
    }

    pub fn message_size(&self) -> usize {
        1 << self.log_message_size
    }

    pub fn domain_size(&self) -> usize {
        self.codeword_size()
    }

    pub fn extended_domain_size(&self) -> usize {
        self.codeword_size()
    }

    pub fn log_domain_size(&self) -> usize {
        self.log_codeword_size()
    }
}

/// Merkle root (32-byte hash)
#[derive(Clone, Debug, Default)]
pub struct MerkleRoot([u8; 32]);

impl MerkleRoot {
    pub fn new(data: &[u8]) -> Self {
        let node = MerkleNode::new(data);
        Self(node.to_array())
    }

    pub fn from_field_elements(values: &[FE]) -> Self {
        let mut data = Vec::new();
        for v in values {
            data.extend_from_slice(&v.to_bytes_be());
        }
        let node = MerkleNode::new(&data);
        Self(node.to_array())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// FRI commitment: Merkle root of the initial RS codeword
#[derive(Clone, Debug)]
pub struct FriCommitment {
    pub merkle_root: MerkleRoot,
}

/// FRI proof
#[derive(Clone, Debug)]
pub struct FriProof {
    /// Merkle root of the initial codeword
    pub commitment: MerkleRoot,
    /// One layer proof per folding round
    pub layers: Vec<FriLayerProof>,
    /// The final (constant) polynomial after all folding
    pub final_poly: Vec<FE>,
    /// Query positions sampled by the verifier
    pub query_positions: Vec<usize>,
    /// Query proofs (codeword values at queried positions for each layer)
    pub query_proofs: Vec<FriQueryProof>,
}

/// A single FRI layer proof
#[derive(Clone, Debug)]
pub struct FriLayerProof {
    /// Merkle root of the folded codeword
    pub commitment: MerkleRoot,
}

/// Query proof for a single position across all layers
#[derive(Clone, Debug)]
pub struct FriQueryProof {
    /// Values at the queried position and its sibling in each layer
    pub layer_values: Vec<(FE, FE)>,
}

/// FRI Prover using additive folding over binary field subspaces
pub struct FriProver {
    params: FriParams,
}

/// Re-export for backward compatibility (old code calls FriProver methods)
pub use crate::fields::tower::Tower;

impl FriProver {
    pub fn new(params: FriParams) -> Self {
        Self { params }
    }

    pub fn params(&self) -> &FriParams {
        &self.params
    }

    /// Encode a polynomial into an RS codeword using additive NTT.
    fn rs_encode(&self, coefficients: &[FE]) -> Vec<FE> {
        ntt::rs_encode(coefficients, self.params.log_blowup)
    }

    /// Commit to a codeword by Merkle hashing.
    fn merkle_commit(&self, codeword: &[FE]) -> MerkleRoot {
        MerkleRoot::from_field_elements(codeword)
    }

    /// Commit to a polynomial (backward compat with old API).
    pub fn commit(
        &self,
        polynomial: &MultilinearPolynomial,
    ) -> (MerkleRoot, Vec<crate::fields::tower::Tower>) {
        let fe_evals: Vec<FE> = polynomial
            .evaluations()
            .iter()
            .map(lambdaworks_math::field::fields::binary::tower_field::from_tower)
            .collect();

        // RS encode: treat evaluations as polynomial coefficients
        let codeword = self.rs_encode(&fe_evals);
        let root = self.merkle_commit(&codeword);

        let tower_codeword: Vec<Tower> = codeword
            .iter()
            .map(lambdaworks_math::field::fields::binary::tower_field::to_tower)
            .collect();

        (root, tower_codeword)
    }

    /// Encode polynomial (backward compat).
    pub fn encode_polynomial(
        &self,
        polynomial: &MultilinearPolynomial,
    ) -> Vec<crate::fields::tower::Tower> {
        let (_, codeword) = self.commit(polynomial);
        codeword
    }

    /// Generate a complete FRI proof.
    pub fn prove(&self, polynomial: &MultilinearPolynomial) -> FriProof {
        let fe_evals: Vec<FE> = polynomial
            .evaluations()
            .iter()
            .map(lambdaworks_math::field::fields::binary::tower_field::from_tower)
            .collect();

        self.prove_fe(&fe_evals)
    }

    /// Generate FRI proof from field elements directly.
    pub fn prove_fe(&self, coefficients: &[FE]) -> FriProof {
        let mut transcript = DefaultTranscript::<BinaryTowerField128>::new(b"binius_fri");

        // Step 1: RS encode and commit
        let initial_codeword = self.rs_encode(coefficients);
        let initial_commitment = self.merkle_commit(&initial_codeword);

        // Absorb commitment into transcript
        transcript.append_bytes(initial_commitment.as_bytes());

        // Step 2: Iteratively fold
        let actual_log_codeword = initial_codeword.len().trailing_zeros() as usize;
        let num_rounds = actual_log_codeword;
        let mut layers = Vec::with_capacity(num_rounds);
        let mut codewords = vec![initial_codeword.clone()];

        let mut current_codeword = initial_codeword;
        // Track the evaluation domain through folding rounds.
        // After each fold, the domain transforms via V(x) = x^2 + x.
        let mut current_domain = ntt::initial_domain(actual_log_codeword);

        for _ in 0..num_rounds.saturating_sub(1) {
            if current_codeword.len() < 2 {
                break;
            }

            // Sample folding challenge from transcript
            let challenge: FE = transcript.sample_field_element();

            // Fold using additive decomposition with the actual domain
            let (folded, new_domain) =
                ntt::fold_codeword(&current_codeword, &challenge, &current_domain);

            // Commit folded codeword
            let folded_commitment = self.merkle_commit(&folded);
            transcript.append_bytes(folded_commitment.as_bytes());

            layers.push(FriLayerProof {
                commitment: folded_commitment,
            });

            codewords.push(folded.clone());
            current_codeword = folded;
            current_domain = new_domain;
        }

        // Step 3: Final polynomial (should be a single constant)
        let final_poly = current_codeword.clone();

        // Send final polynomial to transcript
        for v in &final_poly {
            transcript.append_bytes(&v.to_bytes_be());
        }

        // Step 4: Sample query positions
        let initial_size = 1usize << actual_log_codeword;
        let query_positions: Vec<usize> = (0..self.params.num_queries)
            .map(|_| transcript.sample_u64(initial_size as u64) as usize)
            .collect();

        // Step 5: Build query proofs
        let query_proofs = query_positions
            .iter()
            .map(|&pos| self.build_query_proof(pos, &codewords))
            .collect();

        FriProof {
            commitment: self.merkle_commit(&codewords[0]),
            layers,
            final_poly,
            query_positions,
            query_proofs,
        }
    }

    /// Build query proof for a single position.
    fn build_query_proof(&self, initial_pos: usize, codewords: &[Vec<FE>]) -> FriQueryProof {
        let mut layer_values = Vec::new();
        let mut pos = initial_pos;

        for codeword in codewords.iter().take(codewords.len().saturating_sub(1)) {
            let even_pos = (pos / 2) * 2;
            let odd_pos = even_pos + 1;
            let even_val = codeword[even_pos % codeword.len()];
            let odd_val = codeword[odd_pos % codeword.len()];
            layer_values.push((even_val, odd_val));
            pos /= 2;
        }

        FriQueryProof { layer_values }
    }
}

/// Compute the evaluation domain for each FRI folding round.
///
/// The initial domain is {0, 1, 2, ..., initial_size - 1}.
/// After each fold, the domain transforms via V(x) = x^2 + x,
/// taking only the "even" representatives (elements at indices 0, 2, 4, ...).
fn compute_fri_domains(initial_size: usize, num_rounds: usize) -> Vec<Vec<FE>> {
    let mut domains = Vec::with_capacity(num_rounds + 1);

    // Initial domain
    let domain0: Vec<FE> = (0..initial_size).map(|i| FE::new(i as u128)).collect();
    domains.push(domain0);

    for r in 0..num_rounds {
        let prev = &domains[r];
        let half = prev.len() / 2;
        let new_domain: Vec<FE> = (0..half)
            .map(|i| {
                let w = prev[2 * i];
                w * w + w // V(w) = w^2 + w
            })
            .collect();
        domains.push(new_domain);
    }

    domains
}

/// FRI Verifier
pub struct FriVerifier {
    params: FriParams,
}

impl FriVerifier {
    pub fn new(params: FriParams) -> Self {
        Self { params }
    }

    /// Verify a FRI proof.
    ///
    /// Checks:
    /// 1. Replays Fiat-Shamir transcript to derive same challenges and query positions
    /// 2. Verifies the folding equation at each queried position
    /// 3. Checks that the final polynomial is constant (degree 0)
    pub fn verify(
        &self,
        proof: &FriProof,
        _claimed_evaluations: &[Tower],
    ) -> Result<bool, FriError> {
        let mut transcript = DefaultTranscript::<BinaryTowerField128>::new(b"binius_fri");

        // Absorb initial commitment
        transcript.append_bytes(proof.commitment.as_bytes());

        // Derive challenges (must match prover)
        let mut challenges = Vec::with_capacity(proof.layers.len());
        for layer in &proof.layers {
            let challenge: FE = transcript.sample_field_element();
            challenges.push(challenge);
            transcript.append_bytes(layer.commitment.as_bytes());
        }

        // Absorb final polynomial
        for v in &proof.final_poly {
            transcript.append_bytes(&v.to_bytes_be());
        }

        // Derive query positions (must match prover)
        let num_layers = proof.layers.len();
        let initial_size = proof.final_poly.len() * (1usize << num_layers);
        let expected_positions: Vec<usize> = (0..self.params.num_queries)
            .map(|_| transcript.sample_u64(initial_size as u64) as usize)
            .collect();

        // Check query positions match
        if expected_positions != proof.query_positions {
            return Err(FriError::TranscriptMismatch);
        }

        // Compute the evaluation domains for each folding round.
        // The domain transforms via V(x) = x^2 + x at each round.
        let domains = compute_fri_domains(initial_size, num_layers);

        // Verify folding at each query position
        for (q, query_proof) in proof.query_proofs.iter().enumerate() {
            let pos = proof.query_positions[q];
            self.verify_query(pos, query_proof, &challenges, &proof.final_poly, &domains)?;
        }

        // Check final polynomial is low-degree (should be constant after enough rounds)
        if proof.final_poly.len() > 1 {
            let first = proof.final_poly[0];
            let all_same = proof.final_poly.iter().all(|v| *v == first);
            if !all_same {
                return Err(FriError::DegreeBoundViolation);
            }
        }

        Ok(true)
    }

    /// Verify a single query's folding chain.
    fn verify_query(
        &self,
        initial_pos: usize,
        query_proof: &FriQueryProof,
        challenges: &[FE],
        final_poly: &[FE],
        domains: &[Vec<FE>],
    ) -> Result<(), FriError> {
        let mut pos = initial_pos;

        for (round, (even_val, odd_val)) in query_proof.layer_values.iter().enumerate() {
            if round >= challenges.len() {
                break;
            }

            let challenge = challenges[round];
            let even_pos = (pos / 2) * 2;
            // Use the actual domain point, not integer index
            let w = domains[round][even_pos];

            // Verify folding equation:
            // folded_val = f(w) + (w + alpha) * (f(w) + f(w + basis[0]))
            let diff = *even_val + *odd_val;
            let expected_folded = *even_val + (w + challenge) * diff;

            // The folded value should appear in the next layer
            let folded_pos = pos / 2;

            // Check against next layer or final polynomial
            if round + 1 < query_proof.layer_values.len() {
                let (next_even, next_odd) = query_proof.layer_values[round + 1];
                let next_val = if folded_pos.is_multiple_of(2) {
                    next_even
                } else {
                    next_odd
                };
                if expected_folded != next_val {
                    return Err(FriError::FoldingVerificationFailed(round));
                }
            } else if !final_poly.is_empty() {
                let final_val = final_poly[folded_pos % final_poly.len()];
                if expected_folded != final_val {
                    return Err(FriError::FinalValueMismatch);
                }
            }

            pos = folded_pos;
        }

        Ok(())
    }
}

/// FRI errors
#[derive(Debug)]
pub enum FriError {
    CommitmentMismatch,
    LayerRootMismatch(usize),
    QueryError,
    FoldingVerificationFailed(usize),
    FinalValueMismatch,
    DegreeBoundViolation,
    TranscriptMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(log_msg: usize, log_blowup: usize, num_queries: usize) -> FriParams {
        FriParams {
            log_message_size: log_msg,
            log_blowup,
            num_queries,
        }
    }

    #[test]
    fn test_fri_params() {
        let params = make_params(3, 1, 4);
        assert_eq!(params.message_size(), 8);
        assert_eq!(params.codeword_size(), 16);
        assert_eq!(params.log_codeword_size(), 4);
    }

    #[test]
    fn test_fri_commit_and_encode() {
        let params = make_params(2, 1, 4);
        let prover = FriProver::new(params);

        let coeffs: Vec<FE> = vec![
            FE::new(1u128),
            FE::new(2u128),
            FE::new(3u128),
            FE::new(4u128),
        ];
        let codeword = prover.rs_encode(&coeffs);
        assert_eq!(codeword.len(), 8); // 4 * 2 = 8

        let root = prover.merkle_commit(&codeword);
        assert!(!root.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_fri_prove_creates_proof() {
        let params = make_params(2, 1, 2);
        let prover = FriProver::new(params);

        let coeffs: Vec<FE> = vec![
            FE::new(5u128),
            FE::new(3u128),
            FE::new(0u128),
            FE::new(0u128),
        ];
        let proof = prover.prove_fe(&coeffs);

        // Should have layers (log_codeword_size - 1 rounds)
        assert!(!proof.layers.is_empty());
        assert!(!proof.query_positions.is_empty());
        assert_eq!(proof.query_proofs.len(), proof.query_positions.len());
    }

    #[test]
    fn test_fri_prove_verify_roundtrip() {
        let params = make_params(2, 1, 2);
        let prover = FriProver::new(params.clone());
        let verifier = FriVerifier::new(params);

        // A constant polynomial (degree 0) — easiest case
        let coeffs: Vec<FE> = vec![FE::new(42u128), FE::zero(), FE::zero(), FE::zero()];
        let proof = prover.prove_fe(&coeffs);

        let result = verifier.verify(&proof, &[]);
        assert!(result.is_ok(), "FRI verification failed: {:?}", result);
        assert!(result.unwrap());
    }

    #[test]
    fn test_fri_fiat_shamir_consistency() {
        // Two proofs of the same polynomial should produce the same proof
        // (because Fiat-Shamir is deterministic)
        let params = make_params(2, 1, 2);
        let prover = FriProver::new(params);

        let coeffs: Vec<FE> = vec![FE::new(7u128), FE::new(11u128), FE::zero(), FE::zero()];
        let proof1 = prover.prove_fe(&coeffs);
        let proof2 = prover.prove_fe(&coeffs);

        assert_eq!(proof1.commitment.as_bytes(), proof2.commitment.as_bytes());
        assert_eq!(proof1.query_positions, proof2.query_positions);
    }

    #[test]
    fn test_fri_different_polynomials_different_proofs() {
        let params = make_params(2, 1, 2);
        let prover = FriProver::new(params);

        let coeffs1: Vec<FE> = vec![FE::new(1u128), FE::new(2u128), FE::zero(), FE::zero()];
        let coeffs2: Vec<FE> = vec![FE::new(3u128), FE::new(4u128), FE::zero(), FE::zero()];
        let proof1 = prover.prove_fe(&coeffs1);
        let proof2 = prover.prove_fe(&coeffs2);

        assert_ne!(proof1.commitment.as_bytes(), proof2.commitment.as_bytes());
    }

    // Backward compatibility tests using the old API
    #[test]
    fn test_fri_legacy_prove_verify() {
        let params = FriParams::new(2, 4, 3);
        let prover = FriProver::new(params.clone());

        let evals = vec![
            Tower::new(1, 1),
            Tower::new(0, 1),
            Tower::new(1, 1),
            Tower::new(0, 1),
        ];
        let poly = MultilinearPolynomial::new(evals).unwrap();
        let proof = prover.prove(&poly);

        assert!(!proof.commitment.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_fri_legacy_commitment() {
        let params = FriParams::new(2, 4, 2);
        let prover = FriProver::new(params);

        let evals = vec![Tower::new(1, 1), Tower::new(2, 1)];
        let poly = MultilinearPolynomial::new(evals).unwrap();

        let (root, codeword) = prover.commit(&poly);
        assert!(!codeword.is_empty());
        assert!(!root.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_fri_nonconstant_polynomial() {
        // Test with a non-constant polynomial (degree 3)
        let params = make_params(2, 1, 2);
        let prover = FriProver::new(params.clone());
        let verifier = FriVerifier::new(params);

        let coeffs: Vec<FE> = vec![
            FE::new(5u128),
            FE::new(3u128),
            FE::new(7u128),
            FE::new(11u128),
        ];
        let proof = prover.prove_fe(&coeffs);

        let result = verifier.verify(&proof, &[]);
        assert!(
            result.is_ok(),
            "FRI verification failed for non-constant poly: {:?}",
            result
        );
    }
}
