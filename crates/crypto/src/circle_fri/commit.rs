extern crate alloc;
use alloc::vec::Vec;

use lambdaworks_math::circle::domain::CircleDomain;
use lambdaworks_math::circle::fold::{fold, reorder_natural_to_butterfly};
use lambdaworks_math::circle::traits::IsCircleFriField;
use lambdaworks_math::circle::twiddles::{get_twiddles, TwiddlesConfig};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::traits::AsBytes;

use crate::fiat_shamir::is_transcript::IsTranscript;
use crate::merkle_tree::backends::types::Keccak256Backend;
use crate::merkle_tree::merkle::MerkleTree;

use super::errors::CircleFriError;

/// A single layer of the Circle FRI commitment.
pub struct CircleFriLayer<F: IsCircleFriField>
where
    FieldElement<F>: AsBytes,
{
    /// Evaluations at this layer in butterfly order (size is a power of two).
    pub evaluations: Vec<FieldElement<F>>,
    /// Merkle tree committing to individual evaluations.
    pub merkle_tree: MerkleTree<Keccak256Backend<F>>,
}

/// Result of the Circle FRI commit phase.
pub struct CircleFriCommitment<F: IsCircleFriField>
where
    FieldElement<F>: AsBytes,
{
    /// All committed layers (one per fold round).
    pub layers: Vec<CircleFriLayer<F>>,
    /// The final constant value after all folds.
    pub final_value: FieldElement<F>,
}

/// Runs the Circle FRI commit phase.
///
/// Given evaluations of a polynomial on a circle domain of size 2^n,
/// repeatedly folds using random challenges from the transcript,
/// committing each intermediate layer into a Merkle tree.
///
/// # Arguments
/// * `evaluations` - Polynomial evaluations in natural coset order (length must be domain.size())
/// * `domain`      - The circle domain on which evaluations are defined
/// * `transcript`  - Fiat-Shamir transcript for challenge generation
///
/// # Returns
/// A `CircleFriCommitment` containing all layers and the final value.
pub fn circle_fri_commit<F: IsCircleFriField>(
    evaluations: &[FieldElement<F>],
    domain: &CircleDomain<F>,
    transcript: &mut impl IsTranscript<F>,
) -> Result<CircleFriCommitment<F>, CircleFriError>
where
    FieldElement<F>: AsBytes,
{
    let n = evaluations.len();
    assert!(n.is_power_of_two() && n >= 2);
    assert_eq!(n, domain.size());

    // Precompute all twiddle layers (interpolation = inverse twiddles)
    let inv_twiddles = get_twiddles(domain.coset.clone(), TwiddlesConfig::Interpolation);

    // Reorder evaluations from natural to butterfly order
    let mut current_evals = evaluations.to_vec();
    reorder_natural_to_butterfly(&mut current_evals);

    let mut layers = Vec::new();

    // Fold through all layers until we reach a single value
    for inv_twiddle_layer in &inv_twiddles {
        // Commit current evaluations
        let merkle_tree = MerkleTree::<Keccak256Backend<F>>::build(&current_evals)
            .ok_or(CircleFriError::MerkleTreeBuildFailed)?;

        // Send Merkle root to transcript
        transcript.append_bytes(&merkle_tree.root);

        layers.push(CircleFriLayer {
            evaluations: current_evals.clone(),
            merkle_tree,
        });

        // Sample folding challenge
        let alpha: FieldElement<F> = transcript.sample_field_element();

        // Fold: halves the number of evaluations
        current_evals = fold(&current_evals, inv_twiddle_layer, &alpha);
    }

    assert_eq!(current_evals.len(), 1);
    let final_value = current_evals[0].clone();

    // Send final value to transcript
    transcript.append_field_element(&final_value);

    Ok(CircleFriCommitment {
        layers,
        final_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::circle::polynomial::evaluate_cfft;
    use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;

    type FE = FieldElement<Mersenne31Field>;

    #[test]
    fn commit_phase_produces_correct_number_of_layers() {
        // 8-point domain: 3 fold layers (y-fold + 2 x-folds)
        let coeffs: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let evals = evaluate_cfft(coeffs);
        let domain = CircleDomain::new_standard(3);
        let mut transcript = DefaultTranscript::<Mersenne31Field>::new(&[]);

        let commitment = circle_fri_commit(&evals, &domain, &mut transcript).unwrap();
        assert_eq!(commitment.layers.len(), 3);
    }

    #[test]
    fn commit_phase_layer_sizes_halve() {
        let coeffs: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
        let evals = evaluate_cfft(coeffs);
        let domain = CircleDomain::new_standard(4);
        let mut transcript = DefaultTranscript::<Mersenne31Field>::new(&[]);

        let commitment = circle_fri_commit(&evals, &domain, &mut transcript).unwrap();
        assert_eq!(commitment.layers.len(), 4);
        assert_eq!(commitment.layers[0].evaluations.len(), 16);
        assert_eq!(commitment.layers[1].evaluations.len(), 8);
        assert_eq!(commitment.layers[2].evaluations.len(), 4);
        assert_eq!(commitment.layers[3].evaluations.len(), 2);
    }

    #[test]
    fn commit_phase_is_deterministic() {
        let coeffs: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let evals = evaluate_cfft(coeffs);
        let domain = CircleDomain::new_standard(3);

        let mut t1 = DefaultTranscript::<Mersenne31Field>::new(&[0x42]);
        let c1 = circle_fri_commit(&evals, &domain, &mut t1).unwrap();

        let mut t2 = DefaultTranscript::<Mersenne31Field>::new(&[0x42]);
        let c2 = circle_fri_commit(&evals, &domain, &mut t2).unwrap();

        assert_eq!(c1.final_value, c2.final_value);
        assert_eq!(c1.layers.len(), c2.layers.len());
        for (l1, l2) in c1.layers.iter().zip(c2.layers.iter()) {
            assert_eq!(l1.merkle_tree.root, l2.merkle_tree.root);
        }
    }
}
