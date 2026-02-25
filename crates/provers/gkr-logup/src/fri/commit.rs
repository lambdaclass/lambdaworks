//! FRI commit phase: iteratively fold + LDE + Merkle commit.

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_crypto::merkle_tree::backends::types::Keccak256Backend;
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::traits::{AsBytes, ByteConversion};

use lambdaworks_math::polynomial::Polynomial;

use super::fold::fold_polynomial;
use super::types::{FriConfig, FriError, FriLayerData};

/// Result of the FRI commit phase (prover side).
pub(crate) struct FriCommitResult<F: IsFFTField>
where
    FieldElement<F>: AsBytes,
    F::BaseType: Send + Sync,
{
    /// Per-layer data: merkle root + evaluations.
    pub layers: Vec<FriLayerData<F>>,
    /// The final constant value.
    pub final_value: FieldElement<F>,
    /// Merkle trees for each layer (needed for query phase).
    pub merkle_trees: Vec<MerkleTree<Keccak256Backend<F>>>,
}

/// Run the FRI commit phase on a polynomial given in coefficient form.
///
/// 1. Evaluate on an extended domain (blowup × degree_bound).
/// 2. Build Merkle tree, append root to transcript.
/// 3. Sample folding challenge beta.
/// 4. Fold polynomial, repeat.
/// 5. When degree is 0, append final value to transcript.
pub(crate) fn fri_commit<F, T>(
    poly: &Polynomial<FieldElement<F>>,
    config: &FriConfig,
    transcript: &mut T,
) -> Result<FriCommitResult<F>, FriError>
where
    F: IsFFTField,
    F::BaseType: Send + Sync,
    FieldElement<F>: AsBytes + ByteConversion + Clone,
    T: IsTranscript<F>,
{
    let blowup = config.blowup_factor();
    let mut current_poly = poly.clone();
    let mut layers = Vec::new();
    let mut merkle_trees = Vec::new();

    loop {
        let degree = current_poly.degree();
        if degree == 0 {
            break;
        }

        // Degree bound for this layer: smallest power of 2 >= coeff_len
        let coeff_len = current_poly.coeff_len().max(1);
        let degree_bound = coeff_len.next_power_of_two();
        let lde_size = degree_bound * blowup;

        // Evaluate on extended domain via FFT
        let lde_evals = Polynomial::evaluate_fft::<F>(&current_poly, blowup, Some(degree_bound))
            .map_err(|e| FriError::FftError(format!("{e}")))?;

        // Truncate to lde_size in case FFT returned more
        let lde_evals: Vec<FieldElement<F>> = lde_evals.into_iter().take(lde_size).collect();

        // Build Merkle tree
        let merkle_tree = MerkleTree::<Keccak256Backend<F>>::build(&lde_evals)
            .ok_or_else(|| FriError::MerkleError("failed to build Merkle tree".into()))?;

        // Append Merkle root to transcript
        transcript.append_bytes(&merkle_tree.root);

        layers.push(FriLayerData {
            merkle_root: merkle_tree.root,
            evaluations: lde_evals,
            domain_size: lde_size,
        });
        merkle_trees.push(merkle_tree);

        // Sample folding challenge
        let beta: FieldElement<F> = transcript.sample_field_element();

        // Fold polynomial
        current_poly = fold_polynomial(&current_poly, &beta);
    }

    // Final constant value
    let final_value = if current_poly.is_zero() {
        FieldElement::zero()
    } else {
        current_poly.coefficients()[0].clone()
    };

    // Append final value to transcript
    transcript.append_field_element(&final_value);

    Ok(FriCommitResult {
        layers,
        final_value,
        merkle_trees,
    })
}
