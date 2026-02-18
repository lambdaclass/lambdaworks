use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::traits::AsBytes;
use stark_platinum_prover::config::{BatchedMerkleTree, Commitment};

/// Build a batched Merkle tree from row-major evaluation data.
/// Currently uses CPU (Keccak256). Will be replaced with GPU Poseidon.
pub fn cpu_batch_commit<F: IsField>(
    vectors: &[Vec<FieldElement<F>>],
) -> Option<(BatchedMerkleTree<F>, Commitment)>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    let tree = BatchedMerkleTree::<F>::build(vectors)?;
    let root = tree.root;
    Some((tree, root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

    type FpE = FieldElement<Goldilocks64Field>;

    #[test]
    fn cpu_batch_commit_produces_valid_tree() {
        let col1: Vec<FpE> = (0..8).map(|i| FpE::from(i as u64)).collect();
        let col2: Vec<FpE> = (0..8).map(|i| FpE::from(i as u64 + 100)).collect();
        let (tree, root) = cpu_batch_commit(&[col1, col2]).unwrap();
        // Root should be non-zero
        assert_ne!(root, [0u8; 32]);
        // Tree should have leaves
        assert!(tree.root != [0u8; 32]);
    }
}
