//! Merkle tree gadgets for proof verification.
//!
//! This module provides gadgets for verifying Merkle proofs in circuits.
//! It uses Poseidon as the underlying hash function.
//!
//! # Usage
//!
//! ```ignore
//! // Verify a Merkle proof
//! let root = MerkleProofVerifier::<8>::synthesize(
//!     &mut builder,
//!     MerkleProofInput {
//!         leaf: leaf_var,
//!         path: sibling_vars,
//!         path_indices: index_vars,
//!     },
//! )?;
//! ```

use crate::dsl::builder::CircuitBuilder;
use crate::dsl::gadgets::poseidon::{DefaultPoseidonParams, PoseidonParams, PoseidonTwoToOne};
use crate::dsl::gadgets::{Gadget, GadgetError};
use crate::dsl::types::{BoolVar, FieldVar};
use lambdaworks_math::field::traits::IsField;

/// Input for Merkle proof verification.
pub struct MerkleProofInput {
    /// The leaf value being proven
    pub leaf: FieldVar,
    /// Sibling hashes along the path (from leaf to root)
    pub path: Vec<FieldVar>,
    /// Path indices indicating position at each level (0 = left, 1 = right)
    pub path_indices: Vec<BoolVar>,
}

/// Merkle proof verifier gadget.
///
/// Verifies that a leaf is part of a Merkle tree by hashing
/// up the authentication path and computing the root.
///
/// # Type Parameters
/// * `DEPTH` - The depth of the Merkle tree (number of levels)
/// * `P` - Poseidon parameters to use for hashing
pub struct MerkleProofVerifier<const DEPTH: usize, P = DefaultPoseidonParams> {
    _params: core::marker::PhantomData<P>,
}

impl<F: IsField, const DEPTH: usize, P: PoseidonParams<F>> Gadget<F>
    for MerkleProofVerifier<DEPTH, P>
{
    type Input = MerkleProofInput;
    type Output = FieldVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        input: Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        // Validate input lengths
        if input.path.len() != DEPTH {
            return Err(GadgetError::InvalidInput(format!(
                "Path length {} does not match tree depth {}",
                input.path.len(),
                DEPTH
            )));
        }
        if input.path_indices.len() != DEPTH {
            return Err(GadgetError::InvalidInput(format!(
                "Path indices length {} does not match tree depth {}",
                input.path_indices.len(),
                DEPTH
            )));
        }

        // Start with the leaf
        let mut current = input.leaf;

        // Hash up the tree
        for i in 0..DEPTH {
            let sibling = &input.path[i];
            let is_right = &input.path_indices[i];

            // If is_right = 0: hash(current, sibling)
            // If is_right = 1: hash(sibling, current)
            // Use select to swap order based on path index

            let left = builder.select(is_right, sibling, &current);
            let right = builder.select(is_right, &current, sibling);

            current = PoseidonTwoToOne::<P>::synthesize(builder, (left, right))?;
        }

        Ok(current)
    }

    fn constraint_count() -> usize {
        // Each level: 2 selects + 1 Poseidon hash
        let poseidon_constraints = PoseidonTwoToOne::<P>::constraint_count();
        let select_constraints = 4; // 2 selects, each ~2 constraints
        DEPTH * (poseidon_constraints + select_constraints)
    }

    fn name() -> &'static str {
        "MerkleProofVerifier"
    }
}

/// Merkle proof verification with root equality check.
///
/// Verifies a Merkle proof and asserts that the computed root
/// matches an expected root value.
pub struct MerkleProofChecker<const DEPTH: usize, P = DefaultPoseidonParams> {
    _params: core::marker::PhantomData<P>,
}

/// Input for Merkle proof checker.
pub struct MerkleProofCheckerInput {
    /// The leaf value being proven
    pub leaf: FieldVar,
    /// Sibling hashes along the path
    pub path: Vec<FieldVar>,
    /// Path indices indicating position at each level
    pub path_indices: Vec<BoolVar>,
    /// Expected Merkle root
    pub expected_root: FieldVar,
}

impl<F: IsField, const DEPTH: usize, P: PoseidonParams<F>> Gadget<F>
    for MerkleProofChecker<DEPTH, P>
{
    type Input = MerkleProofCheckerInput;
    type Output = BoolVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        input: Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        // Compute the root from the proof
        let computed_root = MerkleProofVerifier::<DEPTH, P>::synthesize(
            builder,
            MerkleProofInput {
                leaf: input.leaf,
                path: input.path,
                path_indices: input.path_indices,
            },
        )?;

        // Check if computed root equals expected root
        use crate::dsl::gadgets::comparison::IsEqual;
        IsEqual::synthesize(builder, (computed_root, input.expected_root))
    }

    fn constraint_count() -> usize {
        // MerkleProofVerifier + IsEqual
        MerkleProofVerifier::<DEPTH, P>::constraint_count() + 4
    }

    fn name() -> &'static str {
        "MerkleProofChecker"
    }
}

/// Computes a Merkle root from leaves.
///
/// This gadget builds a Merkle tree from a power-of-two number of leaves
/// and returns the root.
pub struct MerkleRoot<const NUM_LEAVES: usize, P = DefaultPoseidonParams> {
    _params: core::marker::PhantomData<P>,
}

impl<F: IsField, const NUM_LEAVES: usize, P: PoseidonParams<F>> Gadget<F>
    for MerkleRoot<NUM_LEAVES, P>
{
    type Input = Vec<FieldVar>;
    type Output = FieldVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        leaves: Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        if leaves.len() != NUM_LEAVES {
            return Err(GadgetError::InvalidInput(format!(
                "Expected {} leaves, got {}",
                NUM_LEAVES,
                leaves.len()
            )));
        }

        // NUM_LEAVES must be a power of 2
        if NUM_LEAVES == 0 || (NUM_LEAVES & (NUM_LEAVES - 1)) != 0 {
            return Err(GadgetError::InvalidInput(
                "Number of leaves must be a power of 2".to_string(),
            ));
        }

        // Build tree bottom-up
        let mut current_level = leaves;

        while current_level.len() > 1 {
            let mut next_level = Vec::with_capacity(current_level.len() / 2);

            for i in (0..current_level.len()).step_by(2) {
                let left = &current_level[i];
                let right = &current_level[i + 1];

                let parent = PoseidonTwoToOne::<P>::synthesize(builder, (*left, *right))?;
                next_level.push(parent);
            }

            current_level = next_level;
        }

        Ok(current_level.into_iter().next().unwrap())
    }

    fn constraint_count() -> usize {
        // Number of internal nodes = NUM_LEAVES - 1
        // Each internal node is one Poseidon hash
        if NUM_LEAVES == 0 {
            0
        } else {
            (NUM_LEAVES - 1) * PoseidonTwoToOne::<P>::constraint_count()
        }
    }

    fn name() -> &'static str {
        "MerkleRoot"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    type F = U64PrimeField<65537>;

    #[test]
    fn test_merkle_proof_verifier() {
        let mut builder = CircuitBuilder::<F>::new();

        // Create a simple proof with depth 3
        let leaf = builder.public_input("leaf");
        let sibling0 = builder.private_input("sibling0");
        let sibling1 = builder.private_input("sibling1");
        let sibling2 = builder.private_input("sibling2");

        let idx0 = builder.private_input("idx0");
        let idx1 = builder.private_input("idx1");
        let idx2 = builder.private_input("idx2");

        let idx0_bool = builder.assert_bool(&idx0);
        let idx1_bool = builder.assert_bool(&idx1);
        let idx2_bool = builder.assert_bool(&idx2);

        let _root = MerkleProofVerifier::<3>::synthesize(
            &mut builder,
            MerkleProofInput {
                leaf,
                path: vec![sibling0, sibling1, sibling2],
                path_indices: vec![idx0_bool, idx1_bool, idx2_bool],
            },
        )
        .unwrap();
    }

    #[test]
    fn test_merkle_proof_checker() {
        let mut builder = CircuitBuilder::<F>::new();

        let leaf = builder.public_input("leaf");
        let expected_root = builder.public_input("expected_root");
        let sibling = builder.private_input("sibling");
        let idx = builder.private_input("idx");
        let idx_bool = builder.assert_bool(&idx);

        let _is_valid = MerkleProofChecker::<1>::synthesize(
            &mut builder,
            MerkleProofCheckerInput {
                leaf,
                path: vec![sibling],
                path_indices: vec![idx_bool],
                expected_root,
            },
        )
        .unwrap();
    }

    #[test]
    fn test_merkle_root_four_leaves() {
        let mut builder = CircuitBuilder::<F>::new();

        let leaf0 = builder.public_input("leaf0");
        let leaf1 = builder.public_input("leaf1");
        let leaf2 = builder.public_input("leaf2");
        let leaf3 = builder.public_input("leaf3");

        let _root =
            MerkleRoot::<4>::synthesize(&mut builder, vec![leaf0, leaf1, leaf2, leaf3]).unwrap();
    }

    #[test]
    fn test_merkle_root_wrong_count() {
        let mut builder = CircuitBuilder::<F>::new();

        let leaf0 = builder.public_input("leaf0");
        let leaf1 = builder.public_input("leaf1");

        // Try to build with wrong number of leaves
        let result = MerkleRoot::<4>::synthesize(&mut builder, vec![leaf0, leaf1]);

        assert!(result.is_err());
    }

    #[test]
    fn test_merkle_proof_wrong_depth() {
        let mut builder = CircuitBuilder::<F>::new();

        let leaf = builder.public_input("leaf");
        let sibling = builder.private_input("sibling");
        let idx = builder.private_input("idx");
        let idx_bool = builder.assert_bool(&idx);

        // Try with wrong path length (2 expected, 1 provided)
        let result = MerkleProofVerifier::<2>::synthesize(
            &mut builder,
            MerkleProofInput {
                leaf,
                path: vec![sibling],
                path_indices: vec![idx_bool],
            },
        );

        assert!(result.is_err());
    }
}
