use std::collections::{BTreeSet, HashMap};

use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::{
    field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::AsBytes,
};
use stark_platinum_prover::{
    config::Commitment,
    domain::Domain,
    proof::{options::ProofOptions, stark::StarkProof},
    traits::AIR,
    transcript::StoneProverTranscript,
    verifier::{IsStarkVerifier, Verifier},
};

use self::{
    ast::IntoAst, stark_config::StarkConfig, stark_public_input::CairoPublicInput, stark_unsent_commitment::StarkUnsentCommitment, stark_witness::StarkWitness
};

pub mod stark_config;
pub mod stark_public_input;
pub mod stark_unsent_commitment;
pub mod stark_witness;
pub mod ast;

pub struct CairoStarkProof {
    pub stark_config: StarkConfig,
    pub public_input: CairoPublicInput,
    pub unsent_commitment: StarkUnsentCommitment,
    pub witness: StarkWitness,
}

/// Serializer compatible with cairoX-verifier stark proof struct
pub struct CairoCompatibleSerializer;

impl CairoCompatibleSerializer {}

impl CairoCompatibleSerializer {
    pub fn convert<A>(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
        public_inputs: &A::PublicInputs,
        options: &ProofOptions,
    ) -> ()
    where
        A: AIR<Field = Stark252PrimeField, FieldExtension = Stark252PrimeField>,
        A::PublicInputs: AsBytes,
    {
        let unsent_commitment = StarkUnsentCommitment::convert(proof).unwrap();
        let fri_query_indexes = Self::get_fri_query_indexes::<A>(proof, public_inputs, options);
        let witness = StarkWitness::convert(proof, &fri_query_indexes);

        let exprs = unsent_commitment.into_ast();
        // let exprs = witness.into_ast();

        let serialized = serde_json::to_string(&exprs).unwrap();

        println!("{:#?}", serialized);
    }

    /// Merges `n` authentication paths for `n` leaves into a list of the minimal number of nodes
    /// needed to reach the Merkle root for all of them. The nodes of the merged authentication
    /// paths are sorted from level 0 to the hightest level of the Merkle tree, and nodes at the
    /// same level are sorted from left to right.
    ///
    /// Let's consider some examples. Suppose the Merkle tree is as follows:
    ///
    /// Root              ABCD
    ///                  /    \
    /// Level 1         AB    CD
    ///                /  \  /  \
    /// Level 0       A   B C    D
    /// Leaf index    0   1 2    3
    ///
    /// All authentication paths are `pA = [B, CD]`, `pB = [A, CD]`, `pC = [D, AB]` and `pD = [C, AB]`.
    ///
    /// (1) `merge_authentication_paths([pA, pB], [0, 1]) = [CD]`.
    /// (2) `merge_authentication_paths([pC, pA], [2, 0]) = [B, D]`.
    /// (3) `merge_authentication_paths([pA, pD], [0, 3]) = [B, C]`.
    /// (4) `merge_authentication_paths([pA, pD, pB], [0, 3, 1]) = [C]`.
    /// (5) `merge_authentication_paths([pA, pB, pC, pD], [0, 1, 2, 3]) = []`.
    ///
    /// Input:
    /// `authentication_paths`: The authentication paths to be merged.
    /// `leaf_indexes`: The leaf indexes corresponding to the authentication paths.
    ///
    /// Output:
    /// The merged authentication paths
    pub fn merge_authentication_paths(
        authentication_paths: &[&Proof<Commitment>],
        leaf_indexes: &[usize],
    ) -> Vec<Commitment> {
        debug_assert_eq!(leaf_indexes.len(), authentication_paths.len());
        let mut merkle_tree: HashMap<(usize, usize), Commitment> = HashMap::new();
        for (index_previous_layer, path) in leaf_indexes.iter().zip(authentication_paths.iter()) {
            let mut node_index = *index_previous_layer;
            for (tree_level, node) in path.merkle_path.iter().enumerate() {
                merkle_tree.insert((tree_level, node_index ^ 1), *node);
                node_index >>= 1;
            }
        }

        let mut result = Vec::new();
        let mut level_indexes: BTreeSet<usize> = leaf_indexes.iter().copied().collect();
        let merkle_tree_height = authentication_paths[0].merkle_path.len();
        for tree_level in 0..merkle_tree_height {
            for node_index in level_indexes.iter() {
                let sibling_index = node_index ^ 1;
                if !level_indexes.contains(&sibling_index) {
                    let node = &merkle_tree[&(tree_level, sibling_index)];
                    result.push(*node);
                }
            }
            level_indexes = level_indexes.iter().map(|index| *index >> 1).collect();
        }
        result
    }

    fn get_fri_query_indexes<A>(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
        public_inputs: &A::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Vec<usize>
    where
        A: AIR<Field = Stark252PrimeField, FieldExtension = Stark252PrimeField>,
        A::PublicInputs: AsBytes,
    {
        let mut transcript = StoneProverTranscript::new(&public_inputs.as_bytes());
        let air = A::new(proof.trace_length, public_inputs, proof_options);
        let domain = Domain::<Stark252PrimeField>::new(&air);
        let challenges = Verifier::step_1_replay_rounds_and_recover_challenges(
            &air,
            proof,
            &domain,
            &mut transcript,
        );
        challenges.iotas
    }
}
