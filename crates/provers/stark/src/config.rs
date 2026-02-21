use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;

// Merkle Trees configuration

// Security of both hashes should match

#[cfg(not(feature = "poseidon-goldilocks"))]
use lambdaworks_crypto::merkle_tree::backends::types::{BatchKeccak256Backend, Keccak256Backend};

#[cfg(feature = "poseidon-goldilocks")]
use lambdaworks_crypto::merkle_tree::backends::poseidon_goldilocks_bytes::{
    BatchPoseidonGoldilocksBytes32Backend, PoseidonGoldilocksBytes32Backend,
};

#[cfg(not(feature = "poseidon-goldilocks"))]
pub type FriMerkleTreeBackend<F> = Keccak256Backend<F>;
#[cfg(feature = "poseidon-goldilocks")]
pub type FriMerkleTreeBackend<F> = PoseidonGoldilocksBytes32Backend<F>;

pub type FriMerkleTree<F> = MerkleTree<FriMerkleTreeBackend<F>>;

// If using hashes with 256-bit security, commitment size should be 32
// If using hashes with 512-bit security, commitment size should be 64
// TODO: Commitment type should be obtained from MerkleTrees
pub const COMMITMENT_SIZE: usize = 32;
pub type Commitment = [u8; COMMITMENT_SIZE];

#[cfg(not(feature = "poseidon-goldilocks"))]
pub type BatchedMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
#[cfg(feature = "poseidon-goldilocks")]
pub type BatchedMerkleTreeBackend<F> = BatchPoseidonGoldilocksBytes32Backend<F>;

pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;
