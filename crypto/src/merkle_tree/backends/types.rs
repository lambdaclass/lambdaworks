use sha2::{Sha256, Sha512};
use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

use super::{
    batch_256_bits::Batch256BitsTree, batch_512_bits::Batch512BitsTree, hash_256_bits::Tree256Bits,
    hash_512_bits::Tree512Bits,
};

// Trees definitions

// - With 256 bit
pub type Sha3_256Tree<F> = Tree256Bits<F, Sha3_256>;
pub type Keccak256Tree<F> = Tree256Bits<F, Keccak256>;
pub type Sha2_256Tree<F> = Tree256Bits<F, Sha256>;

// - With 512 bit
pub type Sha3_512Tree<F> = Tree512Bits<F, Sha3_512>;
pub type Keccak512Tree<F> = Tree512Bits<F, Keccak512>;
pub type Sha2_512Tree<F> = Tree512Bits<F, Sha512>;

// Batch trees definitions

// - With 256 bit
pub type BatchSha3_256Tree<F> = Batch256BitsTree<F, Sha3_256>;
pub type BatchKeccak256Tree<F> = Batch256BitsTree<F, Keccak256>;
pub type BatchSha2_256Tree<F> = Batch256BitsTree<F, Sha256>;

// - With 512 bit
pub type BatchSha3_512Tree<F> = Batch512BitsTree<F, Sha3_512>;
pub type BatchKeccak512Tree<F> = Batch512BitsTree<F, Keccak512>;
pub type BatchSha2_512Tree<F> = Batch512BitsTree<F, Sha512>;
