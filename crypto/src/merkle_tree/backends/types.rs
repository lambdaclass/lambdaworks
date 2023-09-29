use sha2::{Sha256, Sha512};
use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

use super::{
    batch::BatchBackend, hash::Backend
};

// Trees definitions

// - With 256 bit
pub type Sha3_256Tree<F> = Backend<F, Sha3_256, 32>;
pub type Keccak256Tree<F> = Backend<F, Keccak256, 32>;
pub type Sha2_256Tree<F> = Backend<F, Sha256, 32>;

// - With 512 bit
pub type Sha3_512Tree<F> = Backend<F, Sha3_512, 64>;
pub type Keccak512Tree<F> = Backend<F, Keccak512, 64>;
pub type Sha2_512Tree<F> = Backend<F, Sha512, 64>;

// Batch trees definitions

// - With 256 bit
pub type BatchSha3_256Tree<F> = BatchBackend<F, Sha3_256, 32>;
pub type BatchKeccak256Tree<F> = BatchBackend<F, Keccak256, 32>;
pub type BatchSha2_256Tree<F> = BatchBackend<F, Sha256, 32>;

// - With 512 bit
pub type BatchSha3_512Tree<F> = BatchBackend<F, Sha3_512, 64>;
pub type BatchKeccak512Tree<F> = BatchBackend<F, Keccak512, 64>;
pub type BatchSha2_512Tree<F> = BatchBackend<F, Sha512, 64>;
