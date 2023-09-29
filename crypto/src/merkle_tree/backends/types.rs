use sha2::{Sha256, Sha512};
use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

use super::{
    batch::BatchBackend, single::SingleBackend
};

// Field element data backend definitions

// - With 256 bit
pub type Sha3_256Backend<F> = SingleBackend<F, Sha3_256, 32>;
pub type Keccak256Backend<F> = SingleBackend<F, Keccak256, 32>;
pub type Sha2_256Backend<F> = SingleBackend<F, Sha256, 32>;

// - With 512 bit
pub type Sha3_512Backend<F> = SingleBackend<F, Sha3_512, 64>;
pub type Keccak512Backend<F> = SingleBackend<F, Keccak512, 64>;
pub type Sha2_512Backend<F> = SingleBackend<F, Sha512, 64>;

// Vector field element data backend definitions

// - With 256 bit
pub type BatchSha3_256Backend<F> = BatchBackend<F, Sha3_256, 32>;
pub type BatchKeccak256Backend<F> = BatchBackend<F, Keccak256, 32>;
pub type BatchSha2_256Backend<F> = BatchBackend<F, Sha256, 32>;

// - With 512 bit
pub type BatchSha3_512Backend<F> = BatchBackend<F, Sha3_512, 64>;
pub type BatchKeccak512Backend<F> = BatchBackend<F, Keccak512, 64>;
pub type BatchSha2_512Backend<F> = BatchBackend<F, Sha512, 64>;
