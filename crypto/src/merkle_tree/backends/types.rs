use sha2::{Sha256, Sha512};
use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

use super::{field_element::FieldElementBackend, field_element_vector::FieldElementVectorBackend};

// Field element backend definitions

// - With 256 bit
pub type Sha3_256Backend<F> = FieldElementBackend<F, Sha3_256, 32>;
pub type Keccak256Backend<F> = FieldElementBackend<F, Keccak256, 32>;
pub type Sha2_256Backend<F> = FieldElementBackend<F, Sha256, 32>;

// - With 512 bit
pub type Sha3_512Backend<F> = FieldElementBackend<F, Sha3_512, 64>;
pub type Keccak512Backend<F> = FieldElementBackend<F, Keccak512, 64>;
pub type Sha2_512Backend<F> = FieldElementBackend<F, Sha512, 64>;

// Vector of field elements backend definitions

// - With 256 bit
pub type BatchSha3_256Backend<F> = FieldElementVectorBackend<F, Sha3_256, 32>;
pub type BatchKeccak256Backend<F> = FieldElementVectorBackend<F, Keccak256, 32>;
pub type BatchSha2_256Backend<F> = FieldElementVectorBackend<F, Sha256, 32>;

// - With 512 bit
pub type BatchSha3_512Backend<F> = FieldElementVectorBackend<F, Sha3_512, 64>;
pub type BatchKeccak512Backend<F> = FieldElementVectorBackend<F, Keccak512, 64>;
pub type BatchSha2_512Backend<F> = FieldElementVectorBackend<F, Sha512, 64>;
