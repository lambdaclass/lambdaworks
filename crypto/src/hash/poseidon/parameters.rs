use lambdaworks_math::field::{element::FieldElement as FE, traits::IsPrimeField};

/// Parameters for Poseidon
/// MDS constants and rounds constants are stored as references to slices
/// representing matrices of `N_MDS_MATRIX_ROWS * N_MDS_MATRIX_COLS` and
/// `N_ROUND_CONSTANTS_ROWS * N_ROUND_CONSTANTS_COLS` respectively.
/// We use this representation rather than an array because we can't use the
/// associated constants for dimension, requiring many generic parameters
/// otherwise.
pub trait PermutationParameters {
    type F: IsPrimeField + 'static;

    const RATE: usize;
    const CAPACITY: usize;
    const ALPHA: u32;
    const N_FULL_ROUNDS: usize;
    const N_PARTIAL_ROUNDS: usize;
    const STATE_SIZE: usize = Self::RATE + Self::CAPACITY;

    const MDS_MATRIX: &'static [FE<Self::F>];
    const N_MDS_MATRIX_ROWS: usize;
    const N_MDS_MATRIX_COLS: usize;

    const ROUND_CONSTANTS: &'static [FE<Self::F>];
    const N_ROUND_CONSTANTS_ROWS: usize;
    const N_ROUND_CONSTANTS_COLS: usize;
}
