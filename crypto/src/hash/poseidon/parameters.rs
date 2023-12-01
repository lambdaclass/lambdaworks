/// Parameters for Poseidon
/// Mds constants and rounds constants should be used for the shared field, even if it technically can work for any field with the same configuration
use lambdaworks_math::field::{
    element::FieldElement as FE,
    traits::IsPrimeField,
};

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
