use lambdaworks_math::field::{element::FieldElement as FE, traits::IsPrimeField};

use crate::hash::poseidon::cairo_poseidon_constants::round_constants::ROUND_CONSTANTS_HEXSTRINGS;

pub struct PermutationParameters<F: IsPrimeField> {
    /// Exponent for the S box
    pub alpha: u32,
    pub n_full_rounds: usize,
    pub n_partial_rounds: usize,
    pub round_constants: Vec<Vec<FE<F>>>,
    pub mds_matrix: Vec<Vec<FE<F>>>,
    pub state_size: usize,
}
pub enum DefaultPoseidonParams {
    /// Poseidon as used by Cairo
    /// with three inputs
    CairoStark252,
}

/// Parameters for Poseidon
/// Mds constants and rounds constants should be used for the shared field, even if it technically can work for any field with the same configuration
impl<F> PermutationParameters<F>
where
    F: IsPrimeField,
{
    pub fn new_with(params: DefaultPoseidonParams) -> Self {
        match params {
            DefaultPoseidonParams::CairoStark252 => Self::cairo_stark_params(),
        }
    }

    fn cairo_stark_params() -> PermutationParameters<F> {
        let round_constants: Vec<Vec<FE<F>>> = ROUND_CONSTANTS_HEXSTRINGS
            .iter()
            .map(|[x0, x1, x2]| {
                [
                    FE::<F>::from_hex(x0).unwrap(),
                    FE::<F>::from_hex(x1).unwrap(),
                    FE::<F>::from_hex(x2).unwrap(),
                ]
                .to_vec()
            })
            .collect();

        let mds_matrix = [
            [FE::<F>::from(3), FE::<F>::from(1), FE::<F>::from(1)].to_vec(),
            [FE::<F>::from(1), -FE::one(), FE::<F>::from(1)].to_vec(),
            [FE::<F>::from(1), FE::<F>::from(1), -FE::<F>::from(2)].to_vec(),
        ]
        .to_vec();

        const RATE: usize = 2;
        const CAPACITY: usize = 1;
        const ALPHA: u32 = 3;
        const N_FULL_ROUNDS: usize = 8;
        const N_PARTIAL_ROUNDS: usize = 83;
        Self {
            alpha: ALPHA,
            n_full_rounds: N_FULL_ROUNDS,
            n_partial_rounds: N_PARTIAL_ROUNDS,
            round_constants,
            mds_matrix,
            state_size: RATE + CAPACITY,
        }
    }
}
