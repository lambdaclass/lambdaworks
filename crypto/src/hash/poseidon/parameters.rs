use lambdaworks_math::{
    field::{element::FieldElement, traits::{IsPrimeField}}
};
pub struct Parameters<F: IsPrimeField> {
    /// Max Input/Output size. More reduces security, and increases performance by reducing the amount of digests/absorptions
    pub rate: usize,
    /// Internal state. More increases security
    pub capacity: usize,
    /// Exponent for the S box
    pub alpha: u32,
    pub n_full_rounds: usize,
    pub n_partial_rounds: usize,
    pub add_round_constants: Vec<FieldElement<F>>,
    pub mds_matrix: Vec<Vec<FieldElement<F>>>,
}

pub enum DefaultPoseidonParams{
    /// Poseidon as used by Cairo
    /// with three inputs
    CairoStark252,
}

/// Parameters for Poseidon
/// Mds constants and rounds constants should be used for the shared field, even if it technically can work for any field with the same configuration
impl <F> Parameters<F> where
F: IsPrimeField  { 
    pub fn new_with(params: DefaultPoseidonParams) -> Self{
        match params {
            DefaultPoseidonParams::CairoStark252 => 
                Self::cairo_stark_params()
        }
    }

    fn cairo_stark_params() -> Parameters<F>{
        const ADD_ROUND_CONSTANTS_CSV: &str = include_str!("cairo_poseidon_constants/round_constants.csv");
        const MDS_MATRIX_CSV: &str = include_str!("cairo_poseidon_constants/mds_matrix.csv");

        let add_round_constants = Self::decode_add_round_constants(ADD_ROUND_CONSTANTS_CSV);
        let mds_matrix = Self::decode_mds_matrix(MDS_MATRIX_CSV);

        const RATE: usize = 1;
        const N_FULL_ROUNDS: usize = 8;
        const N_PARTIAL_ROUNDS: usize = 56;
        const ALPHA: u32 = 5;
        const CAPACITY: usize = 1;
        
        Self{
            rate: RATE,
            capacity: CAPACITY,
            /// Exponent for the S box
            alpha: ALPHA,
            n_full_rounds: N_FULL_ROUNDS,
            n_partial_rounds: N_PARTIAL_ROUNDS,
            add_round_constants,
            mds_matrix,
        }
    }

    fn decode_add_round_constants(
        round_constants_csv: &str,
    ) -> Vec<FieldElement<F>>
    where F: IsPrimeField
    {
        let arc: Vec<FieldElement<F>> = round_constants_csv
            .split(',')
            .map(|string| string.trim())
            .map(
                |hex| 
                FieldElement::<F>::from_hex(hex).expect("Wrong hex in arc file"))
            .collect();
        arc
    }

    fn decode_mds_matrix(
        mds_constants_csv: &str,
    ) -> Vec<Vec<FieldElement<F>>>
    where F: IsPrimeField
    {
        let mut mds_matrix: Vec<Vec<FieldElement<F>>> = vec![];

        for line in mds_constants_csv.lines() {
            let matrix_line = line
                .split(',')
                .map(|string| string.trim())
                .map(
                    |hex| FieldElement::<F>::from_hex(hex).expect("Wrong hex in mds file"))
                .collect();

            mds_matrix.push(matrix_line);
        }
        mds_matrix
    }
}
