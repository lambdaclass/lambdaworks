use lambdaworks_math::{
    field::{element::FieldElement, traits::{IsPrimeField}},
};

type PoseidonConstants<F> = (Vec<FieldElement<F>>, Vec<Vec<FieldElement<F>>>);

pub struct Parameters<F: IsPrimeField> {
    /// Max Input/Output size. More reduces security, and increases performance by reducing the amount of digests/absorptions
    pub rate: usize,
    /// Internal state. More increases security
    pub capacity: usize,
    /// Exponent for the S box
    pub alpha: u32,
    pub n_full_rounds: usize,
    pub n_partial_rounds: usize,
    pub round_constants: Vec<FieldElement<F>>,
    pub mds_matrix: Vec<Vec<FieldElement<F>>>,
}

pub enum DefaultPoseidonParams{
    /// Poseidon as used by Cairo
    /// with three inputs
    CairoStark252,
}

/// Parameters for Poseidon
/// Mds constants and rounds constants should be used for the shared field, even if it technically can work for any field with the same configuration
impl <F: IsPrimeField>Parameters<F> {
    pub fn new_with(params: DefaultPoseidonParams){
        match params {
            DefaultPoseidonParams::CairoStark252 => 
                Self::cairo_stark_params()

        }
    }

    fn cairo_stark_params() -> Parameters<F>{
        const round_constants: &str = include_str!("cairo_poseidon_constants/t2/round_constants.csv");
        const mds_matrix: &str = include_str!("cairo_poseidon_constants/t2/mds_matrix.csv");


        const rate: usize = 1;
        const n_full_rounds: usize = 8;
        const n_partial_rounds: usize = 56;
        const alpha: u32 = 5;
        
        Self{
            rate,
            capacity,
            /// Exponent for the S box
            alpha,
            n_full_rounds,
            n_partial_rounds,
            round_constants,
            mds_matrix,
        }

    }
}
