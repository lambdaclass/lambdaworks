// t = 2 means width of input is 1
// Params: n=381, t=2, alpha=5, M=128, R_F=8, R_P=56
// Modulus = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
// Number of S-boxes: 72
// Number of round constants: 128
// Round constants for GF(p):

// t = 3 means width of input is 2
// Params: n=381, t=3, alpha=5, M=128, R_F=8, R_P=56
// Modulus = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
// Number of S-boxes: 80
// Number of round constants: 192

use lambdaworks_math::{
    elliptic_curve::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::{element::FieldElement, traits::IsField},
};

type PoseidonConstants = (
    Vec<FieldElement<BLS12381PrimeField>>,
    Vec<Vec<FieldElement<BLS12381PrimeField>>>,
);

pub struct Parameters<F: IsField> {
    pub s_boxes: usize,
    /// Full Rounds
    pub n_full_rounds: usize,
    /// Partial rounds
    pub n_partial_rounds: usize,
    pub round_constants: Vec<FieldElement<F>>,
    pub mds_matrix: Vec<Vec<FieldElement<F>>>,
}

impl Parameters<BLS12381PrimeField> {
    pub fn for_two_elements() -> Result<Parameters<BLS12381PrimeField>, String> {
        let round_constants_csv = include_str!("t3/round_constants.csv");
        let mds_constants_csv = include_str!("t3/mds_matrix.csv");
        let (round_constants, mds_matrix) = Self::parse(round_constants_csv, mds_constants_csv)?;
        Ok(Parameters {
            s_boxes: 80,
            n_full_rounds: 8,
            n_partial_rounds: 56,
            round_constants,
            mds_matrix,
        })
    }

    pub fn for_one_element() -> Result<Parameters<BLS12381PrimeField>, String> {
        let round_constants_csv = include_str!("t2/round_constants.csv");
        let mds_constants_csv = include_str!("t2/mds_matrix.csv");
        let (round_constants, mds_matrix) = Self::parse(round_constants_csv, mds_constants_csv)?;

        Ok(Parameters {
            s_boxes: 72,
            n_full_rounds: 8,
            n_partial_rounds: 56,
            round_constants,
            mds_matrix,
        })
    }

    fn parse(
        round_constants_csv: &str,
        mds_constants_csv: &str,
    ) -> Result<PoseidonConstants, String> {
        let round_constants = round_constants_csv
            .split(',')
            .map(FieldElement::<BLS12381PrimeField>::new_base)
            .collect();

        let mut mds_matrix = vec![];

        for line in mds_constants_csv.lines() {
            let matrix_line = line
                .split(',')
                .map(FieldElement::<BLS12381PrimeField>::new_base)
                .collect();

            mds_matrix.push(matrix_line);
        }

        Ok((round_constants, mds_matrix))
    }
}
