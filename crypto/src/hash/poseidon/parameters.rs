use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::{
        element::FieldElement,
        fields::{
            fft_friendly::stark_252_prime_field::Stark252PrimeField,
            montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        },
        traits::IsField,
    },
    unsigned_integer::element::{UnsignedInteger, U256},
};
use std::fmt::Debug;

type PoseidonConstants<F> = (Vec<FieldElement<F>>, Vec<Vec<FieldElement<F>>>);

#[derive(Clone)]
pub struct Parameters<F: IsField> {
    pub rate: usize,
    pub capacity: usize,
    pub alpha: u32,
    pub n_full_rounds: usize,
    pub n_partial_rounds: usize,
    pub round_constants: Vec<FieldElement<F>>,
    pub mds_matrix: Vec<Vec<FieldElement<F>>>,
}

/// Implements hashing for BLS 12381's field.
/// Alpha = 5 and parameters are predefined for secure implementations
impl Parameters<BLS12381PrimeField> {
    // t = 3 means width of input is 2
    // sage generate_params_poseidon.sage 1 0 381 3 5 128
    // Params: n=381, t=3, alpha=5, M=128, R_F=8, R_P=56
    pub fn with_t3() -> Result<Self, String> {
        let round_constants_csv = include_str!("bls12381/t3/round_constants.csv");
        let mds_constants_csv = include_str!("bls12381/t3/mds_matrix.csv");

        let (round_constants, mds_matrix) = Self::parse(round_constants_csv, mds_constants_csv)?;
        Ok(Parameters {
            rate: 2,
            capacity: 1,
            alpha: 5,
            n_full_rounds: 8,
            n_partial_rounds: 56,
            round_constants,
            mds_matrix,
        })
    }

    // t = 2 means width of input size is 1
    // sage generate_params_poseidon.sage 1 0 381 2 5 128
    // Params: n=381, t=2, alpha=5, M=128, R_F=8, R_P=56
    pub fn with_t2() -> Result<Parameters<BLS12381PrimeField>, String> {
        let round_constants_csv = include_str!("bls12381/t2/round_constants.csv");
        let mds_constants_csv = include_str!("bls12381/t2/mds_matrix.csv");

        let (round_constants, mds_matrix) = Self::parse(round_constants_csv, mds_constants_csv)?;

        Ok(Parameters {
            rate: 1,
            capacity: 1,
            alpha: 5,
            n_full_rounds: 8,
            n_partial_rounds: 56,
            round_constants,
            mds_matrix,
        })
    }

    pub fn parse(
        round_constants_csv: &str,
        mds_constants_csv: &str,
    ) -> Result<PoseidonConstants<BLS12381PrimeField>, String> {
        let round_constants = round_constants_csv
            .split(',')
            .map(|c| FieldElement::<BLS12381PrimeField>::new_base(c.trim()))
            .collect();

        let mut mds_matrix = vec![];

        for line in mds_constants_csv.lines() {
            let matrix_line = line
                .split(',')
                .map(|c| FieldElement::<BLS12381PrimeField>::new_base(c.trim()))
                .collect();

            mds_matrix.push(matrix_line);
        }

        Ok((round_constants, mds_matrix))
    }
}

impl<M, const NUM_LIMBS: usize> Parameters<MontgomeryBackendPrimeField<M, NUM_LIMBS>>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    pub fn default_parse(
        round_constants_csv: &str,
        mds_constants_csv: &str,
    ) -> Result<PoseidonConstants<MontgomeryBackendPrimeField<M, NUM_LIMBS>>, String> {
        let round_constants: Vec<_> = round_constants_csv
            .split(',')
            .map(|c| {
                FieldElement::<MontgomeryBackendPrimeField<M, NUM_LIMBS>>::from_hex_unchecked(
                    c.trim(),
                )
            })
            .collect();

        let mut mds_matrix = vec![];

        for line in mds_constants_csv.lines() {
            let matrix_line: Vec<_> = line
                .split(',')
                .map(|c| {
                    FieldElement::<MontgomeryBackendPrimeField<M, NUM_LIMBS>>::from_hex_unchecked(
                        c.trim(),
                    )
                })
                .collect();

            mds_matrix.push(matrix_line);
        }

        Ok((round_constants, mds_matrix))

        // Ok(Parameters {
        //     rate: 2,
        //     capacity: 1,
        //     alpha: 3,
        //     n_full_rounds: 8,
        //     n_partial_rounds: 83,
        //     round_constants,
        //     mds_matrix,
        // })
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;

//     #[test]
//     fn test_starkfield_params() {
//         let round_constants_csv = include_str!("s128b/round_constants.csv");
//         let mds_constants_csv = include_str!("s128b/mds_matrix.csv");

//         let f = FieldElement::<Stark252PrimeField>::new(U256::from_hex_unchecked(
//             "6861759ea5568bb39dd92f9562a30b9e58e2ad98109ae4780b7fd8eac77fe81",
//             // "0x6861759ea5568bb39dd92f9562a30b9e58e2ad98109ae4780b7fd8eac77fe81",
//         ));

//         println!("EFE: {}", f);

//         let params =
//             Parameters::<Stark252PrimeField>::parse(round_constants_csv, mds_constants_csv);
//     }
// }
