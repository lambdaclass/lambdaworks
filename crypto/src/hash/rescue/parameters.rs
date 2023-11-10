use lambdaworks_math::field::{
    element::FieldElement, traits::IsField,
};

type RescuePrimeConstants<F> = (Vec<FieldElement<F>>, Vec<Vec<FieldElement<F>>>);

// A Rescue Prime Hash Family is fully determined by the tuple of primary parameters which are:
// p - Prime Field
// m - width of the hash field 
// c_p - capacity 
// s - security level (80 <= s <= 512)

#[derive(Clone)]
pub struct Parameters<F: IsField, const WIDTH: usize, const CAPACITY: usize, const SECURITY_LEVEL: usize> {
    pub m: usize, // WIDTH
    pub capacity: usize, //Cp
    pub rate: usize, // r = m - Cp
    pub n: usize, // no of rounds
    pub alpha: u32,
    pub alpha_inv: usize,
    pub round_constants: Vec<FieldElement<F>>,
    pub mds_matrix: Vec<Vec<FieldElement<F>>>,
}

impl<F: IsField, const WIDTH: usize, const CAPACITY: usize, const SECURITY_LEVEL: usize> Parameters<F, WIDTH, CAPACITY, SECURITY_LEVEL> 
// where <F as lambdaworks_math::field::traits::IsField>::BaseType: std::convert::From<&str>
where 
    <F as IsField>::BaseType: for<'a>std::convert::From<& 'a str>,
{
    pub fn new() -> Result<Self, String> {
        let round_constants_csv = include_str!("stark252/round_constants.csv");
        let mds_constants_csv = include_str!("stark252/mds_matrix.csv");
        let (round_constants, mds_matrix) = Self::parse(round_constants_csv, mds_constants_csv)?;
        let alpha_inv: usize = 1717986917;
        Ok(Parameters {
            m: 12,
            rate: 8,
            capacity: 4,
            n: 20,
            alpha: 3,
            alpha_inv,
            round_constants,
            mds_matrix,
        })
    }

    fn parse(
        round_constants_csv: &str,
        mds_constants_csv: &str,
    ) -> Result<RescuePrimeConstants<F>, String> {
        let round_constants = round_constants_csv
            .split(',')
            .map(|c| FieldElement::<F>::new(c.trim().into()))
            .collect();

        let mut mds_matrix = vec![];

        for line in mds_constants_csv.lines() {
            let matrix_line = line
                .split(',')
                .map(|c| FieldElement::<F>::new(c.trim().into()))
                .collect();
            mds_matrix.push(matrix_line);
        }

        Ok((round_constants, mds_matrix))
    }
}
