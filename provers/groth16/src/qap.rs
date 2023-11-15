use lambdaworks_math::{
    fft::polynomial::FFTPoly,
    polynomial::{traits::polynomial::IsPolynomial, univariate::UnivariatePolynomial},
};

use crate::common::*;

#[derive(Debug)]
pub struct QuadraticArithmeticProgram {
    pub num_of_public_inputs: usize,
    pub l: Vec<UnivariatePolynomial<FrField>>,
    pub r: Vec<UnivariatePolynomial<FrField>>,
    pub o: Vec<UnivariatePolynomial<FrField>>,
}

impl QuadraticArithmeticProgram {
    pub fn from_variable_matrices(
        num_of_public_inputs: usize,
        l: &[Vec<FrElement>],
        r: &[Vec<FrElement>],
        o: &[Vec<FrElement>],
    ) -> Self {
        let num_of_total_inputs = l.len();
        assert_eq!(num_of_total_inputs, r.len());
        assert_eq!(num_of_total_inputs, o.len());
        assert!(num_of_total_inputs > 0);
        assert!(num_of_public_inputs <= num_of_total_inputs);

        let num_of_gates = l[0].len();
        let pad_zeroes = num_of_gates.next_power_of_two() - num_of_gates;
        let l = Self::apply_padding(l, pad_zeroes);
        let r = Self::apply_padding(r, pad_zeroes);
        let o = Self::apply_padding(o, pad_zeroes);

        Self {
            num_of_public_inputs,
            l: Self::build_variable_polynomials(&l),
            r: Self::build_variable_polynomials(&r),
            o: Self::build_variable_polynomials(&o),
        }
    }

    pub fn num_of_gates(&self) -> usize {
        self.l[0].degree() + 1
    }

    pub fn num_of_private_inputs(&self) -> usize {
        self.l.len() - self.num_of_public_inputs
    }

    pub fn num_of_total_inputs(&self) -> usize {
        self.l.len()
    }

    pub fn calculate_h_coefficients(&self, w: &[FrElement]) -> Vec<FrElement> {
        let offset = &ORDER_R_MINUS_1_ROOT_UNITY;
        let degree = self.num_of_gates() * 2;

        let [l, r, o] = self.scale_and_accumulate_variable_polynomials(w, degree, offset);

        // TODO: Change to a vector of offsetted evaluations of x^N-1
        let mut t = (UnivariatePolynomial::new_monomial(FrElement::one(), self.num_of_gates())
            - FrElement::one())
        .evaluate_offset_fft(1, Some(degree), offset)
        .unwrap();
        FrElement::inplace_batch_inverse(&mut t).unwrap();

        let h_evaluated = l
            .iter()
            .zip(&r)
            .zip(&o)
            .zip(&t)
            .map(|(((l, r), o), t)| (l * r - o) * t)
            .collect::<Vec<_>>();

        UnivariatePolynomial::interpolate_offset_fft(&h_evaluated, offset)
            .unwrap()
            .coeffs()
            .to_vec()
    }

    fn apply_padding(columns: &[Vec<FrElement>], pad_zeroes: usize) -> Vec<Vec<FrElement>> {
        let from_slice = vec![FrElement::zero(); pad_zeroes];
        columns
            .iter()
            .map(|column| {
                let mut new_column = column.clone();
                new_column.extend_from_slice(&from_slice);
                new_column
            })
            .collect::<Vec<_>>()
    }

    fn build_variable_polynomials(
        from_matrix: &[Vec<FrElement>],
    ) -> Vec<UnivariatePolynomial<FrField>> {
        from_matrix
            .iter()
            .map(|row| UnivariatePolynomial::interpolate_fft(row).unwrap())
            .collect()
    }

    // Compute A.s by summing up polynomials A[0].s, A[1].s, ..., A[n].s
    // In other words, assign the witness coefficients / execution values
    // Similarly for B.s and C.s
    fn scale_and_accumulate_variable_polynomials(
        &self,
        w: &[FrElement],
        degree: usize,
        offset: &FrElement,
    ) -> [Vec<FrElement>; 3] {
        [&self.l, &self.r, &self.o].map(|var_polynomials| {
            var_polynomials
                .iter()
                .zip(w)
                .map(|(poly, coeff)| {
                    poly.mul_with_ref(&UnivariatePolynomial::new_monomial(coeff.clone(), 0))
                })
                .reduce(|poly1, poly2| poly1 + poly2)
                .unwrap()
                .evaluate_offset_fft(1, Some(degree), offset)
                .unwrap()
        })
    }
}
