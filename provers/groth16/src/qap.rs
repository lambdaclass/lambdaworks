use crate::common::*;
use lambdaworks_math::{fft::polynomial::FFTPoly, polynomial::Polynomial};

#[derive(Debug)]
pub struct QuadraticArithmeticProgram {
    pub num_of_public_inputs: usize,
    pub l: Vec<Polynomial<FrElement>>,
    pub r: Vec<Polynomial<FrElement>>,
    pub o: Vec<Polynomial<FrElement>>,
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
        assert!(l[0].len() <= num_of_total_inputs);

        Self {
            num_of_public_inputs,
            l: Self::build_variable_polynomials(l),
            r: Self::build_variable_polynomials(r),
            o: Self::build_variable_polynomials(o),
        }
    }

    pub fn num_of_gates(&self) -> usize {
        self.l[0].degree() + 1
    }

    pub fn num_of_private_inputs(&self) -> usize {
        self.l.len() - self.num_of_public_inputs
    }

    pub fn calculate_h_coefficients(&self, w: &[FrElement]) -> Vec<FrElement> {
        let offset = &ORDER_R_MINUS_1_ROOT_UNITY;
        let degree = self.num_of_gates() * 2;

        let [l, r, o] = self.scale_and_accumulate_variable_polynomials(w, degree, offset);

        // TODO: Change to a vector of offsetted evaluations of x^N-1
        let mut t = (Polynomial::new_monomial(FrElement::one(), self.num_of_gates())
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

        Polynomial::interpolate_offset_fft(&h_evaluated, offset)
            .unwrap()
            .coefficients()
            .to_vec()
    }

    fn build_variable_polynomials(from_matrix: &[Vec<FrElement>]) -> Vec<Polynomial<FrElement>> {
        from_matrix
            .iter()
            .map(|row| Polynomial::interpolate_fft(row).unwrap())
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
                .map(|(poly, coeff)| poly.mul_with_ref(&Polynomial::new_monomial(coeff.clone(), 0)))
                .reduce(|poly1, poly2| poly1 + poly2)
                .unwrap()
                .evaluate_offset_fft(1, Some(degree), offset)
                .unwrap()
        })
    }
}
