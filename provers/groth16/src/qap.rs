use crate::common::*;
use lambdaworks_math::{fft::polynomial::FFTPoly, polynomial::Polynomial};

#[derive(Debug)]
pub struct QAP {
    pub num_of_public_inputs: usize,
    pub l: Vec<Polynomial<FrElement>>,
    pub r: Vec<Polynomial<FrElement>>,
    pub o: Vec<Polynomial<FrElement>>,
}

impl QAP {
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

    pub fn calculate_h_coefficients(&self, w: &[FrElement]) -> Vec<FrElement> {
        // h(x) = p(x) / t(x)
        let (h, remainder) = self
            .calculate_p(w)
            .long_division_with_remainder(&self.vanishing_polynomial());
        assert_eq!(0, remainder.degree()); // must have no remainder

        h.coefficients().to_vec()
    }

    // t(X) = Z_H(x) = x^k-1 as our domain is roots of unity
    pub fn vanishing_polynomial(&self) -> Polynomial<FrElement> {
        Polynomial::new_monomial(FrElement::one(), self.num_of_gates()) - FrElement::one()
    }

    pub fn num_of_gates(&self) -> usize {
        self.l[0].degree() + 1
    }

    pub fn num_of_private_inputs(&self) -> usize {
        self.l.len() - self.num_of_public_inputs
    }

    fn calculate_p(&self, w: &[FrElement]) -> Polynomial<FrElement> {
        let l = Self::scale_and_accumulate_variable_polynomial(&self.l, w);
        let r = Self::scale_and_accumulate_variable_polynomial(&self.r, w);
        let o = Self::scale_and_accumulate_variable_polynomial(&self.o, w);

        &l * &r - &o
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
    fn scale_and_accumulate_variable_polynomial(
        var_polynomials: &[Polynomial<FrElement>],
        w: &[FrElement],
    ) -> Polynomial<FrElement> {
        var_polynomials
            .iter()
            .zip(w)
            .map(|(poly, coeff)| poly.mul_with_ref(&Polynomial::new_monomial(coeff.clone(), 0)))
            .reduce(|poly1, poly2| poly1 + poly2)
            .unwrap()
    }
}
