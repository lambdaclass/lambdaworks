use crate::common::*;
use lambdaworks_math::polynomial::Polynomial;

#[derive(Debug)]
pub struct QAP {
    pub num_of_public_inputs: usize,
    pub num_of_total_inputs: usize,
    pub num_of_gates: usize,
    pub l: Vec<Polynomial<FrElement>>,
    pub r: Vec<Polynomial<FrElement>>,
    pub o: Vec<Polynomial<FrElement>>,
    pub t: Polynomial<FrElement>,
}

impl QAP {
    pub fn from_hex_matrices(
        num_of_public_inputs: usize,
        domain: Vec<FrElement>,
        l: Vec<Vec<&str>>,
        r: Vec<Vec<&str>>,
        o: Vec<Vec<&str>>,
    ) -> Self {
        assert!(l.len() == r.len() && r.len() == o.len() && num_of_public_inputs <= l.len());
        let num_of_total_inputs = l.len();
        let num_of_gates = domain.len();
        assert!(num_of_gates < num_of_total_inputs);

        Self {
            num_of_public_inputs,
            num_of_total_inputs,
            num_of_gates,
            l: Self::build_variable_polynomial(&domain, &l),
            r: Self::build_variable_polynomial(&domain, &r),
            o: Self::build_variable_polynomial(&domain, &o),
            t: Self::build_vanishing_polynomial(&domain),
        }
    }

    pub fn calculate_h_coefficients(&self, w: &[FrElement]) -> Vec<FrElement> {
        let p = self.calculate_p(w);

        // h(x) = p(x) / t(x)
        let (h, remainder) = p.long_division_with_remainder(&self.t);
        assert_eq!(0, remainder.degree()); // must have no remainder

        h.coefficients().to_vec()
    }

    fn calculate_p(&self, w: &[FrElement]) -> Polynomial<FrElement> {
        // Compute A.s by summing up polynomials A[0].s, A[1].s, ..., A[n].s
        // In other words, assign the witness coefficients / execution values
        // Similarly for B.s and C.s

        let l = Self::assign_to_variable_polynomials(&self.l, w);
        let r = Self::assign_to_variable_polynomials(&self.r, w);
        let o = Self::assign_to_variable_polynomials(&self.o, w);

        &l * &r - &o
    }

    fn build_vanishing_polynomial(domain: &[FrElement]) -> Polynomial<FrElement> {
        let mut target_poly = Polynomial::new(&[FrElement::one()]);
        for gate_index in domain {
            target_poly = target_poly * Polynomial::new(&[-gate_index, FrElement::one()]);
        }
        target_poly
    }

    fn build_variable_polynomial(
        domain: &Vec<FrElement>,
        from_hex_matrix: &Vec<Vec<&str>>,
    ) -> Vec<Polynomial<FrElement>> {
        let mut polynomials = vec![];
        for i in 0..from_hex_matrix.len() {
            let mut y_indices = vec![];
            for string in &from_hex_matrix[i] {
                y_indices.push(FrElement::from_hex_unchecked(*string));
            }
            polynomials.push(Polynomial::interpolate(domain, &y_indices).unwrap());
        }
        polynomials
    }

    fn assign_to_variable_polynomials(
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
