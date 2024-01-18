use lambdaworks_math::polynomial::Polynomial;

use crate::{common::*, r1cs::R1CS};

#[derive(Debug)]
pub struct QuadraticArithmeticProgram {
    pub num_of_public_inputs: usize,
    pub num_of_gates: usize,
    pub l: Vec<Polynomial<FrElement>>,
    pub r: Vec<Polynomial<FrElement>>,
    pub o: Vec<Polynomial<FrElement>>,
}

impl QuadraticArithmeticProgram {
    pub fn calculate_h_coefficients(&self, w: &[FrElement]) -> Vec<FrElement> {
        let offset = &ORDER_R_MINUS_1_ROOT_UNITY;
        let degree = self.num_of_gates * 2;

        let [l, r, o] = self.scale_and_accumulate_variable_polynomials(w, degree, offset);

        // TODO: Change to a vector of offsetted evaluations of x^N-1
        let t_poly =
            Polynomial::new_monomial(FrElement::one(), self.num_of_gates) - FrElement::one();
        let mut t = Polynomial::evaluate_offset_fft(&t_poly, 1, Some(degree), offset).unwrap();
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
            Polynomial::evaluate_offset_fft(
                &(var_polynomials
                    .iter()
                    .zip(w)
                    .map(|(poly, coeff)| {
                        poly.mul_with_ref(&Polynomial::new_monomial(coeff.clone(), 0))
                    })
                    .reduce(|poly1, poly2| poly1 + poly2)
                    .unwrap()),
                1,
                Some(degree),
                offset,
            )
            .unwrap()
        })
    }

    pub fn num_of_private_inputs(&self) -> usize {
        self.l.len() - self.num_of_public_inputs
    }

    pub fn from_r1cs(r1cs: R1CS) -> QuadraticArithmeticProgram {
        let num_gates = r1cs.number_of_constraints();
        let next_power_of_two = num_gates.next_power_of_two();
        let pad_zeroes = next_power_of_two - num_gates;

        let mut l: Vec<Polynomial<FrElement>> = vec![];
        let mut r: Vec<Polynomial<FrElement>> = vec![];
        let mut o: Vec<Polynomial<FrElement>> = vec![];
        for i in 0..r1cs.witness_size() {
            let [l_poly, r_poly, o_poly] =
                get_variable_lro_polynomials_from_r1cs(&r1cs, i, pad_zeroes);
            l.push(l_poly);
            r.push(r_poly);
            o.push(o_poly);
        }

        QuadraticArithmeticProgram {
            l,
            r,
            o,
            num_of_gates: next_power_of_two,
            num_of_public_inputs: r1cs.number_of_inputs,
        }
    }
}

#[inline]
fn get_variable_lro_polynomials_from_r1cs(
    r1cs: &R1CS,
    var_idx: usize,
    pad_zeroes: usize,
) -> [Polynomial<FrElement>; 3] {
    let cap = r1cs.number_of_constraints() + pad_zeroes;
    let mut current_var_l = vec![FrElement::zero(); cap];
    let mut current_var_r = vec![FrElement::zero(); cap];
    let mut current_var_o = vec![FrElement::zero(); cap];

    for (i, c) in r1cs.constraints.iter().enumerate() {
        current_var_l[i] = c.a[var_idx].clone();
        current_var_r[i] = c.b[var_idx].clone();
        current_var_o[i] = c.c[var_idx].clone();
    }

    [current_var_l, current_var_r, current_var_o]
        .map(|e| Polynomial::interpolate_fft::<FrField>(&e).unwrap())
}
