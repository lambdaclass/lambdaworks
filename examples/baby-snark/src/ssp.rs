use lambdaworks_math::polynomial::Polynomial;

use crate::{common::*, scs::SquareConstraintSystem};

#[derive(Debug)]
pub struct SquareSpanProgram {
    pub num_of_public_inputs: usize,
    pub num_of_gates: usize,
    pub u_poly: Vec<Polynomial<FrElement>>,
}

impl SquareSpanProgram {
    pub fn calculate_h_coefficients(&self, input: &[FrElement]) -> Vec<FrElement> {
        let offset = &ORDER_R_MINUS_1_ROOT_UNITY;
        let p_degree = 2 * self.num_of_gates - 2;
        let h_degree = self.num_of_gates - 2;

        let u = self.scale_and_accumulate_variable_polynomials(input, p_degree, offset);

        let t_poly =
            Polynomial::new_monomial(FrElement::one(), self.num_of_gates) - FrElement::one();
        let mut t = Polynomial::evaluate_offset_fft(&t_poly, 1, Some(p_degree), offset).unwrap();
        FrElement::inplace_batch_inverse(&mut t).unwrap();

        let h_evaluated = u
            .iter()
            .zip(&t)
            .map(|(u, t)| (u * u - FrElement::one()) * t)
            .collect::<Vec<_>>();

        let mut h_coefficients = Polynomial::interpolate_offset_fft(&h_evaluated, offset)
            .unwrap()
            .coefficients()
            .to_vec();

        let pad = vec![FrElement::zero(); h_degree + 1 - h_coefficients.len()];
        h_coefficients.extend(pad);

        h_coefficients
    }

    // Compute U.w by summing up polynomials U[0].w_0, U[1].w_1, ..., U[n].w_n
    fn scale_and_accumulate_variable_polynomials(
        &self,
        w: &[FrElement],
        degree: usize,
        offset: &FrElement,
    ) -> Vec<FrElement> {
        let scaled_and_accumulated = self
            .u_poly
            .iter()
            .zip(w)
            .map(|(poly, coeff)| poly.mul_with_ref(&Polynomial::new_monomial(coeff.clone(), 0)))
            .reduce(|poly1, poly2| poly1 + poly2)
            .unwrap();

        Polynomial::evaluate_offset_fft(&scaled_and_accumulated, 1, Some(degree), offset).unwrap()
    }

    pub fn from_scs(scs: SquareConstraintSystem) -> Self {
        let num_of_gates = scs.number_of_constraints().next_power_of_two();

        let mut u_poly: Vec<Polynomial<FrElement>> = vec![];

        for var_idx in 0..scs.input_size() {
            let poly = get_var_poly_from_scs(&scs, var_idx, num_of_gates);

            u_poly.push(poly);
        }

        Self {
            num_of_public_inputs: scs.number_of_public_inputs,
            num_of_gates,
            u_poly,
        }
    }

    pub fn num_of_private_inputs(&self) -> usize {
        self.u_poly.len() - self.num_of_public_inputs
    }
}

#[inline]
fn get_var_poly_from_scs(
    scs: &SquareConstraintSystem,
    var_idx: usize,
    num_of_gates: usize,
) -> Polynomial<FrElement> {
    let mut var_u = vec![FrElement::zero(); num_of_gates];

    for (constraint_idx, constraint) in scs.constraints.iter().enumerate() {
        var_u[constraint_idx] = constraint[var_idx].clone();
    }

    Polynomial::interpolate_fft::<FrField>(&var_u).unwrap()
}
