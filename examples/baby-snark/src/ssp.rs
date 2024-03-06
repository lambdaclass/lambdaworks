use lambdaworks_math::polynomial::Polynomial;

use crate::{common::*, scs::SquareConstraintSystem};

#[derive(Debug)]
pub struct SquareSpanProgram {
    pub number_of_public_inputs: usize,
    pub number_of_constraints: usize,
    pub u_polynomials: Vec<Polynomial<FrElement>>,
    pub matrix: Vec<Vec<FrElement>>,
}

impl SquareSpanProgram {
    pub fn calculate_h_coefficients(
        &self,
        input: &[FrElement],
        delta: &FrElement,
    ) -> Vec<FrElement> {
        let offset = &ORDER_R_MINUS_1_ROOT_UNITY;
        let p_degree = 2 * self.number_of_constraints;

        let u_evaluated = self.evaluate_scaled_and_acumulated_u(input, p_degree, offset);
        let (t_evaluated, t_inv_evaluated) = self.evaluate_t(p_degree, offset);

        let h_degree = self.number_of_constraints + 1;

        calculate_h_coefficients(
            u_evaluated,
            t_evaluated,
            t_inv_evaluated,
            h_degree,
            offset,
            delta,
        )
    }

    fn evaluate_t(&self, degree: usize, offset: &FrElement) -> (Vec<FrElement>, Vec<FrElement>) {
        let one_polynomial = Polynomial::new_monomial(FrElement::one(), self.number_of_constraints);
        let t_polynomial = one_polynomial - FrElement::one();

        let t_evaluated =
            Polynomial::evaluate_offset_fft(&t_polynomial, 1, Some(degree), offset).unwrap();
        let mut t_inv_evaluated = t_evaluated.clone();
        FrElement::inplace_batch_inverse(&mut t_inv_evaluated).unwrap();

        (t_evaluated, t_inv_evaluated)
    }

    // Compute U.w by summing up polynomials U[0].w_0, U[1].w_1, ..., U[n].w_n
    fn evaluate_scaled_and_acumulated_u(
        &self,
        w: &[FrElement],
        degree: usize,
        offset: &FrElement,
    ) -> Vec<FrElement> {
        let scaled_and_accumulated = self
            .u_polynomials
            .iter()
            .zip(w)
            .map(|(poly, coeff)| poly.mul_with_ref(&Polynomial::new_monomial(coeff.clone(), 0)))
            .reduce(|poly1, poly2| poly1 + poly2)
            .unwrap();

        Polynomial::evaluate_offset_fft(&scaled_and_accumulated, 1, Some(degree), offset).unwrap()
    }

    pub fn from_scs(scs: SquareConstraintSystem) -> Self {
        let number_of_constraints = scs.number_of_constraints().next_power_of_two();

        let u_polynomials = (0..scs.input_size())
            .map(|polynomial_index| {
                get_u_polynomial_from_scs(&scs, polynomial_index, number_of_constraints)
            })
            .collect();

        let matrix = scs.constraints;
        Self {
            number_of_public_inputs: scs.number_of_public_inputs,
            number_of_constraints,
            u_polynomials,
            matrix,
        }
    }

    pub fn number_of_private_inputs(&self) -> usize {
        self.u_polynomials.len() - self.number_of_public_inputs
    }

    pub fn check_valid(&self, input: &[FrElement]) -> bool {
        for row in &self.matrix {
            let coef = row
                .iter()
                .zip(input)
                .map(|(a, b)| a * b)
                .reduce(|a, b| a + b)
                .unwrap();

            if (&coef * &coef).ne(&FrElement::one()) {
                return false;
            }
        }
        true
    }
}

fn calculate_h_coefficients(
    u_evaluated: Vec<FrElement>,
    t_evaluated: Vec<FrElement>,
    t_inv_evaluated: Vec<FrElement>,
    degree: usize,
    offset: &FrElement,
    delta: &FrElement,
) -> Vec<FrElement> {
    let h_evaluated: Vec<_> = u_evaluated
        .iter()
        .zip(&t_evaluated)
        .zip(&t_inv_evaluated)
        .map(|((ui, ti), ti_inv)| {
            (ui * ui - FrElement::one()) * ti_inv
                + FrElement::from(2) * delta * ui
                + delta * delta * ti
        })
        .collect();

    let mut h_coefficients = Polynomial::interpolate_offset_fft(&h_evaluated, offset)
        .unwrap()
        .coefficients()
        .to_vec();

    let pad = vec![FrElement::zero(); degree - h_coefficients.len()];
    h_coefficients.extend(pad);

    h_coefficients
}

#[inline]
fn get_u_polynomial_from_scs(
    scs: &SquareConstraintSystem,
    polynomial_index: usize,
    number_of_constraints: usize,
) -> Polynomial<FrElement> {
    let mut u_polynomial = if polynomial_index == 0 {
        vec![FrElement::one(); number_of_constraints]
    } else {
        vec![FrElement::zero(); number_of_constraints]
    };

    for (constraint_index, constraint) in scs.constraints.iter().enumerate() {
        u_polynomial[constraint_index] = constraint[polynomial_index].clone();
    }

    Polynomial::interpolate_fft::<FrField>(&u_polynomial).unwrap()
}
