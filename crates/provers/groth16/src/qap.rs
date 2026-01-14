use lambdaworks_math::polynomial::Polynomial;

use crate::{common::*, errors::Groth16Error, r1cs::R1CS};

#[derive(Debug)]
pub struct QuadraticArithmeticProgram {
    pub num_of_public_inputs: usize,
    pub num_of_gates: usize,
    pub l: Vec<Polynomial<FrElement>>,
    pub r: Vec<Polynomial<FrElement>>,
    pub o: Vec<Polynomial<FrElement>>,
}

impl QuadraticArithmeticProgram {
    /// Computes the quotient polynomial
    pub fn calculate_h_coefficients(&self, w: &[FrElement]) -> Result<Vec<FrElement>, Groth16Error> {
        let offset = &ORDER_R_MINUS_1_ROOT_UNITY;
        let degree = self.num_of_gates * 2;

        let [l, r, o] = self.scale_and_accumulate_variable_polynomials(w, degree, offset)?;

        // TODO: Change to a vector of offsetted evaluations of x^N-1
        let t_poly =
            Polynomial::new_monomial(FrElement::one(), self.num_of_gates) - FrElement::one();
        let mut t = Polynomial::evaluate_offset_fft(&t_poly, 1, Some(degree), offset)
            .map_err(|e| Groth16Error::FFTError(format!("{:?}", e)))?;
        FrElement::inplace_batch_inverse(&mut t)
            .map_err(|_| Groth16Error::BatchInversionFailed)?;

        let h_evaluated = l
            .iter()
            .zip(&r)
            .zip(&o)
            .zip(&t)
            .map(|(((l, r), o), t)| (l * r - o) * t)
            .collect::<Vec<_>>();

        Ok(Polynomial::interpolate_offset_fft(&h_evaluated, offset)
            .map_err(|e| Groth16Error::FFTError(format!("{:?}", e)))?
            .coefficients()
            .to_vec())
    }

    // Compute A.s by summing up polynomials A[0].s, A[1].s, ..., A[n].s
    // In other words, assign the witness coefficients / execution values
    // Similarly for B.s and C.s
    fn scale_and_accumulate_variable_polynomials(
        &self,
        w: &[FrElement],
        degree: usize,
        offset: &FrElement,
    ) -> Result<[Vec<FrElement>; 3], Groth16Error> {
        let mut results = Vec::with_capacity(3);
        for var_polynomials in [&self.l, &self.r, &self.o] {
            let accumulated = var_polynomials
                .iter()
                .zip(w)
                .map(|(poly, coeff)| poly.mul_with_ref(&Polynomial::new_monomial(coeff.clone(), 0)))
                .reduce(|poly1, poly2| poly1 + poly2)
                .ok_or_else(|| Groth16Error::QAPError("Empty polynomial list".to_string()))?;
            let evaluated = Polynomial::evaluate_offset_fft(&accumulated, 1, Some(degree), offset)
                .map_err(|e| Groth16Error::FFTError(format!("{:?}", e)))?;
            results.push(evaluated);
        }
        Ok([
            results.remove(0),
            results.remove(0),
            results.remove(0),
        ])
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

    pub fn from_variable_matrices(
        num_of_public_inputs: usize,
        l: &[Vec<FrElement>],
        r: &[Vec<FrElement>],
        o: &[Vec<FrElement>],
    ) -> QuadraticArithmeticProgram {
        let num_of_vars = l.len();
        assert!(num_of_vars > 0);
        assert_eq!(num_of_vars, r.len());
        assert_eq!(num_of_vars, o.len());
        assert!(num_of_public_inputs <= num_of_vars);

        let num_of_gates = l[0].len();
        let next_power_of_two = num_of_gates.next_power_of_two();
        let pad_zeroes = next_power_of_two - num_of_gates;

        QuadraticArithmeticProgram {
            num_of_public_inputs,
            num_of_gates: next_power_of_two,
            l: build_variable_polynomials(&apply_padding(l, pad_zeroes)),
            r: build_variable_polynomials(&apply_padding(r, pad_zeroes)),
            o: build_variable_polynomials(&apply_padding(o, pad_zeroes)),
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

#[inline]
fn build_variable_polynomials(from_matrix: &[Vec<FrElement>]) -> Vec<Polynomial<FrElement>> {
    from_matrix
        .iter()
        .map(|row| Polynomial::interpolate_fft::<FrField>(row).unwrap())
        .collect()
}

/// Pads the columns so that the length is a multiple of 2 and we can use radix-2 FFT
#[inline]
fn apply_padding(columns: &[Vec<FrElement>], pad_zeroes: usize) -> Vec<Vec<FrElement>> {
    columns
        .iter()
        .map(|column| {
            let mut new_column = column.clone();
            new_column.extend(vec![FrElement::zero(); pad_zeroes]);
            new_column
        })
        .collect()
}
