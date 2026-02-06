use lambdaworks_math::polynomial::Polynomial;

use crate::{common::*, errors::Groth16Error, r1cs::R1CS};

/// Quadratic Arithmetic Program representation of a circuit.
///
/// A QAP encodes an R1CS constraint system as polynomials, enabling
/// efficient proof generation. Each variable in the circuit corresponds
/// to a polynomial in the L, R, and O vectors.
#[derive(Debug)]
pub struct QuadraticArithmeticProgram {
    /// Number of public inputs (first `num_of_public_inputs` variables are public)
    pub num_of_public_inputs: usize,
    /// Number of gates (constraints), padded to a power of two
    pub num_of_gates: usize,
    /// Left input polynomials, one per variable
    pub l: Vec<Polynomial<FrElement>>,
    /// Right input polynomials, one per variable
    pub r: Vec<Polynomial<FrElement>>,
    /// Output polynomials, one per variable
    pub o: Vec<Polynomial<FrElement>>,
}

impl QuadraticArithmeticProgram {
    /// Computes the coefficients of the quotient polynomial h(x).
    ///
    /// The quotient polynomial satisfies: L(x) * R(x) - O(x) = h(x) * t(x)
    /// where t(x) = x^n - 1 is the vanishing polynomial over the evaluation domain.
    ///
    /// # Arguments
    ///
    /// * `w` - The witness (all variable assignments including public inputs)
    ///
    /// # Returns
    ///
    /// The coefficients of h(x) on success.
    pub fn calculate_h_coefficients(
        &self,
        w: &[FrElement],
    ) -> Result<Vec<FrElement>, Groth16Error> {
        let offset = &ORDER_R_MINUS_1_ROOT_UNITY;
        let degree = self.num_of_gates * 2;

        let [l_evals, r_evals, o_evals] =
            self.scale_and_accumulate_variable_polynomials(w, degree, offset)?;

        // Compute t(x) = x^n - 1 inverse evaluations.
        //
        // Key insight: On a coset of size 2n with primitive root ω (where ω^(2n) = 1),
        // we have ω^n = -1. Therefore t(offset·ω^i) = offset^n·(-1)^i - 1, giving only
        // two distinct values that alternate:
        //   - Even indices: offset^n - 1
        //   - Odd indices:  -offset^n - 1
        //
        // This avoids an O(n log n) FFT and O(n) batch inverse, replacing them with
        // just one exponentiation and two field inversions.
        let offset_to_n = offset.pow(self.num_of_gates);
        let mut t_inv = [
            &offset_to_n - FrElement::one(),  // t_even = offset^n - 1
            -&offset_to_n - FrElement::one(), // t_odd = -offset^n - 1
        ];
        FrElement::inplace_batch_inverse(&mut t_inv)
            .map_err(|_| Groth16Error::BatchInversionFailed)?;
        let [t_even_inv, t_odd_inv] = t_inv;

        // Compute h(x) evaluations: (L * R - O) / t
        let h_evaluated: Vec<_> = l_evals
            .iter()
            .zip(&r_evals)
            .zip(&o_evals)
            .enumerate()
            .map(|(i, ((l, r), o))| {
                let t_inv = if i % 2 == 0 { &t_even_inv } else { &t_odd_inv };
                (l * r - o) * t_inv
            })
            .collect();

        Polynomial::interpolate_offset_fft(&h_evaluated, offset)
            .map_err(|e| Groth16Error::FFTError(format!("{:?}", e)))
            .map(|p| p.coefficients().to_vec())
    }

    /// Computes the evaluated polynomials L(x), R(x), O(x) with witness assignments.
    ///
    /// For each type (L, R, O), computes: sum_i w[i] * poly[i](x)
    /// evaluated at the offset coset of size `degree`.
    fn scale_and_accumulate_variable_polynomials(
        &self,
        w: &[FrElement],
        degree: usize,
        offset: &FrElement,
    ) -> Result<[Vec<FrElement>; 3], Groth16Error> {
        let compute_accumulated = |var_polynomials: &[Polynomial<FrElement>]| {
            let accumulated = var_polynomials
                .iter()
                .zip(w)
                .map(|(poly, coeff)| poly * coeff.clone())
                .reduce(|acc, p| acc + p)
                .ok_or_else(|| Groth16Error::QAPError("Empty polynomial list".into()))?;

            Polynomial::evaluate_offset_fft(&accumulated, 1, Some(degree), offset)
                .map_err(|e| Groth16Error::FFTError(format!("{:?}", e)))
        };

        Ok([
            compute_accumulated(&self.l)?,
            compute_accumulated(&self.r)?,
            compute_accumulated(&self.o)?,
        ])
    }

    /// Returns the number of private inputs (witness variables excluding public inputs).
    #[inline]
    pub fn num_of_private_inputs(&self) -> usize {
        self.l.len() - self.num_of_public_inputs
    }

    /// Creates a QAP from an R1CS constraint system.
    ///
    /// # Arguments
    ///
    /// * `r1cs` - The R1CS constraint system
    ///
    /// # Returns
    ///
    /// A QAP representing the same constraints, with polynomials interpolated
    /// over a domain of size equal to the next power of two of the constraint count.
    pub fn from_r1cs(r1cs: R1CS) -> Result<QuadraticArithmeticProgram, Groth16Error> {
        let num_gates = r1cs.number_of_constraints();
        let next_power_of_two = num_gates.next_power_of_two();
        let pad_zeroes = next_power_of_two - num_gates;

        let mut l = Vec::with_capacity(r1cs.witness_size());
        let mut r = Vec::with_capacity(r1cs.witness_size());
        let mut o = Vec::with_capacity(r1cs.witness_size());

        for i in 0..r1cs.witness_size() {
            let [l_poly, r_poly, o_poly] =
                get_variable_lro_polynomials_from_r1cs(&r1cs, i, pad_zeroes)?;
            l.push(l_poly);
            r.push(r_poly);
            o.push(o_poly);
        }

        Ok(QuadraticArithmeticProgram {
            l,
            r,
            o,
            num_of_gates: next_power_of_two,
            num_of_public_inputs: r1cs.number_of_inputs,
        })
    }

    /// Creates a QAP from variable matrices.
    ///
    /// # Arguments
    ///
    /// * `num_of_public_inputs` - Number of public input variables
    /// * `l` - Left input matrix (variables × constraints)
    /// * `r` - Right input matrix (variables × constraints)
    /// * `o` - Output matrix (variables × constraints)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The matrices are empty
    /// - The matrices have inconsistent outer dimensions
    /// - Inner vectors (rows) have inconsistent lengths
    /// - `num_of_public_inputs` exceeds the number of variables
    pub fn from_variable_matrices(
        num_of_public_inputs: usize,
        l: &[Vec<FrElement>],
        r: &[Vec<FrElement>],
        o: &[Vec<FrElement>],
    ) -> Result<QuadraticArithmeticProgram, Groth16Error> {
        let num_of_vars = l.len();
        if num_of_vars == 0 {
            return Err(Groth16Error::QAPError("Empty variable matrices".into()));
        }
        if num_of_vars != r.len() || num_of_vars != o.len() {
            return Err(Groth16Error::QAPError(
                "Variable matrices have inconsistent sizes".into(),
            ));
        }
        if num_of_public_inputs > num_of_vars {
            return Err(Groth16Error::QAPError(
                "More public inputs than variables".into(),
            ));
        }

        let num_of_gates = l[0].len();

        // Validate all rows have the same length
        let validate_inner_lengths = |matrix: &[Vec<FrElement>], name: &str| {
            for (i, row) in matrix.iter().enumerate() {
                if row.len() != num_of_gates {
                    return Err(Groth16Error::QAPError(format!(
                        "Inconsistent constraint count in {} matrix: row {} has {} constraints, expected {}",
                        name, i, row.len(), num_of_gates
                    )));
                }
            }
            Ok(())
        };

        validate_inner_lengths(l, "L")?;
        validate_inner_lengths(r, "R")?;
        validate_inner_lengths(o, "O")?;

        let next_power_of_two = num_of_gates.next_power_of_two();
        let pad_zeroes = next_power_of_two - num_of_gates;

        Ok(QuadraticArithmeticProgram {
            num_of_public_inputs,
            num_of_gates: next_power_of_two,
            l: build_variable_polynomials(l, pad_zeroes)?,
            r: build_variable_polynomials(r, pad_zeroes)?,
            o: build_variable_polynomials(o, pad_zeroes)?,
        })
    }
}

/// Extracts L, R, O polynomials for a single variable from R1CS.
#[inline]
fn get_variable_lro_polynomials_from_r1cs(
    r1cs: &R1CS,
    var_idx: usize,
    pad_zeroes: usize,
) -> Result<[Polynomial<FrElement>; 3], Groth16Error> {
    let cap = r1cs.number_of_constraints() + pad_zeroes;
    let mut current_var_l = vec![FrElement::zero(); cap];
    let mut current_var_r = vec![FrElement::zero(); cap];
    let mut current_var_o = vec![FrElement::zero(); cap];

    for (i, c) in r1cs.constraints.iter().enumerate() {
        current_var_l[i].clone_from(&c.a[var_idx]);
        current_var_r[i].clone_from(&c.b[var_idx]);
        current_var_o[i].clone_from(&c.c[var_idx]);
    }

    let interpolate = |evals: Vec<FrElement>| {
        Polynomial::interpolate_fft::<FrField>(&evals)
            .map_err(|e| Groth16Error::FFTError(format!("{:?}", e)))
    };

    Ok([
        interpolate(current_var_l)?,
        interpolate(current_var_r)?,
        interpolate(current_var_o)?,
    ])
}

/// Builds polynomials from matrix rows, applying padding for FFT.
#[inline]
fn build_variable_polynomials(
    from_matrix: &[Vec<FrElement>],
    pad_zeroes: usize,
) -> Result<Vec<Polynomial<FrElement>>, Groth16Error> {
    from_matrix
        .iter()
        .map(|row| {
            let mut padded = row.clone();
            padded.resize(row.len() + pad_zeroes, FrElement::zero());
            Polynomial::interpolate_fft::<FrField>(&padded)
                .map_err(|e| Groth16Error::FFTError(format!("{:?}", e)))
        })
        .collect()
}
