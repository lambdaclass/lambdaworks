//! Generic Rank-1 Constraint System (R1CS).
//!
//! Unlike the Groth16-specific R1CS in crates/provers/groth16/src/r1cs.rs,
//! this version is generic over any field F.

use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use crate::errors::SpartanError;

/// A Rank-1 Constraint System over a generic field F.
///
/// An R1CS instance consists of matrices A, B, C ∈ F^{m×n} and a witness z ∈ F^n
/// such that (Az) ∘ (Bz) = Cz (element-wise product of the matrix-vector products).
///
/// The witness z = (1, x, w) where:
/// - 1 is the constant (always the first element)
/// - x are the public inputs
/// - w are the private witness values
#[derive(Clone, Debug)]
pub struct R1CS<F: IsField>
where
    F::BaseType: Send + Sync,
{
    /// Left input matrix (m × n, row = constraint, col = variable)
    pub a: Vec<Vec<FieldElement<F>>>,
    /// Right input matrix (m × n)
    pub b: Vec<Vec<FieldElement<F>>>,
    /// Output matrix (m × n)
    pub c: Vec<Vec<FieldElement<F>>>,
    /// Number of constraints (m)
    pub num_constraints: usize,
    /// Number of variables including the constant 1 (n)
    pub num_variables: usize,
    /// Number of public inputs (x in the witness z = (1, x, w))
    pub num_public_inputs: usize,
}

impl<F: IsField> R1CS<F>
where
    F::BaseType: Send + Sync,
{
    /// Creates a new R1CS from matrices A, B, C and the number of public inputs.
    ///
    /// Validates that all three matrices have the same dimensions.
    pub fn new(
        a: Vec<Vec<FieldElement<F>>>,
        b: Vec<Vec<FieldElement<F>>>,
        c: Vec<Vec<FieldElement<F>>>,
        num_public_inputs: usize,
    ) -> Result<Self, SpartanError> {
        let num_constraints = a.len();
        if b.len() != num_constraints || c.len() != num_constraints {
            return Err(SpartanError::R1CSError(format!(
                "R1CS matrices have inconsistent row counts: a={}, b={}, c={}",
                a.len(),
                b.len(),
                c.len()
            )));
        }

        if num_constraints == 0 {
            return Err(SpartanError::R1CSError(
                "R1CS must have at least one constraint".to_string(),
            ));
        }

        let num_variables = a[0].len();
        if num_variables == 0 {
            return Err(SpartanError::R1CSError(
                "R1CS must have at least one variable".to_string(),
            ));
        }

        for (i, (row_a, (row_b, row_c))) in a.iter().zip(b.iter().zip(c.iter())).enumerate() {
            if row_a.len() != num_variables
                || row_b.len() != num_variables
                || row_c.len() != num_variables
            {
                return Err(SpartanError::R1CSError(format!(
                    "R1CS row {} has inconsistent column counts: a={}, b={}, c={}",
                    i,
                    row_a.len(),
                    row_b.len(),
                    row_c.len()
                )));
            }
        }

        // z = (1, x, w): the constant 1 always occupies variable 0, so public inputs
        // can use at most num_variables - 1 slots.
        if num_public_inputs >= num_variables {
            return Err(SpartanError::R1CSError(format!(
                "num_public_inputs ({num_public_inputs}) must be less than num_variables \
                 ({num_variables}): variable 0 is reserved for the constant 1 in z = (1, x, w)"
            )));
        }

        Ok(Self {
            a,
            b,
            c,
            num_constraints,
            num_variables,
            num_public_inputs,
        })
    }

    /// Checks whether the witness z satisfies the R1CS constraints.
    ///
    /// Returns true if (Az) ∘ (Bz) = Cz for all constraints.
    pub fn is_satisfied(&self, z: &[FieldElement<F>]) -> bool {
        if z.len() != self.num_variables {
            return false;
        }

        for i in 0..self.num_constraints {
            // Compute <A[i], z>
            let az_i: FieldElement<F> = self.a[i]
                .iter()
                .zip(z.iter())
                .map(|(a_ij, z_j)| a_ij * z_j)
                .fold(FieldElement::zero(), |acc, x| acc + x);

            // Compute <B[i], z>
            let bz_i: FieldElement<F> = self.b[i]
                .iter()
                .zip(z.iter())
                .map(|(b_ij, z_j)| b_ij * z_j)
                .fold(FieldElement::zero(), |acc, x| acc + x);

            // Compute <C[i], z>
            let cz_i: FieldElement<F> = self.c[i]
                .iter()
                .zip(z.iter())
                .map(|(c_ij, z_j)| c_ij * z_j)
                .fold(FieldElement::zero(), |acc, x| acc + x);

            if az_i * bz_i != cz_i {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    /// Build a simple circuit: x * y = z
    /// witness z = [1, 9, 3, 3]  (constant=1, output=9, x=3, y=3)
    /// A[0] = [0, 0, 1, 0]  (picks x=3)
    /// B[0] = [0, 0, 0, 1]  (picks y=3)
    /// C[0] = [0, 1, 0, 0]  (picks output=9)
    fn multiplication_r1cs() -> (R1CS<F>, Vec<FE>) {
        let zero = FE::zero();
        let one = FE::one();

        let a = vec![vec![zero, zero, one, zero]];
        let b = vec![vec![zero, zero, zero, one]];
        let c = vec![vec![zero, one, zero, zero]];

        let r1cs = R1CS::new(a, b, c, 1).unwrap();

        // witness: [1, 9, 3, 3]
        let witness = vec![FE::one(), FE::from(9u64), FE::from(3u64), FE::from(3u64)];

        (r1cs, witness)
    }

    #[test]
    fn test_r1cs_satisfied() {
        let (r1cs, witness) = multiplication_r1cs();
        assert!(r1cs.is_satisfied(&witness));
    }

    #[test]
    fn test_r1cs_not_satisfied_wrong_witness() {
        let (r1cs, mut witness) = multiplication_r1cs();
        // Set output to wrong value
        witness[1] = FE::from(7u64);
        assert!(!r1cs.is_satisfied(&witness));
    }

    #[test]
    fn test_r1cs_wrong_witness_length() {
        let (r1cs, _) = multiplication_r1cs();
        let short_witness = vec![FE::one(), FE::from(2u64)];
        assert!(!r1cs.is_satisfied(&short_witness));
    }

    #[test]
    fn test_r1cs_new_dimension_mismatch() {
        let zero = FE::zero();
        let one = FE::one();

        let a = vec![vec![zero, one]];
        let b = vec![vec![zero]]; // wrong column count
        let c = vec![vec![zero, one]];

        // This should fail because row 0 of b has wrong length
        let result = R1CS::new(a, b, c, 0);
        assert!(result.is_err());
    }
}
