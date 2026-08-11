//! Generic Rank-1 Constraint System (R1CS).
//!
//! Unlike the Groth16-specific R1CS in crates/provers/groth16/src/r1cs.rs,
//! this version is generic over any field F.

use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use crate::errors::SpartanError;
use crate::sparse_matrix::SparseMatrix;

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
    pub a: SparseMatrix<F>,
    /// Right input matrix (m × n)
    pub b: SparseMatrix<F>,
    /// Output matrix (m × n)
    pub c: SparseMatrix<F>,
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
    /// Creates a new R1CS from sparse matrices.
    pub fn new(
        a: SparseMatrix<F>,
        b: SparseMatrix<F>,
        c: SparseMatrix<F>,
        num_public_inputs: usize,
    ) -> Result<Self, SpartanError> {
        let num_constraints = a.num_rows;
        let num_variables = a.num_cols;

        if b.num_rows != num_constraints
            || c.num_rows != num_constraints
            || b.num_cols != num_variables
            || c.num_cols != num_variables
        {
            return Err(SpartanError::R1CSError(format!(
                "R1CS matrices have inconsistent dimensions: a={}x{}, b={}x{}, c={}x{}",
                a.num_rows, a.num_cols, b.num_rows, b.num_cols, c.num_rows, c.num_cols
            )));
        }
        if num_constraints == 0 {
            return Err(SpartanError::R1CSError(
                "R1CS must have at least one constraint".to_string(),
            ));
        }
        if num_variables == 0 {
            return Err(SpartanError::R1CSError(
                "R1CS must have at least one variable".to_string(),
            ));
        }
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

    /// Creates R1CS from dense matrices (backward compatibility).
    pub fn from_dense(
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
        let sa = SparseMatrix::from_dense(&a);
        let sb = SparseMatrix::from_dense(&b);
        let sc = SparseMatrix::from_dense(&c);
        Self::new(sa, sb, sc, num_public_inputs)
    }

    /// Checks whether the witness z satisfies the R1CS constraints.
    ///
    /// Returns true if (Az) ∘ (Bz) = Cz for all constraints.
    pub fn is_satisfied(&self, z: &[FieldElement<F>]) -> bool {
        if z.len() != self.num_variables {
            return false;
        }

        let mut az = vec![FieldElement::<F>::zero(); self.num_constraints];
        let mut bz = vec![FieldElement::<F>::zero(); self.num_constraints];
        let mut cz = vec![FieldElement::<F>::zero(); self.num_constraints];

        for e in &self.a.entries {
            az[e.row] += &e.val * &z[e.col];
        }
        for e in &self.b.entries {
            bz[e.row] += &e.val * &z[e.col];
        }
        for e in &self.c.entries {
            cz[e.row] += &e.val * &z[e.col];
        }
        (0..self.num_constraints).all(|i| &az[i] * &bz[i] == cz[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_matrix::SparseEntry;
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

        let r1cs = R1CS::from_dense(a, b, c, 1).unwrap();

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
        let result = R1CS::from_dense(a, b, c, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_r1cs_satisfied() {
        let one = FE::one();
        // x * y = z: A picks x (col 2), B picks y (col 3), C picks output (col 1)
        let a = SparseMatrix::new(
            vec![SparseEntry {
                row: 0,
                col: 2,
                val: one,
            }],
            1,
            4,
        )
        .unwrap();
        let b = SparseMatrix::new(
            vec![SparseEntry {
                row: 0,
                col: 3,
                val: one,
            }],
            1,
            4,
        )
        .unwrap();
        let c = SparseMatrix::new(
            vec![SparseEntry {
                row: 0,
                col: 1,
                val: one,
            }],
            1,
            4,
        )
        .unwrap();
        let r1cs = R1CS::new(a, b, c, 1).unwrap();
        let witness = vec![FE::one(), FE::from(9u64), FE::from(3u64), FE::from(3u64)];
        assert!(r1cs.is_satisfied(&witness));
    }
}
