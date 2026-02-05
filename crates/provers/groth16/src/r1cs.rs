//! Rank-1 Constraint System (R1CS) representation.
//!
//! R1CS is the standard format for expressing arithmetic circuits in zkSNARKs.
//! Each constraint has the form: `<A, w> * <B, w> = <C, w>` where `w` is the witness.

use crate::common::FrElement;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

/// A constraint system combining R1CS constraints with a witness.
///
/// This is a basic front-end representation that pairs constraints with their
/// satisfying witness assignment.
// TODO: Use ConstraintSystem in Groth16 tests instead of plain QAP for better ergonomics
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintSystem<F: IsField> {
    /// The R1CS constraint matrices
    pub constraints: R1CS,
    /// The witness assignment satisfying the constraints
    pub witness: Vec<FieldElement<F>>,
}

/// A single R1CS constraint: `<a, w> * <b, w> = <c, w>`.
///
/// The constraint is satisfied when the inner product of `a` with the witness,
/// multiplied by the inner product of `b` with the witness, equals the inner
/// product of `c` with the witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    /// Left input coefficients
    pub a: Vec<FrElement>,
    /// Right input coefficients
    pub b: Vec<FrElement>,
    /// Output coefficients
    pub c: Vec<FrElement>,
}

/// Rank-1 Constraint System.
///
/// An R1CS consists of a set of constraints of the form `A·w ∘ B·w = C·w`
/// where `w` is the witness vector and `∘` denotes element-wise multiplication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1CS {
    /// The list of constraints
    pub constraints: Vec<Constraint>,
    /// Number of public inputs (first elements of witness after the constant 1)
    pub number_of_inputs: usize,
}

impl R1CS {
    /// Creates an R1CS from the A, B, C matrices.
    ///
    /// # Arguments
    ///
    /// * `a` - Left input matrix (rows are constraints, columns are variables)
    /// * `b` - Right input matrix
    /// * `c` - Output matrix
    /// * `number_of_inputs` - Number of public inputs
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different numbers of rows.
    pub fn from_matrices(
        a: Vec<Vec<FrElement>>,
        b: Vec<Vec<FrElement>>,
        c: Vec<Vec<FrElement>>,
        number_of_inputs: usize,
    ) -> Self {
        Self {
            constraints: (0..a.len())
                .map(|i| Constraint {
                    a: a[i].clone(),
                    b: b[i].clone(),
                    c: c[i].clone(),
                })
                .collect(),
            number_of_inputs,
        }
    }

    /// Returns the number of constraints in the system.
    pub fn number_of_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Returns the size of the witness vector (number of variables).
    ///
    /// # Panics
    ///
    /// Panics if the constraint system is empty.
    pub fn witness_size(&self) -> usize {
        self.constraints[0].a.len()
    }
}
