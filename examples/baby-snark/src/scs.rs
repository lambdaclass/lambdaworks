use crate::common::FrElement;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

// To be improved with a front-end implementation
// TODO: Use CS in Groth16 tests instead of a plain QAP
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintSystem<F: IsField> {
    pub constraints: SquareConstraintSystem,
    pub witness: Vec<FieldElement<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub u: Vec<FrElement>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SquareConstraintSystem {
    /// A constraint is a row in the matrix
    pub constraints: Vec<Constraint>,
    pub number_of_inputs: usize,
}

impl SquareConstraintSystem {
    pub fn from_matrices(u: Vec<Vec<FrElement>>, number_of_inputs: usize) -> Self {
        Self {
            constraints: (0..u.len())
                .map(|i| Constraint { u: u[i].clone() })
                .collect(),
            number_of_inputs,
        }
    }

    pub fn number_of_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// The size of a constraint represents the number of u polynomials
    pub fn witness_size(&self) -> usize {
        self.constraints[0].u.len()
    }
}
