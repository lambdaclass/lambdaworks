use crate::common::FrElement;

pub type Constraint = Vec<FrElement>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SquareConstraintSystem {
    pub constraints: Vec<Constraint>,
    pub number_of_public_inputs: usize,
}

impl SquareConstraintSystem {
    pub fn from_matrix(matrix: Vec<Vec<FrElement>>, number_of_public_inputs: usize) -> Self {
        Self {
            constraints: matrix,
            number_of_public_inputs,
        }
    }

    pub fn number_of_constraints(&self) -> usize {
        self.constraints.len()
    }

    pub fn input_size(&self) -> usize {
        self.constraints[0].len()
    }
}
