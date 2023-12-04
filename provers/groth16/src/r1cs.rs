use crate::common::FrElement;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub a: Vec<FrElement>,
    pub b: Vec<FrElement>,
    pub c: Vec<FrElement>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1CS {
    pub constraints: Vec<Constraint>,
    pub number_of_inputs: usize,
    pub number_of_outputs: usize,
}

impl R1CS {
    pub fn from_matrices(
        a: Vec<Vec<FrElement>>,
        b: Vec<Vec<FrElement>>,
        c: Vec<Vec<FrElement>>,
        number_of_inputs: usize,
        number_of_outputs: usize,
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
            number_of_outputs,
        }
    }

    pub fn number_of_constraints(&self) -> usize {
        self.constraints.len()
    }

    pub fn witness_size(&self) -> usize {
        self.constraints[0].a.len()
    }
}
