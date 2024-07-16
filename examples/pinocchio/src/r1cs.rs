use crate::common::FE;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

#[derive(Debug, PartialEq, Eq)]
pub enum CreationError {
    VectorsSizeMismatch,
    MatrixesSizeMismatch,
    /// Number of IOs should be less than witness size - 1
    InputOutputTooBig,
}

/// R1CS representation of an Arithmetic Program
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub a: Vec<FE>,
    pub b: Vec<FE>,
    pub c: Vec<FE>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
/// R1CS represented as a vector of constraints/gates
/// Noticing joining all the first vectors of the constraints results
/// in the first Matrix of the R1CS
/// joining the second vectors results on the second matrix, and so on
///
pub struct R1CS {
    pub constraints: Vec<Constraint>,
    // These are not strictly part of a R1CS
    // But they are data of the constraint system
    // that is needed to generate the proof
    pub number_of_inputs: usize,
    pub number_of_outputs: usize,
}

impl R1CS {
    #[allow(dead_code)]
    pub fn new(
        constraints: Vec<Constraint>,
        number_of_inputs: usize,
        number_of_outputs: usize,
    ) -> Result<Self, CreationError> {
        let witness_size = constraints[0].a.len();
        // Each constraints already has the correct dimensionality
        // a check is needed before creatin the r1cs
        // to be sure all constraints have the same dimension
        let all_same_length = constraints
            .iter()
            .all(|v| v.a.len() == constraints[0].a.len());

        if !all_same_length {
            Err(CreationError::VectorsSizeMismatch)
        } else if number_of_inputs + number_of_outputs > witness_size - 1 {
            Err(CreationError::InputOutputTooBig)
        } else {
            Ok(Self {
                constraints,
                number_of_inputs,
                number_of_outputs,
            })
        }
    }

    pub fn new_with_matrixes(
        a: Vec<Vec<FE>>,
        b: Vec<Vec<FE>>,
        c: Vec<Vec<FE>>,
        num_inputs: usize,
        num_outputs: usize,
    ) -> Result<Self, CreationError> {
        let mut constraints: Vec<Constraint> = Vec::with_capacity(a.len());
        // TO DO:
        // - Check if sizes match
        // - Remove clones
        for i in 0..a.len() {
            constraints.push(Constraint::new(a[i].clone(), b[i].clone(), c[i].clone()).unwrap())
        }
        R1CS::new(constraints, num_inputs, num_outputs)
    }

    #[allow(dead_code)]
    pub fn verify_solution(self, s: &[FE]) -> bool {
        for constraint in self.constraints {
            if !constraint.verify_solution(s) {
                return false;
            }
        }
        true
    }

    pub fn number_of_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Returns the size of the witness
    /// This is the constant part, plus the of inputs + intermediate values +
    /// outputs
    pub fn witness_size(&self) -> usize {
        // all constraints have the same size
        // this is enforced in the creation
        self.constraints[0].a.len() //gives the number of columns?
    }
}

impl Constraint {
    /// Creates a new constraint for a,b,c vectors
    #[allow(dead_code)]
    pub fn new(a: Vec<FE>, b: Vec<FE>, c: Vec<FE>) -> Result<Self, CreationError> {
        if a.len() != b.len() || a.len() != c.len() || b.len() != c.len() {
            Err(CreationError::VectorsSizeMismatch)
        } else {
            Ok(Self { a, b, c })
        }
    }

    #[allow(dead_code)]
    pub fn verify_solution(self, s: &[FE]) -> bool {
        inner_product(&self.a, s) * inner_product(&self.b, s) == inner_product(&self.c, s)
    }
}

pub fn inner_product(v1: &[FE], v2: &[FE]) -> FE {
    v1.iter()
        .zip(v2)
        .map(|(x, y)| x * y)
        .fold(FE::from(0), |x, y| x + y)
}

// v1.iter().zip(v2).map(|(x, y)| *x * *y).fold(FE::new(0), |x,y| x + y)

/*#[cfg(test)]
pub mod tests {
    use crate::circuits::test_utils::{
        new_test_first_constraint, new_test_r1cs, new_test_second_constraint,
    };

    use super::*;

    #[test]
    fn mul_vectors_2_2_3_3_equals_12() {
        let v1 = &[FE::new(2), FE::new(2)];
        let v2 = &[FE::new(3), FE::new(3)];

        assert_eq!(inner_product(v1, v2), FE::new(12));
    }

    #[test]
    fn mul_vectors_3_5_equals_15() {
        let v1 = &[FE::new(3)];
        let v2 = &[FE::new(5)];

        assert_eq!(inner_product(v1, v2), FE::new(15));
    }

    #[test]
    fn verify_solution_with_test_circuit_c5_constraints() {
        assert!(new_test_second_constraint().verify_solution(&test_solution()));
    }

    #[test]
    fn verify_solution_with_test_circuit_c6_constraints() {
        assert!(new_test_second_constraint().verify_solution(&test_solution()));
    }

    #[test]
    fn verify_bad_solution_with_test_circuit_c5_constraints() {
        let solution = vec![
            FE::new(0),
            FE::new(0),
            FE::new(0),
            FE::new(4),
            FE::new(1),
            FE::new(0),
            FE::new(0),
        ];
        assert!(!new_test_first_constraint().verify_solution(&solution));
    }

    #[test]
    fn verify_bad_solution_with_test_circuit_c6_constraints() {
        let solution = vec![
            FE::new(0),
            FE::new(2),
            FE::new(1),
            FE::new(4),
            FE::new(5),
            FE::new(2),
            FE::new(2),
        ];
        assert!(!new_test_second_constraint().verify_solution(&solution));
    }

    #[test]
    fn verify_solution_with_new_test_r1cs() {
        assert!(new_test_r1cs().verify_solution(&test_solution()))
    }

    #[test]
    fn verify_bad_solution_with_new_test_r1cs() {
        let solution = vec![
            FE::new(0),
            FE::new(2),
            FE::new(1),
            FE::new(4),
            FE::new(5),
            FE::new(2),
            FE::new(2),
        ];

        assert!(!new_test_r1cs().verify_solution(&solution))
    }

    #[test]
    fn verify_bad_solution_because_of_second_constraint_with_new_test_r1cs() {
        let solution = vec![
            FE::new(0),  // c0
            FE::new(2),  // c1
            FE::new(1),  // c2
            FE::new(5),  // c3
            FE::new(10), // c4
            FE::new(50), // c5 = c4 * c3
            FE::new(2),  // c6 != c5 * (c1+c2), so this should fail
        ];
        assert!(!new_test_r1cs().verify_solution(&solution))
    }

    #[test]
    fn verify_bad_solution_because_of_first_constraint_with_new_test_r1cs() {
        let solution = vec![
            FE::new(0),  // c0
            FE::new(1),  // c1
            FE::new(1),  // c2
            FE::new(5),  // c3
            FE::new(10), // c4
            FE::new(10), // c5 != c4 * c3
            FE::new(19), // c6 = c5 * (c1+c2), so this should fail
        ];
        assert!(!new_test_r1cs().verify_solution(&solution))
    }

    fn test_solution() -> Vec<FE> {
        vec![
            FE::new(0),
            FE::new(1),
            FE::new(2),
            FE::new(3),
            FE::new(4),
            FE::new(12),
            FE::new(36),
        ]
    }
}
*/