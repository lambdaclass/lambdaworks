use super::common::FE;
//use crate::common::FE;

use crate::{
    qap::QuadraticArithmeticProgram as QAP,
    r1cs::{Constraint, R1CS},
};

pub fn test_qap_solver(inputs: [FE; 4]) -> (FE, FE) {
    let c5 = &inputs[2] * &inputs[3];
    let c6 = (&inputs[0] + &inputs[1]) * c5.clone();
    (c5, c6)
}

pub fn new_test_r1cs() -> R1CS {
    let constraints = vec![new_test_first_constraint(), new_test_second_constraint()];
    R1CS::new(constraints, 4, 1).unwrap()
}

pub fn new_test_first_constraint() -> Constraint {
    Constraint {
        a: vec![
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(1),
            FE::from(0),
            FE::from(0),
            FE::from(0),
        ],
        b: vec![
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(1),
            FE::from(0),
            FE::from(0),
        ],
        c: vec![
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(1),
            FE::from(0),
        ],
    }
}

pub fn new_test_second_constraint() -> Constraint {
    Constraint {
        a: vec![
            FE::from(0),
            FE::from(1),
            FE::from(1),
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(0),
        ],
        b: vec![
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(1),
            FE::from(0),
        ],
        c: vec![
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(0),
            FE::from(1),
        ],
    }
}