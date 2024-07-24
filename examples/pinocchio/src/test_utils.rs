use super::common::FE;
use crate::{
    qap::QuadraticArithmeticProgram as QAP,
    r1cs::{Constraint, R1CS},
};
use lambdaworks_math::polynomial::Polynomial;

// r5 and r6 are exposed to help testing
pub fn test_qap_r5() -> FE {
    FE::from(0)
}

pub fn test_qap_r6() -> FE {
    FE::from(1)
}

/// This is a solver for the test qap
/// Inputs: c1,c2,c3,c4 circuit inputs
/// Outputs: c5 intermediate result, c6 result
pub fn test_qap_solver(inputs: [FE; 4]) -> (FE, FE) {
    let c5 = &inputs[2] * &inputs[3];
    let c6 = (&inputs[0] + &inputs[1]) * c5.clone();
    (c5, c6)
}

/// Test qap based on pinocchios paper example.
pub fn new_test_qap() -> QAP {
    let r5: FE = test_qap_r5();
    let r6: FE = test_qap_r6();

    let t: Polynomial<FE> =
        Polynomial::new(&[-&r5, FE::from(1)]) * Polynomial::new(&[-&r6, FE::from(1)]);

    let vs: Vec<Polynomial<FE>> = vec![
        //v0
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        //v1
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(1)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(1)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(1), FE::from(0)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
    ];

    let ws: Vec<Polynomial<FE>> = vec![
        //w0
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        //w1
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(1), FE::from(0)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(1)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
    ];

    let ys: Vec<Polynomial<FE>> = vec![
        //y0
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        //y1
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(0), FE::from(0)]).unwrap(),
        Polynomial::interpolate(&[r5.clone(), r6.clone()], &[FE::from(1), FE::from(0)]).unwrap(),
        Polynomial::interpolate(
            &[r5.clone(), r6.clone().clone()],
            &[FE::from(0), FE::from(1)],
        )
        .unwrap(),
    ];

    QAP::new(vs, ws, ys, t, 4, 1).unwrap()
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
