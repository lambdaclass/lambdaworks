use std::ops::Neg;

use baby_snark::{
    self, common::FrElement, scs::SquareConstraintSystem, setup, ssp::SquareSpanProgram, verify,
    Prover,
};
use lambdaworks_math::unsigned_integer;
#[test]
fn test_simple_circuit() {
    let mut u = vec![vec![FrElement::zero(); 4]; 4];
    u[0][0] = FrElement::from(1).neg();
    u[1][0] = FrElement::from(1).neg();
    u[2][0] = FrElement::from(1).neg();
    u[3][0] = FrElement::from(4);
    u[0][1] = FrElement::from(2);
    u[1][2] = FrElement::from(2);
    u[3][2] = FrElement::from(2);
    u[2][3] = FrElement::from(2);
    u[3][3] = FrElement::from(5).neg();

    let ssp = SquareSpanProgram::from_scs(SquareConstraintSystem::from_matrices(u, 1));
    let (pk, vk) = setup(&ssp);
    let proof = Prover::prove(
        &[FrElement::one(), FrElement::one(), FrElement::zero()],
        &ssp,
        &pk,
    );
    assert!(verify(&vk, &proof, &[FrElement::one()]));
}
