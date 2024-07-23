use lambdaworks_math::{
    cyclic_group::IsGroup, msm::pippenger::msm,
    elliptic_curve::{
        traits::{IsEllipticCurve, IsPairing},
    }
};
//use pinocchio::test_utils::{new_test_r1cs, execute_qap};
use crate::common::{
        sample_fr_elem, Curve, G1Point, G2Point, Pairing, TwistedCurve, FE};

use crate::qap::QuadraticArithmeticProgram;

use crate::prover::Proof;
use crate::setup::VerificationKey;

pub fn verify(verification_key:&VerificationKey,
    proof:&Proof,
    c_inputs_outputs:&[FE]
) -> bool {
    let b1 = check_divisibility(verification_key, proof, c_inputs_outputs);
    let b2 = check_appropriate_spans(verification_key, proof);
    let b3 = check_same_linear_combinations(verification_key, proof);
    b1 && b2 && b3

}

pub fn check_divisibility(
    verification_key: &VerificationKey,
    proof: &Proof,
    c_io: &[FE],
) -> bool {
    // We transform c_io into UnsignedIntegers.
    let c_io = c_io
    .iter()
    .map(|elem| elem.representative())
    .collect::<Vec<_>>();

    let v_io = verification_key.g1_vk[0]
        .operate_with(&msm(&c_io, &verification_key.g1_vk[1..]).unwrap());
    let w_io = verification_key.g2_wk[0]
        .operate_with(&msm(&c_io, &verification_key.g2_wk[1..]).unwrap());
    let y_io = verification_key.g1_yk[0]
        .operate_with(&msm(&c_io, &verification_key.g1_yk[1..]).unwrap());
    Pairing::compute(
        &v_io.operate_with(&proof.v),
        &w_io.operate_with(&proof.w2)
    ).unwrap()
    == Pairing::compute(
        &verification_key.g1y_t,
        &proof.h
    ).unwrap()
    * Pairing::compute(
        &y_io.operate_with(&proof.y),
        &verification_key.g2
    ).unwrap()
}

// We check that v (from the proof) is indeed g multpiplied by a linear combination of the g1_vk.
// The same with w and y.
pub fn check_appropriate_spans(
    verification_key: &VerificationKey,
    proof: &Proof
) -> bool {
    let b1 = Pairing::compute(&proof.v_prime, &verification_key.g2) 
        == Pairing::compute(&proof.v, &verification_key.g2_alpha_v);
    let b2 = Pairing::compute(&proof.w_prime, &verification_key.g2) 
        == Pairing::compute(&proof.w1, &verification_key.g2_alpha_w);
    let b3 = Pairing::compute(&proof.y_prime, &verification_key.g2) 
        == Pairing::compute(&proof.y, &verification_key.g2_alpha_y);
    b1 && b2 && b3
}

// We check that the same coefficients were used for the linear combination of v, w and y.
pub fn check_same_linear_combinations(
    verification_key: &VerificationKey,
    proof: &Proof
) -> bool {
    Pairing::compute(&proof.z, &verification_key.g2_gamma)
    == Pairing::compute(
        &proof.v
            .operate_with(&proof.w1)
            .operate_with(&proof.y),
        &verification_key.g2_beta_gamma
    )
}

//