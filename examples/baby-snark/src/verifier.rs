use std::ops::Mul;

use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::{elliptic_curve::traits::IsPairing, msm::pippenger::msm};

use crate::common::{Curve, FrElement, Pairing, TwistedCurve};
use crate::prover::Proof;
use crate::setup::VerifyingKey;

pub fn verify(vk: &VerifyingKey, proof: &Proof, pub_inputs: &[FrElement]) -> bool {
    let v_w = &proof.v_w;
    let v_w_prime = &proof.v_w_prime;
    let h = &proof.h;
    let b_w = &proof.b_w;

    if Pairing::compute(b_w, &vk.gamma_g2) != Pairing::compute(&vk.beta_gamma_g1, v_w_prime) {
        return false;
    }
    if Pairing::compute(v_w, &TwistedCurve::generator())
        != Pairing::compute(&Curve::generator(), v_w_prime)
    {
        return false;
    }

    let canonical_inputs: Vec<_> = pub_inputs.iter().map(|elem| elem.canonical()).collect();
    let v_u = msm(&canonical_inputs, &vk.u_tau_g1).unwrap();
    let v_u_prime = msm(&canonical_inputs, &vk.u_tau_g2).unwrap();

    Pairing::compute(&v_u.operate_with(v_w), &v_u_prime.operate_with(v_w_prime))
        .unwrap()
        .mul(&vk.inv_pairing_g1_g2)
        == Pairing::compute(h, &vk.t_tau_g2).unwrap()
}
