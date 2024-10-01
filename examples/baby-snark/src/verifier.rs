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

    let mut accept = true;
    accept &= Pairing::compute(b_w, &vk.gamma_g2) == Pairing::compute(&vk.beta_gamma_g1, v_w_prime);
    accept &= Pairing::compute(v_w, &TwistedCurve::generator())
        == Pairing::compute(&Curve::generator(), v_w_prime);
    let v_u = msm(
        &pub_inputs
            .iter()
            .map(|elem| elem.representative())
            .collect::<Vec<_>>(),
        &vk.u_tau_g1,
    )
    .unwrap();
    let v_u_prime = msm(
        &pub_inputs
            .iter()
            .map(|elem| elem.representative())
            .collect::<Vec<_>>(),
        &vk.u_tau_g2,
    )
    .unwrap();

    accept &= Pairing::compute(&v_u.operate_with(v_w), &v_u_prime.operate_with(v_w_prime))
        .unwrap()
        .mul(&vk.inv_pairing_g1_g2)
        == Pairing::compute(h, &vk.t_tau_g2).unwrap();
    accept
}
