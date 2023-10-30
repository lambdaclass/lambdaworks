use crate::{common::*, Proof, VerifyingKey};
use lambdaworks_math::{cyclic_group::IsGroup, elliptic_curve::traits::IsPairing};

pub fn verify(vk: &VerifyingKey, proof: &Proof, pub_inputs: &[FrElement]) -> bool {
    // [γ^{-1} * (β*l(τ) + α*r(τ) + o(τ))]_1
    let k_tau_assigned_verifier_g1 = (0..pub_inputs.len())
        .map(|i| vk.verifier_k_tau_g1[i].operate_with_self(pub_inputs[i].representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // let mut w_representatives = vec![];
    // w.iter()
    //     .for_each(|i| w_representatives.push(i.representative()));

    // let k_tau_assigned_verifier_g1 = msm(
    //     &w_representatives,
    //     &vk.verifier_k_tau_g1,
    // )
    // .unwrap();

    Pairing::compute(&proof.pi3, &vk.delta_g2)
        * vk.alpha_g1_times_beta_g2.clone()
        * Pairing::compute(&k_tau_assigned_verifier_g1, &vk.gamma_g2)
        == Pairing::compute(&proof.pi1, &proof.pi2)
}
