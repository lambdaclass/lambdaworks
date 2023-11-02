use crate::{common::*, ProvingKey, QAP};
use lambdaworks_math::{cyclic_group::IsGroup, msm::pippenger::msm};

pub struct Proof {
    pub pi1: G1Point,
    pub pi2: G2Point,
    pub pi3: G1Point,
}

pub fn generate_proof(w: &[FrElement], qap: &QAP, pk: &ProvingKey) -> Proof {
    let h_coefficients = qap
        .calculate_h_coefficients(&w)
        .into_iter()
        .map(|elem| elem.representative())
        .collect::<Vec<_>>();

    let w = w
        .into_iter()
        .map(|elem| elem.representative())
        .collect::<Vec<_>>();

    // Sample randomness for hiding
    let r = sample_fr_elem();
    let s = sample_fr_elem();

    // [π_1]_1
    let pi1 = msm(&w, &pk.l_tau_g1)
        .unwrap()
        .operate_with(&pk.alpha_g1)
        .operate_with(&pk.delta_g1.operate_with_self(r.representative()));

    // [π_2]_2
    let pi2 = msm(&w, &pk.r_tau_g2)
        .unwrap()
        .operate_with(&pk.beta_g2)
        .operate_with(&pk.delta_g2.operate_with_self(s.representative()));

    // [ƍ^{-1} * t(τ)*h(τ)]_1
    let t_tau_h_tau_assigned_g1 = msm(
        &h_coefficients,
        &pk.z_powers_of_tau_g1[..h_coefficients.len()],
    )
    .unwrap();
    // [ƍ^{-1} * (β*l(τ) + α*r(τ) + o(τ))]_1
    let num_of_total_inputs = qap.l.len();
    let num_of_private_inputs = num_of_total_inputs - qap.num_of_public_inputs;
    let k_tau_assigned_prover_g1 = msm(
        &w[qap.num_of_public_inputs..num_of_total_inputs],
        &pk.prover_k_tau_g1[0..num_of_private_inputs],
    )
    .unwrap();

    // [π_2]_1
    let pi2_g1 = msm(&w, &pk.r_tau_g1)
        .unwrap()
        .operate_with(&pk.beta_g1)
        .operate_with(&pk.delta_g1.operate_with_self(s.representative()));

    // [π_3]_1
    let pi3 = k_tau_assigned_prover_g1
        .operate_with(&t_tau_h_tau_assigned_g1)
        // s[π_1]_1
        .operate_with(&pi1.operate_with_self(s.representative()))
        // r[π_2]_1
        .operate_with(&pi2_g1.operate_with_self(r.representative()))
        // -rs[ƍ]_1
        .operate_with(&pk.delta_g1.operate_with_self((-(&r * &s)).representative()));

    Proof { pi1, pi2, pi3 }
}
