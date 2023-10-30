use crate::common::*;
use crate::qap::QAP;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::traits::{IsEllipticCurve, IsPairing},
};

pub struct VerifyingKey {
    // e([alpha]_1, [beta]_2) computed during setup as it's a constant
    pub alpha_g1_times_beta_g2: PairingOutput,
    pub delta_g2: G2Point,
    pub gamma_g2: G2Point,
    // [K_0(τ)]_1, [K_1(τ)]_1, ..., [K_k(τ)]_1
    // where K_i(τ) = γ^{-1} * (β*l(τ) + α*r(τ) + o(τ))
    // and "k" is the number of public inputs
    pub verifier_k_tau_g1: Vec<G1Point>,
}

pub struct ProvingKey {
    pub alpha_g1: G1Point,
    pub beta_g1: G1Point,
    pub beta_g2: G2Point,
    pub delta_g1: G1Point,
    pub delta_g2: G2Point,
    // [A_0(τ)]_1, [A_1(τ)]_1, ..., [A_n(τ)]_1
    pub l_tau_g1: Vec<G1Point>,
    // [B_0(τ)]_1, [B_1(τ)]_1, ..., [B_n(τ)]_1
    pub r_tau_g1: Vec<G1Point>,
    // [B_0(τ)]_2, [B_1(τ)]_2, ..., [B_n(τ)]_2
    pub r_tau_g2: Vec<G2Point>,
    // [K_{k+1}(τ)]_1, [K_{k+2}(τ)]_1, ..., [K_n(τ)]_1
    // where K_i(τ) = ƍ^{-1} * (β*l(τ) + α*r(τ) + o(τ))
    // and "k" is the number of public inputs
    pub prover_k_tau_g1: Vec<G1Point>,
    // [delta^{-1} * t(τ) * tau^0]_1, [delta^{-1} * t(τ) * τ^1]_1, ..., [delta^{-1} * t(τ) * τ^m]_1
    pub z_powers_of_tau_g1: Vec<G1Point>,
}

struct ToxicWaste {
    tau: FrElement,
    alpha: FrElement,
    beta: FrElement,
    gamma: FrElement,
    delta: FrElement,
}

impl ToxicWaste {
    pub fn new() -> Self {
        Self {
            tau: sample_fr_elem(),
            alpha: sample_fr_elem(),
            beta: sample_fr_elem(),
            gamma: sample_fr_elem(),
            delta: sample_fr_elem(),
        }
    }
}

pub fn setup(qap: &QAP) -> (ProvingKey, VerifyingKey) {
    let g1: G1Point = Curve::generator();
    let g2: G2Point = TwistedCurve::generator();

    let tw = ToxicWaste::new();

    let delta_inv = tw.delta.inv().unwrap();
    let gamma_inv = tw.gamma.inv().unwrap();

    // [A_i(τ)]_1, [B_i(τ)]_1, [B_i(τ)]_2
    let mut l_tau_g1: Vec<G1Point> = vec![];
    let mut r_tau_g1: Vec<G1Point> = vec![];
    let mut r_tau_g2: Vec<G2Point> = vec![];
    let mut verifier_k_tau_g1: Vec<G1Point> = vec![];
    let mut prover_k_tau_g1: Vec<G1Point> = vec![];

    // Public variables
    for i in 0..qap.num_of_public_inputs {
        let l_i_tau = qap.l[i].evaluate(&tw.tau);
        let r_i_tau = qap.r[i].evaluate(&tw.tau);
        let o_i_tau = qap.o[i].evaluate(&tw.tau);
        let k_i_tau = &gamma_inv * (&tw.beta * &l_i_tau + &tw.alpha * &r_i_tau + &o_i_tau);

        l_tau_g1.push(g1.operate_with_self(l_i_tau.representative()));
        r_tau_g1.push(g1.operate_with_self(r_i_tau.representative()));
        r_tau_g2.push(g2.operate_with_self(r_i_tau.representative()));
        verifier_k_tau_g1.push(g1.operate_with_self(k_i_tau.representative()));
    }
    // Private variables
    for i in qap.num_of_public_inputs..qap.num_of_total_inputs {
        let l_i_tau = qap.l[i].evaluate(&tw.tau);
        let r_i_tau = qap.r[i].evaluate(&tw.tau);
        let o_i_tau = qap.o[i].evaluate(&tw.tau);
        let k_i_tau = &delta_inv * (&tw.beta * &l_i_tau + &tw.alpha * &r_i_tau + &o_i_tau);

        l_tau_g1.push(g1.operate_with_self(l_i_tau.representative()));
        r_tau_g1.push(g1.operate_with_self(r_i_tau.representative()));
        r_tau_g2.push(g2.operate_with_self(r_i_tau.representative()));
        prover_k_tau_g1.push(g1.operate_with_self(k_i_tau.representative()));
    }

    // [delta^{-1} * t(τ) * τ^0]_1, [delta^{-1} * t(τ) * τ^1]_1, ..., [delta^{-1} * t(τ) * τ^m]_1
    let t_tau_times_delta_inv = &delta_inv * qap.t.evaluate(&tw.tau);
    let z_powers_of_tau_g1: Vec<G1Point> = (0..qap.num_of_gates + 1)
        .map(|exp: usize| {
            g1.operate_with_self((tw.tau.pow(exp as u128)).representative())
                .operate_with_self((&t_tau_times_delta_inv).representative())
        })
        .collect();

    let alpha_g1 = g1.operate_with_self(tw.alpha.representative());
    let beta_g2 = g2.operate_with_self(tw.beta.representative());
    let delta_g2 = g2.operate_with_self(tw.delta.representative());

    let pk = ProvingKey {
        alpha_g1: alpha_g1.clone(),
        beta_g1: g1.operate_with_self(tw.beta.representative()),
        beta_g2: beta_g2.clone(),
        delta_g1: g1.operate_with_self(tw.delta.representative()),
        delta_g2: delta_g2.clone(),
        l_tau_g1,
        r_tau_g1,
        r_tau_g2,
        prover_k_tau_g1,
        z_powers_of_tau_g1,
    };

    let vk = VerifyingKey {
        alpha_g1_times_beta_g2: Pairing::compute(&alpha_g1, &beta_g2),
        delta_g2,
        gamma_g2: g2.operate_with_self(tw.gamma.representative()),
        verifier_k_tau_g1,
    };

    (pk, vk)
}
