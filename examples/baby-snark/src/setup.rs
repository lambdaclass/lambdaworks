use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::traits::{IsEllipticCurve, IsPairing},
};

use crate::{
    common::{
        sample_fr_elem, Curve, FrElement, G1Point, G2Point, Pairing, PairingOutput, TwistedCurve,
    },
    ssp::SquareSpanProgram,
};

pub struct VerifyingKey {
    // Ui(τ) * g1, 0 <= i < l
    pub u_tau_g1: Vec<G1Point>,
    // Ui(τ) * g2, 0 <= i < l
    pub u_tau_g2: Vec<G2Point>,
    // t(τ) * g2
    pub t_tau_g2: G2Point,
    // e(g1, g2)^-1
    pub inv_pairing_g1_g2: PairingOutput,
    // β * γ * g1
    pub beta_gamma_g1: G1Point,
    // γ * g2
    pub gamma_g2: G2Point,
}

pub struct ProvingKey {
    // (τ^k) * g1, 0 <= k < m
    pub k_powers_of_tau_g1: Vec<G1Point>,
    // Ui(τ) * g1, l <= i <= m
    pub u_tau_g1: Vec<G1Point>,
    // Ui(τ) * g2, l <= i <= m
    pub u_tau_g2: Vec<G2Point>,
    // β * Ui(τ) * g1, l <= i <= m
    pub beta_u_tau_g1: Vec<G1Point>,
    // t(τ) * g1
    pub t_tau_g1: G1Point,
    // β * t(τ) * g1
    pub beta_t_tau_g1: G1Point,
    // t(τ) * g2
    pub t_tau_g2: G2Point,
}

struct ToxicWaste {
    tau: FrElement,
    beta: FrElement,
    gamma: FrElement,
}

impl ToxicWaste {
    pub fn new() -> Self {
        Self {
            tau: sample_fr_elem(),
            beta: sample_fr_elem(),
            gamma: sample_fr_elem(),
        }
    }
}

pub fn setup(u: &SquareSpanProgram) -> (ProvingKey, VerifyingKey) {
    let g1: G1Point = Curve::generator();
    let g2: G2Point = TwistedCurve::generator();

    let tw = ToxicWaste::new();

    let u_tau: Vec<_> = u
        .u_polynomials
        .iter()
        .map(|p| p.evaluate(&tw.tau))
        .collect();
    let t_tau = tw.tau.pow(u.number_of_constraints) - FrElement::one();

    let pub_slice = &u_tau[..u.number_of_public_inputs];
    let priv_end = (u.number_of_constraints + 1).min(u_tau.len());
    let priv_slice = &u_tau[u.number_of_public_inputs..priv_end];

    let vk = VerifyingKey {
        u_tau_g1: pub_slice
            .iter()
            .map(|ui| g1.operate_with_self(ui.canonical()))
            .collect(),
        u_tau_g2: pub_slice
            .iter()
            .map(|ui| g2.operate_with_self(ui.canonical()))
            .collect(),
        t_tau_g2: g2.operate_with_self(t_tau.canonical()),
        inv_pairing_g1_g2: Pairing::compute(&g1, &g2).unwrap().inv().unwrap(),
        beta_gamma_g1: g1.operate_with_self((&tw.beta * &tw.gamma).canonical()),
        gamma_g2: g2.operate_with_self(tw.gamma.canonical()),
    };

    let pk = ProvingKey {
        k_powers_of_tau_g1: (0..u.number_of_constraints + 1)
            .map(|k| g1.operate_with_self(tw.tau.pow(k).canonical()))
            .collect(),
        u_tau_g1: priv_slice
            .iter()
            .map(|ui| g1.operate_with_self(ui.canonical()))
            .collect(),
        u_tau_g2: priv_slice
            .iter()
            .map(|ui| g2.operate_with_self(ui.canonical()))
            .collect(),
        beta_u_tau_g1: priv_slice
            .iter()
            .map(|ui| g1.operate_with_self((ui * &tw.beta).canonical()))
            .collect(),
        t_tau_g1: g1.operate_with_self(t_tau.canonical()),
        beta_t_tau_g1: g1.operate_with_self((&tw.beta * &t_tau).canonical()),
        t_tau_g2: g2.operate_with_self(t_tau.canonical()),
    };

    (pk, vk)
}
