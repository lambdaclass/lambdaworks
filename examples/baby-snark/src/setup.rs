use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_381::curve::BLS12381Curve,
            point::ShortWeierstrassProjectivePoint,
        },
        traits::IsEllipticCurve,
    },
};

use crate::{
    common::{ sample_fr_elem, Curve, FrElement, G1Point, G2Point, PairingOutput, TwistedCurve },
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
    // β * γ * g2
    pub beta_gamma_g2: G2Point,
    // γ * g1
    pub gamma_g1: G1Point,
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


