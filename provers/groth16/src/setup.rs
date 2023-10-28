use rand::Rng;

use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::curves::bls12_381::{
        curve::BLS12381Curve, default_types::FrField, pairing::BLS12381AtePairing,
        twist::BLS12381TwistCurve,
    },
    elliptic_curve::traits::{IsEllipticCurve, IsPairing},
    field::element::FieldElement,
};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::unsigned_integer::element::U256;

use crate::qap::QAP;

pub type Curve = BLS12381Curve;
pub type TwistedCurve = BLS12381TwistCurve;
pub type Pairing = BLS12381AtePairing;
pub type KZG = KateZaveruchaGoldberg<FrField, Pairing>;

pub type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
pub type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

pub struct VerifyingKey {
    // e([alpha]_1, [beta]_2) computed during setup as it's a constant
    pub alpha_g1_times_beta_g2: FieldElement<<Pairing as IsPairing>::OutputField>,
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

pub struct KeyWrapper {
    pub verifying_key: VerifyingKey,
    pub proving_key: ProvingKey,
}

pub struct Witness<F: IsField> {
    pub a: Vec<FieldElement<F>>,
    pub b: Vec<FieldElement<F>>,
    pub c: Vec<FieldElement<F>>,
}

pub struct ToxicWaste {
    pub tau: FrElement,
    pub alpha: FrElement,
    pub beta: FrElement,
    pub gamma: FrElement,
    pub delta: FrElement,
}

impl ToxicWaste {
    pub fn new() -> Self {
        Self {
            tau: sample_field_elem(),
            alpha: sample_field_elem(),
            beta: sample_field_elem(),
            gamma: sample_field_elem(),
            delta: sample_field_elem(),
        }
    }
}

fn sample_field_elem() -> FrElement {
    let mut rng = rand::thread_rng();
    FrElement::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    })
}

pub fn setup(qap: &QAP) -> KeyWrapper {
    let g1: G1Point = BLS12381Curve::generator();
    let g2: G2Point = BLS12381TwistCurve::generator();

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

    KeyWrapper {
        verifying_key: vk,
        proving_key: pk,
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::elliptic_curve::traits::IsPairing;

    use crate::qap::QAP;
    use crate::setup::{Pairing, setup};

    #[test]
    fn test_verifying_key() {
        let qap = QAP::build_example();
        let key_wrapper = setup(&qap);
        let expected_alpha_g1_times_beta_g2 = Pairing::compute(&key_wrapper.proving_key.alpha_g1, &key_wrapper.proving_key.beta_g2);
        assert_eq!(key_wrapper.verifying_key.alpha_g1_times_beta_g2, expected_alpha_g1_times_beta_g2);
    }
}
