use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::{elliptic_curve::traits::IsPairing, msm::pippenger::msm};

use crate::common::{Curve, FrElement, Pairing, TwistedCurve};
use crate::prover::Proof;
use crate::setup::VerifyingKey;

pub fn verify(vk: &VerifyingKey, proof: &Proof, pub_inputs: &[FrElement]) -> bool {
    let v_w = &proof.V_w;
    let v_w_prime = &proof.V_w_prime;
    let h = &proof.H;
    let b_w = &proof.B_w;

    let mut accept = true;
    accept &= Pairing::compute(b_w, &vk.gamma_g2) == Pairing::compute(&vk.beta_gamma_g1, v_w_prime);
    accept &= Pairing::compute(v_w, &TwistedCurve::generator())
        == Pairing::compute(&Curve::generator(), v_w_prime);

    accept
}
