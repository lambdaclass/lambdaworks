use crate::common::{G1Point, G2Point, FE};
use crate::qap::QuadraticArithmeticProgram;
use crate::setup::EvaluationKey;
use lambdaworks_math::msm::pippenger::msm;

pub struct Proof {
    pub v: G1Point,
    pub w1: G1Point,
    pub w2: G2Point,
    pub y: G1Point,
    pub h: G2Point,
    pub v_prime: G1Point,
    pub w_prime: G1Point,
    pub y_prime: G1Point,
    pub z: G1Point,
}

pub fn generate_proof(
    evaluation_key: &EvaluationKey,
    qap: &QuadraticArithmeticProgram,
    qap_c_coefficients: &[FE],
) -> Proof {
    let cmid =
        &qap_c_coefficients[qap.number_of_inputs..qap_c_coefficients.len() - qap.number_of_outputs];
    // We transform each FieldElement of the cmid into an UnsignedInteger so we can multiply them to g1.
    let c_mid = cmid.iter().map(|elem| elem.canonical()).collect::<Vec<_>>();

    let h_polynomial = qap.h_polynomial(qap_c_coefficients);
    // We transform h_polynomial into UnsignedIntegers.
    let h_coefficients = h_polynomial
        .coefficients
        .iter()
        .map(|elem| elem.canonical())
        .collect::<Vec<_>>();
    let h_degree = h_polynomial.degree();

    Proof {
        v: msm(&c_mid, &evaluation_key.g1_vk).unwrap(),
        w1: msm(&c_mid, &evaluation_key.g1_wk).unwrap(),
        w2: msm(&c_mid, &evaluation_key.g2_wk).unwrap(),
        y: msm(&c_mid, &evaluation_key.g1_yk).unwrap(),
        v_prime: msm(&c_mid, &evaluation_key.g1_alpha_vk).unwrap(),
        w_prime: msm(&c_mid, &evaluation_key.g1_alpha_wk).unwrap(),
        y_prime: msm(&c_mid, &evaluation_key.g1_alpha_yk).unwrap(),
        z: msm(&c_mid, &evaluation_key.g1_beta).unwrap(),
        h: msm(&h_coefficients, &evaluation_key.g2_s_i[..h_degree]).unwrap(),
    }
}
