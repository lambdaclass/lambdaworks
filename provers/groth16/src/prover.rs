use crate::{common::*, ProvingKey, QAP};
use lambdaworks_math::{cyclic_group::IsGroup, msm::pippenger::msm, polynomial::Polynomial};

pub struct Proof {
    pub pi1: G1Point,
    pub pi2: G2Point,
    pub pi3: G1Point,
}

pub fn generate_proof(is_zk: bool, w: &[FrElement], qap: &QAP, pk: &ProvingKey) -> Proof {
    let h_coefficients = calculate_h(&qap, &w)
        .into_iter()
        .map(|elem| elem.representative())
        .collect::<Vec<_>>();

    let w = w
        .into_iter()
        .map(|elem| elem.representative())
        .collect::<Vec<_>>();

    // [π_1]_1
    let mut pi1 = msm(&w, &pk.l_tau_g1).unwrap().operate_with(&pk.alpha_g1);
    // [π_2]_2
    let mut pi2 = msm(&w, &pk.r_tau_g2).unwrap().operate_with(&pk.beta_g2);

    // [ƍ^{-1} * t(τ)*h(τ)]_1
    let t_tau_h_tau_assigned_g1 = msm(
        &h_coefficients,
        &pk.z_powers_of_tau_g1[..h_coefficients.len()],
    )
    .unwrap();

    // [ƍ^{-1} * (β*l(τ) + α*r(τ) + o(τ))]_1
    let num_of_private_inputs = qap.num_of_total_inputs - qap.num_of_public_inputs;
    let k_tau_assigned_prover_g1 = msm(
        &w[qap.num_of_public_inputs..qap.num_of_total_inputs],
        &pk.prover_k_tau_g1[0..num_of_private_inputs],
    )
    .unwrap();

    // [π_3]_1
    let mut pi3 = t_tau_h_tau_assigned_g1.operate_with(&k_tau_assigned_prover_g1);

    if is_zk {
        let r = sample_fr_elem();
        let s = sample_fr_elem();

        pi1 = pi1.operate_with(&pk.delta_g1.operate_with_self(r.representative()));
        pi2 = pi2.operate_with(&pk.delta_g2.operate_with_self(s.representative()));

        // [π_2]_1
        let pi2_g1 = msm(&w, &pk.r_tau_g1)
            .unwrap()
            .operate_with(&pk.beta_g1)
            .operate_with(&pk.delta_g1.operate_with_self(s.representative()));

        pi3 = pi3
            // s[π_1]_1
            .operate_with(&pi1.operate_with_self(s.representative()))
            // r[π_2]_1
            .operate_with(&pi2_g1.operate_with_self(r.representative()))
            // -rs[ƍ]_1
            .operate_with(&pk.delta_g1.operate_with_self((-(&r * &s)).representative()));
    }

    Proof { pi1, pi2, pi3 }
}

fn calculate_h<'a>(qap: &QAP, w: &[FrElement]) -> Vec<FrElement> {
    // Compute A.s by summing up polynomials A[0].s, A[1].s, ..., A[n].s
    // In other words, assign the witness coefficients / execution values
    // Similarly for B.s and C.s
    let mut l_coeffs = vec![FrElement::from_hex_unchecked("0"); qap.num_of_gates];
    let mut r_coeffs = vec![FrElement::from_hex_unchecked("0"); qap.num_of_gates];
    let mut o_coeffs = vec![FrElement::from_hex_unchecked("0"); qap.num_of_gates];
    for row in 0..qap.num_of_gates {
        for col in 0..qap.num_of_total_inputs {
            let current_l_assigned = &qap.l[col] * &w[col];
            let current_l_coeffs = current_l_assigned.coefficients();
            if current_l_coeffs.len() != 0 {
                l_coeffs[row] += current_l_coeffs[row].clone();
            }

            let current_r_assigned = &qap.r[col] * &w[col];
            let current_r_coeffs = current_r_assigned.coefficients();
            if current_r_coeffs.len() != 0 {
                r_coeffs[row] += current_r_coeffs[row].clone();
            }

            let current_o_assigned = &qap.o[col] * &w[col];
            let current_o_coeffs = current_o_assigned.coefficients();
            if current_o_coeffs.len() != 0 {
                o_coeffs[row] += current_o_coeffs[row].clone();
            }
        }
    }
    let l_assigned = Polynomial::new(&l_coeffs);
    let r_assigned = Polynomial::new(&r_coeffs);
    let o_assigned = Polynomial::new(&o_coeffs);

    // h(x) = p(x) / t(x) = (A.s * B.s - C.s) / t(x)
    let (h, remainder) =
        (&l_assigned * &r_assigned - &o_assigned).long_division_with_remainder(&qap.t);
    assert_eq!(0, remainder.degree()); // must have no remainder

    h.coefficients().to_vec()
}
