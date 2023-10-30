use crate::setup::ProvingKey;
use crate::{common::*, qap::QAP};
use lambdaworks_math::msm::naive::msm;
use lambdaworks_math::{cyclic_group::IsGroup, polynomial::Polynomial};

pub struct Proof {
    pub pi1: G1Point,
    pub pi2: G2Point,
    pub pi3: G1Point,
}

fn calculate_h(qap: &QAP, w: &[FrElement]) -> Polynomial<FrElement> {
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

    h
}

pub fn generate_proof(w: &[FrElement], qap: &QAP, pk: &ProvingKey) -> Proof {
    let is_zk = true;

    let h = calculate_h(&qap, &w);

    // [π_1]_1
    let mut pi1 = w
        .iter()
        .enumerate()
        .map(|(i, coeff)| pk.l_tau_g1[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap()
        .operate_with(&pk.alpha_g1);

    // [π_2]_2
    let mut pi2 = w
        .iter()
        .enumerate()
        .map(|(i, coeff)| pk.r_tau_g2[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap()
        .operate_with(&pk.beta_g2);

    // [ƍ^{-1} * t(τ)*h(τ)]_1
    let t_tau_h_tau_assigned_g1 = h
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, coeff)| pk.z_powers_of_tau_g1[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // [ƍ^{-1} * (β*l(τ) + α*r(τ) + o(τ))]_1
    let k_tau_assigned_prover_g1 = (qap.num_of_public_inputs..qap.num_of_total_inputs)
        .map(|i| {
            pk.prover_k_tau_g1[i - qap.num_of_public_inputs]
                .operate_with_self(w[i].representative())
        })
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // [π_3]_1
    let mut pi3 = t_tau_h_tau_assigned_g1.operate_with(&k_tau_assigned_prover_g1);

    if is_zk {
        let r = sample_field_elem();
        let s = sample_field_elem();

        pi1 = pi1.operate_with(&pk.delta_g1.operate_with_self(r.representative()));
        pi2 = pi2.operate_with(&pk.delta_g2.operate_with_self(s.representative()));

        // [π_2]_1
        let pi2_g1 = w
            .iter()
            .enumerate()
            .map(|(i, coeff)| pk.r_tau_g1[i].operate_with_self(coeff.representative()))
            .reduce(|acc, x| acc.operate_with(&x))
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

#[cfg(test)]
mod tests {}
