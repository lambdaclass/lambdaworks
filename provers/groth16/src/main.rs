use rand::Rng;

use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_groth16::qap::QAP;
use lambdaworks_groth16::setup::setup;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::curves::bls12_381::{
        curve::BLS12381Curve,
        default_types::{FrElement, FrField},
        pairing::BLS12381AtePairing,
        twist::BLS12381TwistCurve,
    },
    elliptic_curve::traits::{IsEllipticCurve, IsPairing},
    polynomial::Polynomial,
    unsigned_integer::element::U256,
};
use lambdaworks_math::msm::naive::msm;

pub type Curve = BLS12381Curve;
pub type TwistedCurve = BLS12381TwistCurve;
pub type Pairing = BLS12381AtePairing;
pub type KZG = KateZaveruchaGoldberg<FrField, Pairing>;

pub type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
pub type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

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

pub type Proof = (G1Point, G2Point, G1Point);

fn main() {
    //////////////////////////////////////////////////////////////////////
    /////////////////////////////// QAP //////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    /*
        sym_1 = x * x
        y = sym_1 * x
        sym_2 = y + x
        ~out = sym_2 + 5
    */

    let qap = QAP::build_example();

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Setup /////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    let key_wrapper = setup(&qap);

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Prove /////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    let is_zk = true;

    // aka the secret assignments, w vector, s vector
    // Includes the public inputs from the beginning.
    let w = ["0x1", "0x3", "0x23", "0x9", "0x1b", "0x1e"]
        .map(|e| FrElement::from_hex_unchecked(e))
        .to_vec();

    let h = calculate_h(&qap, &w);

    // [π_1]_1
    let mut pi1_g1 = w
        .iter()
        .enumerate()
        .map(|(i, coeff)| {
            key_wrapper.proving_key.l_tau_g1[i].operate_with_self(coeff.representative())
        })
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap()
        .operate_with(&key_wrapper.proving_key.alpha_g1);

    // [π_2]_2
    let mut pi2_g2 = w
        .iter()
        .enumerate()
        .map(|(i, coeff)| {
            key_wrapper.proving_key.r_tau_g2[i].operate_with_self(coeff.representative())
        })
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap()
        .operate_with(&key_wrapper.proving_key.beta_g2);

    // [ƍ^{-1} * t(τ)*h(τ)]_1
    let t_tau_h_tau_assigned_g1 = h
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, coeff)| {
            key_wrapper.proving_key.z_powers_of_tau_g1[i].operate_with_self(coeff.representative())
        })
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // [ƍ^{-1} * (β*l(τ) + α*r(τ) + o(τ))]_1
    let k_tau_assigned_prover_g1 = (qap.num_of_public_inputs..qap.num_of_total_inputs)
        .map(|i| {
            key_wrapper.proving_key.prover_k_tau_g1[i - qap.num_of_public_inputs]
                .operate_with_self(w[i].representative())
        })
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // [π_3]_1
    let mut pi3_g1 = t_tau_h_tau_assigned_g1.operate_with(&k_tau_assigned_prover_g1);

    if is_zk {
        let r = sample_field_elem();
        let s = sample_field_elem();

        pi1_g1 = pi1_g1.operate_with(
            &key_wrapper
                .proving_key
                .delta_g1
                .operate_with_self(r.representative()),
        );
        pi2_g2 = pi2_g2.operate_with(
            &key_wrapper
                .proving_key
                .delta_g2
                .operate_with_self(s.representative()),
        );

        // [π_2]_1
        let pi2_g1 = w
            .iter()
            .enumerate()
            .map(|(i, coeff)| {
                key_wrapper.proving_key.r_tau_g1[i].operate_with_self(coeff.representative())
            })
            .reduce(|acc, x| acc.operate_with(&x))
            .unwrap()
            .operate_with(&key_wrapper.proving_key.beta_g1)
            .operate_with(
                &key_wrapper
                    .proving_key
                    .delta_g1
                    .operate_with_self(s.representative()),
            );

        pi3_g1 = pi3_g1
            // s[π_1]_1
            .operate_with(&pi1_g1.operate_with_self(s.representative()))
            // r[π_2]_1
            .operate_with(&pi2_g1.operate_with_self(r.representative()))
            // -rs[ƍ]_1
            .operate_with(
                &key_wrapper
                    .proving_key
                    .delta_g1
                    .operate_with_self((-(&r * &s)).representative()),
            );
    }

    // //////////////////////////////////////////////////////////////////////
    // ////////////////////////////// Verify ////////////////////////////////
    // //////////////////////////////////////////////////////////////////////

    // [γ^{-1} * (β*l(τ) + α*r(τ) + o(τ))]_1

    let mut w_representatives = vec![];
    w.iter().for_each(|i| w_representatives.push(i.representative()));

    let k_tau_assigned_verifier_g1 = msm(
        &w_representatives,
        &key_wrapper.verifying_key.verifier_k_tau_g1,
    )
        .unwrap();

    assert_eq!(
        Pairing::compute(&pi3_g1, &key_wrapper.verifying_key.delta_g2)
            * key_wrapper.verifying_key.alpha_g1_times_beta_g2
            * Pairing::compute(
            &k_tau_assigned_verifier_g1,
            &key_wrapper.verifying_key.gamma_g2,
        ),
        Pairing::compute(&pi1_g1, &pi2_g2),
    );
}
