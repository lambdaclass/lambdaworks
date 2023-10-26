use std::ops::Mul;

// use groth16::prover::Groth16Prover;
// use groth16::setup::setup;
// use groth16::verifier;
// use verifier::Verifier;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::curves::bls12_381::{
        curve::BLS12381Curve, pairing::BLS12381AtePairing, twist::BLS12381TwistCurve,
    },
    elliptic_curve::traits::{IsEllipticCurve, IsPairing},
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::Polynomial,
    unsigned_integer::element::{UnsignedInteger, U384},
};

use lambdaworks_math::{
    elliptic_curve::short_weierstrass::{
        curves::bls12_381::default_types::{FrElement, FrField},
        point::ShortWeierstrassProjectivePoint,
    },
    traits::{Deserializable, Serializable},
    unsigned_integer::element::U256,
};

use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;

use rand::Rng;

pub type Curve = BLS12381Curve;
pub type TwistedCurve = BLS12381TwistCurve;
pub type Pairing = BLS12381AtePairing;
pub type KZG = KateZaveruchaGoldberg<FrField, Pairing>;

pub type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
pub type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

pub struct ToxicWaste {
    pub tau: FrElement,
    pub alpha: FrElement,
    pub beta: FrElement,
    pub gamma: FrElement,
    pub delta: FrElement,
}

impl ToxicWaste {
    fn sample_field_elem() -> FrElement {
        // Config
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

    pub fn new() -> Self {
        Self {
            tau: Self::sample_field_elem(),
            alpha: Self::sample_field_elem(),
            beta: Self::sample_field_elem(),
            gamma: Self::sample_field_elem(),
            delta: Self::sample_field_elem(),
        }
    }
}

impl Default for ToxicWaste {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ProvingKey {
    pub alpha_g1: G1Point,
    pub beta_g1: G1Point,
    pub beta_g2: G2Point,
    pub delta_g1: G1Point,
    pub delta_g2: G2Point,
    // [A_0(tau)]_1, [A_1(tau)]_1, ..., [A_n(tau)]_1
    pub l_tau_g1: Vec<G1Point>,
    // [B_0(tau)]_1, [B_1(tau)]_1, ..., [B_n(tau)]_1
    pub r_tau_g1: Vec<G1Point>,
    // [B_0(tau)]_2, [B_1(tau)]_2, ..., [B_n(tau)]_2
    pub r_tau_g2: Vec<G2Point>,
    // [K_{k+1}(tau)]_1, [K_{k+2}(tau)]_1, ..., [K_n(tau)]_1
    // where K_i(tau) = ƍ^{-1} * (β*l(tau) + α*r(tau) + o(tau))
    // and "k" is the number of public inputs
    pub prover_k_tau_g1: Vec<G1Point>,
    // [delta^{-1} * t(tau) * tau^0]_1, [delta^{-1} * t(tau) * tau^1]_1, ..., [delta^{-1} * t(tau) * tau^m]_1
    pub z_powers_of_tau_g1: Vec<G1Point>,
}

pub struct VerifyingKey {
    // e([alpha]_1, [beta]_2) computed during setup as it's a constant
    pub alpha_g1_times_beta_g2: FieldElement<<Pairing as IsPairing>::OutputField>,
    pub delta_g2: G2Point,
    pub gamma_g2: G2Point,
    // [K_0(tau)]_1, [K_1(tau)]_1, ..., [K_k(tau)]_1
    // where K_i(tau) = ƍ^{-1} * (β*l(tau) + α*r(tau) + o(tau))
    // and "k" is the number of public inputs
    pub verifier_k_tau_g1: Vec<G1Point>,
}

pub type Proof = (G1Point, G2Point, G1Point);

fn get_test_QAP_L(gate_indices: &[FrElement]) -> Vec<Polynomial<FrElement>> {
    vec![
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("5"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("1"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("1"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("1"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("1"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("1"),
            ],
        )
        .unwrap(),
    ]
}

fn get_test_QAP_R(gate_indices: &[FrElement]) -> Vec<Polynomial<FrElement>> {
    vec![
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("1"),
                FrElement::from_hex_unchecked("1"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("1"),
                FrElement::from_hex_unchecked("1"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
    ]
}

fn get_test_QAP_O(gate_indices: &[FrElement]) -> Vec<Polynomial<FrElement>> {
    vec![
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("1"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("1"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("1"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
        Polynomial::interpolate(
            &gate_indices,
            &[
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("0"),
                FrElement::from_hex_unchecked("1"),
                FrElement::from_hex_unchecked("0"),
            ],
        )
        .unwrap(),
    ]
}

fn main() {
    //////////////////////////////////////////////////////////////////////
    /////////////////////////////// QAP //////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    /*
        sym_1 = x * x
        y = sym_1 * x
        sym_2 = y + x
        ~out = sym_2 + 5

        A                   B                   C
        [0, 1, 0, 0, 0, 0]  [0, 1, 0, 0, 0, 0]  [0, 0, 0, 1, 0, 0]
        [0, 0, 0, 1, 0, 0]  [0, 1, 0, 0, 0, 0]  [0, 0, 0, 0, 1, 0]
        [0, 1, 0, 0, 1, 0]  [1, 0, 0, 0, 0, 0]  [0, 0, 0, 0, 0, 1]
        [5, 0, 0, 0, 0, 1]  [1, 0, 0, 0, 0, 0]  [0, 0, 1, 0, 0, 0]

        m+1 rows, n+1 cols
    */

    let number_of_public_vars = 1;
    let number_of_private_vars = 5;
    let number_of_total_vars = number_of_public_vars + number_of_private_vars;

    // Gate indices. Will correspond to rows of QAP matrices. TODO: Roots of unity
    let gate_indices = [
        FrElement::from_hex_unchecked("1"),
        FrElement::from_hex_unchecked("2"),
        FrElement::from_hex_unchecked("3"),
        FrElement::from_hex_unchecked("4"),
    ];
    let l = get_test_QAP_L(&gate_indices);
    let r = get_test_QAP_R(&gate_indices);
    let o = get_test_QAP_O(&gate_indices);
    {
        assert_eq!(l.len(), r.len());
        assert_eq!(r.len(), o.len());
        assert_eq!(number_of_total_vars, o.len());
    }

    let num_of_gates = l[0].degree() + 1;

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Setup /////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    let g1: G1Point = BLS12381Curve::generator();
    let g2: G2Point = BLS12381TwistCurve::generator();

    // t(x) = (x-1)(x-2)(x-3)(x-4) = [24, -50, 35, -10, 1]
    let t = Polynomial::new(&[
        FrElement::from_hex_unchecked("0x18"),
        -FrElement::from_hex_unchecked("0x32"),
        FrElement::from_hex_unchecked("0x23"),
        -FrElement::from_hex_unchecked("0xa"),
        FrElement::from_hex_unchecked("0x1"),
    ]);

    let toxic_waste = ToxicWaste::default();

    // Point of evaluation.
    let tau = toxic_waste.tau;

    let alpha = toxic_waste.alpha;
    let beta = toxic_waste.beta;

    let delta = toxic_waste.delta;
    let delta_inv = delta.inv().unwrap();

    let gamma = toxic_waste.gamma;
    let gamma_inv = gamma.inv().unwrap();

    let alpha_g1 = g1.operate_with_self(alpha.representative());
    let beta_g2 = g2.operate_with_self(beta.representative());
    let delta_g2 = g2.operate_with_self(delta.representative());

    // [A_i(tau)]_1, [B_i(tau)]_1, [B_i(tau)]_2
    let mut l_tau_g1: Vec<G1Point> = vec![];
    let mut r_tau_g1: Vec<G1Point> = vec![];
    let mut o_tau_g1_temp: Vec<G1Point> = vec![]; // TODO:remove

    let mut r_tau_g2: Vec<G2Point> = vec![];
    let mut verifier_k_tau_g1: Vec<G1Point> = vec![];
    let mut prover_k_tau_g1: Vec<G1Point> = vec![];

    // Public variables
    for i in 0..number_of_public_vars {
        let l_i_tau = l[i].evaluate(&tau);
        let r_i_tau = r[i].evaluate(&tau);
        let o_i_tau = o[i].evaluate(&tau);
        let k_i_tau = &gamma_inv * (&beta * &l_i_tau + &alpha * &r_i_tau + &o_i_tau);

        o_tau_g1_temp.push(g1.operate_with_self(o_i_tau.representative())); // TODO:remove

        l_tau_g1.push(g1.operate_with_self(l_i_tau.representative()));
        r_tau_g1.push(g1.operate_with_self(r_i_tau.representative()));
        r_tau_g2.push(g2.operate_with_self(r_i_tau.representative()));
        verifier_k_tau_g1.push(g1.operate_with_self(k_i_tau.representative()));
    }
    // Private variables
    for i in number_of_public_vars..number_of_total_vars {
        let l_i_tau = l[i].evaluate(&tau);
        let r_i_tau = r[i].evaluate(&tau);
        let o_i_tau = o[i].evaluate(&tau);
        let k_i_tau = &delta_inv * (&beta * &l_i_tau + &alpha * &r_i_tau + &o_i_tau);

        o_tau_g1_temp.push(g1.operate_with_self(o_i_tau.representative())); // TODO:remove

        l_tau_g1.push(g1.operate_with_self(l_i_tau.representative()));
        r_tau_g1.push(g1.operate_with_self(r_i_tau.representative()));
        r_tau_g2.push(g2.operate_with_self(r_i_tau.representative()));
        prover_k_tau_g1.push(g1.operate_with_self(k_i_tau.representative()));
    }

    // [delta^{-1} * t(tau) * tau^0]_1, [delta^{-1} * t(tau) * tau^1]_1, ..., [delta^{-1} * t(tau) * tau^m]_1
    let t_tau_times_delta_inv = &delta_inv * t.evaluate(&tau);
    let z_powers_of_tau_temp_g1: Vec<G1Point> = (0..num_of_gates + 1) // TODO:remove
        .map(|exp: usize| {
            g1.operate_with_self((&t.evaluate(&tau) * tau.pow(exp as u128)).representative())
        })
        .collect();

    let z_powers_of_tau_g1: Vec<G1Point> = (0..num_of_gates + 1)
        .map(|exp: usize| {
            g1.operate_with_self((&t_tau_times_delta_inv * tau.pow(exp as u128)).representative())
        })
        .collect();

    let prov_key = ProvingKey {
        alpha_g1: alpha_g1.clone(),
        beta_g1: g1.operate_with_self(beta.representative()),
        beta_g2: beta_g2.clone(),
        delta_g1: g1.operate_with_self(delta.representative()),
        delta_g2: delta_g2.clone(),
        l_tau_g1: l_tau_g1.clone(),
        r_tau_g1: r_tau_g1.clone(),
        r_tau_g2: r_tau_g2.clone(),
        prover_k_tau_g1: prover_k_tau_g1.clone(),
        z_powers_of_tau_g1: z_powers_of_tau_g1.clone(),
    };

    let verif_key = VerifyingKey {
        alpha_g1_times_beta_g2: Pairing::compute(&alpha_g1, &beta_g2),
        delta_g2: delta_g2.clone(),
        gamma_g2: g2.operate_with_self(gamma.representative()),
        verifier_k_tau_g1: verifier_k_tau_g1.clone(),
    };

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Prove /////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    // aka the secret assignments, w vector, s vector
    // Includes the public inputs from the beginning.
    let witness_vector = ["0x1", "0x3", "0x23", "0x9", "0x1b", "0x1e"]
        .map(|e| FrElement::from_hex_unchecked(e))
        .to_vec();

    // Compute A.s by summing up polynomials A[0].s, A[1].s, ..., A[n].s
    // In other words, assign the witness coefficients / execution values
    // Similarly for B.s and C.s
    let mut A_s_coeffs = vec![FrElement::from_hex_unchecked("0"); num_of_gates];
    let mut B_s_coeffs = vec![FrElement::from_hex_unchecked("0"); num_of_gates];
    let mut C_s_coeffs = vec![FrElement::from_hex_unchecked("0"); num_of_gates];
    for row in 0..num_of_gates {
        for col in 0..number_of_total_vars {
            let current_variable_assigned_l = &l[col] * &witness_vector[col];
            let l_current_poly_coeffs = current_variable_assigned_l.coefficients();
            if l_current_poly_coeffs.len() != 0 {
                A_s_coeffs[row] += l_current_poly_coeffs[row].clone();
            }

            let current_variable_assigned_r = &r[col] * &witness_vector[col];
            let r_current_poly_coeffs = current_variable_assigned_r.coefficients();
            if r_current_poly_coeffs.len() != 0 {
                B_s_coeffs[row] += r_current_poly_coeffs[row].clone();
            }

            let current_variable_assigned_o = &o[col] * &witness_vector[col];
            let o_current_poly_coeffs = current_variable_assigned_o.coefficients();
            if o_current_poly_coeffs.len() != 0 {
                C_s_coeffs[row] += o_current_poly_coeffs[row].clone();
            }
        }
    }
    let A_s = Polynomial::new(&A_s_coeffs);
    let B_s = Polynomial::new(&B_s_coeffs);
    let C_s = Polynomial::new(&C_s_coeffs);
    // Assert correctness of assignments
    {
        assert_eq!(
            A_s.evaluate(&gate_indices[0]) * B_s.evaluate(&gate_indices[0]),
            C_s.evaluate(&gate_indices[0])
        );
        assert_eq!(
            A_s.evaluate(&gate_indices[1]) * B_s.evaluate(&gate_indices[1]),
            C_s.evaluate(&gate_indices[1])
        );
        assert_eq!(
            A_s.evaluate(&gate_indices[2]) * B_s.evaluate(&gate_indices[2]),
            C_s.evaluate(&gate_indices[2])
        );
        assert_eq!(
            A_s.evaluate(&gate_indices[3]) * B_s.evaluate(&gate_indices[3]),
            C_s.evaluate(&gate_indices[3])
        );
    }

    // h(x) = p(x) / t(x) = (A.s * B.s - C.s) / t(x)
    let (h, remainder) = (&A_s * &B_s - &C_s).long_division_with_remainder(&t);
    assert_eq!(0, remainder.degree()); // must have no remainder

    // Did we assign coefficients correctly?
    assert_eq!(
        A_s.evaluate(&tau) * B_s.evaluate(&tau),
        C_s.evaluate(&tau) + h.evaluate(&tau) * t.evaluate(&tau)
    );

    let A_s_g1 = witness_vector
        .iter()
        .enumerate()
        .map(|(i, coeff)| l_tau_g1[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // Was encrypted evaluation correct?
    assert_eq!(
        A_s_g1,
        g1.operate_with_self(A_s.evaluate(&tau).representative())
    );

    let B_s_temp_g1 = witness_vector
        .iter()
        .enumerate()
        .map(|(i, coeff)| r_tau_g1[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    let B_s_g2 = witness_vector
        .iter()
        .enumerate()
        .map(|(i, coeff)| r_tau_g2[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // Was encrypted evaluation correct?
    assert_eq!(
        B_s_g2,
        g2.operate_with_self(B_s.evaluate(&tau).representative())
    );

    let C_s_temp_g1 = witness_vector
        .iter()
        .enumerate()
        .map(|(i, coeff)| o_tau_g1_temp[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // Was encrypted evaluation correct?
    assert_eq!(
        C_s_temp_g1,
        g1.operate_with_self(C_s.evaluate(&tau).representative())
    );

    let t_tau_h_tau_temp_g1 = h
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, coeff)| z_powers_of_tau_temp_g1[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // Was encrypted evaluation correct?
    assert_eq!(
        t_tau_h_tau_temp_g1,
        g1.operate_with_self((h.evaluate(&tau) * t.evaluate(&tau)).representative())
    );

    assert_eq!(
        Pairing::compute(&A_s_g1, &B_s_g2),
        Pairing::compute(&C_s_temp_g1.operate_with(&t_tau_h_tau_temp_g1), &g2)
    );

    ////////////////// introduce shifts

    let t_tau_h_tau_g1 = h
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, coeff)| z_powers_of_tau_g1[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // Did we construct t_tau_h_tau_g1 correctly?
    // Remove the shifting to make sure
    assert_eq!(
        Pairing::compute(&t_tau_h_tau_temp_g1, &g2),
        Pairing::compute(&t_tau_h_tau_g1, &verif_key.delta_g2)
    );

    let K_s_verifier_g1 = (0..number_of_public_vars)
        .map(|i| verifier_k_tau_g1[i].operate_with_self(witness_vector[i].representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    let K_s_prover_g1 = (number_of_public_vars..number_of_total_vars)
        .map(|i| {
            prover_k_tau_g1[i - number_of_public_vars]
                .operate_with_self(witness_vector[i].representative())
        })
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // Is K correctly constructed?
    assert_eq!(
        Pairing::compute(&K_s_verifier_g1, &verif_key.gamma_g2)
            * Pairing::compute(&K_s_prover_g1, &verif_key.delta_g2),
        Pairing::compute(
            &(A_s_g1
                .operate_with_self(beta.representative())
                .operate_with(&B_s_temp_g1.operate_with_self(alpha.representative()))
                .operate_with(&C_s_temp_g1)),
            &g2
        )
    );

    // Ultimate check
    assert_eq!(
        verif_key.alpha_g1_times_beta_g2
            * Pairing::compute(&K_s_verifier_g1, &verif_key.gamma_g2)
            * Pairing::compute(&K_s_prover_g1, &verif_key.delta_g2)
            * Pairing::compute(&t_tau_h_tau_g1, &verif_key.delta_g2),
        Pairing::compute(
            &A_s_g1.operate_with(&prov_key.alpha_g1),
            &B_s_g2.operate_with(&prov_key.beta_g2)
        ),
    );

    //////////////////

    //////////////////

    // // Zk(t) = delta^{-1} * t(tau) * tau^k
    // // [h(tau)t(tau)]_1 = h_i * [Zk(tau)]_1
    // let delta_inv_t_tau_h_tau_g1 = h
    //     .coefficients()
    //     .iter()
    //     .enumerate()
    //     .map(|(i, coeff)| {
    //         powers_of_tau_for_h_g1[i].operate_with_self(coeff.representative())
    //         // .operate_with_self((&delta_shift).representative())
    //     })
    //     .reduce(|acc, x| acc.operate_with(&x))
    //     .unwrap();

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Verify ////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    // // check alpha shift
    // assert_eq!(
    //     Pairing::compute(&p_evaluated_g1, &alpha_g2,),
    //     Pairing::compute(&p_evaluated_shifted_g1, &g2,)
    // );

    // // check computational integrity - polynomial divisibility
    // assert_eq!(
    //     Pairing::compute(&p_evaluated_g1, &g2),
    //     Pairing::compute(&t_evaluated_g1, &z_k_evaluated_g1)
    // );

    // let one = FrElement::new(U384 {
    //     limbs: [0, 0, 0, 0, 0, 1],
    // });
}
