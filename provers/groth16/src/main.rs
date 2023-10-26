// use groth16::prover::Groth16Prover;
// use groth16::setup::setup;
// use groth16::verifier;
// use verifier::Verifier;
use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::curves::bls12_381::{
        curve::BLS12381Curve,
        default_types::{FrElement, FrField},
        pairing::BLS12381AtePairing,
        twist::BLS12381TwistCurve,
    },
    elliptic_curve::traits::{IsEllipticCurve, IsPairing},
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
    unsigned_integer::element::U256,
};
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

struct QAP {
    pub num_of_public_inputs: usize,
    pub num_of_total_inputs: usize,
    pub num_of_gates: usize,
    pub l: Vec<Polynomial<FrElement>>,
    pub r: Vec<Polynomial<FrElement>>,
    pub o: Vec<Polynomial<FrElement>>,
    pub t: Polynomial<FrElement>,
}

impl QAP {
    fn new(
        num_of_public_inputs: usize,
        gate_indices: Vec<FrElement>,
        l: Vec<Vec<&str>>,
        r: Vec<Vec<&str>>,
        o: Vec<Vec<&str>>,
    ) -> Self {
        let l_len = l.len();
        let r_len = r.len();
        assert!(l_len == r_len && r_len == o.len() && num_of_public_inputs <= l_len);

        let mut target_poly = Polynomial::new(&[FrElement::one()]);
        for gate_index in &gate_indices {
            target_poly = target_poly * Polynomial::new(&[-gate_index, FieldElement::one()]);
        }

        Self {
            num_of_public_inputs,
            num_of_total_inputs: l_len,
            num_of_gates: gate_indices.len(),
            l: Self::build_test_variable_polynomial(&gate_indices, &l),
            r: Self::build_test_variable_polynomial(&gate_indices, &r),
            o: Self::build_test_variable_polynomial(&gate_indices, &o),
            t: target_poly,
        }
    }

    fn build_test_variable_polynomial(
        gate_indices: &Vec<FrElement>,
        from_matrix: &Vec<Vec<&str>>,
    ) -> Vec<Polynomial<FrElement>> {
        let mut polynomials = vec![];
        for i in 0..from_matrix.len() {
            let mut y_indices = vec![];
            for string in &from_matrix[i] {
                y_indices.push(FrElement::from_hex_unchecked(*string));
            }
            polynomials.push(Polynomial::interpolate(gate_indices, &y_indices).unwrap());
        }
        polynomials
    }
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
    */

    let qap = QAP::new(
        1,
        // TODO: Roots of unity
        ["0x1", "0x2", "0x3", "0x4"]
            .map(|e| FrElement::from_hex_unchecked(e))
            .to_vec(),
        [
            ["0", "0", "0", "5"],
            ["1", "0", "1", "0"],
            ["0", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "1", "0"],
            ["0", "0", "0", "1"],
        ]
        .map(|col| col.to_vec())
        .to_vec(),
        [
            ["0", "0", "1", "1"],
            ["1", "1", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
        ]
        .map(|col| col.to_vec())
        .to_vec(),
        [
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "1"],
            ["1", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "1", "0"],
        ]
        .map(|col| col.to_vec())
        .to_vec(),
    );

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Setup /////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    let g1: G1Point = BLS12381Curve::generator();
    let g2: G2Point = BLS12381TwistCurve::generator();

    let tw = ToxicWaste::default();

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
            g1.operate_with_self(
                (&t_tau_times_delta_inv * tw.tau.pow(exp as u128)).representative(),
            )
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
        z_powers_of_tau_g1: z_powers_of_tau_g1.clone(),
    };

    let vk = VerifyingKey {
        alpha_g1_times_beta_g2: Pairing::compute(&alpha_g1, &beta_g2),
        delta_g2,
        gamma_g2: g2.operate_with_self(tw.gamma.representative()),
        verifier_k_tau_g1,
    };

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Prove /////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    // aka the secret assignments, w vector, s vector
    // Includes the public inputs from the beginning.
    let w = ["0x1", "0x3", "0x23", "0x9", "0x1b", "0x1e"]
        .map(|e| FrElement::from_hex_unchecked(e))
        .to_vec();

    let h = calculate_h(&qap, &w);

    let t_tau_h_tau_assigned_g1 = h
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, coeff)| z_powers_of_tau_g1[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    let l_tau_assigned_g1 = w
        .iter()
        .enumerate()
        .map(|(i, coeff)| pk.l_tau_g1[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    let r_tau_assigned_g2 = w
        .iter()
        .enumerate()
        .map(|(i, coeff)| pk.r_tau_g2[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    // [γ^{-1} * (β*l(τ) + α*r(τ) + o(τ))]_1
    let k_tau_assigned_verifier_g1 = (0..qap.num_of_public_inputs)
        .map(|i| vk.verifier_k_tau_g1[i].operate_with_self(w[i].representative()))
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

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Verify ////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    /*
        SNARK verification without ZK

        (A.s + α) * (B.s + β) =
            αβ +
            δ * δ^{-1} * (
                t(τ)*h(τ) + β*A(τ) + α*B(τ) + C(τ)
            ) +
            γ * γ^{-1} * (
                β*A(τ) + α*B(τ) + C(τ)
            )
    */
    assert_eq!(
        vk.alpha_g1_times_beta_g2
            * Pairing::compute(&t_tau_h_tau_assigned_g1, &vk.delta_g2)
            * Pairing::compute(&k_tau_assigned_prover_g1, &vk.delta_g2)
            * Pairing::compute(&k_tau_assigned_verifier_g1, &vk.gamma_g2),
        Pairing::compute(
            &l_tau_assigned_g1.operate_with(&pk.alpha_g1),
            &r_tau_assigned_g2.operate_with(&pk.beta_g2)
        ),
    );
}
