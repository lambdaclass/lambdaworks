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

use rand::Rng; // 0.8.5

pub type Curve = BLS12381Curve;
pub type TwistedCurve = BLS12381TwistCurve;
pub type Pairing = BLS12381AtePairing;
pub type KZG = KateZaveruchaGoldberg<FrField, Pairing>;

pub type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
pub type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

fn get_test_QAP_L(gate_indices: &Vec<FrElement>) -> Vec<Polynomial<FrElement>> {
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

fn get_test_QAP_R(gate_indices: &Vec<FrElement>) -> Vec<Polynomial<FrElement>> {
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

fn get_test_QAP_O(gate_indices: &Vec<FrElement>) -> Vec<Polynomial<FrElement>> {
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
    */

    let number_of_public_vars = 1;
    let number_of_private_vars = 5;
    let number_of_total_vars = number_of_public_vars + number_of_private_vars;

    let number_of_gates = 4;

    let max_degree = number_of_total_vars + 1;

    // Gate indices. Will correspond to rows of QAP matrices. TODO: Roots of unity
    let gate_indices = vec![
        FrElement::from_hex_unchecked("1"),
        FrElement::from_hex_unchecked("2"),
        FrElement::from_hex_unchecked("3"),
        FrElement::from_hex_unchecked("4"),
    ];
    let mut l = get_test_QAP_L(&gate_indices);
    let mut r = get_test_QAP_R(&gate_indices);
    let mut o = get_test_QAP_O(&gate_indices);
    {
        assert_eq!(l.len(), r.len());
        assert_eq!(r.len(), o.len());
        assert_eq!(number_of_total_vars, o.len());
    }

    // t(x) = (x-1)(x-2)(x-3)(x-4) = [24, -50, 35, -10, 1]
    let t = Polynomial::new(&[
        FrElement::from_hex_unchecked("0x18"),
        -FrElement::from_hex_unchecked("0x32"),
        FrElement::from_hex_unchecked("0x23"),
        -FrElement::from_hex_unchecked("0xa"),
        FrElement::from_hex_unchecked("0x1"),
    ]);

    let num_of_variables = l.len();
    let num_of_gates = l[0].degree() + 1;

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Setup /////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    // Config
    let mut rng = rand::thread_rng();

    let g1: G1Point = BLS12381Curve::generator();
    let g2: G2Point = BLS12381TwistCurve::generator();

    // Point of evaluation. TODO: Fiat-Shamir
    let tau: FrElement = FrElement::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    });

    // [t(tau)]_1
    let t_evaluated_g1 = g1.operate_with_self(t.evaluate(&tau).representative());

    let alpha_shift = FrElement::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    });
    // [alpha]_2
    let alpha_g2 = g2.operate_with_self(alpha_shift.representative());

    // [tau]_1,[tau^2]_1,[tau^3]_1,...,[tau^n]_1
    let powers_of_tau_g1: Vec<G1Point> = (0..max_degree + 1)
        .map(|exp: usize| g1.operate_with_self(tau.pow(exp as u128).representative()))
        .collect();
    // [alpha*tau]_1,[alpha*tau^2]_1,[alpha*tau^3]_1,...,[alpha*tau^n]_1
    let shifted_powers_of_tau_g1: Vec<G1Point> = (0..max_degree + 1)
        .map(|exp| {
            g1.operate_with_self(tau.pow(exp as u128).representative())
                .operate_with_self((&alpha_shift).representative())
        })
        .collect();

    // [tau]_2,[tau^2]_2,[tau^3]_2,...,[tau^n]_2
    let powers_of_tau_g2: Vec<G2Point> = (0..max_degree + 1)
        .map(|exp| g2.operate_with_self(tau.pow(exp as u128).representative()))
        .collect();

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Prove /////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    // aka the secret assignments, w vector, s vector
    let witness_vector =
        ["0x1", "0x3", "0x23", "0x9", "0x1b", "0x1e"].map(|e| FrElement::from_hex_unchecked(e));

    // Compute A[0].s, A[1].s,..., B[0].s, ..., C[0].s, C[1].s, ..., C[n].s
    for i in 0..num_of_variables {
        // A[i] *= s[i]
        l[i] = l[i].scale_coeffs(&witness_vector[i]);
        // B[i] *= s[i]
        r[i] = r[i].scale_coeffs(&witness_vector[i]);
        // C[i] *= s[i]
        o[i] = o[i].scale_coeffs(&witness_vector[i]);
    }

    // Now compute A.s by summing up polynomials A[0].s, A[1].s, ..., A[n].s
    // Similarly for B.s and C.s
    let mut l_dot_s_coeffs = vec![FrElement::from_hex_unchecked("0"); num_of_gates];
    let mut r_dot_s_coeffs = vec![FrElement::from_hex_unchecked("0"); num_of_gates];
    let mut o_dot_s_coeffs = vec![FrElement::from_hex_unchecked("0"); num_of_gates];
    for row in 0..num_of_gates {
        for col in 0..num_of_variables {
            let l_current_poly_coeffs = l[col].coefficients();
            if l_current_poly_coeffs.len() != 0 {
                l_dot_s_coeffs[row] += l_current_poly_coeffs[row].clone();
            }

            let r_current_poly_coeffs = r[col].coefficients();
            if r_current_poly_coeffs.len() != 0 {
                r_dot_s_coeffs[row] += r_current_poly_coeffs[row].clone();
            }

            let o_current_poly_coeffs = o[col].coefficients();
            if o_current_poly_coeffs.len() != 0 {
                o_dot_s_coeffs[row] += o_current_poly_coeffs[row].clone();
            }
        }
    }
    let l_dot_s = Polynomial::new(&l_dot_s_coeffs);
    let r_dot_s = Polynomial::new(&r_dot_s_coeffs);
    let o_dot_s = Polynomial::new(&o_dot_s_coeffs);
    // Assert correctness of assignments
    {
        assert_eq!(
            l_dot_s.evaluate(&gate_indices[0]) * r_dot_s.evaluate(&gate_indices[0]),
            o_dot_s.evaluate(&gate_indices[0])
        );
        assert_eq!(
            l_dot_s.evaluate(&gate_indices[1]) * r_dot_s.evaluate(&gate_indices[1]),
            o_dot_s.evaluate(&gate_indices[1])
        );
        assert_eq!(
            l_dot_s.evaluate(&gate_indices[2]) * r_dot_s.evaluate(&gate_indices[2]),
            o_dot_s.evaluate(&gate_indices[2])
        );
        assert_eq!(
            l_dot_s.evaluate(&gate_indices[3]) * r_dot_s.evaluate(&gate_indices[3]),
            o_dot_s.evaluate(&gate_indices[3])
        );
    }

    // Sample delta for zk-ness
    // let delta_shift = FrElement::new(U256 {
    //     limbs: [
    //         rng.gen::<u64>(),
    //         rng.gen::<u64>(),
    //         rng.gen::<u64>(),
    //         rng.gen::<u64>(),
    //     ],
    // });

    // p(x) = A.s * B.s - C.s
    let p = &l_dot_s * &r_dot_s - &o_dot_s;
    // h(x) = p(x) / t(x) = (A.s * B.s - C.s) / t(x)
    let (h, remainder) = &p.clone().long_division_with_remainder(&t);
    // must have no remainder
    assert_eq!(0, remainder.degree());

    let p_evaluated_g1 = p
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, coeff)| {
            powers_of_tau_g1[i].operate_with_self(coeff.representative())
            // .operate_with_self((&delta_shift).representative())
        })
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    let p_evaluated_shifted_g1 = p
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, coeff)| {
            shifted_powers_of_tau_g1[i].operate_with_self(coeff.representative())
            // .operate_with_self((&delta_shift).representative())
        })
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    let h_evaluated_g2 = h
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, coeff)| {
            powers_of_tau_g2[i].operate_with_self(coeff.representative())
            // .operate_with_self((&delta_shift).representative())
        })
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Verify ////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    // check alpha shift
    assert_eq!(
        Pairing::compute(&p_evaluated_g1, &alpha_g2,),
        Pairing::compute(&p_evaluated_shifted_g1, &g2,)
    );

    // check computational integrity - polynomial divisibility
    assert_eq!(
        Pairing::compute(&p_evaluated_g1, &g2),
        Pairing::compute(&t_evaluated_g1, &h_evaluated_g2)
    );

    // let one = FrElement::new(U384 {
    //     limbs: [0, 0, 0, 0, 0, 1],
    // });
}
