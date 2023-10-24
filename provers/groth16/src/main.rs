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

/// Generates a test SRS for the BLS12381 curve
/// n is the number of constraints in the system.

// pub fn test_srs(n: usize) -> StructuredReferenceString<G1Point, G2Point> {
//     let s = FrElement::from(2);
//     let g1 = <BLS12381Curve as IsEllipticCurve>::generator();
//     let g2 = <BLS12381TwistCurve as IsEllipticCurve>::generator();

//     let powers_main_group: Vec<G1Point> = (0..n + 3)
//         .map(|exp| g1.operate_with_self(s.pow(exp as u64).representative()))
//         .collect();
//     let powers_secondary_group = [g2.clone(), g2.operate_with_self(s.representative())];

//     StructuredReferenceString::new(&powers_main_group, &powers_secondary_group)
// }

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
        A                   B                   C
        [0, 1, 0, 0, 0, 0]  [0, 1, 0, 0, 0, 0]  [0, 0, 0, 1, 0, 0]
        [0, 0, 0, 1, 0, 0]  [0, 1, 0, 0, 0, 0]  [0, 0, 0, 0, 1, 0]
        [0, 1, 0, 0, 1, 0]  [1, 0, 0, 0, 0, 0]  [0, 0, 0, 0, 0, 1]
        [5, 0, 0, 0, 0, 1]  [1, 0, 0, 0, 0, 0]  [0, 0, 1, 0, 0, 0]
    */
    // Will correspond to rows of QAP matrices
    let gate_indices = vec![
        FrElement::from_hex_unchecked("1"),
        FrElement::from_hex_unchecked("2"),
        FrElement::from_hex_unchecked("3"),
        FrElement::from_hex_unchecked("4"),
    ];
    // t(x) = (x-1)(x-2)(x-3)(x-4) = [24, -50, 35, -10, 1]
    let t_coefficients = vec![
        FrElement::from_hex_unchecked("0x18"),
        -FrElement::from_hex_unchecked("0x32"),
        FrElement::from_hex_unchecked("0x23"),
        -FrElement::from_hex_unchecked("0xa"),
        FrElement::from_hex_unchecked("0x1"),
    ];
    let t = Polynomial::new(&t_coefficients);

    let mut l_variable_polynomials = get_test_QAP_L(&gate_indices);
    let mut r_variable_polynomials = get_test_QAP_R(&gate_indices);
    let mut o_variable_polynomials = get_test_QAP_O(&gate_indices);

    // aka the secret assignments, w vector, s vector
    let witness_vector =
        ["0x1", "0x3", "0x23", "0x9", "0x1b", "0x1e"].map(|e| FrElement::from_hex_unchecked(e));

    // Assert input integrity
    {
        assert_eq!(l_variable_polynomials.len(), r_variable_polynomials.len());
        assert_eq!(r_variable_polynomials.len(), o_variable_polynomials.len());
        assert_eq!(witness_vector.len(), o_variable_polynomials.len());
    }

    let num_of_variables = l_variable_polynomials.len();
    let num_of_gates = l_variable_polynomials[0].degree() + 1;

    // Compute A[0].s, A[1].s,..., B[0].s, ..., C[0].s, C[1].s, ..., C[n].s
    for i in 0..num_of_variables {
        // A[i] *= s[i]
        l_variable_polynomials[i] = l_variable_polynomials[i].scale_coeffs(&witness_vector[i]);
        // B[i] *= s[i]
        r_variable_polynomials[i] = r_variable_polynomials[i].scale_coeffs(&witness_vector[i]);
        // C[i] *= s[i]
        o_variable_polynomials[i] = o_variable_polynomials[i].scale_coeffs(&witness_vector[i]);
    }
    // Now compute A.s by summing up polynomials A[0].s, A[1].s, ..., A[n].s
    // Similarly for B.s and C.s

    let mut l_dot_s_coeffs = vec![FrElement::from_hex_unchecked("0"); num_of_gates];
    let mut r_dot_s_coeffs = vec![FrElement::from_hex_unchecked("0"); num_of_gates];
    let mut o_dot_s_coeffs = vec![FrElement::from_hex_unchecked("0"); num_of_gates];
    for row in 0..num_of_gates {
        for col in 0..num_of_variables {
            let l_current_poly_coeffs = l_variable_polynomials[col].coefficients();
            if l_current_poly_coeffs.len() != 0 {
                l_dot_s_coeffs[row] += l_current_poly_coeffs[row].clone();
            }

            let r_current_poly_coeffs = r_variable_polynomials[col].coefficients();
            if r_current_poly_coeffs.len() != 0 {
                r_dot_s_coeffs[row] += r_current_poly_coeffs[row].clone();
            }

            let o_current_poly_coeffs = o_variable_polynomials[col].coefficients();
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

    // p(x) = A.s * B.s - C.s
    let p = l_dot_s * r_dot_s - o_dot_s;
    let p_degree = p.degree();
    // h(x) = (A.s * B.s - C.s) / t(x)
    let (h, remainder) = &p.clone().long_division_with_remainder(&t);
    // must have no remainder
    assert_eq!(0, remainder.degree());

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Setup /////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    // Config
    let mut rng = rand::thread_rng();

    let g1: G1Point = BLS12381Curve::generator();
    let g2: G2Point = BLS12381TwistCurve::generator();

    let tau: FrElement = FrElement::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    });

    let t_eval = t.evaluate(&tau);
    let encrypted_powers_of_tau: Vec<G1Point> = (0..p_degree + 1)
        .map(|exp| g1.operate_with_self(tau.pow(exp as u128).representative()))
        .collect();

    //////////////////////////////////////////////////////////////////////
    ////////////////////////////// Prove /////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    let p_evaluated_encrypted = p
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, coeff)| encrypted_powers_of_tau[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    let h_evaluated_encrypted = h
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, coeff)| encrypted_powers_of_tau[i].operate_with_self(coeff.representative()))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap();

    assert_eq!(
        p_evaluated_encrypted,
        h_evaluated_encrypted.operate_with_self(t_eval.representative()),
    );

    // let one = FrElement::new(U384 {
    //     limbs: [0, 0, 0, 0, 0, 1],
    // });
}
