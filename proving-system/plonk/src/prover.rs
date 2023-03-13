use crate::{
    config::{
        FrElement, G1Point, G2Point, KZG, MAXIMUM_DEGREE, ORDER_4_ROOT_UNITY,
        ORDER_R_MINUS_1_ROOT_UNITY,
    },
    setup::{Circuit, CommonPreprocessedInput, VerificationKey, Witness},
};
use lambdaworks_crypto::{
    commitments::kzg::{KateZaveruchaGoldberg, StructuredReferenceString},
    fiat_shamir::transcript::{self, Transcript},
};
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_math::{
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing, traits::IsPairing,
    },
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{
            IsMontgomeryConfiguration, MontgomeryBackendPrimeField, U256PrimeField,
        },
    },
    polynomial::{self, Polynomial},
    unsigned_integer::element::{UnsignedInteger, U256},
};

struct Proof {
    // Round 1
    a_1: G1Point, // [a(x)]₁ (commitment to left wire polynomial)
    b_1: G1Point, // [b(x)]₁ (commitment to right wire polynomial)
    c_1: G1Point, // [c(x)]₁ (commitment to output wire polynomial)

    // Round 2
    z_1: G1Point, // [z(x)]₁ (commitment to permutation polynomial)

    // Round 3
    t_lo_1: G1Point, // [t_lo(x)]₁ (commitment to t_lo(X), the low chunk of the quotient polynomial t(X))
    t_mid_1: G1Point, // [t_mid(x)]₁ (commitment to t_mid(X), the middle chunk of the quotient polynomial t(X))
    t_hi_1: G1Point, // [t_hi(x)]₁ (commitment to t_hi(X), the high chunk of the quotient polynomial t(X))

    // Round 4
    a_eval: FrElement,         // Evaluation of a(X) at evaluation challenge ζ
    b_eval: FrElement,         // Evaluation of b(X) at evaluation challenge ζ
    c_eval: FrElement,         // Evaluation of c(X) at evaluation challenge ζ
    s1_eval: FrElement, // Evaluation of the first permutation polynomial S_σ1(X) at evaluation challenge ζ
    s2_eval: FrElement, // Evaluation of the second permutation polynomial S_σ2(X) at evaluation challenge ζ
    z_shifted_eval: FrElement, // Evaluation of the shifted permutation polynomial z(X) at the shifted evaluation challenge ζω

    // Round 5
    W_z_1: G1Point,  // [W_ζ(X)]₁ (commitment to the opening proof polynomial)
    W_zw_1: G1Point, // [W_ζω(X)]₁ (commitment to the opening proof polynomial)
}

fn round_1(
    witness: &Witness,
    common_preprocesed_input: &CommonPreprocessedInput,
    kzg: &KZG,
) -> (
    G1Point,
    G1Point,
    G1Point,
    Polynomial<FrElement>,
    Polynomial<FrElement>,
    Polynomial<FrElement>,
) {
    let domain = &common_preprocesed_input.domain;

    let polynomial_a = Polynomial::interpolate(&domain, &witness.a);
    let polynomial_b = Polynomial::interpolate(&domain, &witness.b);
    let polynomial_c = Polynomial::interpolate(&domain, &witness.c);

    let a_1 = kzg.commit(&polynomial_a);
    let b_1 = kzg.commit(&polynomial_b);
    let c_1 = kzg.commit(&polynomial_c);

    (a_1, b_1, c_1, polynomial_a, polynomial_b, polynomial_c)
}

fn linearize_pair(
    witness_value: &FrElement,
    eta: &FrElement,
    beta: &FrElement,
    gamma: &FrElement,
) -> FrElement {
    witness_value + beta * eta + gamma
}

fn round_2(
    witness: &Witness,
    common_preprocesed_input: &CommonPreprocessedInput,
    kzg: &KZG,
    beta: &FrElement,
    gamma: &FrElement,
) -> (G1Point, Polynomial<FrElement>) {
    let mut coefficients: Vec<FrElement> = vec![FrElement::one()];
    let n = common_preprocesed_input.number_constraints;
    let domain = &common_preprocesed_input.domain;

    let S1 = &common_preprocesed_input.S1_lagrange;
    let S2 = &common_preprocesed_input.S2_lagrange;
    let S3 = &common_preprocesed_input.S3_lagrange;

    let k1 = ORDER_R_MINUS_1_ROOT_UNITY;
    let k2 = ORDER_R_MINUS_1_ROOT_UNITY * &k1;

    for i in 1..n {
        let a_i = &witness.a[i];
        let b_i = &witness.b[i];
        let c_i = &witness.c[i];
        let num = linearize_pair(&a_i, &domain[i], beta, gamma)
            * linearize_pair(&b_i, &(&domain[i] * &k1), beta, gamma)
            * linearize_pair(&c_i, &(&domain[i] * &k2), beta, gamma);
        let den = linearize_pair(&a_i, &S1[i], beta, gamma)
            * linearize_pair(&b_i, &S2[i], beta, gamma)
            * linearize_pair(&c_i, &S3[i], beta, gamma);
        let new_factor = num / den;
        let new_term = coefficients.last().unwrap() * &new_factor;
        coefficients.push(new_term);
    }

    let z_polynomial = Polynomial::interpolate(&common_preprocesed_input.domain, &coefficients);
    let z_1 = kzg.commit(&z_polynomial);
    (z_1, z_polynomial)
}

fn round_3(
    witness: &Witness,
    common_preprocesed_input: &CommonPreprocessedInput,
    kzg: &KZG,
    polynomial_a: &Polynomial<FrElement>,
    polynomial_b: &Polynomial<FrElement>,
    polynomial_c: &Polynomial<FrElement>,
    polynomial_z: &Polynomial<FrElement>,
    alpha: &FrElement,
    beta: &FrElement,
    gamma: &FrElement,
) -> (
    G1Point,
    G1Point,
    G1Point,
    Polynomial<FrElement>,
    Polynomial<FrElement>,
    Polynomial<FrElement>,
) {
    let a = polynomial_a;
    let b = polynomial_b;
    let c = polynomial_c;
    let n = common_preprocesed_input.number_constraints;
    let k1 = ORDER_R_MINUS_1_ROOT_UNITY;
    let k2 = ORDER_R_MINUS_1_ROOT_UNITY * &k1;
    let z = polynomial_z;

    let one = Polynomial::new_monomial(FieldElement::one(), 0);
    let domain = &common_preprocesed_input.domain;
    let Zh = Polynomial::new_monomial(FrElement::one(), n) - &one;
    let beta_x = Polynomial::new_monomial(beta.clone(), 1);
    let gamma_1 = Polynomial::new_monomial(gamma.clone(), 0);
    let beta_1 = Polynomial::new_monomial(beta.clone(), 0);
    let alpha_1 = Polynomial::new_monomial(alpha.clone(), 0);
    let beta_x_k1 = Polynomial::new_monomial(beta * k1, 1);
    let beta_x_k2 = Polynomial::new_monomial(beta * k2, 1);
    let z_x_w_coefficients: Vec<FrElement> = polynomial_z
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, x)| x * &domain[i])
        .collect();
    let z_x_w = Polynomial::new(&z_x_w_coefficients);
    let mut e1 = vec![FrElement::zero(); domain.len()];
    e1[0] = FrElement::one();
    let l1 = Polynomial::interpolate(&domain, &e1);

    let Qm = &common_preprocesed_input.Qm;
    let Ql = &common_preprocesed_input.Ql;
    let Qr = &common_preprocesed_input.Qr;
    let Qo = &common_preprocesed_input.Qo;
    let Qc = &common_preprocesed_input.Qc;
    let S1 = &common_preprocesed_input.S1_monomial;
    let S2 = &common_preprocesed_input.S2_monomial;
    let S3 = &common_preprocesed_input.S3_monomial;

    let p_constraints = (a * b * Qm + a * Ql + b * Qr + c * Qo + Qc);
    let p_permutation_1_1 =
        (a + beta_x + &gamma_1) * (b + beta_x_k1 + &gamma_1) * (c + beta_x_k2 + &gamma_1) * z;
    let p_permutation_1_2 = (a + &beta_1 * S1 + &gamma_1)
        * (b * &beta_1 * S2 + &gamma_1)
        * (c + beta_1 * S3 + gamma_1)
        * z_x_w;
    let p_permutation_2 = (z - one) * l1;
    let p = p_constraints
        + &alpha_1 * (p_permutation_1_1 - p_permutation_1_2)
        + &alpha_1 * &alpha_1 * p_permutation_2;

    let t = p / Zh;

    let t_lo = Polynomial::new(&t.coefficients[..n]);
    let t_mid = Polynomial::new(&t.coefficients[n..2 * n]);
    let t_hi = Polynomial::new(&t.coefficients[2 * n..]);

    let t_lo_1 = kzg.commit(&t_lo);
    let t_mid_1 = kzg.commit(&t_mid);
    let t_hi_1 = kzg.commit(&t_hi);

    (t_lo_1, t_mid_1, t_hi_1, t_lo, t_mid, t_hi)
}

fn round_4(
    common_preprocesed_input: &CommonPreprocessedInput,
    polynomial_a: &Polynomial<FrElement>,
    polynomial_b: &Polynomial<FrElement>,
    polynomial_c: &Polynomial<FrElement>,
    polynomial_z: &Polynomial<FrElement>,
    zeta: &FrElement,
) -> (FrElement, FrElement, FrElement, FrElement, FrElement, FrElement) {
    let omega = ORDER_4_ROOT_UNITY;
    let a_value = polynomial_a.evaluate(zeta);
    let b_value = polynomial_b.evaluate(zeta);
    let c_value = polynomial_c.evaluate(zeta);
    let s1_value = common_preprocesed_input.S1_monomial.evaluate(zeta);
    let s2_value = common_preprocesed_input.S2_monomial.evaluate(zeta);
    let z_value = polynomial_z.evaluate(&(zeta * omega));
    (a_value, b_value, c_value, s1_value, s2_value, z_value)
}

fn prove(
    circuit: &Circuit,
    common_preprocesed_input: &CommonPreprocessedInput,
    srs: &StructuredReferenceString<MAXIMUM_DEGREE, G1Point, G2Point>,
) {
    let mut transcript = Transcript::new();
    let kzg = KZG::new(srs.clone());
    let witness = circuit.get_witness();

    // Round 1
    let (a_1, b_1, c_1, polynomial_a, polynomial_b, polynomial_c) =
        round_1(&witness, &common_preprocesed_input, &kzg);
    transcript.append(&a_1.to_bytes_be());
    transcript.append(&b_1.to_bytes_be());
    transcript.append(&c_1.to_bytes_be());

    // Round 2
    // TODO: Handle error
    let beta = FrElement::from_bytes_be(&transcript.challenge()).unwrap();
    let gamma = FrElement::from_bytes_be(&transcript.challenge()).unwrap();

    let (z_1, polynomial_z) = round_2(&witness, &common_preprocesed_input, &kzg, &beta, &gamma);
    transcript.append(&z_1.to_bytes_be());

    // Round 3
    let alpha = FrElement::from_bytes_be(&transcript.challenge()).unwrap();
    let (t_lo_1, t_mid_1, t_hi_1, t_lo, t_mid, t_hi) = round_3(
        &witness,
        &common_preprocesed_input,
        &kzg,
        &polynomial_a,
        &polynomial_b,
        &polynomial_c,
        &polynomial_z,
        &alpha,
        &beta,
        &gamma,
    );
    transcript.append(&t_lo_1.to_bytes_be());
    transcript.append(&t_mid_1.to_bytes_be());
    transcript.append(&t_hi_1.to_bytes_be());

    // Round 4
    let zeta = FrElement::from_bytes_be(&transcript.challenge()).unwrap();
    let (a_value, b_value, c_value, s1_value, s2_value, z_value) = round_4(&common_preprocesed_input, &polynomial_a, &polynomial_b, &polynomial_c, &polynomial_z, &zeta);
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{test_circuit, test_srs};

    use super::*;

    #[test]
    fn test_round_1() {
        let test_circuit = test_circuit();
        let witness = test_circuit.get_witness();
        let common_preprocesed_input = CommonPreprocessedInput::for_this(&test_circuit);
        let srs = test_srs();
        let kzg = KZG::new(srs);
        round_1(&witness, &common_preprocesed_input, &kzg);
    }

    #[test]
    fn test_round_2() {
        let test_circuit = test_circuit();
        let witness = test_circuit.get_witness();
        let common_preprocesed_input = CommonPreprocessedInput::for_this(&test_circuit);
        let srs = test_srs();
        let kzg = KZG::new(srs);
        // TODO: put gnark challenges
        let beta = FrElement::one();
        let gamma = FrElement::one();
        round_2(&witness, &common_preprocesed_input, &kzg, &beta, &gamma);
    }
}
