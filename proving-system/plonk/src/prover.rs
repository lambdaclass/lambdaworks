use crate::{
    config::{
        FrElement, G1Point, G2Point, KZG, MAXIMUM_DEGREE, ORDER_4_ROOT_UNITY,
        ORDER_R_MINUS_1_ROOT_UNITY,
    },
    setup::{Circuit, CommonPreprocessedInput, Witness},
};
use lambdaworks_crypto::{
    commitments::kzg::StructuredReferenceString, fiat_shamir::transcript::Transcript,
};
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_math::{field::element::FieldElement, polynomial::Polynomial};

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
    let k2 = &k1 * &k1;

    for i in 0..n - 1 {
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
    let z_x_omega_coefficients: Vec<FrElement> = polynomial_z
        .coefficients()
        .iter()
        .enumerate()
        .map(|(i, x)| x * &domain[i])
        .collect();
    let z_x_omega = Polynomial::new(&z_x_omega_coefficients);
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

    let p_constraints = a * b * Qm + a * Ql + b * Qr + c * Qo + Qc;
    let f = (a + beta_x + &gamma_1) * (b + beta_x_k1 + &gamma_1) * (c + beta_x_k2 + &gamma_1);
    let g = (a + &beta_1 * S1 + &gamma_1) * (b + &beta_1 * S2 + &gamma_1) * (c + beta_1 * S3 + gamma_1);
    let p_permutation_1 = g * z_x_omega - f * z; // TODO: Paper says this term is minus the term found on gnark. This doesn't affect the protocol.
    let p_permutation_2 = (z - one) * &l1;

    let p = ((p_permutation_2 * &alpha_1) + p_permutation_1) * alpha_1 + p_constraints;

    let mut t = p / Zh;

    // (*) TODO: This is written this way to cross validate with `gnark`.
    // But it's different from what the paper describes
    // https://eprint.iacr.org/2019/953.pdf page 29
    // In particular, this way `t` is different from
    // `t_lo + t_mid * X^n + t_hi * X^{2n} (see TODO (**) below)
    Polynomial::pad_with_zero_coefficients_to_length(&mut t, 3 * (n + 2));
    let t_lo = Polynomial::new(&t.coefficients[..n + 2]);
    let t_mid = Polynomial::new(&t.coefficients[n + 2..2 * (n + 2)]);
    let t_hi = Polynomial::new(&t.coefficients[2 * (n + 2)..3 * (n + 2)]);

    // // (**) TODO: Shouldn't this pass? Failing now
    // let xn = Polynomial::new_monomial(FrElement::one(), n);
    // let t_expected = &t_lo + &t_mid * &xn + &t_hi * &xn * &xn;
    // let t = Polynomial::new(&t.coefficients); // trim zeroes
    // assert_eq!(t, t_expected);

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
) -> (
    FrElement,
    FrElement,
    FrElement,
    FrElement,
    FrElement,
    FrElement,
) {
    let omega = ORDER_4_ROOT_UNITY;
    let a_value = polynomial_a.evaluate(zeta);
    let b_value = polynomial_b.evaluate(zeta);
    let c_value = polynomial_c.evaluate(zeta);
    let s1_value = common_preprocesed_input.S1_monomial.evaluate(zeta);
    let s2_value = common_preprocesed_input.S2_monomial.evaluate(zeta);
    let z_value = polynomial_z.evaluate(&(zeta * omega));
    (a_value, b_value, c_value, s1_value, s2_value, z_value)
}

fn round_5(
    common_preprocesed_input: &CommonPreprocessedInput,
    kzg: &KZG,
    polynomial_a: &Polynomial<FrElement>,
    polynomial_b: &Polynomial<FrElement>,
    polynomial_c: &Polynomial<FrElement>,
    polynomial_z: &Polynomial<FrElement>,
    t_lo: &Polynomial<FrElement>,
    t_mid: &Polynomial<FrElement>,
    t_hi: &Polynomial<FrElement>,
    alpha: &FrElement,
    beta: &FrElement,
    gamma: &FrElement,
    zeta: &FrElement,
    upsilon: &FrElement,
    a_value: &FrElement,
    b_value: &FrElement,
    c_value: &FrElement,
    s1_value: &FrElement,
    s2_value: &FrElement,
    z_value: &FrElement,
) -> (G1Point, G1Point) {
    // dbg!(&a_value, &b_value, &c_value, &s1_value, &s2_value, &z_value);

    let n = common_preprocesed_input.number_constraints; // TODO: Is this the correct value?

    let a_value_1 = Polynomial::new_monomial(a_value.clone(), 0);
    let b_value_1 = Polynomial::new_monomial(b_value.clone(), 0);
    let c_value_1 = Polynomial::new_monomial(c_value.clone(), 0);
    let s1_value_1 = Polynomial::new_monomial(s1_value.clone(), 0);
    let s2_value_1 = Polynomial::new_monomial(s2_value.clone(), 0);
    let z_omega_value_1 = Polynomial::new_monomial(z_value.clone(), 0);
    
    let domain = &common_preprocesed_input.domain;
    let mut e1 = vec![FrElement::zero(); domain.len()];
    e1[0] = FrElement::one();
    let l1 = Polynomial::interpolate(&domain, &e1);

    let z = polynomial_z;
  
    let k1 = ORDER_R_MINUS_1_ROOT_UNITY;
    let k2 = &k1 * &k1;
    let k1_1 = Polynomial::new_monomial(k1, 0);
    let k2_1 = Polynomial::new_monomial(k2, 0);
    let one = Polynomial::new_monomial(FrElement::one(), 0);

    let zeta_1 = Polynomial::new_monomial(zeta.clone(), 0);
    let alpha_1 = Polynomial::new_monomial(alpha.clone(), 0);
    let beta_1 = Polynomial::new_monomial(beta.clone(), 0);
    let gamma_1 = Polynomial::new_monomial(gamma.clone(), 0);

    let Zh = Polynomial::new_monomial(FrElement::one(), n) - &one;

    let Qm = &common_preprocesed_input.Qm;
    let Ql = &common_preprocesed_input.Ql;
    let Qr = &common_preprocesed_input.Qr;
    let Qo = &common_preprocesed_input.Qo;
    let Qc = &common_preprocesed_input.Qc;
    let S1 = &common_preprocesed_input.S1_monomial;
    let S2 = &common_preprocesed_input.S2_monomial;
    let S3 = &common_preprocesed_input.S3_monomial;

    let zeta_raised_n = Polynomial::new_monomial(zeta.pow(n + 2), 0); // TODO: Paper says n and 2n, but Gnark uses n+2 and 2n+4 (see the TODO(*))
    let zeta_raised_2n = Polynomial::new_monomial(zeta.pow(2 * n + 4), 0);
    dbg!((&a_value_1 + &beta_1 * &s1_value_1 + &gamma_1) * (&b_value_1 + &beta_1 * &s2_value_1 + &gamma_1) *  &z_omega_value_1 );

    dbg!(
        (z.clone() * l1.evaluate(zeta)) * &alpha_1 * &alpha_1 +
        ((&a_value_1 + &beta_1 * &s1_value_1 + &gamma_1) * (&b_value_1 + &beta_1 * &s2_value_1 + &gamma_1) *  &z_omega_value_1 * S3 - z * (&a_value_1 + &beta_1 * &zeta_1 + &gamma_1) * (&b_value_1 + &beta_1 * &k1_1 * &zeta_1 + &gamma_1) * (&c_value_1 + &beta_1 * &k2_1 * &zeta_1 + &gamma_1)) * &alpha_1 +
        &a_value_1 * &b_value_1 * Qm + &a_value_1 * Ql + &b_value_1 * Qr + &c_value_1 * Qo + Qc
    );
    let r_1 = &a_value_1 * &b_value_1 * Qm + &a_value_1 * Ql + &b_value_1 * Qr + &c_value_1 * Qo + Qc; // TODO paper says PI(z), but GNark does this.
    let r_2_1 = (&a_value_1 + &beta_1 * &zeta_1 + &gamma_1) * (&b_value_1 + &beta_1 * k1_1 * &zeta_1 + &gamma_1) * (&c_value_1 + &beta_1 * &k2_1 * &zeta_1 + &gamma_1) * polynomial_z;
    let r_2_2 = (&a_value_1 + &beta_1 * &s1_value_1 + &gamma_1) * (&b_value_1 + &beta_1 * &s2_value_1 + &gamma_1) * (&c_value_1 + beta_1 * S3 + gamma_1) * &z_omega_value_1;
    let r_3 = (z - &one) * l1.evaluate(zeta);
    let r_4 = (t_lo + zeta_raised_n * t_mid + zeta_raised_2n * t_hi) * Zh.evaluate(zeta);
    dbg!(&r_1 + &alpha_1 * (-&r_2_1 + &r_2_2) + &alpha_1 * &alpha_1  * &r_3);
    let r = r_1 + &alpha_1 * (-r_2_1 + r_2_2) + &alpha_1 * &alpha_1  * r_3 - r_4; // TODO: Paper says second term is minus the term found on gnark. This doesn't affect the protocol.


    let w_zeta_den = Polynomial::new(&[-zeta, FrElement::one()]);
    let w_zeta_num = r + (polynomial_a - a_value_1) * upsilon.clone() + (polynomial_b - b_value_1) * upsilon.pow(2_u64) + (polynomial_c - c_value_1) * upsilon.pow(3_u64) + (S1 - s1_value_1) * upsilon.pow(4_u64) + (S2 - s2_value_1) * upsilon.pow(5_u64);
    //let w_zeta = w_zeta_num / w_zeta_den;
    let (w_zeta, remainder) = w_zeta_num.long_division_with_remainder(&w_zeta_den);
    assert_eq!(remainder, Polynomial::zero(), "w_zeta_den does not divide w_zeta_num");

    let w_zeta_omega_num = polynomial_z - &z_omega_value_1;
    let w_zeta_omega_den = Polynomial::new(&[-zeta * ORDER_4_ROOT_UNITY, FrElement::one()]);
    let (w_zeta_omega, remainder) = w_zeta_omega_num.long_division_with_remainder(&w_zeta_omega_den);
    assert_eq!(remainder, Polynomial::zero());

    let w_zeta_1 = kzg.commit(&w_zeta);
    let w_zeta_omega_1 = kzg.commit(&w_zeta_omega);

    (w_zeta_1, w_zeta_omega_1)
}

fn prove(
    circuit: &Circuit,
    common_preprocesed_input: &CommonPreprocessedInput,
    srs: &StructuredReferenceString<MAXIMUM_DEGREE, G1Point, G2Point>,
) {
    // TODO: use strong Fiat-Shamir (e.g.: add public inputs and statement)
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
    let (a_value, b_value, c_value, s1_value, s2_value, z_value) = round_4(
        &common_preprocesed_input,
        &polynomial_a,
        &polynomial_b,
        &polynomial_c,
        &polynomial_z,
        &zeta,
    );

    // TODO: Should we append something to the transcript here? This step does not return any commitment.
    // But the next step receives a new challenge `upsilon`

    // Round 5
    let upsilon = FrElement::from_bytes_be(&transcript.challenge()).unwrap();
    let (w_zeta_1, w_zeta_omega_1) = round_5(
        &common_preprocesed_input,
        &kzg,
        &polynomial_a,
        &polynomial_b,
        &polynomial_c,
        &polynomial_z,
        &t_lo,
        &t_mid,
        &t_hi,
        &alpha,
        &beta,
        &gamma,
        &zeta,
        &upsilon,
        &a_value,
        &b_value,
        &c_value,
        &s1_value,
        &s2_value,
        &z_value,
    );
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bls12_381::curve::BLS12381Curve, point::ShortWeierstrassProjectivePoint,
            },
            traits::IsEllipticCurve,
        },
    };

    use crate::{
        config::FpElement,
        test_utils::{test_circuit, test_srs},
    };

    use super::*;

    fn alpha() -> FrElement {
        FrElement::from_hex("583cfb0df2ef98f2131d717bc6aadd571c5302597c135cab7c00435817bf6e50")
    }

    fn beta() -> FrElement {
        FrElement::from_hex("bdda7414bdf5bf42b77cbb3af4a82f32ec7622dd6c71575bede021e6e4609d4")
    }

    fn gamma() -> FrElement {
        FrElement::from_hex("58f6690d9b36e62e4a0aef27612819288df2a3ff5bf01597cf06779503f51583")
    }

    fn zeta() -> FrElement {
        FrElement::from_hex("2a4040abb941ee5e2a42602a7a60d282a430a4cf099fa3bb0ba8f4da628ec59a")
    }

    fn upsilon() -> FrElement {
        // TODO: put the correct value
        FrElement::from_hex("1")
    }

    #[test]
    fn test_round_1() {
        let test_circuit = test_circuit();
        let witness = test_circuit.get_witness();
        let common_preprocesed_input = CommonPreprocessedInput::for_this(&test_circuit);
        let srs = test_srs();
        let kzg = KZG::new(srs);
        let (a_1, b_1, c_1, _, _, _) = round_1(&witness, &common_preprocesed_input, &kzg);
        let a_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"),
            FpElement::from_hex("114d1d6855d545a8aa7d76c8cf2e21f267816aef1db507c96655b9d5caac42364e6f38ba0ecb751bad54dcd6b939c2ca"),
        ).unwrap();
        let b_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("44ed7c3ed015c6a39c350cd06d03b48d3e1f5eaf7a256c5b6203886e6e78cd9b76623d163da4dfb0f2491e7cc06408"),
            FpElement::from_hex("14c4464d2556fdfdc8e31068ef8d953608e511569a236c825f2ddab4fe04af03aba29e38b9b2b6221243124d235f4c67"),
        ).unwrap();
        let c_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("7726dc031bd26122395153ca428d5e6dea0a64c1f9b3b1bb2f2508a5eb6ea0ea0363294fad3160858bc87e46d3422fd"),
            FpElement::from_hex("8db0c15bfd77df7fe66284c3b04e6043eaba99ef6a845d4f7255fd0da95f2fb8e474df2e7f8e1a38829f7a9612a9b87"),
        ).unwrap();
        assert_eq!(a_1, a_1_expected);
        assert_eq!(b_1, b_1_expected);
        assert_eq!(c_1, c_1_expected);
    }

    #[test]
    fn test_round_2() {
        let test_circuit = test_circuit();
        let witness = test_circuit.get_witness();
        let common_preprocesed_input = CommonPreprocessedInput::for_this(&test_circuit);
        let srs = test_srs();
        let kzg = KZG::new(srs);

        let (z_1, _) = round_2(&witness, &common_preprocesed_input, &kzg, &beta(), &gamma());
        let z_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("3e8322968c3496cf1b5786d4d71d158a646ec90c14edf04e758038e1f88dcdfe8443fcecbb75f3074a872a380391742"),
            FpElement::from_hex("11eac40d09796ff150004e7b858d83ddd9fe995dced0b3fbd7535d6e361729b25d488799da61fdf1d7b5022684053327"),
        ).unwrap();
        assert_eq!(z_1, z_1_expected);
    }

    #[test]
    fn test_round_3() {
        // This test is subject to TODO (*) above.
        let test_circuit = test_circuit();
        let witness = test_circuit.get_witness();
        let common_preprocesed_input = CommonPreprocessedInput::for_this(&test_circuit);
        let srs = test_srs();
        let kzg = KZG::new(srs);

        let (_, _, _, polynomial_a, polynomial_b, polynomial_c) =
            round_1(&witness, &common_preprocesed_input, &kzg);
        let (_, polynomial_z) = round_2(&witness, &common_preprocesed_input, &kzg, &beta(), &gamma());
        let (t_lo_1, t_mid_1, t_hi_1, _, _, _) = round_3(
            &witness,
            &common_preprocesed_input,
            &kzg,
            &polynomial_a,
            &polynomial_b,
            &polynomial_c,
            &polynomial_z,
            &alpha(),
            &beta(),
            &gamma(),
        );

        let t_lo_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("9f511a769e77e87537b0749d65f467532fbf0f9dc1bcc912c333741be9d0a613f61e5fe595996964646ce30794701e5"),
            FpElement::from_hex("89fd6bb571323912210517237d6121144fc01ba2756f47c12c9cc94fc9197313867d68530f152dc8d447f10fcf75a6c"),
        ).unwrap();
        let t_mid_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("f96d8a93f3f5be2ab2819891f41c9f883cacea63da423e6ed1701765fcd659fc11e056a48c554f5df3a9c6603d48ca8"),
            FpElement::from_hex("14fa74fa049b7276007b739f3b8cfeac09e8cfabd4f858b6b99798c81124c34851960bebda90133cb03c981c08c8b6d3"),
        ).unwrap();
        let t_hi_1_expected = ShortWeierstrassProjectivePoint::<BLS12381Curve>::neutral_element();

        assert_eq!(t_lo_1, t_lo_1_expected);
        assert_eq!(t_mid_1, t_mid_1_expected);
        assert_eq!(t_hi_1, t_hi_1_expected);
    }

    #[test]
    fn test_round_4() {
        let test_circuit = test_circuit();
        let witness = test_circuit.get_witness();
        let common_preprocesed_input = CommonPreprocessedInput::for_this(&test_circuit);
        let srs = test_srs();
        let kzg = KZG::new(srs);

        let (_, _, _, polynomial_a, polynomial_b, polynomial_c) =
            round_1(&witness, &common_preprocesed_input, &kzg);
        let (_, polynomial_z) = round_2(&witness, &common_preprocesed_input, &kzg, &beta(), &gamma());

        let (a_value, b_value, c_value, s1_value, s2_value, z_value) = round_4(
            &common_preprocesed_input,
            &polynomial_a,
            &polynomial_b,
            &polynomial_c,
            &polynomial_z,
            &zeta(),
        );
        let expected_a_value = FrElement::from_hex("2c090a95b57f1f493b7b747bba34fef7772fd72f97d718ed69549641a823eb2e");
        let expected_b_value = FrElement::from_hex("5975959d91369ba4e7a03c6ae94b7fe98e8b61b7bf9af63c8ae0759e17ac0c7e");
        let expected_c_value = FrElement::from_hex("6bf31edeb4344b7d2df2cb1bd40b4d13e182d9cb09f89591fa043c1a34b4a93");
        let expected_z_value = FrElement::from_hex("38e2ec8e7c3dab29e2b8e9c8ea152914b8fe4612e91f2902c80238efcf21f4ee");
        let expected_s1_value = FrElement::from_hex("472f66db4fb6947d9ed9808241fe82324bc08aa2a54be93179db8e564e1137d4");
        let expected_s2_value = FrElement::from_hex("5588f1239c24efe0538868d0f716984e69c6980e586864f615e4b0621fdc6f81");

        assert_eq!(a_value, expected_a_value);
        assert_eq!(b_value, expected_b_value);
        assert_eq!(c_value, expected_c_value);
        assert_eq!(z_value, expected_z_value);
        assert_eq!(s1_value, expected_s1_value);
        assert_eq!(s2_value, expected_s2_value);
    }

    #[test]
    fn test_round_5() {
        let test_circuit = test_circuit();
        let witness = test_circuit.get_witness();
        let common_preprocesed_input = CommonPreprocessedInput::for_this(&test_circuit);
        let srs = test_srs();
        let kzg = KZG::new(srs);

        let (_, _, _, polynomial_a, polynomial_b, polynomial_c) =
            round_1(&witness, &common_preprocesed_input, &kzg);
        let (_, polynomial_z) = round_2(&witness, &common_preprocesed_input, &kzg, &beta(), &gamma());

        let (_, _, _, t_lo, t_mid, t_hi) = round_3(
            &witness,
            &common_preprocesed_input,
            &kzg,
            &polynomial_a,
            &polynomial_b,
            &polynomial_c,
            &polynomial_z,
            &alpha(),
            &beta(),
            &gamma(),
        );

        let (a_value, b_value, c_value, s1_value, s2_value, z_value) = round_4(
            &common_preprocesed_input,
            &polynomial_a,
            &polynomial_b,
            &polynomial_c,
            &polynomial_z,
            &zeta(),
        );

        let expected_w_zeta_1 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("fa6250b80a418f0548b132ac264ff9915b2076c0c2548da9316ae19ffa35bbcf905d9f02f9274739608045ef83a4757"),
            FpElement::from_hex("17713ade2dbd66e923d4092a5d2da98202959dd65a15e9f7791fab3c0dd08788aa9b4a1cb21d04e0c43bd29225472145"),
        ).unwrap();
        //let expected_w_zeta_omega_1 = BLS12381Curve::create_point_from_affine(
        //    FpElement::from_hex(""),
        //    FpElement::from_hex(""),
        //).unwrap();

        let (w_zeta_1, w_zeta_omega_1) = round_5(
            &common_preprocesed_input,
            &kzg,
            &polynomial_a,
            &polynomial_b,
            &polynomial_c,
            &polynomial_z,
            &t_lo,
            &t_mid,
            &t_hi,
            &alpha(),
            &beta(),
            &gamma(),
            &zeta(),
            &upsilon(),
            &a_value,
            &b_value,
            &c_value,
            &s1_value,
            &s2_value,
            &z_value,
        );
        assert_eq!(w_zeta_1, expected_w_zeta_1);
    }
}
