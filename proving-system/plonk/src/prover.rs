use std::marker::PhantomData;

use crate::setup::{CommonPreprocessedInput, Witness};
use lambdaworks_crypto::{
    commitments::traits::IsCommitmentScheme, fiat_shamir::transcript::Transcript,
};
use lambdaworks_math::{field::element::FieldElement, polynomial::Polynomial};
use lambdaworks_math::{field::traits::IsField, traits::ByteConversion};

#[allow(unused)]
pub struct Proof<F: IsField, CS: IsCommitmentScheme<F>> {
    // Round 1
    pub a_1: CS::Commitment, // [a(x)]₁ (commitment to left wire polynomial)
    pub b_1: CS::Commitment, // [b(x)]₁ (commitment to right wire polynomial)
    pub c_1: CS::Commitment, // [c(x)]₁ (commitment to output wire polynomial)

    // Round 2
    pub z_1: CS::Commitment, // [z(x)]₁ (commitment to permutation polynomial)

    // Round 3
    pub t_lo_1: CS::Commitment, // [t_lo(x)]₁ (commitment to t_lo(X), the low chunk of the quotient polynomial t(X))
    pub t_mid_1: CS::Commitment, // [t_mid(x)]₁ (commitment to t_mid(X), the middle chunk of the quotient polynomial t(X))
    pub t_hi_1: CS::Commitment, // [t_hi(x)]₁ (commitment to t_hi(X), the high chunk of the quotient polynomial t(X))

    // Round 4
    pub a_zeta: FieldElement<F>, // Evaluation of a(X) at evaluation challenge ζ
    pub b_zeta: FieldElement<F>, // Evaluation of b(X) at evaluation challenge ζ
    pub c_zeta: FieldElement<F>, // Evaluation of c(X) at evaluation challenge ζ
    pub s1_zeta: FieldElement<F>, // Evaluation of the first permutation polynomial S_σ1(X) at evaluation challenge ζ
    pub s2_zeta: FieldElement<F>, // Evaluation of the second permutation polynomial S_σ2(X) at evaluation challenge ζ
    pub z_zeta_omega: FieldElement<F>, // Evaluation of the shifted permutation polynomial z(X) at the shifted evaluation challenge ζω

    // Round 5
    pub w_zeta_1: CS::Commitment, // [W_ζ(X)]₁ (commitment to the opening proof polynomial)
    pub w_zeta_omega_1: CS::Commitment, // [W_ζω(X)]₁ (commitment to the opening proof polynomial)
    pub p_non_constant_zeta: FieldElement<F>,
    pub t_zeta: FieldElement<F>,
}

pub struct Prover<F: IsField, CS: IsCommitmentScheme<F>> {
    commitment_scheme: CS,
    phantom: PhantomData<F>,
}

struct Round1Result<F: IsField, Hiding> {
    a_1: Hiding,
    b_1: Hiding,
    c_1: Hiding,
    p_a: Polynomial<FieldElement<F>>,
    p_b: Polynomial<FieldElement<F>>,
    p_c: Polynomial<FieldElement<F>>,
}

struct Round2Result<F: IsField, Hiding> {
    z_1: Hiding,
    p_z: Polynomial<FieldElement<F>>,
    beta: FieldElement<F>,
    gamma: FieldElement<F>,
}

struct Round3Result<F: IsField, Hiding> {
    t_lo_1: Hiding,
    t_mid_1: Hiding,
    t_hi_1: Hiding,
    p_t_lo: Polynomial<FieldElement<F>>,
    p_t_mid: Polynomial<FieldElement<F>>,
    p_t_hi: Polynomial<FieldElement<F>>,
    alpha: FieldElement<F>,
}

struct Round4Result<F: IsField> {
    a_zeta: FieldElement<F>,
    b_zeta: FieldElement<F>,
    c_zeta: FieldElement<F>,
    s1_zeta: FieldElement<F>,
    s2_zeta: FieldElement<F>,
    z_zeta_omega: FieldElement<F>,
    zeta: FieldElement<F>,
}

struct Round5Result<F: IsField, Hiding> {
    w_zeta_1: Hiding,
    w_zeta_omega_1: Hiding,
    p_non_constant_zeta: FieldElement<F>,
    t_zeta: FieldElement<F>,
}

impl<F, CS> Prover<F, CS>
where
    F: IsField,
    CS: IsCommitmentScheme<F>,
    FieldElement<F>: ByteConversion,
    CS::Commitment: ByteConversion,
{
    #[allow(unused)]
    pub fn new(commitment_scheme: CS) -> Self {
        Self {
            commitment_scheme,
            phantom: PhantomData,
        }
    }

    fn round_1(
        &self,
        witness: &Witness<F>,
        common_preprocesed_input: &CommonPreprocessedInput<F>,
    ) -> Round1Result<F, CS::Commitment> {
        let domain = &common_preprocesed_input.domain;

        let p_a = Polynomial::interpolate(domain, &witness.a);
        let p_b = Polynomial::interpolate(domain, &witness.b);
        let p_c = Polynomial::interpolate(domain, &witness.c);

        let a_1 = self.commitment_scheme.commit(&p_a);
        let b_1 = self.commitment_scheme.commit(&p_b);
        let c_1 = self.commitment_scheme.commit(&p_c);

        Round1Result {
            a_1,
            b_1,
            c_1,
            p_a,
            p_b,
            p_c,
        }
    }

    fn round_2(
        &self,
        witness: &Witness<F>,
        CommonPreprocessedInput {
            n,
            domain,
            s1_lagrange,
            s2_lagrange,
            s3_lagrange,
            k1,
            ..
        }: &CommonPreprocessedInput<F>,
        beta: &FieldElement<F>,
        gamma: &FieldElement<F>,
    ) -> Round2Result<F, CS::Commitment> {
        let mut coefficients: Vec<FieldElement<F>> = vec![FieldElement::one()];
        let (s1, s2, s3) = (&s1_lagrange, &s2_lagrange, &s3_lagrange);

        let k2 = k1 * k1;

        let lp = |w: &FieldElement<F>, eta: &FieldElement<F>| w + beta * eta + gamma;

        for i in 0..n - 1 {
            let (a_i, b_i, c_i) = (&witness.a[i], &witness.b[i], &witness.c[i]);
            let num =
                lp(a_i, &domain[i]) * lp(b_i, &(&domain[i] * k1)) * lp(c_i, &(&domain[i] * &k2));
            let den = lp(a_i, &s1[i]) * lp(b_i, &s2[i]) * lp(c_i, &s3[i]);
            let new_factor = num / den;
            let new_term = coefficients.last().unwrap() * &new_factor;
            coefficients.push(new_term);
        }

        let p_z = Polynomial::interpolate(domain, &coefficients);
        let z_1 = self.commitment_scheme.commit(&p_z);
        Round2Result {
            z_1,
            p_z,
            beta: beta.clone(),
            gamma: gamma.clone(),
        }
    }

    fn round_3(
        &self,
        CommonPreprocessedInput {
            ql,
            qr,
            qo,
            qm,
            qc,
            s1,
            s2,
            s3,
            n,
            k1,
            domain,
            ..
        }: &CommonPreprocessedInput<F>,
        public_input: &[FieldElement<F>],
        Round1Result { p_a, p_b, p_c, .. }: &Round1Result<F, CS::Commitment>,
        Round2Result {
            p_z, beta, gamma, ..
        }: &Round2Result<F, CS::Commitment>,
        alpha: &FieldElement<F>,
    ) -> Round3Result<F, CS::Commitment> {
        let k2 = k1 * k1;

        let one = Polynomial::new_monomial(FieldElement::one(), 0);
        let p_x = &Polynomial::new_monomial(FieldElement::one(), 1);
        let zh = Polynomial::new_monomial(FieldElement::one(), *n) - &one;

        let z_x_omega_coefficients: Vec<FieldElement<F>> = p_z
            .coefficients()
            .iter()
            .enumerate()
            .map(|(i, x)| x * &domain[i])
            .collect();
        let z_x_omega = Polynomial::new(&z_x_omega_coefficients);
        let mut e1 = vec![FieldElement::zero(); domain.len()];
        e1[0] = FieldElement::one();
        let l1 = Polynomial::interpolate(domain, &e1);
        let mut p_pi_y = public_input.to_vec();
        p_pi_y.append(&mut vec![FieldElement::zero(); n - public_input.len()]);
        let p_pi = Polynomial::interpolate(domain, &p_pi_y);

        let p_constraints = p_a * p_b * qm + p_a * ql + p_b * qr + p_c * qo + qc + p_pi;
        let f = (p_a + p_x * beta + gamma)
            * (p_b + p_x * beta * k1 + gamma)
            * (p_c + p_x * beta * k2 + gamma);
        let g = (p_a + s1 * beta + gamma) * (p_b + s2 * beta + gamma) * (p_c + s3 * beta + gamma);
        let p_permutation_1 = g * z_x_omega - f * p_z; // TODO: Paper says this term is minus the term found on gnark. This doesn't affect the protocol.
        let p_permutation_2 = (p_z - one) * &l1;

        let p = ((&p_permutation_2 * alpha) + p_permutation_1) * alpha + p_constraints;

        //let mut t = p / zh;
        let (mut t, remainder) = p.long_division_with_remainder(&zh);
        assert_eq!(remainder, Polynomial::zero());

        Polynomial::pad_with_zero_coefficients_to_length(&mut t, 3 * (n + 2));
        let p_t_lo = Polynomial::new(&t.coefficients[..n + 2]);
        let p_t_mid = Polynomial::new(&t.coefficients[n + 2..2 * (n + 2)]);
        let p_t_hi = Polynomial::new(&t.coefficients[2 * (n + 2)..3 * (n + 2)]);

        let t_lo_1 = self.commitment_scheme.commit(&p_t_lo);
        let t_mid_1 = self.commitment_scheme.commit(&p_t_mid);
        let t_hi_1 = self.commitment_scheme.commit(&p_t_hi);

        Round3Result {
            t_lo_1,
            t_mid_1,
            t_hi_1,
            p_t_lo,
            p_t_mid,
            p_t_hi,
            alpha: alpha.clone(),
        }
    }

    fn round_4(
        &self,
        CommonPreprocessedInput { s1, s2, omega, .. }: &CommonPreprocessedInput<F>,
        Round1Result { p_a, p_b, p_c, .. }: &Round1Result<F, CS::Commitment>,
        Round2Result { p_z, .. }: &Round2Result<F, CS::Commitment>,
        zeta: &FieldElement<F>,
    ) -> Round4Result<F> {
        let a_zeta = p_a.evaluate(zeta);
        let b_zeta = p_b.evaluate(zeta);
        let c_zeta = p_c.evaluate(zeta);
        let s1_zeta = s1.evaluate(zeta);
        let s2_zeta = s2.evaluate(zeta);
        let z_zeta_omega = p_z.evaluate(&(zeta * omega));
        Round4Result {
            a_zeta,
            b_zeta,
            c_zeta,
            s1_zeta,
            s2_zeta,
            z_zeta_omega,
            zeta: zeta.clone(),
        }
    }

    fn round_5(
        &self,
        CommonPreprocessedInput {
            ql,
            qr,
            qo,
            qm,
            qc,
            s1,
            s2,
            s3,
            n,
            k1,
            omega,
            ..
        }: &CommonPreprocessedInput<F>,
        Round1Result { p_a, p_b, p_c, .. }: &Round1Result<F, CS::Commitment>,
        Round2Result {
            p_z, beta, gamma, ..
        }: &Round2Result<F, CS::Commitment>,
        Round3Result {
            p_t_lo,
            p_t_mid,
            p_t_hi,
            alpha,
            ..
        }: &Round3Result<F, CS::Commitment>,
        Round4Result {
            a_zeta,
            b_zeta,
            c_zeta,
            s1_zeta,
            s2_zeta,
            z_zeta_omega,
            zeta,
        }: &Round4Result<F>,
        upsilon: &FieldElement<F>,
    ) -> Round5Result<F, CS::Commitment> {
        // Precompute variables
        let k2 = k1 * k1;
        let zeta_raised_n = Polynomial::new_monomial(zeta.pow(n + 2), 0); // TODO: Paper says n and 2n, but Gnark uses n+2 and 2n+4 (see the TODO(*))
        let zeta_raised_2n = Polynomial::new_monomial(zeta.pow(2 * n + 4), 0);

        let l1_zeta = (zeta.pow(*n as u64) - FieldElement::one())
            / (zeta - FieldElement::one())
            / FieldElement::from(*n as u64);

        // This is called linearized polynomial in Gnark
        let mut p_non_constant = qm * a_zeta * b_zeta + a_zeta * ql + b_zeta * qr + c_zeta * qo + qc;

        let r_2_1 = (a_zeta + beta * zeta + gamma)
            * (b_zeta + beta * k1 * zeta + gamma)
            * (c_zeta + beta * &k2 * zeta + gamma)
            * p_z;
        let r_2_2 = (a_zeta + beta * s1_zeta + gamma)
            * (b_zeta + beta * s2_zeta + gamma)
            * beta
            * z_zeta_omega
            * s3;
        p_non_constant = p_non_constant + (r_2_2 - r_2_1) * alpha;

        let r_3 = p_z * l1_zeta;
        p_non_constant = p_non_constant + (r_3 * alpha * alpha);

        let partial_t = p_t_lo + zeta_raised_n * p_t_mid + zeta_raised_2n * p_t_hi;

        // TODO: Refactor to remove clones.
        let polynomials = vec![partial_t, p_non_constant, p_a.clone(), p_b.clone(), p_c.clone(), s1.clone(), s2.clone()];
        let ys: Vec<FieldElement<F>> = polynomials.iter().map(|p| p.evaluate(zeta)).collect();
        let w_zeta_1 = self.commitment_scheme.open_batch(
            zeta,
            &ys,
            &polynomials,
            upsilon
        );

        let w_zeta_omega_1 = self.commitment_scheme.open(&(zeta * omega), z_zeta_omega, p_z);

        Round5Result {
            w_zeta_1,
            w_zeta_omega_1,
            p_non_constant_zeta: ys[1].clone(),
            t_zeta: ys[0].clone(),
        }
    }

    #[allow(unused)]
    pub fn prove(
        &self,
        witness: &Witness<F>,
        public_input: &[FieldElement<F>],
        common_preprocesed_input: &CommonPreprocessedInput<F>,
    ) -> Proof<F, CS> {
        // TODO: use strong Fiat-Shamir (e.g.: add public inputs and statement)
        let mut transcript = Transcript::new();

        // Round 1
        let round_1 = self.round_1(witness, common_preprocesed_input);
        transcript.append(&round_1.a_1.to_bytes_be());
        transcript.append(&round_1.b_1.to_bytes_be());
        transcript.append(&round_1.c_1.to_bytes_be());

        // Round 2
        // TODO: Handle error
        let beta = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
        let gamma = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();

        let round_2 = self.round_2(witness, common_preprocesed_input, &beta, &gamma);
        transcript.append(&round_2.z_1.to_bytes_be());

        // Round 3
        let alpha = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
        let round_3 = self.round_3(
            common_preprocesed_input,
            public_input,
            &round_1,
            &round_2,
            &alpha,
        );
        transcript.append(&round_3.t_lo_1.to_bytes_be());
        transcript.append(&round_3.t_mid_1.to_bytes_be());
        transcript.append(&round_3.t_hi_1.to_bytes_be());

        // Round 4
        let zeta = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
        let round_4 = self.round_4(common_preprocesed_input, &round_1, &round_2, &zeta);

        // TODO: Should we append something to the transcript here? This step does not return any commitment.
        // But the next step receives a new challenge `upsilon`

        // Round 5
        let upsilon = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
        let round_5 = self.round_5(
            common_preprocesed_input,
            &round_1,
            &round_2,
            &round_3,
            &round_4,
            &upsilon,
        );

        Proof {
            a_1: round_1.a_1,
            b_1: round_1.b_1,
            c_1: round_1.c_1,
            z_1: round_2.z_1,
            t_lo_1: round_3.t_lo_1,
            t_mid_1: round_3.t_mid_1,
            t_hi_1: round_3.t_hi_1,
            a_zeta: round_4.a_zeta,
            b_zeta: round_4.b_zeta,
            c_zeta: round_4.c_zeta,
            s1_zeta: round_4.s1_zeta,
            s2_zeta: round_4.s2_zeta,
            z_zeta_omega: round_4.z_zeta_omega,
            w_zeta_1: round_5.w_zeta_1,
            w_zeta_omega_1: round_5.w_zeta_omega_1,
            p_non_constant_zeta: round_5.p_non_constant_zeta,
            t_zeta: round_5.t_zeta,
        }
    }
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
        test_utils::FpElement,
        test_utils::{
            test_common_preprocessed_input_1, test_srs_1, FrElement, KZG, test_witness_1,
        },
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
        FrElement::from_hex("2d15959489a2a8e44693221ca7cbdcab15253d6bae9fd7fe0664cff02fe4f1cf")
    }

    #[test]
    fn test_round_1() {
        let witness = test_witness_1();
        let common_preprocesed_input = test_common_preprocessed_input_1();
        let srs = test_srs_1();
        let kzg = KZG::new(srs);
        let prover = Prover::new(kzg);
        let round_1 = prover.round_1(&witness, &common_preprocesed_input);
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
        assert_eq!(round_1.a_1, a_1_expected);
        assert_eq!(round_1.b_1, b_1_expected);
        assert_eq!(round_1.c_1, c_1_expected);
    }

    #[test]
    fn test_round_2() {
        let witness = test_witness_1();
        let common_preprocesed_input = test_common_preprocessed_input_1();
        let srs = test_srs_1();
        let kzg = KZG::new(srs);
        let prover = Prover::new(kzg);

        let result_2 = prover.round_2(&witness, &common_preprocesed_input, &beta(), &gamma());
        let z_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("3e8322968c3496cf1b5786d4d71d158a646ec90c14edf04e758038e1f88dcdfe8443fcecbb75f3074a872a380391742"),
            FpElement::from_hex("11eac40d09796ff150004e7b858d83ddd9fe995dced0b3fbd7535d6e361729b25d488799da61fdf1d7b5022684053327"),
        ).unwrap();
        assert_eq!(result_2.z_1, z_1_expected);
    }

    #[test]
    fn test_round_3() {
        let witness = test_witness_1();
        let common_preprocesed_input = test_common_preprocessed_input_1();
        let srs = test_srs_1();
        let kzg = KZG::new(srs);
        let public_input = vec![FieldElement::from(2_u64), FieldElement::from(4)];
        let prover = Prover::new(kzg);
        let round_1 = prover.round_1(&witness, &common_preprocesed_input);
        let round_2 = prover.round_2(&witness, &common_preprocesed_input, &beta(), &gamma());
        let round_3 = prover.round_3(
            &common_preprocesed_input,
            &public_input,
            &round_1,
            &round_2,
            &alpha(),
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

        assert_eq!(round_3.t_lo_1, t_lo_1_expected);
        assert_eq!(round_3.t_mid_1, t_mid_1_expected);
        assert_eq!(round_3.t_hi_1, t_hi_1_expected);
    }

    #[test]
    fn test_round_4() {
        let witness = test_witness_1();
        let common_preprocesed_input = test_common_preprocessed_input_1();
        let srs = test_srs_1();
        let kzg = KZG::new(srs);
        let prover = Prover::new(kzg);

        let round_1 = prover.round_1(&witness, &common_preprocesed_input);
        let round_2 = prover.round_2(&witness, &common_preprocesed_input, &beta(), &gamma());

        let round_4 = prover.round_4(&common_preprocesed_input, &round_1, &round_2, &zeta());
        let expected_a_value =
            FrElement::from_hex("2c090a95b57f1f493b7b747bba34fef7772fd72f97d718ed69549641a823eb2e");
        let expected_b_value =
            FrElement::from_hex("5975959d91369ba4e7a03c6ae94b7fe98e8b61b7bf9af63c8ae0759e17ac0c7e");
        let expected_c_value =
            FrElement::from_hex("6bf31edeb4344b7d2df2cb1bd40b4d13e182d9cb09f89591fa043c1a34b4a93");
        let expected_z_value =
            FrElement::from_hex("38e2ec8e7c3dab29e2b8e9c8ea152914b8fe4612e91f2902c80238efcf21f4ee");
        let expected_s1_value =
            FrElement::from_hex("472f66db4fb6947d9ed9808241fe82324bc08aa2a54be93179db8e564e1137d4");
        let expected_s2_value =
            FrElement::from_hex("5588f1239c24efe0538868d0f716984e69c6980e586864f615e4b0621fdc6f81");

        assert_eq!(round_4.a_zeta, expected_a_value);
        assert_eq!(round_4.b_zeta, expected_b_value);
        assert_eq!(round_4.c_zeta, expected_c_value);
        assert_eq!(round_4.z_zeta_omega, expected_z_value);
        assert_eq!(round_4.s1_zeta, expected_s1_value);
        assert_eq!(round_4.s2_zeta, expected_s2_value);
    }

    #[test]
    fn test_round_5() {
        let witness = test_witness_1();
        let common_preprocesed_input = test_common_preprocessed_input_1();
        let srs = test_srs_1();
        let kzg = KZG::new(srs);
        let public_input = vec![FieldElement::from(2_u64), FieldElement::from(4)];

        let prover = Prover::new(kzg);

        let round_1 = prover.round_1(&witness, &common_preprocesed_input);
        let round_2 = prover.round_2(&witness, &common_preprocesed_input, &beta(), &gamma());

        let round_3 = prover.round_3(
            &common_preprocesed_input,
            &public_input,
            &round_1,
            &round_2,
            &alpha(),
        );

        let round_4 = prover.round_4(&common_preprocesed_input, &round_1, &round_2, &zeta());

        let expected_w_zeta_1 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("fa6250b80a418f0548b132ac264ff9915b2076c0c2548da9316ae19ffa35bbcf905d9f02f9274739608045ef83a4757"),
            FpElement::from_hex("17713ade2dbd66e923d4092a5d2da98202959dd65a15e9f7791fab3c0dd08788aa9b4a1cb21d04e0c43bd29225472145"),
        ).unwrap();
        let expected_w_zeta_omega_1 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("4484f08f8eaccf28bab8ee9539e6e7f4059cb1ce77b9b18e9e452f387163dc0b845f4874bf6445399e650d362799ff5"),
            FpElement::from_hex("1254347a0fa2ac856917825a5cff5f9583d39a52edbc2be5bb10fabd0c04d23019bcb963404345743120310fd734a61a"),
        ).unwrap();

        let round_5 = prover.round_5(
            &common_preprocesed_input,
            &round_1,
            &round_2,
            &round_3,
            &round_4,
            &upsilon(),
        );
        assert_eq!(round_5.w_zeta_1, expected_w_zeta_1);
        assert_eq!(round_5.w_zeta_omega_1, expected_w_zeta_omega_1);
    }
}
