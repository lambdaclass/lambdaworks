use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;

use crate::setup::{Circuit, CommonPreprocessedInput};
use crate::prover::Proof;

fn compute_challenges<F, CS>(
    p: &Proof<F, CS>,
) -> (FieldElement<F>, FieldElement<F>, FieldElement<F>, FieldElement<F>, FieldElement<F>)
where
    F: IsField,
    CS: IsCommitmentScheme<F>,
    CS::Hiding: ByteConversion,
    FieldElement<F>: ByteConversion
{
    let mut transcript = Transcript::new();
    transcript.append(&p.a_1.to_bytes_be());
    transcript.append(&p.b_1.to_bytes_be());
    transcript.append(&p.c_1.to_bytes_be());
    let beta = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
    let gamma = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();

    transcript.append(&p.z_1.to_bytes_be());
    let alpha = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();

    transcript.append(&p.t_lo_1.to_bytes_be());
    transcript.append(&p.t_mid_1.to_bytes_be());
    transcript.append(&p.t_hi_1.to_bytes_be());

    let zeta = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
    let upsilon = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
    (beta, gamma, alpha, zeta, upsilon)
}

fn verify<F, CS>(
    p: &Proof<F, CS>,
    circuit: &Circuit,
    public_input: &[FieldElement<F>],
    input: &CommonPreprocessedInput<F>,
) -> bool
where
    F: IsField,
    CS: IsCommitmentScheme<F>,
    CS::Hiding: ByteConversion,
    FieldElement<F>: ByteConversion
{
    // TODO: First three steps are validations: belonging to main subgroup, belonging to prime field.
    let (beta, gamma, alpha, zeta, upsilon) = compute_challenges(p);
    let zh_zeta = zeta.pow(input.n) - FieldElement::one();
    let mut p_pi_y = public_input.to_vec();
    p_pi_y.append(&mut vec![FieldElement::zero(); input.n - public_input.len()]);

    let l1_zeta = (zeta.pow(input.n as u64) - FieldElement::one()) / (&zeta - FieldElement::one()) / FieldElement::from(input.n as u64);
    let p_pi_zeta = Polynomial::interpolate(&input.domain, &p_pi_y).evaluate(&zeta);

    let mut p_remaining = &alpha * &p.z_zeta_omega * (&p.c_zeta + &gamma) * (&p.a_zeta + &beta * &p.s1_zeta + &gamma) * (&p.b_zeta + beta * &p.s2_zeta + gamma);
    p_remaining = p_remaining - l1_zeta * &alpha * alpha;
    p_remaining = p_remaining + p_pi_zeta;

    let p_zeta = p_remaining + &p.partial_p_zeta;

    let verify = p_zeta - (zh_zeta * &p.partial_t_zeta);

    verify == FieldElement::zero()
}

mod tests {
    use super::*;

    use crate::{
        test_utils::{test_circuit, test_srs, FrElement, ORDER_4_ROOT_UNITY, ORDER_R_MINUS_1_ROOT_UNITY, KZG, test_common_preprocessed_input}, prover::Prover,
    };

    #[test]
    fn test_verifier() {
        let test_circuit = test_circuit();
        let common_preprocesed_input = test_common_preprocessed_input();
        let srs = test_srs();
        let kzg = KZG::new(srs);
        let public_input = vec![FieldElement::from(2_u64), FieldElement::from(4)];
        let prover = Prover::new(kzg, ORDER_4_ROOT_UNITY, ORDER_R_MINUS_1_ROOT_UNITY);
        let proof = prover.prove(&test_circuit, &public_input, &common_preprocesed_input);
        assert!(verify(&proof, &test_circuit, &public_input, &common_preprocesed_input));
    }
}