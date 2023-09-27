use lambdaworks_crypto::{
    commitments::traits::IsCommitmentScheme, fiat_shamir::transcript::Transcript,
};
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::{ByteConversion, Serializable},
};
use lambdaworks_plonk::{
    prover::Proof,
    setup::{new_strong_fiat_shamir_transcript, CommonPreprocessedInput, VerificationKey},
};

#[allow(unused)]
fn forge_y_for_valid_proof<F: IsField, CS: IsCommitmentScheme<F>>(
    proof: &Proof<F, CS>,
    vk: &VerificationKey<CS::Commitment>,
    common_preprocessed_input: CommonPreprocessedInput<F>,
) -> FieldElement<F>
where
    CS::Commitment: Serializable,
    FieldElement<F>: ByteConversion,
{
    // Replay interactions like the verifier
    let mut transcript = new_strong_fiat_shamir_transcript::<F, CS>(vk, &[]);

    transcript.append(&proof.a_1.serialize());
    transcript.append(&proof.b_1.serialize());
    transcript.append(&proof.c_1.serialize());
    let beta = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
    let gamma = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();

    transcript.append(&proof.z_1.serialize());
    let alpha = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();

    transcript.append(&proof.t_lo_1.serialize());
    transcript.append(&proof.t_mid_1.serialize());
    transcript.append(&proof.t_hi_1.serialize());
    let zeta = &FieldElement::from_bytes_be(&transcript.challenge()).unwrap();

    // Forge public input
    let zh_zeta = zeta.pow(common_preprocessed_input.n) - FieldElement::one();

    let omega = &common_preprocessed_input.omega;
    let n = common_preprocessed_input.n as u64;
    let one = &FieldElement::one();

    let l1_zeta = ((zeta.pow(n) - one) / (zeta - one)) / FieldElement::from(n);

    let l2_zeta = omega * &l1_zeta * (zeta - one) / (zeta - omega);

    let mut p_constant_zeta = &alpha
        * &proof.z_zeta_omega
        * (&proof.c_zeta + &gamma)
        * (&proof.a_zeta + &beta * &proof.s1_zeta + &gamma)
        * (&proof.b_zeta + &beta * &proof.s2_zeta + &gamma);
    p_constant_zeta = p_constant_zeta - &l1_zeta * &alpha * &alpha;

    let p_zeta = p_constant_zeta + &proof.p_non_constant_zeta;
    -(p_zeta + l1_zeta * one - (&zh_zeta * &proof.t_zeta)) / l2_zeta
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_plonk::{prover::Prover, setup::setup};

    use crate::{
        circuit::{circuit_common_preprocessed_input, circuit_witness},
        server::{generate_srs, server_endpoint_verify, TestRandomFieldGenerator, FLAG, KZG},
        solution::forge_y_for_valid_proof,
    };

    #[test]
    fn test_challenge() {
        // This is the circuit for `ASSERT 0 == y ** 2 - x ** 3 - 4`
        let cpi = circuit_common_preprocessed_input();
        let srs = generate_srs(cpi.n);
        let kzg = KZG::new(srs.clone());
        let verifying_key = setup(&cpi.clone(), &kzg);

        let x = FieldElement::from(0);
        let y = FieldElement::from(2);

        let public_input = vec![x.clone(), y.clone()];
        let witness = circuit_witness(&x, &y);

        let random_generator = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg.clone(), random_generator);
        let proof = prover.prove(&witness, &public_input, &cpi, &verifying_key);

        let response_valid =
            server_endpoint_verify(srs.clone(), cpi.clone(), &verifying_key, &x, &y, &proof);
        assert_eq!("Valid Proof. Congrats!".to_string(), response_valid);

        let response_invalid = server_endpoint_verify(
            srs.clone(),
            cpi.clone(),
            &verifying_key,
            &FieldElement::one(),
            &y,
            &proof,
        );
        assert_eq!("Invalid Proof".to_string(), response_invalid);

        // Use the real proof to modify the public input
        // and make it pass for `x = 1`
        let forged_y = forge_y_for_valid_proof(&proof, &verifying_key, cpi.clone());

        let response_solution = server_endpoint_verify(
            srs.clone(),
            cpi.clone(),
            &verifying_key,
            &FieldElement::one(),
            &forged_y,
            &proof,
        );
        assert_eq!(FLAG.to_string(), response_solution);
    }
}
