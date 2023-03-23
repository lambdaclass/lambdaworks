use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;
use std::marker::PhantomData;

use crate::prover::Proof;
use crate::setup::{CommonPreprocessedInput, VerificationKey};

struct Verifier<F: IsField, CS: IsCommitmentScheme<F>> {
    commitment_scheme: CS,
    phantom: PhantomData<F>,
}

impl<F: IsField, CS: IsCommitmentScheme<F>> Verifier<F, CS> {
    #[allow(unused)]
    pub fn new(commitment_scheme: CS) -> Self {
        Self {
            commitment_scheme,
            phantom: PhantomData,
        }
    }

    fn compute_challenges(&self, p: &Proof<F, CS>) -> [FieldElement<F>; 5]
    where
        F: IsField,
        CS: IsCommitmentScheme<F>,
        CS::Commitment: ByteConversion,
        FieldElement<F>: ByteConversion,
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
        [beta, gamma, alpha, zeta, upsilon]
    }

    #[allow(unused)]
    fn verify(
        &self,
        p: &Proof<F, CS>,
        public_input: &[FieldElement<F>],
        input: &CommonPreprocessedInput<F>,
        vk: &VerificationKey<CS::Commitment>,
    ) -> bool
    where
        F: IsPrimeField,
        CS: IsCommitmentScheme<F>,
        CS::Commitment: ByteConversion + IsGroup,
        FieldElement<F>: ByteConversion,
    {
        // TODO: First three steps are validations: belonging to main subgroup, belonging to prime field.
        let [beta, gamma, alpha, zeta, upsilon] = self.compute_challenges(p);
        let zh_zeta = zeta.pow(input.n) - FieldElement::one();
        let mut p_pi_y = public_input.to_vec();
        p_pi_y.append(&mut vec![
            FieldElement::zero();
            input.n - public_input.len()
        ]);

        let k1 = &input.k1;
        let k2 = k1 * k1;

        let l1_zeta = (zeta.pow(input.n as u64) - FieldElement::one())
            / (&zeta - FieldElement::one())
            / FieldElement::from(input.n as u64);
        let p_pi_zeta = Polynomial::interpolate(&input.domain, &p_pi_y).evaluate(&zeta);

        let mut p_constant_zeta = &alpha
            * &p.z_zeta_omega
            * (&p.c_zeta + &gamma)
            * (&p.a_zeta + &beta * &p.s1_zeta + &gamma)
            * (&p.b_zeta + &beta * &p.s2_zeta + &gamma);
        p_constant_zeta = p_constant_zeta - &l1_zeta * &alpha * &alpha;
        p_constant_zeta += p_pi_zeta;

        let p_zeta = p_constant_zeta + &p.p_non_constant_zeta;

        let constraints_check = p_zeta - (&zh_zeta * &p.t_zeta) == FieldElement::zero();

        // Compute commitment of partial evaluation of t (p = zh * t)
        let partial_t_1 = p
            .t_lo_1
            .operate_with(
                &p.t_mid_1
                    .operate_with_self(zeta.pow(input.n + 2).representative()),
            )
            .operate_with(
                &p.t_hi_1
                    .operate_with_self(zeta.pow(2 * input.n + 4).representative()),
            );

        // Compute commitment of the non constant part of the linearization of p
        // The first term corresponds to the gates constraints
        let mut first_term = vk
            .qm_1
            .operate_with_self((&p.a_zeta * &p.b_zeta).representative());
        first_term = first_term.operate_with(&vk.ql_1.operate_with_self(p.a_zeta.representative()));
        first_term = first_term.operate_with(&vk.qr_1.operate_with_self(p.b_zeta.representative()));
        first_term = first_term.operate_with(&vk.qo_1.operate_with_self(p.c_zeta.representative()));
        first_term = first_term.operate_with(&vk.qc_1);

        // Second and third terms correspond to copy constraints
        // + α*((l(ζ)+β*s1(ζ)+γ)*(r(ζ)+β*s2(ζ)+γ)*Z(μζ)*β*s3(X) - Z(X)*(l(ζ)+β*id1(ζ)+γ)*(r(ζ)+β*id2(ζ)+γ)*(o(ζ)+β*id3(ζ)+γ))
        let z_coefficient = -(&p.a_zeta + &beta * &zeta + &gamma)
            * (&p.b_zeta + &beta * k1 * &zeta + &gamma)
            * (&p.c_zeta + &beta * k2 * &zeta + &gamma);
        let s3_coefficient = (&p.a_zeta + &beta * &p.s1_zeta + &gamma)
            * (&p.b_zeta + &beta * &p.s2_zeta + &gamma)
            * beta
            * &p.z_zeta_omega;
        let second_term = p
            .z_1
            .operate_with_self(z_coefficient.representative())
            .operate_with(&vk.s3_1.operate_with_self(s3_coefficient.representative()))
            .operate_with_self(alpha.representative());
        // α²*L₁(ζ)*Z(X)
        let third_term = p
            .z_1
            .operate_with_self((&alpha * &alpha * l1_zeta).representative());

        let p_non_constant_1 = first_term
            .operate_with(&second_term)
            .operate_with(&third_term);

        let ys = [
            p.t_zeta.clone(),
            p.p_non_constant_zeta.clone(),
            p.a_zeta.clone(),
            p.b_zeta.clone(),
            p.c_zeta.clone(),
            p.s1_zeta.clone(),
            p.s2_zeta.clone(),
        ];
        let commitments = [
            partial_t_1,
            p_non_constant_1,
            p.a_1.clone(),
            p.b_1.clone(),
            p.c_1.clone(),
            vk.s1_1.clone(),
            vk.s2_1.clone(),
        ];
        let batch_openings_check =
            self.commitment_scheme
                .verify_batch(&zeta, &ys, &commitments, &p.w_zeta_1, &upsilon);

        let single_opening_check = self.commitment_scheme.verify(
            &(zeta * &input.omega),
            &p.z_zeta_omega,
            &p.z_1,
            &p.w_zeta_omega_1,
        );

        constraints_check && batch_openings_check && single_opening_check
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        prover::Prover,
        setup::setup,
        test_utils::{
            test_common_preprocessed_input_1, test_common_preprocessed_input_2, test_srs_1,
            test_srs_2, test_witness_1, test_witness_2, KZG,
        },
    };

    #[test]
    fn test_verifier() {
        let common_preprocesed_input = test_common_preprocessed_input_1();
        let srs = test_srs_1();
        let witness = test_witness_1();

        let public_input = vec![FieldElement::from(2_u64), FieldElement::from(4)];

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocesed_input, &kzg);

        let prover = Prover::new(kzg.clone());
        let proof = prover.prove(&witness, &public_input, &common_preprocesed_input);

        let verifier = Verifier::new(kzg.clone());
        assert!(verifier.verify(
            &proof,
            &public_input,
            &common_preprocesed_input,
            &verifying_key
        ));
    }

    #[test]
    fn test_verifier_2() {
        let common_preprocesed_input = test_common_preprocessed_input_2();
        let srs = test_srs_2();
        let witness = test_witness_2();
        let public_input = vec![FieldElement::from(2_u64), FieldElement::from(11)];

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocesed_input, &kzg);

        let prover = Prover::new(kzg.clone());
        let proof = prover.prove(&witness, &public_input, &common_preprocesed_input);

        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(
            &proof,
            &public_input,
            &common_preprocesed_input,
            &verifying_key
        ));
    }
}
