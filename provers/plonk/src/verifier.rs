use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsFFTField, IsField, IsPrimeField};
use lambdaworks_math::traits::{AsBytes, ByteConversion};
use std::marker::PhantomData;

use crate::prover::Proof;
use crate::setup::{new_strong_fiat_shamir_transcript, CommonPreprocessedInput, VerificationKey};

pub struct Verifier<F: IsField, CS: IsCommitmentScheme<F>> {
    commitment_scheme: CS,
    phantom: PhantomData<F>,
}

impl<F: IsField + IsFFTField + HasDefaultTranscript, CS: IsCommitmentScheme<F>> Verifier<F, CS> {
    pub fn new(commitment_scheme: CS) -> Self {
        Self {
            commitment_scheme,
            phantom: PhantomData,
        }
    }

    fn compute_challenges(
        &self,
        p: &Proof<F, CS>,
        vk: &VerificationKey<CS::Commitment>,
        public_input: &[FieldElement<F>],
    ) -> [FieldElement<F>; 5]
    where
        F: IsField,
        CS: IsCommitmentScheme<F>,
        CS::Commitment: AsBytes,
        FieldElement<F>: ByteConversion,
    {
        let mut transcript = new_strong_fiat_shamir_transcript::<F, CS>(vk, public_input);

        transcript.append_bytes(&p.a_1.as_bytes());
        transcript.append_bytes(&p.b_1.as_bytes());
        transcript.append_bytes(&p.c_1.as_bytes());
        let beta = transcript.sample_field_element();
        let gamma = transcript.sample_field_element();

        transcript.append_bytes(&p.z_1.as_bytes());
        let alpha = transcript.sample_field_element();

        transcript.append_bytes(&p.t_lo_1.as_bytes());
        transcript.append_bytes(&p.t_mid_1.as_bytes());
        transcript.append_bytes(&p.t_hi_1.as_bytes());
        let zeta = transcript.sample_field_element();

        transcript.append_field_element(&p.a_zeta);
        transcript.append_field_element(&p.b_zeta);
        transcript.append_field_element(&p.c_zeta);
        transcript.append_field_element(&p.s1_zeta);
        transcript.append_field_element(&p.s2_zeta);
        transcript.append_field_element(&p.z_zeta_omega);
        let upsilon = transcript.sample_field_element();

        [beta, gamma, alpha, zeta, upsilon]
    }

    pub fn verify(
        &self,
        p: &Proof<F, CS>,
        public_input: &[FieldElement<F>],
        input: &CommonPreprocessedInput<F>,
        vk: &VerificationKey<CS::Commitment>,
    ) -> bool
    where
        F: IsPrimeField + IsFFTField,
        CS: IsCommitmentScheme<F>,
        CS::Commitment: AsBytes + IsGroup,
        FieldElement<F>: ByteConversion,
    {
        // TODO: First three steps are validations: belonging to main subgroup, belonging to prime field.
        let [beta, gamma, alpha, zeta, upsilon] = self.compute_challenges(p, vk, public_input);
        let zh_zeta = zeta.pow(input.n) - FieldElement::<F>::one();

        let k1 = &input.k1;
        let k2 = k1 * k1;

        // We are using that zeta != 0 because is sampled outside the set of roots of unity,
        // and n != 0 because is the length of the trace.
        let l1_zeta = ((zeta.pow(input.n as u64) - FieldElement::<F>::one())
            / ((&zeta - FieldElement::<F>::one()) * FieldElement::from(input.n as u64)))
        .unwrap();

        // Use the following equality to compute PI(ζ)
        // without interpolating:
        // Lᵢ₊₁ = ω Lᵢ (X − ωⁱ) / (X − ωⁱ⁺¹)
        // Here Lᵢ is the i-th polynomial of the Lagrange basis.
        let p_pi_zeta = if public_input.is_empty() {
            FieldElement::zero()
        } else {
            let mut p_pi_zeta = &l1_zeta * &public_input[0];
            let mut li_zeta = l1_zeta.clone();
            for (i, value) in public_input.iter().enumerate().skip(1) {
                li_zeta = &input.omega
                    * &li_zeta
                    // We are using that zeta is sampled outside the domain.
                    * ((&zeta - &input.domain[i - 1]) / (&zeta - &input.domain[i])).unwrap();
                p_pi_zeta = &p_pi_zeta + value * &li_zeta;
            }
            p_pi_zeta
        };

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
    use lambdaworks_math::traits::Deserializable;

    use super::*;

    use crate::{
        prover::Prover,
        setup::setup,
        test_utils::circuit_1::{test_common_preprocessed_input_1, test_witness_1},
        test_utils::circuit_2::{test_common_preprocessed_input_2, test_witness_2},
        test_utils::circuit_json::common_preprocessed_input_from_json,
        test_utils::utils::{test_srs, TestRandomFieldGenerator, KZG},
    };

    #[test]
    fn test_happy_path_for_circuit_1() {
        // This is the circuit for x * e == y
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);

        // Public input
        let x = FieldElement::from(4_u64);
        let y = FieldElement::from(12_u64);

        // Private variable
        let e = FieldElement::from(3_u64);

        let public_input = vec![x.clone(), y];
        let witness = test_witness_1(x, e);

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);
        let random_generator = TestRandomFieldGenerator {};

        let prover = Prover::new(kzg.clone(), random_generator);
        let proof = prover.prove(
            &witness,
            &public_input,
            &common_preprocessed_input,
            &verifying_key,
        );

        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(
            &proof,
            &public_input,
            &common_preprocessed_input,
            &verifying_key
        ));
    }

    #[test]
    fn test_happy_path_for_circuit_2() {
        // This is the circuit for x * e + 5 == y
        let common_preprocessed_input = test_common_preprocessed_input_2();
        let srs = test_srs(common_preprocessed_input.n);

        // Public input
        let x = FieldElement::from(2_u64);
        let y = FieldElement::from(11_u64);

        // Private variable
        let e = FieldElement::from(3_u64);

        let public_input = vec![x.clone(), y];
        let witness = test_witness_2(x, e);

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);
        let random_generator = TestRandomFieldGenerator {};

        let prover = Prover::new(kzg.clone(), random_generator);
        let proof = prover.prove(
            &witness,
            &public_input,
            &common_preprocessed_input,
            &verifying_key,
        );

        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(
            &proof,
            &public_input,
            &common_preprocessed_input,
            &verifying_key
        ));
    }

    #[test]
    fn test_happy_path_from_json() {
        let (witness, common_preprocessed_input, public_input) =
            common_preprocessed_input_from_json(
                r#"{
            "N": 4,
            "N_Padded": 4,
            "Omega": "8d51ccce760304d0ec030002760300000001000000000000",
             "Input": [
             "2",
             "4"
            ],
            "Ql": [
             "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000",
             "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000",
             "0",
             "1"
            ],
            "Qr": [
             "0",
             "0",
             "0",
             "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000"
            ],
            "Qm": [
             "0",
             "0",
             "1",
             "0"
            ],
            "Qo": [
             "0",
             "0",
             "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000",
             "0"
            ],
            "Qc": [
             "0",
             "0",
             "0",
             "0"
            ],
            "A": [
             "2",
             "4",
             "2",
             "4"
            ],
            "B": [
             "2",
             "2",
             "2",
             "4"
            ],
            "C": [
             "2",
             "2",
             "4",
             "2"
            ],
            "Permutation": [
             11,
             3,
             2,
             1,
             0,
             4,
             5,
             10,
             6,
             8,
             7,
             9
            ]
           }"#,
            );
        let srs = test_srs(common_preprocessed_input.n);

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);
        let random_generator = TestRandomFieldGenerator {};

        let prover = Prover::new(kzg.clone(), random_generator);
        let proof = prover.prove(
            &witness,
            &public_input,
            &common_preprocessed_input,
            &verifying_key,
        );

        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(
            &proof,
            &public_input,
            &common_preprocessed_input,
            &verifying_key
        ));
    }

    #[test]
    fn test_serialize_proof() {
        // This is the circuit for x * e == y
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);

        // Public input
        let x = FieldElement::from(4_u64);
        let y = FieldElement::from(12_u64);

        // Private variable
        let e = FieldElement::from(3_u64);

        let public_input = vec![x.clone(), y];
        let witness = test_witness_1(x, e);

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);
        let random_generator = TestRandomFieldGenerator {};

        let prover = Prover::new(kzg.clone(), random_generator);
        let proof = prover.prove(
            &witness,
            &public_input,
            &common_preprocessed_input,
            &verifying_key,
        );

        let serialized_proof = proof.as_bytes();
        let deserialized_proof = Proof::deserialize(&serialized_proof).unwrap();

        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(
            &deserialized_proof,
            &public_input,
            &common_preprocessed_input,
            &verifying_key
        ));
    }
}
