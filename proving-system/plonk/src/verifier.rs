use std::marker::PhantomData;

use lambdaworks_crypto::commitments::kzg::Opening;
use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;

use crate::prover::Proof;
use crate::setup::{Circuit, CommonPreprocessedInput, VerificationKey};

struct Verifier<F: IsField, CS: IsCommitmentScheme<F>> {
    commitment_scheme: CS,
    phantom: PhantomData<F>,
}

impl<F: IsField, CS: IsCommitmentScheme<F>> Verifier<F, CS> {
    #[allow(unused)]
    pub fn new(commitment_scheme: CS) -> Self {
        Self {
            commitment_scheme: commitment_scheme,
            phantom: PhantomData,
        }
    }

    fn compute_challenges(
        &self,
        p: &Proof<F, CS>,
    ) -> (
        FieldElement<F>,
        FieldElement<F>,
        FieldElement<F>,
        FieldElement<F>,
        FieldElement<F>,
    )
    where
        F: IsField,
        CS: IsCommitmentScheme<F>,
        CS::Hiding: ByteConversion,
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
        (beta, gamma, alpha, zeta, upsilon)
    }

    fn verify(
        &self,
        p: &Proof<F, CS>,
        circuit: &Circuit,
        public_input: &[FieldElement<F>],
        input: &CommonPreprocessedInput<F>,
        vk: &VerificationKey<CS::Hiding>,
    ) -> bool
    where
        F: IsPrimeField,
        CS: IsCommitmentScheme<F>,
        CS::Hiding: ByteConversion + IsGroup,
        FieldElement<F>: ByteConversion,
    {
        // TODO: First three steps are validations: belonging to main subgroup, belonging to prime field.
        let (beta, gamma, alpha, zeta, upsilon) = self.compute_challenges(p);
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

        let mut p_remaining = &alpha
            * &p.z_zeta_omega
            * (&p.c_zeta + &gamma)
            * (&p.a_zeta + &beta * &p.s1_zeta + &gamma)
            * (&p.b_zeta + &beta * &p.s2_zeta + &gamma);
        p_remaining = p_remaining - &l1_zeta * &alpha * &alpha;
        p_remaining = p_remaining + p_pi_zeta;

        let p_zeta = p_remaining + &p.partial_p_zeta;

        let check_constraints = p_zeta - (&zh_zeta * &p.t_zeta) == FieldElement::zero();

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

        // + l(ζ)*Ql(X) + l(ζ)r(ζ)*Qm(X) + r(ζ)*Qr(X) + o(ζ)*Qo(X) + Qk(X)
        let mut partial_p_1 = vk
            .qm_1
            .operate_with_self((&p.a_zeta * &p.b_zeta).representative());
        partial_p_1 =
            partial_p_1.operate_with(&vk.ql_1.operate_with_self(p.a_zeta.representative()));
        partial_p_1 =
            partial_p_1.operate_with(&vk.qr_1.operate_with_self(p.b_zeta.representative()));
        partial_p_1 =
            partial_p_1.operate_with(&vk.qo_1.operate_with_self(p.c_zeta.representative()));
        partial_p_1 = partial_p_1.operate_with(&vk.qc_1);

        // α²*L₁(ζ)*Z(X)
        partial_p_1 = partial_p_1.operate_with(
            &p.z_1
                .operate_with_self((&alpha * &alpha * l1_zeta).representative()),
        );

        // + α*( (l(ζ)+β*s1(ζ)+γ)*(r(ζ)+β*s2(ζ)+γ)*Z(μζ)*β*s3(X) - Z(X)*(l(ζ)+β*id1(ζ)+γ)*(r(ζ)+β*id2(ζ)+γ)*(o(ζ)+β*id3(ζ)+γ))
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

        partial_p_1 = partial_p_1.operate_with(&second_term);

        let mut f_1 = partial_t_1;
        f_1 = f_1.operate_with(&partial_p_1.operate_with_self(upsilon.representative()));
        f_1 = f_1.operate_with(&p.a_1.operate_with_self(upsilon.pow(2_u64).representative()));
        f_1 = f_1.operate_with(&p.b_1.operate_with_self(upsilon.pow(3_u64).representative()));
        f_1 = f_1.operate_with(&p.c_1.operate_with_self(upsilon.pow(4_u64).representative()));
        f_1 = f_1.operate_with(
            &vk.s1_1
                .operate_with_self(upsilon.pow(5_u64).representative()),
        );
        f_1 = f_1.operate_with(
            &vk.s2_1
                .operate_with_self(upsilon.pow(6_u64).representative()),
        );

        let mut e_1_coefficient = p.t_zeta.clone();
        e_1_coefficient = e_1_coefficient + &p.partial_p_zeta * &upsilon;
        e_1_coefficient = e_1_coefficient + &p.a_zeta * &upsilon.pow(2_u64);
        e_1_coefficient = e_1_coefficient + &p.b_zeta * &upsilon.pow(3_u64);
        e_1_coefficient = e_1_coefficient + &p.c_zeta * &upsilon.pow(4_u64);
        e_1_coefficient = e_1_coefficient + &p.s1_zeta * &upsilon.pow(5_u64);
        e_1_coefficient = e_1_coefficient + &p.s2_zeta * &upsilon.pow(6_u64);

        let e_1 = vk.g1.operate_with_self(e_1_coefficient.representative());

        check_constraints
    }
}

mod tests {
    use super::*;

    use crate::{
        prover::Prover,
        setup::setup,
        test_utils::{
            test_circuit, test_common_preprocessed_input, test_srs, FrElement, KZG,
            ORDER_4_ROOT_UNITY, ORDER_R_MINUS_1_ROOT_UNITY,
        },
    };

    #[test]
    fn test_verifier() {
        let test_circuit = test_circuit();
        let common_preprocesed_input = test_common_preprocessed_input();
        let srs = test_srs();

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocesed_input, &kzg, &test_circuit);
        let public_input = vec![FieldElement::from(2_u64), FieldElement::from(4)];

        let prover = Prover::new(kzg.clone());
        let proof = prover.prove(&test_circuit, &public_input, &common_preprocesed_input);

        let verifier = Verifier::new(kzg.clone());
        assert!(verifier.verify(
            &proof,
            &test_circuit,
            &public_input,
            &common_preprocesed_input,
            &verifying_key
        ));
    }
}
