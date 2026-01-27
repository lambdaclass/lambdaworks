use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsFFTField, IsField, IsPrimeField};
use lambdaworks_math::traits::{AsBytes, ByteConversion};
use lambdaworks_math::unsigned_integer::traits::IsUnsignedInteger;
use std::marker::PhantomData;

use crate::prover::Proof;
use crate::setup::{new_strong_fiat_shamir_transcript, CommonPreprocessedInput, VerificationKey};

/// Errors that can occur during PLONK verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifierError {
    /// A commitment point is not on the curve
    InvalidCommitment(&'static str),
    /// A commitment point is not in the prime-order subgroup
    CommitmentNotInSubgroup(&'static str),
    /// Constraint check failed
    ConstraintCheckFailed,
    /// Batch opening verification failed
    BatchOpeningFailed,
    /// Single opening verification failed
    SingleOpeningFailed,
}

impl std::fmt::Display for VerifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifierError::InvalidCommitment(name) => {
                write!(f, "Invalid commitment: {} is not on curve", name)
            }
            VerifierError::CommitmentNotInSubgroup(name) => {
                write!(f, "Commitment {} is not in prime-order subgroup", name)
            }
            VerifierError::ConstraintCheckFailed => {
                write!(f, "Constraint equation check failed")
            }
            VerifierError::BatchOpeningFailed => {
                write!(f, "Batch opening proof verification failed")
            }
            VerifierError::SingleOpeningFailed => {
                write!(f, "Single opening proof verification failed")
            }
        }
    }
}

impl std::error::Error for VerifierError {}

/// Trait for commitments that can be validated for subgroup membership.
/// This is used to ensure proof elements are in the correct prime-order subgroup.
pub trait SubgroupCheck: IsGroup {
    /// The type representing the subgroup order (e.g., U256)
    type Order: IsUnsignedInteger;

    /// Returns the order of the prime-order subgroup
    fn subgroup_order() -> Self::Order;

    /// Checks if the element is in the prime-order subgroup.
    /// This is done by verifying that multiplying by the subgroup order yields the identity.
    fn is_in_subgroup(&self) -> bool {
        self.operate_with_self(Self::subgroup_order())
            .is_neutral_element()
    }
}

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

    /// Validates that all commitment elements in the proof are in the prime-order subgroup.
    /// This is a critical security check that prevents attacks using points from the wrong subgroup.
    ///
    /// # Arguments
    /// * `p` - The proof to validate
    ///
    /// # Returns
    /// * `Ok(())` if all commitments are valid
    /// * `Err(VerifierError)` if any commitment is invalid
    pub fn validate_proof_commitments(p: &Proof<F, CS>) -> Result<(), VerifierError>
    where
        CS::Commitment: SubgroupCheck,
    {
        // Validate Round 1 commitments (wire polynomials)
        if !p.a_1.is_in_subgroup() {
            return Err(VerifierError::CommitmentNotInSubgroup("a_1"));
        }
        if !p.b_1.is_in_subgroup() {
            return Err(VerifierError::CommitmentNotInSubgroup("b_1"));
        }
        if !p.c_1.is_in_subgroup() {
            return Err(VerifierError::CommitmentNotInSubgroup("c_1"));
        }

        // Validate Round 2 commitment (permutation polynomial)
        if !p.z_1.is_in_subgroup() {
            return Err(VerifierError::CommitmentNotInSubgroup("z_1"));
        }

        // Validate Round 3 commitments (quotient polynomial parts)
        if !p.t_lo_1.is_in_subgroup() {
            return Err(VerifierError::CommitmentNotInSubgroup("t_lo_1"));
        }
        if !p.t_mid_1.is_in_subgroup() {
            return Err(VerifierError::CommitmentNotInSubgroup("t_mid_1"));
        }
        if !p.t_hi_1.is_in_subgroup() {
            return Err(VerifierError::CommitmentNotInSubgroup("t_hi_1"));
        }

        // Validate Round 5 commitments (opening proofs)
        if !p.w_zeta_1.is_in_subgroup() {
            return Err(VerifierError::CommitmentNotInSubgroup("w_zeta_1"));
        }
        if !p.w_zeta_omega_1.is_in_subgroup() {
            return Err(VerifierError::CommitmentNotInSubgroup("w_zeta_omega_1"));
        }

        Ok(())
    }

    /// Verifies a PLONK proof with full validation.
    ///
    /// This method performs:
    /// 1. Subgroup membership checks on all commitments (when SubgroupCheck is implemented)
    /// 2. Challenge derivation via Fiat-Shamir
    /// 3. Constraint equation verification
    /// 4. Opening proof verification
    ///
    /// # Arguments
    /// * `p` - The proof to verify
    /// * `public_input` - The public inputs to the circuit
    /// * `input` - Common preprocessed input from setup
    /// * `vk` - The verification key
    ///
    /// # Returns
    /// * `Ok(())` if the proof is valid
    /// * `Err(VerifierError)` describing why verification failed
    pub fn verify_with_result(
        &self,
        p: &Proof<F, CS>,
        public_input: &[FieldElement<F>],
        input: &CommonPreprocessedInput<F>,
        vk: &VerificationKey<CS::Commitment>,
    ) -> Result<(), VerifierError>
    where
        F: IsPrimeField + IsFFTField,
        CS: IsCommitmentScheme<F>,
        CS::Commitment: AsBytes + IsGroup + SubgroupCheck,
        FieldElement<F>: ByteConversion,
    {
        // Step 1: Validate subgroup membership of all commitments
        Self::validate_proof_commitments(p)?;

        // Steps 2-5: Perform the actual verification
        let [beta, gamma, alpha, zeta, upsilon] = self.compute_challenges(p, vk, public_input);
        let zh_zeta = zeta.pow(input.n) - FieldElement::<F>::one();

        let k1 = &input.k1;
        let k2 = k1 * k1;

        // Compute L1(zeta) - Lagrange polynomial at first point
        // Using zeta != 0 (sampled outside roots of unity) and n != 0 (trace length)
        let l1_zeta = ((zeta.pow(input.n as u64) - FieldElement::<F>::one())
            / ((&zeta - FieldElement::<F>::one()) * FieldElement::from(input.n as u64)))
        .map_err(|_| VerifierError::ConstraintCheckFailed)?;

        // Compute PI(zeta) using Lagrange basis recurrence
        let p_pi_zeta = if public_input.is_empty() {
            FieldElement::zero()
        } else {
            let mut p_pi_zeta = &l1_zeta * &public_input[0];
            let mut li_zeta = l1_zeta.clone();
            for (i, value) in public_input.iter().enumerate().skip(1) {
                li_zeta = &input.omega
                    * &li_zeta
                    * ((&zeta - &input.domain[i - 1]) / (&zeta - &input.domain[i]))
                        .map_err(|_| VerifierError::ConstraintCheckFailed)?;
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

        // Check constraint equation: p(zeta) = zh(zeta) * t(zeta)
        if p_zeta - (&zh_zeta * &p.t_zeta) != FieldElement::zero() {
            return Err(VerifierError::ConstraintCheckFailed);
        }

        // Compute commitment of partial evaluation of t(ζ)
        // The quotient polynomial was split as: t(X) = t_lo + X^(n+2)·t_mid + X^(2n+4)·t_hi
        // Following gnark's approach (accounts for blinding polynomials)
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

        // Compute commitment of the non-constant part of the linearization
        let mut first_term = vk
            .qm_1
            .operate_with_self((&p.a_zeta * &p.b_zeta).representative());
        first_term = first_term.operate_with(&vk.ql_1.operate_with_self(p.a_zeta.representative()));
        first_term = first_term.operate_with(&vk.qr_1.operate_with_self(p.b_zeta.representative()));
        first_term = first_term.operate_with(&vk.qo_1.operate_with_self(p.c_zeta.representative()));
        first_term = first_term.operate_with(&vk.qc_1);

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

        if !self
            .commitment_scheme
            .verify_batch(&zeta, &ys, &commitments, &p.w_zeta_1, &upsilon)
        {
            return Err(VerifierError::BatchOpeningFailed);
        }

        if !self.commitment_scheme.verify(
            &(zeta * &input.omega),
            &p.z_zeta_omega,
            &p.z_1,
            &p.w_zeta_omega_1,
        ) {
            return Err(VerifierError::SingleOpeningFailed);
        }

        Ok(())
    }

    /// Verifies a PLONK proof (legacy interface returning bool).
    ///
    /// This method is kept for backwards compatibility. For new code, prefer
    /// `verify_with_result` which provides detailed error information.
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
        // Note: This legacy method skips subgroup checks for backwards compatibility.
        // Use verify_with_result for full validation.
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
        let proof = prover
            .prove(
                &witness,
                &public_input,
                &common_preprocessed_input,
                &verifying_key,
            )
            .unwrap();

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
        let proof = prover
            .prove(
                &witness,
                &public_input,
                &common_preprocessed_input,
                &verifying_key,
            )
            .unwrap();

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
        let proof = prover
            .prove(
                &witness,
                &public_input,
                &common_preprocessed_input,
                &verifying_key,
            )
            .unwrap();

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
        let proof = prover
            .prove(
                &witness,
                &public_input,
                &common_preprocessed_input,
                &verifying_key,
            )
            .unwrap();

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

    // ============================================
    // Negative tests - verifier must reject invalid proofs
    // ============================================

    #[test]
    fn test_rejects_wrong_public_input() {
        // x * e == y where x=4, e=3, y=12
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);

        let x = FieldElement::from(4_u64);
        let y = FieldElement::from(12_u64);
        let e = FieldElement::from(3_u64);

        let public_input = vec![x.clone(), y];
        let witness = test_witness_1(x, e);

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);
        let random_generator = TestRandomFieldGenerator {};

        let prover = Prover::new(kzg.clone(), random_generator);
        let proof = prover
            .prove(
                &witness,
                &public_input,
                &common_preprocessed_input,
                &verifying_key,
            )
            .unwrap();

        // Try to verify with wrong public input
        let wrong_public_input = vec![FieldElement::from(4_u64), FieldElement::from(13_u64)];

        let verifier = Verifier::new(kzg);
        assert!(
            !verifier.verify(
                &proof,
                &wrong_public_input,
                &common_preprocessed_input,
                &verifying_key
            ),
            "Verifier should reject proof with wrong public input"
        );
    }

    #[test]
    fn test_rejects_tampered_a_zeta() {
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);

        let x = FieldElement::from(4_u64);
        let y = FieldElement::from(12_u64);
        let e = FieldElement::from(3_u64);

        let public_input = vec![x.clone(), y];
        let witness = test_witness_1(x, e);

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);
        let random_generator = TestRandomFieldGenerator {};

        let prover = Prover::new(kzg.clone(), random_generator);
        let mut proof = prover
            .prove(
                &witness,
                &public_input,
                &common_preprocessed_input,
                &verifying_key,
            )
            .unwrap();

        // Tamper with a_zeta (polynomial evaluation)
        proof.a_zeta = &proof.a_zeta + FieldElement::one();

        let verifier = Verifier::new(kzg);
        assert!(
            !verifier.verify(
                &proof,
                &public_input,
                &common_preprocessed_input,
                &verifying_key
            ),
            "Verifier should reject proof with tampered a_zeta"
        );
    }

    #[test]
    fn test_rejects_tampered_t_zeta() {
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);

        let x = FieldElement::from(4_u64);
        let y = FieldElement::from(12_u64);
        let e = FieldElement::from(3_u64);

        let public_input = vec![x.clone(), y];
        let witness = test_witness_1(x, e);

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);
        let random_generator = TestRandomFieldGenerator {};

        let prover = Prover::new(kzg.clone(), random_generator);
        let mut proof = prover
            .prove(
                &witness,
                &public_input,
                &common_preprocessed_input,
                &verifying_key,
            )
            .unwrap();

        // Tamper with t_zeta (quotient polynomial evaluation)
        proof.t_zeta = &proof.t_zeta + FieldElement::one();

        let verifier = Verifier::new(kzg);
        assert!(
            !verifier.verify(
                &proof,
                &public_input,
                &common_preprocessed_input,
                &verifying_key
            ),
            "Verifier should reject proof with tampered t_zeta"
        );
    }

    #[test]
    fn test_rejects_tampered_z_zeta_omega() {
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);

        let x = FieldElement::from(4_u64);
        let y = FieldElement::from(12_u64);
        let e = FieldElement::from(3_u64);

        let public_input = vec![x.clone(), y];
        let witness = test_witness_1(x, e);

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);
        let random_generator = TestRandomFieldGenerator {};

        let prover = Prover::new(kzg.clone(), random_generator);
        let mut proof = prover
            .prove(
                &witness,
                &public_input,
                &common_preprocessed_input,
                &verifying_key,
            )
            .unwrap();

        // Tamper with z_zeta_omega (permutation polynomial shifted evaluation)
        proof.z_zeta_omega = &proof.z_zeta_omega + FieldElement::one();

        let verifier = Verifier::new(kzg);
        assert!(
            !verifier.verify(
                &proof,
                &public_input,
                &common_preprocessed_input,
                &verifying_key
            ),
            "Verifier should reject proof with tampered z_zeta_omega"
        );
    }

    #[test]
    fn test_rejects_swapped_commitments() {
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);

        let x = FieldElement::from(4_u64);
        let y = FieldElement::from(12_u64);
        let e = FieldElement::from(3_u64);

        let public_input = vec![x.clone(), y];
        let witness = test_witness_1(x, e);

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);
        let random_generator = TestRandomFieldGenerator {};

        let prover = Prover::new(kzg.clone(), random_generator);
        let mut proof = prover
            .prove(
                &witness,
                &public_input,
                &common_preprocessed_input,
                &verifying_key,
            )
            .unwrap();

        // Swap a_1 and b_1 commitments
        std::mem::swap(&mut proof.a_1, &mut proof.b_1);

        let verifier = Verifier::new(kzg);
        assert!(
            !verifier.verify(
                &proof,
                &public_input,
                &common_preprocessed_input,
                &verifying_key
            ),
            "Verifier should reject proof with swapped commitments"
        );
    }

    #[test]
    fn test_rejects_empty_public_input_for_circuit_with_public_inputs() {
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);

        let x = FieldElement::from(4_u64);
        let y = FieldElement::from(12_u64);
        let e = FieldElement::from(3_u64);

        let public_input = vec![x.clone(), y];
        let witness = test_witness_1(x, e);

        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);
        let random_generator = TestRandomFieldGenerator {};

        let prover = Prover::new(kzg.clone(), random_generator);
        let proof = prover
            .prove(
                &witness,
                &public_input,
                &common_preprocessed_input,
                &verifying_key,
            )
            .unwrap();

        // Try to verify with empty public input
        let empty_public_input: Vec<FieldElement<_>> = vec![];

        let verifier = Verifier::new(kzg);
        assert!(
            !verifier.verify(
                &proof,
                &empty_public_input,
                &common_preprocessed_input,
                &verifying_key
            ),
            "Verifier should reject proof with missing public inputs"
        );
    }
}
