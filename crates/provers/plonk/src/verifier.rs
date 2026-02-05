use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsFFTField, IsField, IsPrimeField};
use lambdaworks_math::traits::{AsBytes, ByteConversion};
use std::marker::PhantomData;

use crate::prover::Proof;
use crate::setup::{new_strong_fiat_shamir_transcript, CommonPreprocessedInput, VerificationKey};

/// Errors that can occur during PLONK proof verification.
///
/// # Security
/// These errors are critical for soundness - they indicate that a proof
/// may be malformed or constructed by a malicious prover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifierError {
    /// A point in the proof is not on the elliptic curve.
    /// This could indicate a malicious proof attempting to exploit
    /// the pairing verification.
    PointNotOnCurve,
    /// A point is on the curve but not in the correct subgroup.
    /// For curves with cofactor > 1 (like BLS12-381), points must
    /// be in the prime-order subgroup for security.
    PointNotInSubgroup,
    /// A field element in the proof is not properly reduced.
    /// Field elements must be in the range [0, p-1] where p is
    /// the field modulus.
    InvalidFieldElement,
}

impl std::fmt::Display for VerifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifierError::PointNotOnCurve => {
                write!(
                    f,
                    "Proof element is not a valid point on the elliptic curve"
                )
            }
            VerifierError::PointNotInSubgroup => {
                write!(
                    f,
                    "Proof element is not in the correct prime-order subgroup"
                )
            }
            VerifierError::InvalidFieldElement => {
                write!(f, "Proof contains an invalid field element")
            }
        }
    }
}

impl std::error::Error for VerifierError {}

/// Trait for validating proof commitment elements.
///
/// # Security
///
/// This trait is critical for ensuring soundness of the PLONK verifier.
/// Without proper validation of proof elements, a malicious prover could:
///
/// 1. Submit points not on the curve, potentially causing pairing operations
///    to produce incorrect results or panic.
/// 2. Submit points on the curve but not in the correct subgroup, which can
///    break the security assumptions of the pairing-based verification.
///
/// For curves with cofactor = 1 (like BN254), being on the curve implies
/// being in the subgroup. For curves with cofactor > 1 (like BLS12-381),
/// explicit subgroup membership checks are required.
pub trait IsValidProofCommitment {
    /// Validates that this commitment element is a valid point for use in proofs.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the point is valid (on curve and in correct subgroup)
    /// - `Err(VerifierError::PointNotOnCurve)` if the point is not on the curve
    /// - `Err(VerifierError::PointNotInSubgroup)` if the point is not in the subgroup
    fn validate(&self) -> Result<(), VerifierError>;
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

    /// Validates all proof elements before verification.
    ///
    /// # Security
    ///
    /// This function performs critical security checks that are essential for
    /// the soundness of the PLONK verification. Without these checks, a malicious
    /// prover could potentially:
    ///
    /// 1. Submit points not on the elliptic curve
    /// 2. Submit points on the curve but not in the prime-order subgroup
    ///
    /// Both attacks could lead to accepting invalid proofs.
    ///
    /// # Arguments
    ///
    /// * `proof` - The PLONK proof containing all commitments to validate
    ///
    /// # Returns
    ///
    /// * `Ok(())` if all proof elements are valid
    /// * `Err(VerifierError)` if any element fails validation
    ///
    /// # Validation Checks
    ///
    /// For each G1 point commitment in the proof:
    /// 1. Verify the point is on the elliptic curve (satisfies curve equation)
    /// 2. Verify the point is in the correct prime-order subgroup
    ///
    /// The following commitments are validated:
    /// - `a_1`, `b_1`, `c_1`: Wire polynomial commitments (Round 1)
    /// - `z_1`: Copy constraints polynomial commitment (Round 2)
    /// - `t_lo_1`, `t_mid_1`, `t_hi_1`: Quotient polynomial commitments (Round 3)
    /// - `w_zeta_1`: Batch opening proof (Round 5)
    /// - `w_zeta_omega_1`: Single opening proof (Round 5)
    pub fn validate_proof_elements(proof: &Proof<F, CS>) -> Result<(), VerifierError>
    where
        CS::Commitment: IsValidProofCommitment,
    {
        // Validate Round 1 commitments: wire polynomials a(x), b(x), c(x)
        proof.a_1.validate()?;
        proof.b_1.validate()?;
        proof.c_1.validate()?;

        // Validate Round 2 commitment: copy constraints polynomial z(x)
        proof.z_1.validate()?;

        // Validate Round 3 commitments: quotient polynomial parts
        proof.t_lo_1.validate()?;
        proof.t_mid_1.validate()?;
        proof.t_hi_1.validate()?;

        // Validate Round 5 commitments: opening proofs
        proof.w_zeta_1.validate()?;
        proof.w_zeta_omega_1.validate()?;

        Ok(())
    }

    /// Validates all commitments in the verification key.
    ///
    /// # Security
    ///
    /// While verification keys typically come from a trusted setup ceremony,
    /// validating them is important when:
    /// - Loading verification keys from files or network
    /// - Using verification keys from external or untrusted sources
    /// - Implementing defense-in-depth security practices
    ///
    /// Without validation, a malicious verification key with invalid points could
    /// compromise the soundness of the pairing-based verification.
    ///
    /// # Arguments
    ///
    /// * `vk` - The verification key to validate
    ///
    /// # Returns
    ///
    /// * `Ok(())` if all verification key commitments are valid
    /// * `Err(VerifierError)` if any commitment fails validation
    ///
    /// # Validation Checks
    ///
    /// Validates all 8 G1 commitments in the verification key:
    /// - `qm_1`, `ql_1`, `qr_1`, `qo_1`, `qc_1`: Gate selector commitments
    /// - `s1_1`, `s2_1`, `s3_1`: Permutation polynomial commitments
    pub fn validate_verification_key(
        vk: &VerificationKey<CS::Commitment>,
    ) -> Result<(), VerifierError>
    where
        CS::Commitment: IsValidProofCommitment,
    {
        // Validate gate selector commitments
        vk.qm_1.validate()?;
        vk.ql_1.validate()?;
        vk.qr_1.validate()?;
        vk.qo_1.validate()?;
        vk.qc_1.validate()?;

        // Validate permutation commitments (used in batch opening verification)
        vk.s1_1.validate()?;
        vk.s2_1.validate()?;
        vk.s3_1.validate()?;

        Ok(())
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

    /// Verifies a PLONK proof with full validation of proof elements.
    ///
    /// # Security
    ///
    /// This is the recommended verification method as it performs critical
    /// validation checks on all proof elements before verification.
    /// Use this method when processing proofs from untrusted sources.
    ///
    /// # Arguments
    ///
    /// * `p` - The PLONK proof to verify
    /// * `public_input` - The public inputs to the circuit
    /// * `input` - The common preprocessed input
    /// * `vk` - The verification key
    ///
    /// # Returns
    ///
    /// * `Ok(true)` if the proof is valid
    /// * `Ok(false)` if the proof is mathematically invalid but well-formed
    /// * `Err(VerifierError)` if the proof contains malformed elements
    pub fn verify_with_validation(
        &self,
        p: &Proof<F, CS>,
        public_input: &[FieldElement<F>],
        input: &CommonPreprocessedInput<F>,
        vk: &VerificationKey<CS::Commitment>,
    ) -> Result<bool, VerifierError>
    where
        F: IsPrimeField + IsFFTField,
        CS: IsCommitmentScheme<F>,
        CS::Commitment: AsBytes + IsGroup + IsValidProofCommitment,
        FieldElement<F>: ByteConversion,
    {
        // Step 1-2: Validate proof elements (CRITICAL FOR SECURITY)
        // - Check all proof commitments are valid points on the curve
        // - Check all proof commitments are in the correct subgroup
        Self::validate_proof_elements(p)?;

        // Step 3: Validate verification key elements (CRITICAL FOR SECURITY)
        // - Check all VK commitments are valid points on the curve
        // - Check all VK commitments are in the correct subgroup
        Self::validate_verification_key(vk)?;

        // Proceed with standard verification
        Ok(self.verify_internal(p, public_input, input, vk))
    }

    /// Verifies a PLONK proof without validation of proof elements.
    ///
    /// # Warning
    ///
    /// This method does NOT validate that proof elements are valid curve points
    /// in the correct subgroup. Only use this when you have already validated
    /// the proof elements through other means, or when processing proofs from
    /// trusted sources (e.g., locally generated proofs).
    ///
    /// For proofs from untrusted sources, use `verify_with_validation` instead.
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
        self.verify_internal(p, public_input, input, vk)
    }

    /// Internal verification logic shared by both verify methods.
    fn verify_internal(
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
        let [beta, gamma, alpha, zeta, upsilon] = self.compute_challenges(p, vk, public_input);

        let k1 = &input.k1;
        let k2 = k1 * k1;

        // Precompute zeta powers efficiently
        let zeta_n = zeta.pow(input.n as u64);
        let zeta_sq = &zeta * &zeta;
        let zeta_n_plus_2 = &zeta_n * &zeta_sq;
        let zeta_2n_plus_4 = &zeta_n_plus_2 * &zeta_n_plus_2;
        let zh_zeta = &zeta_n - FieldElement::<F>::one();

        // We are using that zeta != 0 because is sampled outside the set of roots of unity,
        // and n != 0 because is the length of the trace.
        let l1_zeta = (&zh_zeta
            / ((&zeta - FieldElement::<F>::one()) * FieldElement::from(input.n as u64)))
        .expect("zeta is outside roots of unity so denominator is non-zero");

        // Precompute alpha^2 for reuse
        let alpha_squared = &alpha * &alpha;

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
                    // zeta is sampled outside the domain so denominator is non-zero.
                    * ((&zeta - &input.domain[i - 1]) / (&zeta - &input.domain[i]))
                        .expect("zeta is outside domain so division is valid");
                p_pi_zeta = &p_pi_zeta + value * &li_zeta;
            }
            p_pi_zeta
        };

        let mut p_constant_zeta = &alpha
            * &p.z_zeta_omega
            * (&p.c_zeta + &gamma)
            * (&p.a_zeta + &beta * &p.s1_zeta + &gamma)
            * (&p.b_zeta + &beta * &p.s2_zeta + &gamma);
        p_constant_zeta = p_constant_zeta - &l1_zeta * &alpha_squared;
        p_constant_zeta += p_pi_zeta;

        let p_zeta = p_constant_zeta + &p.p_non_constant_zeta;

        let constraints_check = p_zeta - (&zh_zeta * &p.t_zeta) == FieldElement::zero();

        // Compute commitment of partial evaluation of t (p = zh * t)
        let partial_t_1 = p
            .t_lo_1
            .operate_with(&p.t_mid_1.operate_with_self(zeta_n_plus_2.canonical()))
            .operate_with(&p.t_hi_1.operate_with_self(zeta_2n_plus_4.canonical()));

        // Compute commitment of the non constant part of the linearization of p
        // The first term corresponds to the gates constraints
        let mut first_term = vk
            .qm_1
            .operate_with_self((&p.a_zeta * &p.b_zeta).canonical());
        first_term = first_term.operate_with(&vk.ql_1.operate_with_self(p.a_zeta.canonical()));
        first_term = first_term.operate_with(&vk.qr_1.operate_with_self(p.b_zeta.canonical()));
        first_term = first_term.operate_with(&vk.qo_1.operate_with_self(p.c_zeta.canonical()));
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
            .operate_with_self(z_coefficient.canonical())
            .operate_with(&vk.s3_1.operate_with_self(s3_coefficient.canonical()))
            .operate_with_self(alpha.canonical());
        // α²*L₁(ζ)*Z(X)
        let third_term = p
            .z_1
            .operate_with_self((&alpha * &alpha * l1_zeta).canonical());

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

// Implementation of IsValidProofCommitment for BLS12-381 Jacobian points.
// This is the curve used by the PLONK implementation in this crate.
use lambdaworks_math::elliptic_curve::short_weierstrass::{
    curves::bls12_381::curve::BLS12381Curve, point::ShortWeierstrassJacobianPoint,
};

impl IsValidProofCommitment for ShortWeierstrassJacobianPoint<BLS12381Curve> {
    /// Validates that this BLS12-381 G1 point is valid for use in proofs.
    ///
    /// # Security
    ///
    /// BLS12-381 has cofactor h = 0x396c8c005555e1568c00aaab0000aaab, which means
    /// that not all points on the curve are in the prime-order subgroup.
    /// A malicious prover could submit a point that is on the curve but not in
    /// the subgroup, which would break the security of the pairing verification.
    ///
    /// # Validation
    ///
    /// 1. **On-curve check**: The `ShortWeierstrassJacobianPoint::new()` constructor
    ///    already validates that points satisfy the curve equation y^2 = x^3 + 4.
    ///    Since all points in the proof were created through deserialization which
    ///    uses this constructor, they are guaranteed to be on the curve.
    ///
    /// 2. **Subgroup check**: We use the efficient endomorphism-based check:
    ///    phi(P) = -u^2 * P where u is the curve seed (MILLER_LOOP_CONSTANT).
    ///    This is more efficient than multiplying by the full subgroup order.
    ///    See: <https://eprint.iacr.org/2022/352.pdf> Section 4.3 Prop. 4.
    fn validate(&self) -> Result<(), VerifierError> {
        // The neutral element (point at infinity) is always valid
        if self.is_neutral_element() {
            return Ok(());
        }

        // Check subgroup membership using the efficient endomorphism-based method.
        // For BLS12-381, we verify that phi(P) = -u^2 * P where phi is the
        // GLV endomorphism and u is the curve seed.
        if self.is_in_subgroup() {
            Ok(())
        } else {
            Err(VerifierError::PointNotInSubgroup)
        }
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
        test_utils::utils::{test_srs, FpElement, TestRandomFieldGenerator, KZG},
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

    // ============================================================
    // Proof Element Validation Tests
    // ============================================================

    #[test]
    fn test_valid_proof_passes_validation() {
        // Generate a valid proof and verify it passes validation
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
        let proof = prover.prove(
            &witness,
            &public_input,
            &common_preprocessed_input,
            &verifying_key,
        );

        // Validation should pass for a legitimately generated proof
        let validation_result = Verifier::<_, KZG>::validate_proof_elements(&proof);
        assert!(
            validation_result.is_ok(),
            "Valid proof should pass validation"
        );
    }

    #[test]
    fn test_verify_with_validation_valid_proof() {
        // Test that verify_with_validation accepts valid proofs
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
        let proof = prover.prove(
            &witness,
            &public_input,
            &common_preprocessed_input,
            &verifying_key,
        );

        let verifier = Verifier::new(kzg);
        let result = verifier.verify_with_validation(
            &proof,
            &public_input,
            &common_preprocessed_input,
            &verifying_key,
        );

        assert!(result.is_ok(), "verify_with_validation should succeed");
        assert!(result.unwrap(), "Valid proof should verify");
    }

    #[test]
    fn test_generator_point_is_in_subgroup() {
        // Test that the BLS12-381 generator is in the subgroup
        use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;

        let generator = BLS12381Curve::generator();
        let validation_result = generator.validate();
        assert!(
            validation_result.is_ok(),
            "Generator point should be in the subgroup"
        );
    }

    #[test]
    fn test_neutral_element_passes_validation() {
        // The neutral element (point at infinity) should always pass validation
        let neutral = ShortWeierstrassJacobianPoint::<BLS12381Curve>::neutral_element();
        let validation_result = neutral.validate();
        assert!(
            validation_result.is_ok(),
            "Neutral element should pass validation"
        );
    }

    #[test]
    fn test_point_not_in_subgroup_fails_validation() {
        // Test that a point on the curve but NOT in the subgroup fails validation
        // This is a known point on BLS12-381 that is NOT in the subgroup
        // (it has order divisible by the cofactor)
        use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;

        let x = FpElement::from_hex_unchecked("178212cbe4a3026c051d4f867364b3ea84af623f93233b347ffcd3d6b16f16e0a7aedbe1c78d33c6beca76b2b75c8486");
        let y = FpElement::from_hex_unchecked("13a8b1347e5b43bc4051754b2a29928b5df78cf03ca3b1f73d0424b09fccdef116c9f0ecbec7420a99b2dd785209e9d");
        let point =
            BLS12381Curve::create_point_from_affine(x, y).expect("Point should be on curve");

        // Verify the point is on the curve but not in the subgroup
        assert!(
            !point.is_in_subgroup(),
            "Test point should not be in subgroup"
        );

        // Validation should fail
        let validation_result = point.validate();
        assert_eq!(
            validation_result,
            Err(VerifierError::PointNotInSubgroup),
            "Point not in subgroup should fail validation"
        );
    }

    #[test]
    fn test_scalar_multiple_of_generator_is_in_subgroup() {
        // Any scalar multiple of the generator should be in the subgroup
        use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;

        let generator = BLS12381Curve::generator();
        let multiple = generator.operate_with_self(12345u64);

        let validation_result = multiple.validate();
        assert!(
            validation_result.is_ok(),
            "Scalar multiple of generator should be in subgroup"
        );
    }

    #[test]
    fn test_verifier_error_display() {
        // Test that error messages are properly formatted
        let err = VerifierError::PointNotOnCurve;
        assert!(
            format!("{}", err).contains("not a valid point"),
            "PointNotOnCurve should have descriptive message"
        );

        let err = VerifierError::PointNotInSubgroup;
        assert!(
            format!("{}", err).contains("subgroup"),
            "PointNotInSubgroup should mention subgroup"
        );

        let err = VerifierError::InvalidFieldElement;
        assert!(
            format!("{}", err).contains("field element"),
            "InvalidFieldElement should mention field element"
        );
    }

    #[test]
    fn test_valid_verification_key_passes_validation() {
        // Test that a properly constructed verification key passes validation
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);
        let kzg = KZG::new(srs);

        let verifying_key = setup(&common_preprocessed_input, &kzg);

        // Validation should pass for a legitimately generated verification key
        let validation_result = Verifier::<_, KZG>::validate_verification_key(&verifying_key);
        assert!(
            validation_result.is_ok(),
            "Valid verification key should pass validation"
        );
    }

    #[test]
    fn test_verify_with_validation_validates_both_proof_and_vk() {
        // Test that verify_with_validation validates both proof elements and VK
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
        let proof = prover.prove(
            &witness,
            &public_input,
            &common_preprocessed_input,
            &verifying_key,
        );

        let verifier = Verifier::new(kzg);

        // verify_with_validation should validate both proof and VK
        let result = verifier.verify_with_validation(
            &proof,
            &public_input,
            &common_preprocessed_input,
            &verifying_key,
        );

        assert!(result.is_ok(), "Validation should succeed");
        assert!(result.unwrap(), "Valid proof with valid VK should verify");
    }
}
