//! Groth16 proof generation.
//!
//! This module contains the prover implementation that generates zero-knowledge
//! proofs from witnesses satisfying an R1CS constraint system.

use crate::{common::*, errors::Groth16Error, ProvingKey, QuadraticArithmeticProgram};
use lambdaworks_math::errors::DeserializationError;
use lambdaworks_math::traits::{deserialize_with_length, serialize_with_length};
use lambdaworks_math::{cyclic_group::IsGroup, msm::pippenger::msm};

/// A Groth16 proof consisting of three group elements.
///
/// The proof is constant-size (3 group elements) regardless of the circuit complexity.
/// - `pi1` (A): Element in G1
/// - `pi2` (B): Element in G2
/// - `pi3` (C): Element in G1
///
/// The proof satisfies the pairing equation:
/// `e(A, B) = e(α, β) · e(L, γ) · e(C, δ)`
/// where L is derived from the public inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proof {
    /// First proof element [π₁]₁ in G1
    pub pi1: G1Point,
    /// Second proof element [π₂]₂ in G2
    pub pi2: G2Point,
    /// Third proof element [π₃]₁ in G1
    pub pi3: G1Point,
}

impl Proof {
    /// Serializes the proof to bytes.
    ///
    /// The serialization format is:
    /// - pi1 (G1 point with length prefix)
    /// - pi2 (G2 point with length prefix)
    /// - pi3 (G1 point with length prefix)
    pub fn serialize(&self) -> Vec<u8> {
        [
            serialize_with_length(&self.pi1),
            serialize_with_length(&self.pi2),
            serialize_with_length(&self.pi3),
        ]
        .concat()
    }

    /// Deserializes a proof from bytes.
    ///
    /// # Errors
    ///
    /// Returns `DeserializationError` if the bytes don't represent a valid proof.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        let (offset, pi1) = deserialize_with_length::<G1Point>(bytes, 0)?;
        let (offset, pi2) = deserialize_with_length::<G2Point>(bytes, offset)?;
        let (_, pi3) = deserialize_with_length::<G1Point>(bytes, offset)?;
        Ok(Self { pi1, pi2, pi3 })
    }
}

/// The Groth16 prover.
///
/// This struct provides the `prove` method to generate zero-knowledge proofs.
pub struct Prover;

impl Prover {
    /// Generates a Groth16 proof for the given witness.
    ///
    /// # Arguments
    ///
    /// * `w` - The full witness vector including public inputs and private values.
    ///   The first `qap.num_of_public_inputs` elements are public inputs.
    /// * `qap` - The Quadratic Arithmetic Program describing the circuit
    /// * `pk` - The proving key from the trusted setup
    ///
    /// # Returns
    ///
    /// A `Proof` that can be verified with the corresponding verifying key.
    ///
    /// # Errors
    ///
    /// Returns `Groth16Error` if:
    /// - QAP coefficient computation fails
    /// - Multi-scalar multiplication fails
    pub fn prove(
        w: &[FrElement],
        qap: &QuadraticArithmeticProgram,
        pk: &ProvingKey,
    ) -> Result<Proof, Groth16Error> {
        // Compute the coefficients of the quotient polynomial
        let h_coefficients: Vec<_> = qap
            .calculate_h_coefficients(w)?
            .into_iter()
            .map(|elem| elem.canonical())
            .collect();

        let w: Vec<_> = w.iter().map(|elem| elem.canonical()).collect();

        // Sample randomness for hiding
        let r = sample_fr_elem();
        let s = sample_fr_elem();

        // [π_1]_1
        let pi1 = msm(&w, &pk.l_tau_g1)
            .map_err(Groth16Error::msm)?
            .operate_with(&pk.alpha_g1)
            .operate_with(&pk.delta_g1.operate_with_self(r.canonical()));

        // [π_2]_2
        let pi2 = msm(&w, &pk.r_tau_g2)
            .map_err(Groth16Error::msm)?
            .operate_with(&pk.beta_g2)
            .operate_with(&pk.delta_g2.operate_with_self(s.canonical()));

        // [ƍ^{-1} * t(τ)*h(τ)]_1
        let t_tau_h_tau_assigned_g1 = msm(
            &h_coefficients,
            &pk.z_powers_of_tau_g1[..h_coefficients.len()],
        )
        .map_err(Groth16Error::msm)?;

        // [ƍ^{-1} * (β*l(τ) + α*r(τ) + o(τ))]_1
        let k_tau_assigned_prover_g1 = msm(
            &w[qap.num_of_public_inputs..],
            &pk.prover_k_tau_g1[..qap.num_of_private_inputs()],
        )
        .map_err(Groth16Error::msm)?;

        // [π_2]_1
        let pi2_g1 = msm(&w, &pk.r_tau_g1)
            .map_err(Groth16Error::msm)?
            .operate_with(&pk.beta_g1)
            .operate_with(&pk.delta_g1.operate_with_self(s.canonical()));

        // [π_3]_1
        let pi3 = k_tau_assigned_prover_g1
            .operate_with(&t_tau_h_tau_assigned_g1)
            // s[π_1]_1
            .operate_with(&pi1.operate_with_self(s.canonical()))
            // r[π_2]_1
            .operate_with(&pi2_g1.operate_with_self(r.canonical()))
            // -rs[ƍ]_1
            .operate_with(&pk.delta_g1.operate_with_self((-(&r * &s)).canonical()));

        Ok(Proof { pi1, pi2, pi3 })
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;

    use super::*;

    #[test]
    fn serde() {
        let proof = Proof {
            pi1: Curve::generator().operate_with_self(sample_fr_elem().canonical()),
            pi2: TwistedCurve::generator().operate_with_self(sample_fr_elem().canonical()),
            pi3: Curve::generator().operate_with_self(sample_fr_elem().canonical()),
        };
        let deserialized_proof = Proof::deserialize(&proof.serialize()).unwrap();

        assert_eq!(proof.pi1, deserialized_proof.pi1);
        assert_eq!(proof.pi2, deserialized_proof.pi2);
        assert_eq!(proof.pi3, deserialized_proof.pi3);
    }
}
