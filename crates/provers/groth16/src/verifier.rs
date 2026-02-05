use lambdaworks_math::{elliptic_curve::traits::IsPairing, msm::pippenger::msm};

use crate::common::{FrElement, Pairing};
use crate::errors::Groth16Error;
use crate::prover::Proof;
use crate::setup::VerifyingKey;

/// Verifies a Groth16 proof against public inputs.
///
/// # Arguments
///
/// * `vk` - The verifying key generated during setup
/// * `proof` - The proof to verify
/// * `pub_inputs` - The public inputs to the circuit
///
/// # Returns
///
/// `Ok(true)` if the proof is valid, `Ok(false)` if invalid,
/// or an error if verification cannot be performed.
pub fn verify(
    vk: &VerifyingKey,
    proof: &Proof,
    pub_inputs: &[FrElement],
) -> Result<bool, Groth16Error> {
    // [γ^{-1} * (β*l(τ) + α*r(τ) + o(τ))]_1
    let pub_inputs_canonical: Vec<_> = pub_inputs.iter().map(|elem| elem.canonical()).collect();
    let k_tau_assigned_verifier_g1 =
        msm(&pub_inputs_canonical, &vk.verifier_k_tau_g1).map_err(Groth16Error::msm)?;

    let lhs = Pairing::compute(&proof.pi3, &vk.delta_g2).map_err(Groth16Error::pairing)?
        * vk.alpha_g1_times_beta_g2.clone()
        * Pairing::compute(&k_tau_assigned_verifier_g1, &vk.gamma_g2)
            .map_err(Groth16Error::pairing)?;

    let rhs = Pairing::compute(&proof.pi1, &proof.pi2).map_err(Groth16Error::pairing)?;

    Ok(lhs == rhs)
}
