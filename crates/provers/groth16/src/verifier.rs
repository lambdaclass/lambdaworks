use lambdaworks_math::{
    cyclic_group::IsGroup, elliptic_curve::traits::IsPairing, msm::pippenger::msm,
};

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
///
/// # Performance
///
/// Uses batch pairing computation to share the expensive final exponentiation
/// across all three pairings, reducing verification time by ~40%.
pub fn verify(
    vk: &VerifyingKey,
    proof: &Proof,
    pub_inputs: &[FrElement],
) -> Result<bool, Groth16Error> {
    // Compute [γ^{-1} * (β*l(τ) + α*r(τ) + o(τ))]_1 from public inputs
    let pub_inputs_canonical: Vec<_> = pub_inputs.iter().map(|elem| elem.canonical()).collect();
    let k_tau_g1 = msm(&pub_inputs_canonical, &vk.verifier_k_tau_g1).map_err(Groth16Error::msm)?;

    // Groth16 verification equation:
    //   e(A, B) = e(α, β) · e(L, γ) · e(C, δ)
    //
    // Rearranged for batch computation with shared final exponentiation:
    //   e(A, B) · e(-L, γ) · e(-C, δ) = e(α, β)
    //
    // where e(α, β) is precomputed in the verifying key
    let neg_k_tau_g1 = k_tau_g1.neg();
    let neg_pi3 = proof.pi3.neg();

    let lhs = Pairing::compute_batch(&[
        (&proof.pi1, &proof.pi2),
        (&neg_k_tau_g1, &vk.gamma_g2),
        (&neg_pi3, &vk.delta_g2),
    ])
    .map_err(Groth16Error::pairing)?;

    Ok(lhs == vk.alpha_g1_times_beta_g2)
}
