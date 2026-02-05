use crate::{common::*, errors::Groth16Error, QuadraticArithmeticProgram};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{point::ShortWeierstrassJacobianPoint, traits::IsShortWeierstrass},
        traits::{IsEllipticCurve, IsPairing},
    },
};

/// The verifying key used to verify Groth16 proofs.
///
/// This key is generated during the setup phase and can be made public.
/// It contains the minimal information needed to verify proofs.
pub struct VerifyingKey {
    /// e([alpha]_1, [beta]_2) computed during setup as it's a constant
    pub alpha_g1_times_beta_g2: PairingOutput,
    /// [delta]_2
    pub delta_g2: G2Point,
    /// [gamma]_2
    pub gamma_g2: G2Point,
    /// [K_0(τ)]_1, [K_1(τ)]_1, ..., [K_k(τ)]_1
    /// where K_i(τ) = γ^{-1} * (β*l(τ) + α*r(τ) + o(τ))
    /// and "k" is the number of public inputs
    pub verifier_k_tau_g1: Vec<G1Point>,
}

/// The proving key used to generate Groth16 proofs.
///
/// This key is generated during the setup phase and must be kept
/// by the prover. It contains all information needed to create proofs.
pub struct ProvingKey {
    /// [alpha]_1
    pub alpha_g1: G1Point,
    /// [beta]_1
    pub beta_g1: G1Point,
    /// [beta]_2
    pub beta_g2: G2Point,
    /// [delta]_1
    pub delta_g1: G1Point,
    /// [delta]_2
    pub delta_g2: G2Point,
    /// [A_0(τ)]_1, [A_1(τ)]_1, ..., [A_n(τ)]_1
    pub l_tau_g1: Vec<G1Point>,
    /// [B_0(τ)]_1, [B_1(τ)]_1, ..., [B_n(τ)]_1
    pub r_tau_g1: Vec<G1Point>,
    /// [B_0(τ)]_2, [B_1(τ)]_2, ..., [B_n(τ)]_2
    pub r_tau_g2: Vec<G2Point>,
    /// [K_{k+1}(τ)]_1, [K_{k+2}(τ)]_1, ..., [K_n(τ)]_1
    /// where K_i(τ) = δ^{-1} * (β*l(τ) + α*r(τ) + o(τ))
    /// and "k" is the number of public inputs
    pub prover_k_tau_g1: Vec<G1Point>,
    /// [delta^{-1} * t(τ) * τ^0]_1, [delta^{-1} * t(τ) * τ^1]_1, ..., [delta^{-1} * t(τ) * τ^m]_1
    pub z_powers_of_tau_g1: Vec<G1Point>,
}

/// Toxic waste from the trusted setup ceremony.
///
/// These values must be securely destroyed after setup.
/// In production, a multi-party computation (MPC) ceremony should be used.
struct ToxicWaste {
    tau: FrElement,
    alpha: FrElement,
    beta: FrElement,
    gamma: FrElement,
    delta: FrElement,
}

impl ToxicWaste {
    /// Creates new random toxic waste from entropy.
    ///
    /// # Warning
    ///
    /// This uses local randomness and is NOT safe for production use.
    /// A proper ceremony like Powers of Tau should be used instead.
    fn new() -> Self {
        Self {
            tau: sample_fr_elem(),
            alpha: sample_fr_elem(),
            beta: sample_fr_elem(),
            gamma: sample_fr_elem(),
            delta: sample_fr_elem(),
        }
    }
}

/// Performs the Groth16 trusted setup ceremony.
///
/// # Arguments
///
/// * `qap` - The Quadratic Arithmetic Program representing the circuit
///
/// # Returns
///
/// A tuple of (ProvingKey, VerifyingKey) on success.
///
/// # Warning
///
/// This function uses local randomness for the toxic waste. In production,
/// use a multi-party computation ceremony to generate the keys securely.
pub fn setup(qap: &QuadraticArithmeticProgram) -> Result<(ProvingKey, VerifyingKey), Groth16Error> {
    let g1: G1Point = Curve::generator();
    let g2: G2Point = TwistedCurve::generator();

    let tw = ToxicWaste::new();

    let l_tau: Vec<_> = qap.l.iter().map(|p| p.evaluate(&tw.tau)).collect();
    let r_tau: Vec<_> = qap.r.iter().map(|p| p.evaluate(&tw.tau)).collect();

    let mut to_be_inversed = [tw.delta.clone(), tw.gamma.clone()];
    FrElement::inplace_batch_inverse(&mut to_be_inversed)
        .map_err(|_| Groth16Error::SetupError("batch inverse failed (zero element?)".into()))?;
    let [delta_inv, gamma_inv] = to_be_inversed;

    let k_tau: Vec<_> = l_tau
        .iter()
        .zip(&r_tau)
        .enumerate()
        .map(|(i, (l, r))| {
            let unshifted = &tw.beta * l + &tw.alpha * r + &qap.o[i].evaluate(&tw.tau);
            if i < qap.num_of_public_inputs {
                &gamma_inv * &unshifted
            } else {
                &delta_inv * &unshifted
            }
        })
        .collect();

    let alpha_g1 = g1.operate_with_self(tw.alpha.canonical());
    let beta_g2 = g2.operate_with_self(tw.beta.canonical());

    let alpha_g1_times_beta_g2 = Pairing::compute(&alpha_g1, &beta_g2)
        .map_err(|e| Groth16Error::PairingError(format!("{:?}", e)))?;

    let delta_g2 = g2.operate_with_self(tw.delta.canonical());

    Ok((
        ProvingKey {
            alpha_g1,
            beta_g1: g1.operate_with_self(tw.beta.canonical()),
            beta_g2,
            delta_g1: g1.operate_with_self(tw.delta.canonical()),
            delta_g2: delta_g2.clone(),
            l_tau_g1: batch_operate(&l_tau, &g1),
            r_tau_g1: batch_operate(&r_tau, &g1),
            r_tau_g2: batch_operate(&r_tau, &g2),
            prover_k_tau_g1: batch_operate(&k_tau[qap.num_of_public_inputs..], &g1),
            z_powers_of_tau_g1: batch_operate(
                &core::iter::successors(
                    // Start from delta^{-1} * t(τ)
                    // Note that t(τ) = (τ^N - 1) because our domain is roots of unity
                    Some(&delta_inv * (&tw.tau.pow(qap.num_of_gates) - FrElement::one())),
                    |prev| Some(prev * &tw.tau),
                )
                .take(qap.num_of_gates * 2)
                .collect::<Vec<_>>(),
                &g1,
            ),
        },
        VerifyingKey {
            alpha_g1_times_beta_g2,
            delta_g2,
            gamma_g2: g2.operate_with_self(tw.gamma.canonical()),
            verifier_k_tau_g1: batch_operate(&k_tau[..qap.num_of_public_inputs], &g1),
        },
    ))
}

fn batch_operate<E: IsShortWeierstrass>(
    elems: &[FrElement],
    point: &ShortWeierstrassJacobianPoint<E>,
) -> Vec<ShortWeierstrassJacobianPoint<E>> {
    elems
        .iter()
        .map(|elem| point.operate_with_self(elem.canonical()))
        .collect()
}
