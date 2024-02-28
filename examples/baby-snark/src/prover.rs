use crate::{common::*, setup::ProvingKey, ssp::SquareSpanProgram};
use lambdaworks_math::msm::pippenger::msm;
pub struct Proof {
    pub H: G1Point,
    pub V_w: G1Point,
    pub V_w_prime: G2Point,
    pub B_w: G1Point,
}

pub struct Prover;
impl Prover {
    pub fn prove(w: &[FrElement], ssp: &SquareSpanProgram, pk: &ProvingKey) -> Proof {
        let h_coefficients = ssp
            .calculate_h_coefficients(w)
            .iter()
            .map(|elem| elem.representative())
            .collect::<Vec<_>>();

        let H = msm(&h_coefficients, &pk.k_powers_of_tau_g1).unwrap();

        let w = w
            .iter()
            .map(|elem| elem.representative())
            .collect::<Vec<_>>();

        let V_w = msm(&w, &pk.u_tau_g1).unwrap();

        let V_w_prime = msm(&w, &pk.u_tau_g2).unwrap();

        let B_w = msm(&w, &pk.beta_u_tau_g1).unwrap();

        Proof {
            H,
            V_w,
            V_w_prime,
            B_w,
        }
    }
}
