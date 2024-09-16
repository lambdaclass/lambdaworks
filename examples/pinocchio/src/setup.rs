use crate::common::{sample_fr_elem, Curve, G1Point, G2Point, TwistedCurve, FE};
use lambdaworks_math::{cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve};

use crate::qap::QuadraticArithmeticProgram;

pub struct EvaluationKey {
    pub g1_vk: Vec<G1Point>,
    pub g1_wk: Vec<G1Point>,
    pub g2_wk: Vec<G2Point>,
    pub g1_yk: Vec<G1Point>,
    pub g1_alpha_vk: Vec<G1Point>,
    pub g1_alpha_wk: Vec<G1Point>,
    pub g1_alpha_yk: Vec<G1Point>,
    pub g1_beta: Vec<G1Point>,
    pub g2_s_i: Vec<G2Point>,
}
pub struct VerificationKey {
    pub g2: G2Point,
    pub g2_alpha_v: G2Point,
    pub g2_alpha_w: G2Point,
    pub g2_alpha_y: G2Point,
    pub g2_gamma: G2Point,
    pub g2_beta_gamma: G2Point,
    pub g1y_t: G1Point,
    pub g1_vk: Vec<G1Point>,
    pub g2_wk: Vec<G2Point>,
    pub g1_yk: Vec<G1Point>,
}
pub struct ToxicWaste {
    rv: FE,
    rw: FE,
    s: FE,
    alpha_v: FE,
    alpha_y: FE,
    alpha_w: FE,
    beta: FE,
    gamma: FE,
}
impl ToxicWaste {
    pub fn ry(&self) -> FE {
        &self.rv * &self.rw
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s: FE,
        alpha_v: FE,
        alpha_w: FE,
        alpha_y: FE,
        beta: FE,
        rv: FE,
        rw: FE,
        gamma: FE,
    ) -> Self {
        Self {
            s,
            alpha_v,
            alpha_w,
            alpha_y,
            beta,
            rv,
            rw,
            gamma,
        }
    }

    pub fn sample() -> Self {
        Self {
            s: sample_fr_elem(),
            alpha_v: sample_fr_elem(),
            alpha_w: sample_fr_elem(),
            alpha_y: sample_fr_elem(),
            beta: sample_fr_elem(),
            rv: sample_fr_elem(),
            rw: sample_fr_elem(),
            gamma: sample_fr_elem(),
        }
    }
}

pub fn generate_verification_key(
    qap: QuadraticArithmeticProgram,
    toxic_waste: &ToxicWaste,
) -> VerificationKey {
    let g1: G1Point = Curve::generator();
    let g2: G2Point = TwistedCurve::generator();
    let s = &toxic_waste.s;
    let alpha_v = &toxic_waste.alpha_v;
    let alpha_w = &toxic_waste.alpha_w;
    let alpha_y = &toxic_waste.alpha_y;
    let beta = &toxic_waste.beta;
    let rv = &toxic_waste.rv;
    let rw = &toxic_waste.rw;
    let gamma = &toxic_waste.gamma;
    let ry = toxic_waste.ry();
    let vector_capacity = qap.number_of_inputs + qap.number_of_outputs + 1;

    // We construct g1_vk.
    let mut g1_vk_io: Vec<G1Point> = Vec::with_capacity(vector_capacity);
    g1_vk_io.push(g1.operate_with_self((rv.clone() * qap.v0().evaluate(s)).representative()));
    g1_vk_io.extend(
        qap.v_input()
            .iter()
            .map(|vk| g1.operate_with_self((rv.clone() * vk.evaluate(s)).representative())),
    );
    g1_vk_io.extend(
        qap.v_output()
            .iter()
            .map(|vk| g1.operate_with_self((rv.clone() * &vk.evaluate(s)).representative())),
    );

    // We construct g2_wk.
    let mut g2_wk_io: Vec<G2Point> = Vec::with_capacity(vector_capacity);
    g2_wk_io.push(g2.operate_with_self((rw.clone() * qap.w0().evaluate(s)).representative()));
    g2_wk_io.extend(
        qap.w_input()
            .iter()
            .map(|wk| g2.operate_with_self((rw.clone() * wk.evaluate(s)).representative())),
    );
    g2_wk_io.extend(
        qap.w_output()
            .iter()
            .map(|wk| g2.operate_with_self((rw.clone() * wk.evaluate(s)).representative())),
    );

    // We construct g1_yk.
    let mut g1_yk_io: Vec<G1Point> = Vec::with_capacity(vector_capacity);
    g1_yk_io.push(g1.operate_with_self((ry.clone() * qap.y0().evaluate(s)).representative()));
    g1_yk_io.extend(
        qap.y_input()
            .iter()
            .map(|yk| g1.operate_with_self((ry.clone() * yk.evaluate(s)).representative())),
    );
    g1_yk_io.extend(
        qap.y_output()
            .iter()
            .map(|yk| g1.operate_with_self((ry.clone() * yk.evaluate(s)).representative())),
    );

    VerificationKey {
        g2: g2.clone(),
        g2_alpha_v: g2.operate_with_self(alpha_v.representative()),
        g2_alpha_w: g2.operate_with_self(alpha_w.representative()),
        g2_alpha_y: g2.operate_with_self(alpha_y.representative()),
        g2_gamma: g2.operate_with_self(gamma.representative()),
        g2_beta_gamma: g2.operate_with_self((beta * gamma).representative()),
        g1y_t: g1.operate_with_self((ry * qap.target.evaluate(s)).representative()),
        g1_vk: g1_vk_io,
        g2_wk: g2_wk_io,
        g1_yk: g1_yk_io,
    }
}

pub fn generate_evaluation_key(
    qap: &QuadraticArithmeticProgram,
    toxic_waste: &ToxicWaste,
) -> EvaluationKey {
    let g1: G1Point = Curve::generator();
    let g2: G2Point = TwistedCurve::generator();
    let (v_mid, w_mid, y_mid) = (qap.v_mid(), qap.w_mid(), qap.y_mid());
    let s = &toxic_waste.s;
    let alpha_v = &toxic_waste.alpha_v;
    let alpha_w = &toxic_waste.alpha_w;
    let alpha_y = &toxic_waste.alpha_y;
    let beta = &toxic_waste.beta;
    let rv = &toxic_waste.rv;
    let rw = &toxic_waste.rw;
    let ry = &toxic_waste.ry();
    let degree = qap.target.degree();

    EvaluationKey {
        g1_vk: v_mid
            .iter()
            .map(|vk| g1.operate_with_self((rv * vk.evaluate(s)).representative()))
            .collect(),
        g1_wk: w_mid
            .iter()
            .map(|wk| g1.operate_with_self((rw * wk.evaluate(s)).representative()))
            .collect(),
        g2_wk: w_mid
            .iter()
            .map(|wk| g2.operate_with_self((rw * wk.evaluate(s)).representative()))
            .collect(),
        g1_yk: y_mid
            .iter()
            .map(|yk| g1.operate_with_self((ry * yk.evaluate(s)).representative()))
            .collect(),
        g1_alpha_vk: v_mid
            .iter()
            .map(|vk| g1.operate_with_self((rv * alpha_v * vk.evaluate(s)).representative()))
            .collect(),
        g1_alpha_wk: w_mid
            .iter()
            .map(|wk| g1.operate_with_self((rw * alpha_w * wk.evaluate(s)).representative()))
            .collect(),
        g1_alpha_yk: y_mid
            .iter()
            .map(|yk| g1.operate_with_self((ry * alpha_y * yk.evaluate(s)).representative()))
            .collect(),
        g2_s_i: (0..degree)
            .map(|i| g2.operate_with_self((s.pow(i)).representative()))
            .collect(),
        g1_beta: v_mid
            .iter()
            .zip(w_mid.iter())
            .zip(y_mid.iter())
            .map(|((vk, wk), yk)| {
                g1.operate_with_self(
                    (rv * beta * vk.evaluate(s)
                        + rw * beta * wk.evaluate(s)
                        + ry * beta * yk.evaluate(s))
                    .representative(),
                )
            })
            .collect(),
    }
}

pub fn setup(
    qap: &QuadraticArithmeticProgram,
    toxic_waste: ToxicWaste,
) -> (EvaluationKey, VerificationKey) {
    (
        generate_evaluation_key(qap, &toxic_waste),
        generate_verification_key(qap.clone(), &toxic_waste),
    )
}
