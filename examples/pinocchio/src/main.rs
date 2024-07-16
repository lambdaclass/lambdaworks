// Pinocchio implemtation
use lambdaworks_math::{
    cyclic_group::IsGroup, msm::pippenger::msm,
    elliptic_curve::{
        traits::{IsEllipticCurve, IsPairing},
        short_weierstrass::{point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass},
    }
};

use pinocchio::common::{
        {sample_fr_elem, Curve, FE, G1Point,  Pairing}};

// use pinocchio::{QuadraticArithmeticProgram};
use pinocchio::qap::QuadraticArithmeticProgram;

// Use G1 point?
struct EvaluationKey {
    //g^vk(s)
    g_vk_s: Vec<G1Point>,
    //g^wk_s
    gw_wk: Vec<G1Point>,
    // g^yk(s)
    g_yk_s: Vec<G1Point>,
    // g^alphavk(s)
    g_alpha_vk_s: Vec<G1Point>,
    // g^alpha_wk(s)
    g_alpha_wk_s: Vec<G1Point>,
    // g^alpha_yk(s)
    g_alpha_yk_s: Vec<G1Point>,
   /* // g^betav_vk(s)
    g_betav_vk_s: Vec<G1Point>,
    // g^betaw_wk(s)
    g_betaw_wk_s: Vec<G1Point>,
    // g^betay_yk(s)
    g_betay_yk_s: Vec<G1Point>,*/
    g_beta: Vec<G1Point>,
    // g^s^i
    g_s_i: Vec<G1Point>,
    // g^alpha_s^i
    //g_alpha_s_i: Vec<G1Point>,
}
struct VerificationKey {
    g1:G1Point,
    g_alpha_v:G1Point,
    g_alpha_w:G1Point,
    g_alpha_y:G1Point,
    g_gamma:G1Point,
    g_beta_gamma:G1Point,
    // gy target on s?
    gy_t:G1Point,
    gv_vk:Vec<G1Point>,
    gw_wk:Vec<G1Point>,
    gy_yk:Vec<G1Point>,
}
struct ToxicWaste {
   rv:FE,
   rw:FE,
   s:FE,
   alpha_v:FE,
   alpha_y:FE,
   alpha_w:FE,
   beta:FE,
   gamma:FE,
}
impl ToxicWaste {
    fn ry(&self) -> FE {
        &self.rv * &self.rw
    }

    fn new(
    s:FE,
    alpha_v:FE,
    alpha_w:FE,
    alpha_y:FE,
    beta:FE,
    rv:FE,
    rw:FE,
    gamma:FE,
    ) -> Self{
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

    fn sample() -> Self {
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

fn generate_verification_key(
    qap: QuadraticArithmeticProgram,
    toxic_waste: &ToxicWaste,
    //generator: &GPoint//?
) -> VerificationKey {
    let g : G1Point = Curve::generator();
    let s = &toxic_waste.s;
    let alpha_v = &toxic_waste.alpha_v;
    let alpha_w = &toxic_waste.alpha_w;
    let alpha_y = &toxic_waste.alpha_y;
    let beta =&toxic_waste.beta;
    let rv = &toxic_waste.rv;
    let rw = &toxic_waste.rw;
    let gamma = &toxic_waste.gamma;
    let ry = toxic_waste.ry();
    // add a variable that tells the capacity of the vector
    // what is the capacity?
    // output + input + 1 or  2 times input + 1 ? 
    let vector_capacity = qap.number_of_inputs + qap.number_of_outputs + 1;
    let mut gv_vk_io: Vec<G1Point> = Vec::with_capacity(vector_capacity);
// why v0 is not an input?
// why representative?
    gv_vk_io.push(
        g.operate_with_self(
            (rv.clone() * qap.v0().evaluate(&s)).representative()
        )
    );
    gv_vk_io.extend(
        qap.v_input().iter()
        .map(|vk| g.operate_with_self(
            (rv.clone() * vk.evaluate(&s)).representative()
            ) 
        )
    );
    gv_vk_io.extend(
        qap.v_output().iter()
        .map(|vk| g.operate_with_self(
            (rv.clone() * &vk.evaluate(&s)).representative()
            )
        )
    );

    let mut  gw_wk_io : Vec<G1Point> = Vec::with_capacity(vector_capacity);
    gw_wk_io.push(
        g.operate_with_self(
            (rw.clone() * qap.w0().evaluate(&s)).representative()
        )
    );
    gw_wk_io.extend(
        qap.w_input().iter()
        .map(|wk| g.operate_with_self(
            (rw.clone() * wk.evaluate(&s)).representative()
            ) 
        )
    );
    gw_wk_io.extend(
        qap.w_output().iter()
        .map(|wk| g.operate_with_self(
            (rw.clone() * wk.evaluate(&s)).representative()
            )
        )
    );

    let mut  gy_yk_io : Vec<G1Point> = Vec::with_capacity(vector_capacity);
    gy_yk_io.push(
        g.operate_with_self(
            (ry.clone() * qap.y0().evaluate(&s)).representative()
        )
    );
    gy_yk_io.extend(
        qap.y_input().iter()
        .map(|yk| g.operate_with_self(
            (ry.clone() * yk.evaluate(&s)).representative()
            ) 
        )
    );
    gy_yk_io.extend(
        qap.y_output().iter()
        .map(|yk| g.operate_with_self(
            (ry.clone() * yk.evaluate(&s)).representative()
            )
        )
    );

/*
   gv_vk_io.push(g.operate_with_self((rv * qap.v0().evaluate(&s)).representative()))
    .chain(
        qap.v_input().iter()
        .map(|vk| g.operate_with_self((rv * vk.evaluate(&s)).representative()) 
    )
    .chain(
        qap.v_output.iter()
            .map(|vk| g.operate_with_self((rv * vk.evaluate(s)).representative()))
        )
    );

    let gw_wk_io : Vec<G1Point> = Vec::with_capacity(vector_capacity);
    gw_wk_io.push(g.operate_with_self((rw *qap.w0.evaluate(s)).representative()))
    .chain(
        qap.v_input.iter()
        .map(|wk| g.operate_with_self((rw * wk.evaluate(s)).representative()))
    .chain(
        qap.v_output.iter()
            .map(|wk| g.operate_with_self((rw* wk.evaluate(s)).representative()))
        )
    );

    let gy_yk_io : Vec<G1Point> = Vec::with_capacity(vector_capacity);
    gy_yk_io.push(g.operate_with_self((ry *qap.y0.evaluate(s)).representative()))
    .chain(
        qap.v_input.iter()
        .map(|yk| g.operate_with_self((ry * yk.evaluate(s)).representative()))
    .chain(
        qap.v_output.iter()
            .map(|yk| g.operate_with_self((ry* yk.evaluate(s)).representative()))
        )
    ); */

    VerificationKey {
        g1: g.clone(),
        g_alpha_v: g.operate_with_self(alpha_v.representative()),
        g_alpha_w: g.operate_with_self(alpha_w.representative()),
        g_alpha_y: g.operate_with_self(alpha_y.representative()),
        g_gamma: g.operate_with_self(gamma.representative()),
        g_beta_gamma: g.operate_with_self((beta * gamma).representative()),
        gy_t: g.operate_with_self((ry * qap.target.evaluate(&s)).representative()),
        gv_vk: gv_vk_io,
        gw_wk: gw_wk_io,
        gy_yk: gy_yk_io,
    }
}

fn generate_evaluation_key(
    qap: &QuadraticArithmeticProgram,
    toxic_waste: &ToxicWaste,
) -> EvaluationKey {
    let g : G1Point = Curve::generator();
    let (vs_mid, ws_mid, ys_mid) = (qap.v_mid(), qap.w_mid(), qap.y_mid());
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
    g_vk_s: vs_mid.iter().map(|vk| g.operate_with_self((rv * vk.evaluate(&s)).representative())).collect(),
    gw_wk: ws_mid.iter().map(|wk| g.operate_with_self((rw * wk.evaluate(&s)).representative())).collect(),
    g_yk_s: ys_mid.iter().map(|yk| g.operate_with_self((ry * yk.evaluate(&s)).representative())).collect(),
    g_alpha_vk_s: vs_mid.iter().map(|vk| g.operate_with_self((rv * alpha_v * vk.evaluate(&s)).representative())).collect(),
    g_alpha_wk_s: ws_mid.iter().map(|wk| g.operate_with_self((rw * alpha_w * wk.evaluate(&s)).representative())).collect(),
    g_alpha_yk_s: ys_mid.iter().map(|yk| g.operate_with_self((ry * alpha_y * yk.evaluate(&s)).representative())).collect(),
    g_s_i: (0..degree).map(|i| g.operate_with_self((s.pow(i)).representative())).collect(),
    g_beta:  vs_mid.iter()
        .zip(ws_mid.iter())
        .zip(ys_mid.iter())
        .map(|((vk, wk), yk)| {

            g.operate_with_self(
                ((rv * beta * vk.evaluate(&s)
                    + rw * beta * wk.evaluate(&s)
                    + ry * beta * yk.evaluate(&s)).representative())
            )
    })
    .collect()
    }
}

// with_capacity vs new for vec
// to do: ask about the difference between with_capacity and new for vec
fn setup(
    qap: &QuadraticArithmeticProgram,
    toxic_waste: ToxicWaste,
) -> (EvaluationKey, VerificationKey) {
    (generate_evaluation_key(&qap, &toxic_waste),
    generate_verification_key(qap.clone(), &toxic_waste))
}
struct Proof {
g_vs:G1Point,
g_ws:G1Point,
g_ys:G1Point,
g_hs:G1Point,
g_alpha_vs:G1Point,
g_alpha_ws:G1Point,
g_alpha_ys:G1Point,
g_beta_vwy:G1Point,
}
fn generate_proof(
    evaluation_key: &EvaluationKey,
    qap: &QuadraticArithmeticProgram,
    qap_c_coefficients: &[FE],
) -> Proof{
    let cmid = &qap_c_coefficients[qap.number_of_inputs
    ..qap_c_coefficients.len() - qap.number_of_outputs];
    let c_mid = cmid
    .iter()
    .map(|elem| elem.representative())
    .collect::<Vec<_>>();

    let h_polynomial = qap.h_polynomial(qap_c_coefficients);
    let h_coefficients = h_polynomial.coefficients
    .iter()
    .map(|elem| elem.representative())
    .collect::<Vec<_>>();
    let h_degree = h_polynomial.degree();

    Proof {
        g_vs: msm(&c_mid, &evaluation_key.g_vk_s).unwrap(),
        g_ws: msm(&c_mid, &evaluation_key.gw_wk).unwrap(),
        g_ys: msm(&c_mid, &evaluation_key.g_yk_s).unwrap(),
        g_alpha_vs: msm(&c_mid, &evaluation_key.g_alpha_vk_s).unwrap(),
        g_alpha_ws: msm(&c_mid, &evaluation_key.g_alpha_wk_s).unwrap(),
        g_alpha_ys: msm(&c_mid, &evaluation_key.g_alpha_yk_s).unwrap(),
        g_beta_vwy: msm(&c_mid, &evaluation_key.g_beta).unwrap(),
        g_hs:msm(
            &h_coefficients,&evaluation_key.g_s_i[..h_degree],
        ).unwrap()
    }
}

pub fn verify(verification_key:&VerificationKey,
    proof:&Proof,
    c_inputs_outputs:&[FE]
) -> bool {
    let b1 = check_divisibility(verification_key, proof, c_inputs_outputs);
    let b2 = check_appropriate_spans(verification_key, proof);
    let b3 = check_same_linear_combinations(verification_key, proof);
    b1 && b2 && b3

}

pub fn check_divisibility(
    verification_key: &VerificationKey,
    proof: &Proof,
    c_inputs_outputs: &[FE],
) -> bool {
    // We will use hiding_v, hiding_w and hiding_y as arguments of the pairings.
    let c_inputs_outputs = c_inputs_outputs
    .iter()
    .map(|elem| elem.representative())
    .collect::<Vec<_>>();

    // hiding_v = ( gv^( 1*v_0(s) + c_1*v_1(s) + ... + c_m*v_m(s) ) ) * gv_mid
    let hiding_v = verification_key.gv_vk[0] // en vez de hacer esto megustaría poder agregar a c_inputs_outputs el FE 1 al principio, pero creo que no se puede.
        .operate_with(&msm(&c_inputs_outputs, &verification_key.gv_vk[1..]).unwrap())
        .operate_with(&proof.g_vs);
    let hiding_w = verification_key.gw_wk[0]
        .operate_with(&msm(&c_inputs_outputs, &verification_key.gw_wk[1..]).unwrap())
        .operate_with(&proof.g_ws);
    let hiding_y = verification_key.gy_yk[0]
        .operate_with(&msm(&c_inputs_outputs, &verification_key.gy_yk[1..]).unwrap())
        .operate_with(&proof.g_ys);

    Pairing::compute(&hiding_v, &hiding_w).unwrap() 
    == Pairing::compute(&verification_key.gy_t, &proof.g_hs).unwrap()
        * Pairing::compute(&hiding_y, verification_key.g1).unwrap()
}

// We check that g_vs (g_{v,mid}) is indeed g multpiplied by a linear combination of the {v_k}_{k mid}.
// The same with g_ws and g_ys.
pub fn check_appropriate_spans(
    verification_key: &VerificationKey,
    proof: &Proof
) -> bool {
    let b1 = Pairing::compute(&proof.g_alpha_vs, &verification_key.g1) 
        == Pairing::compute(&proof.g_vs, &verification_key.g_alpha_v);
    let b2 = Pairing::compute(&proof.g_alpha_ws, &verification_key.g1) 
        == Pairing::compute(&proof.g_ws, &verification_key.g_alpha_w);
    let b3 = Pairing::compute(&proof.g_alpha_ys, &verification_key.g1) 
        == Pairing::compute(&proof.g_ys, &verification_key.g_alpha_y);
    b1 && b2 && b3
}

// We check that the same coefficients were used for the linear combination of v, w and y.
pub fn check_same_linear_combinations(
    verification_key: &VerificationKey,
    proof: &Proof
) -> bool {
    Pairing::compute(&proof.g_beta_vwy, &verification_key.g_gamma)
        == Pairing::compute(
            &proof.g_vs
                .operate_with(&proof.g_ws)
                .operate_with(&proof.g_ys),
            &verification_key.g_beta_gamma
        )
}

//







fn main() {
    println!("Hello, world!");
}
