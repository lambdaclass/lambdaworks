// Pinocchio implemtation
use lambdaworks_math::{
    cyclic_group::IsGroup, msm::pippenger::msm,
    elliptic_curve::{
        traits::{IsEllipticCurve, IsPairing},
        short_weierstrass::{point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass},
    }
};

use pinocchio::common::{
        sample_fr_elem, Curve, G1Point, G2Point, Pairing, TwistedCurve, FE};

// use pinocchio::{QuadraticArithmeticProgram};
use pinocchio::qap::QuadraticArithmeticProgram;

// Use G1 point?
struct EvaluationKey {
    //g^vk(s)
    g1_vk: Vec<G1Point>,
    //g^wk_s
    g1_wk: Vec<G1Point>,
    g2_wk: Vec<G2Point>,
    // g^yk(s)
    g1_yk: Vec<G1Point>,
    // g^alphavk(s)
    g1_alpha_vk: Vec<G1Point>,
    // g^alpha_wk(s)
    g1_alpha_wk: Vec<G1Point>,
    // g^alpha_yk(s)
    g1_alpha_yk: Vec<G1Point>,
   /* // g^betav_vk(s)
    g_betav_vk_s: Vec<G1Point>,
    // g^betaw_wk(s)
    g_betaw_wk_s: Vec<G1Point>,
    // g^betay_yk(s)
    g_betay_yk_s: Vec<G1Point>,*/
    g1_beta: Vec<G1Point>,
    // g^s^i
    g2_s_i: Vec<G2Point>,
    // g^alpha_s^i
    //g_alpha_s_i: Vec<G1Point>,
}
struct VerificationKey {
    g2:G2Point,
    g2_alpha_v:G2Point,
    g2_alpha_w:G2Point,
    g2_alpha_y:G2Point,
    g2_gamma:G2Point,
    g2_beta_gamma:G2Point,
    // gy target on s?
    g1y_t:G1Point,
    g1_vk:Vec<G1Point>,
    g2_wk:Vec<G2Point>,
    g1_yk:Vec<G1Point>,
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
    let g1 : G1Point = Curve::generator();
    let g2: G2Point = TwistedCurve::generator();
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
    let mut g1v_vk_io: Vec<G1Point> = Vec::with_capacity(vector_capacity);
// why v0 is not an input?
// why representative?
    g1v_vk_io.push(
        g1.operate_with_self(
            (rv.clone() * qap.v0().evaluate(&s)).representative()
        )
    );
    g1v_vk_io.extend(
        qap.v_input().iter()
        .map(|vk| g1.operate_with_self(
            (rv.clone() * vk.evaluate(&s)).representative()
            ) 
        )
    );
    g1v_vk_io.extend(
        qap.v_output().iter()
        .map(|vk| g1.operate_with_self(
            (rv.clone() * &vk.evaluate(&s)).representative()
            )
        )
    );

    let mut  g2_wk_io : Vec<G2Point> = Vec::with_capacity(vector_capacity);
    g2_wk_io.push(
        g2.operate_with_self(
            (rw.clone() * qap.w0().evaluate(&s)).representative()
        )
    );
    g2_wk_io.extend(
        qap.w_input().iter()
        .map(|wk| g2.operate_with_self(
            (rw.clone() * wk.evaluate(&s)).representative()
            ) 
        )
    );
    g2_wk_io.extend(
        qap.w_output().iter()
        .map(|wk| g2.operate_with_self(
            (rw.clone() * wk.evaluate(&s)).representative()
            )
        )
    );

    let mut  g1_yk_io : Vec<G1Point> = Vec::with_capacity(vector_capacity);
    g1_yk_io.push(
        g1.operate_with_self(
            (ry.clone() * qap.y0().evaluate(&s)).representative()
        )
    );
    g1_yk_io.extend(
        qap.y_input().iter()
        .map(|yk| g1.operate_with_self(
            (ry.clone() * yk.evaluate(&s)).representative()
            ) 
        )
    );
    g1_yk_io.extend(
        qap.y_output().iter()
        .map(|yk| g1.operate_with_self(
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
        g2: g2.clone(),
        g2_alpha_v: g2.operate_with_self(alpha_v.representative()),
        g2_alpha_w: g2.operate_with_self(alpha_w.representative()),
        g2_alpha_y: g2.operate_with_self(alpha_y.representative()),
        g2_gamma: g2.operate_with_self(gamma.representative()),
        g2_beta_gamma: g2.operate_with_self((beta * gamma).representative()),
        g1y_t: g1.operate_with_self((ry * qap.target.evaluate(&s)).representative()),
        g1_vk: g1v_vk_io,
        g2_wk: g2_wk_io,
        g1_yk: g1_yk_io,
    }
}

fn generate_evaluation_key(
    qap: &QuadraticArithmeticProgram,
    toxic_waste: &ToxicWaste,
) -> EvaluationKey {
    let g1 : G1Point = Curve::generator();
    let g2 :G2Point = TwistedCurve::generator();
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
    g1_vk: vs_mid.iter().map(|vk| g1.operate_with_self((rv * vk.evaluate(&s)).representative())).collect(),
    g1_wk: ws_mid.iter().map(|wk| g1.operate_with_self((rw * wk.evaluate(&s)).representative())).collect(),
    g2_wk: ws_mid.iter().map(|wk| g2.operate_with_self((rw * wk.evaluate(&s)).representative())).collect(),
    g1_yk: ys_mid.iter().map(|yk| g1.operate_with_self((ry * yk.evaluate(&s)).representative())).collect(),
    g1_alpha_vk: vs_mid.iter().map(|vk| g1.operate_with_self((rv * alpha_v * vk.evaluate(&s)).representative())).collect(),
    g1_alpha_wk: ws_mid.iter().map(|wk| g1.operate_with_self((rw * alpha_w * wk.evaluate(&s)).representative())).collect(),
    g1_alpha_yk: ys_mid.iter().map(|yk| g1.operate_with_self((ry * alpha_y * yk.evaluate(&s)).representative())).collect(),
    g2_s_i: (0..degree).map(|i| g2.operate_with_self((s.pow(i)).representative())).collect(),
    g1_beta:  vs_mid.iter()
        .zip(ws_mid.iter())
        .zip(ys_mid.iter())
        .map(|((vk, wk), yk)| {

            g1.operate_with_self(
                (rv * beta * vk.evaluate(&s)
                    + rw * beta * wk.evaluate(&s)
                    + ry * beta * yk.evaluate(&s)).representative()
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
v: G1Point,
w1: G1Point,
w2: G2Point,
y: G1Point,
h: G2Point,
v_prime: G1Point,
w_prime: G1Point,
y_prime: G1Point,
z: G1Point,
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
        v: msm(&c_mid, &evaluation_key.g1_vk).unwrap(),
        w1: msm(&c_mid, &evaluation_key.g1_wk).unwrap(),
        w2: msm(&c_mid, &evaluation_key.g2_wk).unwrap(),
        y: msm(&c_mid, &evaluation_key.g1_yk).unwrap(),
        v_prime: msm(&c_mid, &evaluation_key.g1_alpha_vk).unwrap(),
        w_prime: msm(&c_mid, &evaluation_key.g1_alpha_wk).unwrap(),
        y_prime: msm(&c_mid, &evaluation_key.g1_alpha_yk).unwrap(),
        z: msm(&c_mid, &evaluation_key.g1_beta).unwrap(),
        h:msm(
            &h_coefficients,&evaluation_key.g2_s_i[..h_degree],
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
    c_io: &[FE],
) -> bool {
    // We transform c_io into UnsignedIntegers.
    let c_io = c_io
    .iter()
    .map(|elem| elem.representative())
    .collect::<Vec<_>>();

    let v_io = verification_key.g1_vk[0]
        .operate_with(&msm(&c_io, &verification_key.g1_vk[1..]).unwrap());
    let w_io = verification_key.g2_wk[0]
        .operate_with(&msm(&c_io, &verification_key.g2_wk[1..]).unwrap());
    let y_io = verification_key.g1_yk[0]
        .operate_with(&msm(&c_io, &verification_key.g1_yk[1..]).unwrap());
    Pairing::compute(
        &v_io.operate_with(&proof.v),
        &w_io.operate_with(&proof.w2)
    ).unwrap()
    == Pairing::compute(
        &verification_key.g1y_t,
        &proof.h
    ).unwrap()
    * Pairing::compute(
        &y_io.operate_with(&proof.y),
        &verification_key.g2
    ).unwrap()
}

// We check that v (from the proof) is indeed g multpiplied by a linear combination of the g1_vk.
// The same with w and y.
pub fn check_appropriate_spans(
    verification_key: &VerificationKey,
    proof: &Proof
) -> bool {
    let b1 = Pairing::compute(&proof.v_prime, &verification_key.g2) 
        == Pairing::compute(&proof.v, &verification_key.g2_alpha_v);
    let b2 = Pairing::compute(&proof.w_prime, &verification_key.g2) 
        == Pairing::compute(&proof.w1, &verification_key.g2_alpha_w);
    let b3 = Pairing::compute(&proof.y_prime, &verification_key.g2) 
        == Pairing::compute(&proof.y, &verification_key.g2_alpha_y);
    b1 && b2 && b3
}

// We check that the same coefficients were used for the linear combination of v, w and y.
pub fn check_same_linear_combinations(
    verification_key: &VerificationKey,
    proof: &Proof
) -> bool {
    Pairing::compute(&proof.z, &verification_key.g2_gamma)
    == Pairing::compute(
        &proof.v
            .operate_with(&proof.w1)
            .operate_with(&proof.y),
        &verification_key.g2_beta_gamma
    )
}

//

fn main() {
    println!("Running Pinocchio test...");

    // Create a test QAP (you'll need to implement this based on your specific needs)
    let test_qap = create_test_qap();

    // Sample a random toxic waste
    let toxic_waste = ToxicWaste::sample();

    // Setup
    let (evaluation_key, verification_key) = setup(&test_qap, toxic_waste);

    // Define inputs (adjust as needed for your specific QAP)
    let inputs = vec![FE::new(1), FE::new(2), FE::new(3), FE::new(4)];

    // Execute the QAP (you'll need to implement this based on your specific QAP)
    let (c_mid, c_output) = execute_qap(&test_qap, &inputs);

    // Construct the full witness vector
    let mut c_vector = inputs.clone();
    c_vector.extend(c_mid);
    c_vector.push(c_output);

    // Generate proof
    let proof = generate_proof(&evaluation_key, &test_qap, &c_vector);

    // Prepare inputs and outputs for verification
    let mut c_io_vector = inputs;
    c_io_vector.push(c_output);

    // Verify the proof
    let accepted = verify(&verification_key, &proof, &c_io_vector);

    if accepted {
        println!("Proof verified successfully!");
    } else {
        println!("Proof verification failed.");
    }
}


