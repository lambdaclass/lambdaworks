use crate::{common::*};

pub struct Proof {
    pub H: G1Point,
    pub V_w: G1Point,
    pub V_w_prime: G2Point,
    pub B_w: G1Point
}
