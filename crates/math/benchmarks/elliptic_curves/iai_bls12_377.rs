use criterion::black_box;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_377::curve::BLS12377Curve, point::ShortWeierstrassProjectivePoint,
        },
        traits::IsEllipticCurve,
    },
};
use rand::{rngs::StdRng, Rng, SeedableRng};
#[allow(dead_code)]
type G1 = ShortWeierstrassProjectivePoint<BLS12377Curve>;

#[allow(dead_code)]
pub fn rand_points_g1() -> (G1, G1, u128, u128) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val = rng.gen();
    let b_val = rng.gen();
    let a = BLS12377Curve::generator().operate_with_self(a_val);
    let b = BLS12377Curve::generator().operate_with_self(b_val);
    (a, b, a_val, b_val)
}

#[allow(dead_code)]
pub fn bls12_377_operate_with_g1() {
    let (a, b, _, _) = rand_points_g1();
    let _ = black_box(black_box(&a).operate_with(black_box(&b)));
}

#[allow(dead_code)]
pub fn bls12_377_operate_with_self_g1() {
    let (a, _, _, b_val) = rand_points_g1();
    let _ = black_box(black_box(&a).operate_with_self(black_box(b_val)));
}

#[allow(dead_code)]
pub fn bls12_377_double_g1() {
    let (a, _, _, _) = rand_points_g1();
    let _ = black_box(black_box(&a).operate_with_self(black_box(2u64)));
}

#[allow(dead_code)]
pub fn bls12_377_neg_g1() {
    let (a, _, _, _) = rand_points_g1();
    let _ = black_box(black_box(&a).neg());
}
