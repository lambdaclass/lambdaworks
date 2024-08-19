use criterion::black_box;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bn_254::{
                curve::BN254Curve,
                pairing::BN254AtePairing,
                twist::BN254TwistCurve,
            },
            point::ShortWeierstrassProjectivePoint,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
};
use rand::{rngs::StdRng, Rng, SeedableRng};
#[allow(dead_code)]
type G1 = ShortWeierstrassProjectivePoint<BN254Curve>;
#[allow(dead_code)]
type G2 = ShortWeierstrassProjectivePoint<BN254TwistCurve>;

#[allow(dead_code)]
pub fn rand_points_g1() -> (G1, G1, u128, u128) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val = rng.gen();
    let b_val = rng.gen();
    let a = BN254Curve::generator().operate_with_self(a_val);
    let b = BN254Curve::generator().operate_with_self(b_val);
    (a, b, a_val, b_val)
}

#[allow(dead_code)]
pub fn rand_points_g2() -> (G2, G2, u128, u128) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val = rng.gen();
    let b_val = rng.gen();
    let a = BN254TwistCurve::generator().operate_with_self(a_val);
    let b = BN254TwistCurve::generator().operate_with_self(b_val);
    (a, b, a_val, b_val)
}

#[allow(dead_code)]
pub fn bn_254_operate_with_g1() {
    let (a, b, _, _) = rand_points_g1();
    let _ = black_box(black_box(&a).operate_with(black_box(&b)));
}

#[allow(dead_code)]
pub fn bn_254_operate_with_g2() {
    let (a, b, _, _) = rand_points_g1();
    let _ = black_box(black_box(&a).operate_with(black_box(&b)));
}

#[allow(dead_code)]
pub fn bn_254_operate_with_self_g1() {
    let (a, _, _, b_val) = rand_points_g1();
    let _ = black_box(black_box(&a).operate_with_self(black_box(b_val)));
}

#[allow(dead_code)]
pub fn bn_254_operate_with_self_g2() {
    let (a, _, _, b_val) = rand_points_g2();
    let _ = black_box(black_box(&a).operate_with_self(black_box(b_val)));
}

#[allow(dead_code)]
pub fn bn_254_double_g1() {
    let (a, _, _, _) = rand_points_g1();
    let _ = black_box(black_box(&a).operate_with_self(black_box(2u64)));
}

#[allow(dead_code)]
pub fn bn_254_double_g2() {
    let (a, _, _, _) = rand_points_g2();
    let _ = black_box(black_box(&a).operate_with_self(black_box(2u64)));
}

#[allow(dead_code)]
pub fn bn_254_neg_g1() {
    let (a, _, _, _) = rand_points_g1();
    let _ = black_box(black_box(&a).neg());
}

#[allow(dead_code)]
pub fn bn_254_neg_g2() {
    let (a, _, _, _) = rand_points_g2();
    let _ = black_box(black_box(&a).neg());
}

#[allow(dead_code)]
pub fn bn_254_subgroup_check_g2() {
    let (a, _, _) = rand_points_g2();
    let _ = black_box(black_box(&a.is_in_subgroup()));
}

#[allow(dead_code)]
pub fn bn_254_ate_pairing() {
    let (a, _, _, _) = rand_points_g1();
    let (_, b, _, _) = rand_points_g2();
    let _ = black_box(BN254AtePairing::compute(black_box(&a), black_box(&b)));
}
