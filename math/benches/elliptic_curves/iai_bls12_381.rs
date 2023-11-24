use criterion::black_box;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{curves::bls12_381::{
            compression::{check_point_is_in_subgroup, compress_g1_point, decompress_g1_point},
            curve::BLS12381Curve,
            pairing::BLS12381AtePairing,
            twist::BLS12381TwistCurve,
        }, point::ShortWeierstrassProjectivePoint},
        traits::{IsEllipticCurve, IsPairing},
    },
};
use rand::{rngs::StdRng, SeedableRng, Rng};
type G1 = ShortWeierstrassProjectivePoint<BLS12381Curve>;
type G2 = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;

pub fn rand_points_g1() -> (G1, G1, u128, u128) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val = rng.gen();
    let b_val = rng.gen();
    let a = BLS12381Curve::generator().operate_with_self(a_val);
    let b = BLS12381Curve::generator().operate_with_self(b_val);
    (a, b, a_val, b_val)
}

pub fn rand_points_g2() -> (G2, G2, u128, u128) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val = rng.gen();
    let b_val = rng.gen();
    let a = BLS12381TwistCurve::generator().operate_with_self(a_val);
    let b = BLS12381TwistCurve::generator().operate_with_self(b_val);
    (a, b, a_val, b_val)
}

#[inline(never)]
pub fn bls12_381_operate_with_g1() {
    let (a, b, _, _) = rand_points_g1();
    let _ = black_box(black_box(&a).operate_with(black_box(&b)));
}

#[inline(never)]
pub fn bls12_381_operate_with_g2() {
    let (a, b, _, _) = rand_points_g1();
    let _ = black_box(black_box(&a).operate_with(black_box(&b)));
}

#[inline(never)]
pub fn bls12_381_operate_with_self_g1() {
    let (a, _, _, b_val) = rand_points_g1();
    let _ = black_box(black_box(&a).operate_with_self(black_box(b_val)));
}

#[inline(never)]
pub fn bls12_381_operate_with_self_g2() {
    let (a, _, _, b_val) = rand_points_g2();
    let _ = black_box(black_box(&a).operate_with_self(black_box(b_val)));
}

#[inline(never)]
pub fn bls12_381_double_g1() {
    let (a, _, _, _) = rand_points_g1();
    let _ = black_box(black_box(&a).operate_with_self(black_box(2u64)));
}

#[inline(never)]
pub fn bls12_381_double_g2() {
    let (a, _, _, _) = rand_points_g2();
    let _ = black_box(black_box(&a).operate_with_self(black_box(2u64)));
}

#[inline(never)]
pub fn bls12_381_neg_g1() {
    let (a, _, _, _) = rand_points_g1();
    let _ = black_box(black_box(&a).neg());
}

#[inline(never)]
pub fn bls12_381_neg_g2() {
    let (a, _, _, _) = rand_points_g2();
    let _ = black_box(black_box(&a).neg());
}

#[inline(never)]
pub fn bls12_381_compress_g1() {
    let (a, _, _, _) = rand_points_g1();
    let _ = black_box(compress_g1_point(black_box(&a)));
}

#[inline(never)]
pub fn bls12_381_decompress_g1() {
    let (a, _, _, _) = rand_points_g1();
        let a: [u8; 48] = compress_g1_point(&a).try_into().unwrap();
    let _ = black_box(decompress_g1_point(&mut black_box(a))).unwrap();
}

#[inline(never)]
pub fn bls12_381_subgroup_check_g1() {
    let (a, _, _, _) = rand_points_g1();
    let _ = black_box(check_point_is_in_subgroup(black_box(&a)));
}

#[inline(never)]
pub fn bls12_381_ate_pairing() {
    let (a, _, _, _) = rand_points_g1();
    let (_, b, _, _) = rand_points_g2();
    black_box(BLS12381AtePairing::compute(
        black_box(&a),
        black_box(&b),
    ));
}