mod elliptic_curves;

use elliptic_curves::{iai_bls12_377::*, iai_bls12_381::*};

iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = bls12_381_operate_with_g1,
    bls12_381_operate_with_g2,
    bls12_381_operate_with_self_g1,
    bls12_381_operate_with_self_g2,
    bls12_381_double_g1,
    bls12_381_double_g2,
    bls12_381_neg_g1,
    bls12_381_neg_g2,
    bls12_381_compress_g1,
    bls12_381_decompress_g1,
    bls12_381_subgroup_check_g1,
    bls12_381_ate_pairing,
    bls12_377_operate_with_g1,
    bls12_377_operate_with_self_g1,
    bls12_377_double_g1,
    bls12_377_neg_g1,
);
