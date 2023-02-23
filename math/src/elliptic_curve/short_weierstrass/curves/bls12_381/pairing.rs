use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint,
    field::element::FieldElement, unsigned_integer::element::U256,
};

use super::{
    curve::BLS12381Curve, field_extension::Order12ExtensionField, twist::BLS12381TwistCurve,
};

const PRIME_R: U256 =
    U256::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

fn miller(
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    q: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
) -> FieldElement<Order12ExtensionField> {
    let mut r = q.clone();
    let mut f = FieldElement::<Order12ExtensionField>::one();
    let mut order_r = PRIME_R;
    let mut order_r_bits: Vec<bool> = vec![];

    // TODO: improve this to avoid using U256 everywhere.
    while order_r > U256::from_u64(0) {
        order_r_bits.insert(0, (order_r & U256::from_u64(1)) == U256::from_u64(0));
        order_r = order_r >> 1;
    }

    for bit in order_r_bits[1..].iter() {
        f = f.pow(2_u64) * line(&r, &r, p);
        r = r.operate_with(&r);
        if *bit {
            f = f * line(&r, q, p);
            r = r.operate_with(q);
        }
    }
    f
}

fn line(
    r_1: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    r_2: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
) -> FieldElement<Order12ExtensionField> {
    todo!()
}
