use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint,
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
    unsigned_integer::element::U256,
};

use super::{
    curve::BLS12381Curve, field_extension::Order12ExtensionField, twist::BLS12381TwistCurve,
};

const PRIME_R: U256 =
    U256::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
const NORMALIZATION_POWER: &[u64] = &[
    0x0000000002ee1db5,
    0xdcc825b7e1bda9c0,
    0x496a1c0a89ee0193,
    0xd4977b3f7d4507d0,
    0x7363baa13f8d14a9,
    0x17848517badc3a43,
    0xd1073776ab353f2c,
    0x30698e8cc7deada9,
    0xc0aadff5e9cfee9a,
    0x074e43b9a660835c,
    0xc872ee83ff3a0f0f,
    0x1c0ad0d6106feaf4,
    0xe347aa68ad49466f,
    0xa927e7bb93753318,
    0x07a0dce2630d9aa4,
    0xb113f414386b0e88,
    0x19328148978e2b0d,
    0xd39099b86e1ab656,
    0xd2670d93e4d7acdd,
    0x350da5359bc73ab6,
    0x1a0c5bf24c374693,
    0xc49f570bcd2b01f3,
    0x077ffb10bf24dde4,
    0x1064837f27611212,
    0x596bc293c8d4c01f,
    0x25118790f4684d0b,
    0x9c40a68eb74bb22a,
    0x40ee7169cdc10412,
    0x96532fef459f1243,
    0x8dfc8e2886ef965e,
    0x61a474c5c85b0129,
    0x127a1b5ad0463434,
    0x724538411d1676a5,
    0x3b5a62eb34c05739,
    0x334f46c02c3f0bd0,
    0xc55d3109cd15948d,
    0x0a1fad20044ce6ad,
    0x4c6bec3ec03ef195,
    0x92004cedd556952c,
    0x6d8823b19dadd7c2,
    0x498345c6e5308f1c,
    0x511291097db60b17,
    0x49bf9b71a9f9e010,
    0x0418a3ef0bc62775,
    0x1bbd81367066bca6,
    0xa4c1b6dcfc5cceb7,
    0x3fc56947a403577d,
    0xfa9e13c24ea820b0,
    0x9c1d9f7c31759c36,
    0x35de3f7a36399917,
    0x08e88adce8817745,
    0x6c49637fd7961be1,
    0xa4c7e79fb02faa73,
    0x2e2f3ec2bea83d19,
    0x6283313492caa9d4,
    0xaff1c910e9622d2a,
    0x73f62537f2701aae,
    0xf6539314043f7bbc,
    0xe5b78c7869aeb218,
    0x1a67e49eeed2161d,
    0xaf3f881bd88592d7,
    0x67f67c4717489119,
    0x226c2f011d4cab80,
    0x3e9d71650a6f8069,
    0x8e2f8491d12191a0,
    0x4406fbc8fbd5f489,
    0x25f98630e68bfb24,
    0xc0bcb9b55df57510,
];

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
        f = f.pow(2_u64) * line(&r, &r, &p);
        r = r.operate_with(&r).to_affine();
        if *bit {
            f = f * line(&r, q, p);
            r = r.operate_with(q).to_affine();
        }
    }
    f
}

fn final_exponentiation(
    base: &FieldElement<Order12ExtensionField>,
) -> FieldElement<Order12ExtensionField> {
    let bit_representation_exponent = NORMALIZATION_POWER;
    let mut pow = FieldElement::one();
    let mut base = base.clone();

    // This is computes the power of base raised to the target_normalization_power
    for (index, limb) in bit_representation_exponent.iter().rev().enumerate() {
        let mut limb = *limb;
        for _bit in 1..=16 {
            if limb & 1 == 1 {
                pow = &pow * &base;
            }
            base = &base * &base;
            let finished = (index == bit_representation_exponent.len() - 1) && (limb == 0);
            if !finished {
                limb >>= 1;
            } else {
                break;
            }
        }
    }
    pow
}

fn ate(
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    q: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
) -> FieldElement<Order12ExtensionField> {
    let base = miller(p, q);
    final_exponentiation(&base)
}

/// Evaluates the Self::line between points `p` and `r` at point `q`
pub fn line(
    p: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    r: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    q: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
) -> FieldElement<Order12ExtensionField> {
    // TODO: Improve error handling.
    debug_assert!(
        !q.is_neutral_element(),
        "q cannot be the point at infinity."
    );
    let [px, py] = p.to_fp12_affine();
    let [rx, ry] = r.to_fp12_affine();
    let [qx_fp, qy_fp, _] = q.coordinates().clone();
    let qx = FieldElement::<Order12ExtensionField>::new([
        FieldElement::new([
            FieldElement::new([qx_fp, FieldElement::zero()]),
            FieldElement::zero(),
            FieldElement::zero(),
        ]),
        FieldElement::zero(),
    ]);
    let qy = FieldElement::<Order12ExtensionField>::new([
        FieldElement::new([
            FieldElement::new([qy_fp, FieldElement::zero()]),
            FieldElement::zero(),
            FieldElement::zero(),
        ]),
        FieldElement::zero(),
    ]);
    let a_of_curve = FieldElement::<Order12ExtensionField>::new([
        FieldElement::new([
            FieldElement::new([BLS12381Curve::a(), FieldElement::zero()]),
            FieldElement::zero(),
            FieldElement::zero(),
        ]),
        FieldElement::zero(),
    ]);

    if p.is_neutral_element() || r.is_neutral_element() {
        if p == r {
            return FieldElement::one();
        }
        if p.is_neutral_element() {
            qx - rx
        } else {
            qx - px
        }
    } else if p != r {
        if px == rx {
            qx - px
        } else {
            let l = (ry - &py) / (rx - &px);
            qy - py - l * (qx - px)
        }
    } else {
        let numerator = FieldElement::from(3) * &px.pow(2_u16) + a_of_curve;
        let denominator = FieldElement::from(2) * &py;
        if denominator == FieldElement::zero() {
            qx - px
        } else {
            let l = numerator / denominator;
            qy - py - l * (qx - px)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{elliptic_curve::traits::IsEllipticCurve, unsigned_integer::element::U384};

    use super::*;

    #[test]
    fn final_exp() {
        /*
        let base = FieldElement::<Order12ExtensionField>::new([
            FieldElement::new([
                FieldElement::new([FieldElement::one(), FieldElement::one()]),
                FieldElement::new([FieldElement::one(), FieldElement::one()]),
                FieldElement::new([FieldElement::one(), FieldElement::one()])
            ]),
            FieldElement::new([
                FieldElement::new([FieldElement::one(), FieldElement::one()]),
                FieldElement::new([FieldElement::one(), FieldElement::one()]),
                FieldElement::new([FieldElement::one(), FieldElement::one()])
            ])
        ]);

        let base_square = FieldElement::<Order12ExtensionField>::new([
            FieldElement::new([
                FieldElement::new([FieldElement::new(U384::from("680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaac")), FieldElement::new(U384::from("680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaae"))]),
                FieldElement::new([FieldElement::new(U384::from("1380cd6fab1fecf3b854bdc8b278c1a18b5978a3b6a3ce0f8d649df8b904b89b1700ffff04feffffcb7f3fffffffc001")), FieldElement::new(U384::from("1380cd6fab1fecf3b854bdc8b278c1a18b5978a3b6a3ce0f8d649df8b904b89b1700ffff04feffffcb7f3fffffffc007"))]),
                FieldElement::new([FieldElement::new(U384::from("680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaab")), FieldElement::new(U384::from("680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeab5"))])
            ]),
            FieldElement::new([
                FieldElement::new([FieldElement::new(U384::from("5")), FieldElement::new(U384::from("1"))]),
                FieldElement::new([FieldElement::new(U384::from("d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd556")), FieldElement::new(U384::from("d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd55e"))]),
                FieldElement::new([FieldElement::new(U384::from("0")), FieldElement::new(U384::from("c"))])
            ])
        ]);
        */
        // Original sage
        // 0x680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaac
        // 0x680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaae
        // 0x5
        // 0x1
        // 0x1380cd6fab1fecf3b854bdc8b278c1a18b5978a3b6a3ce0f8d649df8b904b89b1700ffff04feffffcb7f3fffffffc001
        // 0x1380cd6fab1fecf3b854bdc8b278c1a18b5978a3b6a3ce0f8d649df8b904b89b1700ffff04feffffcb7f3fffffffc007
        // 0xd0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd556
        // 0xd0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd55e
        // 0x680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaab
        // 0x680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeab5
        // 0x0
        // 0xc

        // Reordenado
        // 0x680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaac
        // 0x680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaae
        // 0x1380cd6fab1fecf3b854bdc8b278c1a18b5978a3b6a3ce0f8d649df8b904b89b1700ffff04feffffcb7f3fffffffc001
        // 0x1380cd6fab1fecf3b854bdc8b278c1a18b5978a3b6a3ce0f8d649df8b904b89b1700ffff04feffffcb7f3fffffffc007
        // 0x680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaab
        // 0x680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeab5
        // 0x5
        // 0x1
        // 0xd0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd556
        // 0xd0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd55e
        // 0x0
        // 0xc

        //assert_eq!(base.pow(2_u16), base_square);
        //assert_eq!(
        //    final_exponentiation(&FieldElement::one()),
        //    FieldElement::one()
        //);

        assert_eq!(
            FieldElement::<Order12ExtensionField>::one().pow(2_u16),
            FieldElement::one()
        );
        assert_eq!(
            FieldElement::new_from_coefficients(&[
                "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"
            ]),
            FieldElement::one()
        );
        assert_eq!(
            FieldElement::new_from_coefficients(&[
                "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0"
            ])
            .pow(2_u16),
            FieldElement::new_from_coefficients(&[
                "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0"
            ]),
        );
        assert_eq!(
            FieldElement::new_from_coefficients(&["0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]).pow(2_u16),
            FieldElement::new_from_coefficients(&["1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaaa", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]),
        );
        assert_eq!(
            FieldElement::new_from_coefficients(&[
                "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"
            ])
            .pow(2_u16),
            FieldElement::new_from_coefficients(&[
                "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"
            ]),
        );

        //FieldElement::new_from_coefficients(["0", "1", ])
    }

    #[test]
    fn ate_pairing() {
        let p = BLS12381Curve::generator().to_affine();
        let q = BLS12381TwistCurve::generator().to_affine();
        let a = U384::from_u64(2);
        let b = U384::from_u64(2);

        assert_eq!(
            ate(
                &p.operate_with_self(a).to_affine(),
                &q.operate_with_self(b).to_affine()
            ),
            ate(&p.operate_with_self(a * b).to_affine(), &q)
        )
        // e(a*P, b*Q) = e(a*b*P, Q) = e(P, a*b*Q)
    }
}
/*
0000000000000000000000000000000017f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
0000000000000000000000000000000008b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1

00000000000000000000000000000000024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8
0000000000000000000000000000000013e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e
000000000000000000000000000000000ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801
000000000000000000000000000000000606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be

0000000000000000000000000000000017f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
0000000000000000000000000000000008b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1

00000000000000000000000000000000024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8
0000000000000000000000000000000013e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e
000000000000000000000000000000000ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801
000000000000000000000000000000000606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be
*/
