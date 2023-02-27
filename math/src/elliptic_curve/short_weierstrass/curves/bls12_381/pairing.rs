use super::{
    curve::BLS12381Curve, field_extension::Order12ExtensionField, twist::BLS12381TwistCurve,
};
use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint,
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
    unsigned_integer::element::UnsignedInteger,
};

const MILLER_LOOP_CONSTANT: u64 = 0xd201000000010000; // This is equal to the frobenius trace minus one.
const NORMALIZATION_POWER: UnsignedInteger<68> = UnsignedInteger::from("2ee1db5dcc825b7e1bda9c0496a1c0a89ee0193d4977b3f7d4507d07363baa13f8d14a917848517badc3a43d1073776ab353f2c30698e8cc7deada9c0aadff5e9cfee9a074e43b9a660835cc872ee83ff3a0f0f1c0ad0d6106feaf4e347aa68ad49466fa927e7bb9375331807a0dce2630d9aa4b113f414386b0e8819328148978e2b0dd39099b86e1ab656d2670d93e4d7acdd350da5359bc73ab61a0c5bf24c374693c49f570bcd2b01f3077ffb10bf24dde41064837f27611212596bc293c8d4c01f25118790f4684d0b9c40a68eb74bb22a40ee7169cdc1041296532fef459f12438dfc8e2886ef965e61a474c5c85b0129127a1b5ad0463434724538411d1676a53b5a62eb34c05739334f46c02c3f0bd0c55d3109cd15948d0a1fad20044ce6ad4c6bec3ec03ef19592004cedd556952c6d8823b19dadd7c2498345c6e5308f1c511291097db60b1749bf9b71a9f9e0100418a3ef0bc627751bbd81367066bca6a4c1b6dcfc5cceb73fc56947a403577dfa9e13c24ea820b09c1d9f7c31759c3635de3f7a3639991708e88adce88177456c49637fd7961be1a4c7e79fb02faa732e2f3ec2bea83d196283313492caa9d4aff1c910e9622d2a73f62537f2701aaef6539314043f7bbce5b78c7869aeb2181a67e49eeed2161daf3f881bd88592d767f67c4717489119226c2f011d4cab803e9d71650a6f80698e2f8491d12191a04406fbc8fbd5f48925f98630e68bfb24c0bcb9b55df57510");

#[allow(unused)]
fn miller(
    q: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
) -> FieldElement<Order12ExtensionField> {
    let mut r = q.clone();
    let mut f = FieldElement::<Order12ExtensionField>::one();
    let mut miller_loop_constant = MILLER_LOOP_CONSTANT;
    let mut miller_loop_constant_bits: Vec<bool> = vec![];

    // TODO: improve this to avoid using U256 everywhere.
    while miller_loop_constant > 0 {
        miller_loop_constant_bits.insert(0, (miller_loop_constant & 1) == 1);
        miller_loop_constant >>= 1;
    }

    for bit in miller_loop_constant_bits[1..].iter() {
        f = f.pow(2_u64) * line(&r, &r, p);
        r = r.operate_with(&r).to_affine();
        if *bit {
            f = f * line(&r, q, p);
            r = r.operate_with(q);
            if !r.is_neutral_element() {
                r = r.to_affine();
            }
        }
    }
    f.inv()
}

#[allow(unused)]
fn final_exponentiation(
    base: &FieldElement<Order12ExtensionField>,
) -> FieldElement<Order12ExtensionField> {
    base.pow(NORMALIZATION_POWER)
}

#[allow(unused)]
fn ate(
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    q: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
) -> FieldElement<Order12ExtensionField> {
    let base = miller(q, p);
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
    use crate::{
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::field_extension::BLS12381_PRIME_FIELD_ORDER,
            traits::IsEllipticCurve,
        },
        unsigned_integer::element::U384,
    };

    use super::*;

    type Fp12E = FieldElement<Order12ExtensionField>;

    #[test]
    fn final_exp() {
        let one = FieldElement::<Order12ExtensionField>::one();
        assert_eq!(final_exponentiation(&one), one);
    }

    #[test]
    fn element_squared_1() {
        // base = 1 + u + (1 + u)v + (1 + u)v^2 + ((1+u) + (1 + u)v + (1+ u)v^2)w
        let element_ones =
            Fp12E::from_coefficients(&["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]);
        let element_ones_squared =
            Fp12E::from_coefficients(&["5", "7", "3", "9", "1", "b", "4", "8", "2", "a", "0", "c"]);
        assert_eq!(element_ones.pow(2_u16), element_ones_squared);
    }

    #[test]
    fn element_squared_2() {
        let element_sequence =
            Fp12E::from_coefficients(&["1", "2", "5", "6", "9", "a", "3", "4", "7", "8", "b", "c"]);

        let element_sequence_squared = Fp12E::from_coefficients(&[
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd61d",
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd66f",
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd62a",
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd6b0",
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd597",
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd6c1",
            "e0",
            "142",
            "a1",
            "167",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa5d",
            "16c"
        ]);

        assert_eq!(element_sequence.pow(2_u16), element_sequence_squared);
    }

    #[test]
    fn element_to_normalization_power() {
        // assert_eq!(element_ones.pow(2_u16), element_ones_squared);

        let element_sequence =
            Fp12E::from_coefficients(&["1", "2", "5", "6", "9", "a", "3", "4", "7", "8", "b", "c"]);

        let element_sequence_raised_to_r_power = Fp12E::from_coefficients(&[
            "6751a492e8604c0297d3015aa840ffa8cf0b1a83adbe144534ed6d99f1ee590fb64e22372146945234d4971e7c59967",
            "af30d65fed6acbfc731d26214f026ca04a67db45fd922f258a9df0971b9fe9035b617f520ec8fadfe81ed99bb4b4f35",
            "10c69fc56477cbcbd2ceceee631bdf5338f50d2351ecdf4de33aa35ef0418ee0d0acbebb93be3f45f6c3a80bf720faed",
            "18035639e29b2ee19db64be512621da650aabd7e6779fd66eff1a899e39e677bd4029fee33c450fdc3046a48adbfea5f",
            "1825c8059c53e9113a66905fdc082e7c8b4f69dc8dff62d521dd65a0c58e46fb3e0ed8c9ec341d83b037d2a90352d9cf",
            "f32b1f2c72faf5f55e2d2d59225f145b8505162a540ad4e89793e984f3fd98c5de40af7fbd4a6c1c9a59bdb430d2c1c",
            "6f98212c1e50d84414f0a3acebdcfc28bd4488782b57e478ab11db99c65df0877b829c177dddfa2f92e1b69bb8dc16c",
            "19b12e43ad1dfae439b0fb9bd338ca54ce46c3638c697cb0f09a28245595ed4b8163691c9adcb706b71680cfe994b806",
            "1a5cc2e093b6bf5c862aaf588881eaba3f510d7ad8e045c2da78a90a1b3b4ac20457771b35187a4f6d35bfe77eb503f",
            "b7ccba6d35d79d7734a7c256d05811145786b2a1d89e960540cc3472ba2c7f55b2dcf54234ea21a3f103d2472938dfe",
            "139f6b151f9ffaedc70eee88cfb298f701f732288674f399b3b1d1e0e71fb981adfcd74397954e591b557646f22f6585",
            "188f6adb85a0caabc76e27b0ce1ec4b4c6d63568e6e0e73114f54b1be5c93a74f413f44c3b2aa1114e61062267a2b52e"
        ]);

        assert_eq!(
            final_exponentiation(&element_sequence),
            element_sequence_raised_to_r_power
        );
    }

    #[test]
    fn inverse_of_u_plus_one() {
        let z =
            Fp12E::from_coefficients(&["0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0"])
                .pow(3_u16);
        let one_plus_u =
            Fp12E::from_coefficients(&["1", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]);
        assert_eq!(z * one_plus_u, FieldElement::one());
    }

    #[test]
    fn to_fp12_affine_computes_correctly() {
        let g = BLS12381TwistCurve::generator();
        let expectedx = Fp12E::from_coefficients(&[
            "0",
            "0",
            "24aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8",
            "13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0"
        ]);
        let expectedy = Fp12E::from_coefficients(&[
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801",
            "606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be",
            "0",
            "0"
        ]);
        let [g_to_fp12_x, g_to_fp12_y] = g.to_fp12_affine();
        assert_eq!(g_to_fp12_x, expectedx);
        assert_eq!(g_to_fp12_y, expectedy);
    }

    #[test]
    fn test_line_1() {
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let expected = Fp12E::from_coefficients(&[
            "8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1",
            "0",
            "0",
            "0",
            "0",
            "0",
            "11bfe7fb8923ace27a0692871443365b3e5e0dd7ef388153c5bb27d923a3dedb6b9f9cbfd022a8eb913e281a830fac9c",
            "13d0fab25f0e7099c6ffaa8ddd3cb036f5c35df322a9359ff8f98a4f5c6a84c50b6007c296eafda2ffa2d20b253a6633",
            "4c208bdb300097927393e963768099390a3f9d581d8828070e39167384e44fdaf716fa49d68b0bdf431a2f53189c109",
            "546ca700477f9c2f9def9691be2f7e6eaa0f1474cb64c53ce3b4d4da03cbdac75933b5468ab4b88cf058f147ba2cda9",
            "0",
            "0"
        ]);
        assert_eq!(line(&g2, &g2, &g1), expected);
    }

    #[test]
    fn test_line_2() {
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let g2_times_5 = g2.operate_with_self(5_u16);
        let expected = Fp12E::from_coefficients(&[
            "8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1",
            "0",
            "0",
            "0",
            "0",
            "0",
            "c61af5cbff75e6cdd8100b22636ec410a1cb914a599775a29590dd2c48af1f18a71e120fb72ddc7c1ca57ce58a8670f",
            "4b7599ae8879affcb68f23ac23ddc7316f81013a2caa8925f33fe978f19ce00effd7e5bd6b51e2d9723e66fd897e125",
            "449f41bddfdc54476c92f4249f1ce75a9693eb8f0b34ba3c57de3edaa29ba069f0d97376eadb1be6c5a5dfdf595302b",
            "ebf57b77d5ffe1476a8c8c0f0e0e73e8d32f070c89f09e465cb39312463bc42768c6028a481fb8ba2f5d2a95cd4eedb",
            "0",
            "0"
        ]);
        assert_eq!(line(&g2, &g2_times_5, &g1), expected);
    }

    #[derive(Clone, Debug)]
    struct BLS12381TestFp12;
    impl IsEllipticCurve for BLS12381TestFp12 {
        type BaseField = Order12ExtensionField;
        type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

        fn generator() -> Self::PointRepresentation {
            Self::PointRepresentation::new([
                FieldElement::<Self::BaseField>::new_base("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"),
                FieldElement::<Self::BaseField>::new_base("8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1"),
                FieldElement::one()
            ])
        }
    }

    impl IsShortWeierstrass for BLS12381TestFp12 {
        fn a() -> FieldElement<Self::BaseField> {
            FieldElement::from(0)
        }

        fn b() -> FieldElement<Self::BaseField> {
            FieldElement::from(4)
        }
    }

    #[test]
    fn test_applying_frobenius_is_the_same_as_adding_point_p_times() {
        // This checks that the generator of the twisted curve belongs to
        // ker(frobenius - [p])
        let [g2_fp12_x, g2_fp12_y] = BLS12381TwistCurve::generator().to_fp12_affine();
        let g2 = BLS12381TestFp12::create_point_from_affine(g2_fp12_x.clone(), g2_fp12_y.clone())
            .unwrap();

        let adding_point_p_times = g2.operate_with_self(BLS12381_PRIME_FIELD_ORDER).to_affine();
        let frobenius = BLS12381TestFp12::create_point_from_affine(
            g2_fp12_x.pow(BLS12381_PRIME_FIELD_ORDER),
            g2_fp12_y.pow(BLS12381_PRIME_FIELD_ORDER),
        )
        .unwrap();

        assert_eq!(adding_point_p_times, frobenius);
    }

    #[test]
    fn ate_pairing_bilinearity() {
        let p = BLS12381Curve::generator().to_affine();
        let q = BLS12381TwistCurve::generator().to_affine();
        let a = U384::from_u64(11);
        let b = U384::from_u64(93);

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
