use crate::elliptic_curve::curves::bls12_381::field_extension::{order_p, order_r, ModP};
use crate::unsigned_integer::UnsignedInteger384 as U384;
use crate::{
    elliptic_curve::traits::IsEllipticCurve,
    field::{element::FieldElement, fields::u384_prime_field::U384PrimeField},
};

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12381Curve;
impl IsEllipticCurve for BLS12381Curve {
    type BaseField = U384PrimeField<ModP>;
    type UIntOrders = U384;

    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(4)
    }

    fn generator_affine_x() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb")
    }

    fn generator_affine_y() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base("8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1")
    }

    fn order_r() -> Self::UIntOrders {
        order_r()
    }

    fn order_p() -> Self::UIntOrders {
        order_p()
    }

    fn target_normalization_power() -> Vec<u64> {
        vec![
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
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsCyclicGroup, elliptic_curve::element::EllipticCurveElement,
        field::element::FieldElement,
    };

    use super::BLS12381Curve;

    type FEE = FieldElement<U384PrimeField<ModP>>;

    fn point_1() -> EllipticCurveElement<BLS12381Curve> {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        EllipticCurveElement::new([x, y, FieldElement::one()])
    }

    fn point_2() -> EllipticCurveElement<BLS12381Curve> {
        let x = FEE::new_base("10cd31f0d0bc539a7322ba3f3523364c6f82001f2ca859a8349f0109a1497365cdaeda13d7cb872e7a3b72906c7c8579");
        let y = FEE::new_base("122fa2aa688cabfb11582271850bb150cf9fd5c9883b4cc37b1b03cd54081c331107beb6da392db87b3f9954e199b7d1");
        EllipticCurveElement::new([x, y, FieldElement::one()])
    }

    fn point_1_times_5() -> EllipticCurveElement<BLS12381Curve> {
        let x = FEE::new_base("32bcce7e71eb50384918e0c9809f73bde357027c6bf15092dd849aa0eac274d43af4c68a65fb2cda381734af5eecd5c");
        let y = FEE::new_base("11e48467b19458aabe7c8a42dc4b67d7390fdf1e150534caadddc7e6f729d8890b68a5ea6885a21b555186452b954d88");
        EllipticCurveElement::new([x, y, FieldElement::one()])
    }

    #[test]
    fn adding_five_times_point_1_works() {
        let point_1 = point_1();
        let point_1_times_5 = point_1_times_5();
        assert_eq!(point_1.operate_with_self(5), point_1_times_5);
    }

    // This tests only apply for the specific curve found in the configuration file.
    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(*p.x(), FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5"));
        assert_eq!(*p.y(), FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0"));
        assert_eq!(*p.z(), FEE::new_base("1"));
    }

    #[test]
    #[should_panic]
    fn create_invalid_points_panicks() {
        EllipticCurveElement::<BLS12381Curve>::new([FEE::from(0), FEE::from(1), FEE::from(1)]);
    }

    #[test]
    fn equality_works() {
        let g = EllipticCurveElement::<BLS12381Curve>::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = EllipticCurveElement::<BLS12381Curve>::generator();
        assert_eq!(g.operate_with(&g).operate_with(&g), g.operate_with_self(3));
    }
}
