use crate::{
    elliptic_curve::traits::HasEllipticCurveOperations,
    field::{
        element::FieldElement,
        extensions::{
            cubic::{CubicExtensionField, HasCubicNonResidue},
            quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
        },
        fields::u384_prime_field::{HasU384Constant, U384PrimeField},
    },
};
use crypto_bigint::U384;

/// Order of the base field (e.g.: order of the coordinates)
const fn order_p() -> U384 {
    U384::from_be_hex("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab")
}

/// Order of the subgroup of the curve.
const fn order_r() -> U384 {
    U384::from_be_hex("0000000000000000000000000000000073eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001")
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModP;
impl HasU384Constant for ModP {
    const VALUE: U384 = order_p();
}

#[derive(Debug, Clone)]
pub struct LevelOneResidue;
impl HasQuadraticNonResidue for LevelOneResidue {
    type BaseField = U384PrimeField<ModP>;

    fn residue() -> FieldElement<U384PrimeField<ModP>> {
        -FieldElement::one()
    }
}

type LevelOneField = QuadraticExtensionField<LevelOneResidue>;

#[derive(Debug, Clone)]
pub struct LevelTwoResidue;
impl HasCubicNonResidue for LevelTwoResidue {
    type BaseField = LevelOneField;

    fn residue() -> FieldElement<LevelOneField> {
        FieldElement::new([FieldElement::from(1), FieldElement::from(1)])
    }
}

type LevelTwoField = CubicExtensionField<LevelTwoResidue>;

#[derive(Debug, Clone)]
pub struct LevelThreeResidue;
impl HasQuadraticNonResidue for LevelThreeResidue {
    type BaseField = LevelTwoField;

    fn residue() -> FieldElement<LevelTwoField> {
        FieldElement::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ])
    }
}

type LevelThreeField = QuadraticExtensionField<LevelThreeResidue>;

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12381Curve;
impl HasEllipticCurveOperations for BLS12381Curve {
    type BaseField = LevelThreeField;
    type UIntOrders = U384;

    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(4)
    }

    fn generator_affine_x() -> FieldElement<Self::BaseField> {
        FieldElement::new([
            FieldElement::new([
                FieldElement::new([FieldElement::new(U384::from_be_hex("3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507")), FieldElement::zero()]),
                FieldElement::zero(),
                FieldElement::zero(),
            ]),
            FieldElement::zero(),
        ])
    }

    fn generator_affine_y() -> FieldElement<Self::BaseField> {
        FieldElement::new([
            FieldElement::new([
                FieldElement::new([FieldElement::new(U384::from_be_hex("1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569")), FieldElement::zero()]),
                FieldElement::zero(),
                FieldElement::zero(),
            ]),
            FieldElement::zero(),
        ])
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
