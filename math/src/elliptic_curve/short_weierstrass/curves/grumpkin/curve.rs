use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::fields::montgomery_backed_prime_fields::{
    IsModulus, MontgomeryBackendPrimeField,
};
use crate::unsigned_integer::element::U256;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

/// Grumpkin an elliptic curve on top of BN254 for SNARK efficient group operations used by the Aztec Protocol.
/// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
/// a = 0
/// b = -17
/// r = 21888242871839275222246405745257275088696311157297823662689037894645226208583
/// Grumpkin is a cycle curve together with BN254 meaning the field and group order of Grumpkin are equal to the group and field order of BN254 G1.
#[derive(Clone, Debug)]
pub struct GrumpkinCurve;

impl IsEllipticCurve for GrumpkinCurve {
    type BaseField = GrumpkinPrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    // G = (1, sprt(-16)) = (1, 17631683881184975370165255887551781615748388533673675138860) = (0x1, 0x2cf135e7506a45d632d270d45f1181294833fc48d823f272c)
    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::one(),
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "0x2cf135e7506a45d632d270d45f1181294833fc48d823f272c",
            ),
            FieldElement::one(),
        ])
    }
}

impl IsShortWeierstrass for GrumpkinCurve {
    // a = 0
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    // b = -17
    fn b() -> FieldElement<Self::BaseField> {
        -FieldElement::from(17)
    }
}

// Grumpkin Fp
// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// Equal tp BN254 Fr
pub type GrumpkinFieldElement = FieldElement<GrumpkinPrimeField>;

pub const GRUMPKIN_PRIME_FIELD_ORDER: U256 =
    U256::from_hex_unchecked("0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

#[derive(Clone, Debug)]
pub struct GrumpkinFieldModulus;
impl IsModulus<U256> for GrumpkinFieldModulus {
    const MODULUS: U256 = GRUMPKIN_PRIME_FIELD_ORDER;
}

pub type GrumpkinPrimeField = MontgomeryBackendPrimeField<GrumpkinFieldModulus, 4>;

impl FieldElement<GrumpkinPrimeField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U256::from(a_hex))
    }
}

#[derive(Clone, Debug)]
pub struct FrConfig;

/// Modulus (Order) of Grumpkin Fr
// r = 21888242871839275222246405745257275088696311157297823662689037894645226208583
impl IsModulus<U256> for FrConfig {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
    );
}

/// Grumpkin Fr
/// Equal to BN254 Fp
pub type FrField = MontgomeryBackendPrimeField<FrConfig, 4>;
/// FrElement using MontgomeryBackend for Grumpkin
pub type FrElement = FieldElement<FrField>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement,
    };

    use super::GrumpkinCurve;

    #[allow(clippy::upper_case_acronyms)]
    type FE = GrumpkinFieldElement;
    type G = ShortWeierstrassProjectivePoint<GrumpkinCurve>;

    /*
    Sage script:
    p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    Fp = GF(p)
    a = Fp(0)
    b = Fp(-17)
    Grumpkin = EllipticCurve(Fp, [a, b])
    A = Grumpkin.random_point()
    A
    (15485117031023686537706709698619053905755909389649581280762364787786480506330 :
    8998283053861550708725041915039948040873858194502192019982314435709819336827 :
    1)
    hex(15485117031023686537706709698619053905755909389649581280762364787786480506330) = 0x223c44015b1ab0705802e079ad06dc25f608633c83192ed0720bd396ab3a55da
    hex(8998283053861550708725041915039948040873858194502192019982314435709819336827) = 0x13e4d9047d76f812c834a27f2bbaab6ca5fd62ed34ac2e1ff1870ab083f2b87b
    */
    fn point_a() -> ShortWeierstrassProjectivePoint<GrumpkinCurve> {
        let x = FE::from_hex_unchecked(
            "0x223c44015b1ab0705802e079ad06dc25f608633c83192ed0720bd396ab3a55da",
        );
        let y = FE::from_hex_unchecked(
            "0x13e4d9047d76f812c834a27f2bbaab6ca5fd62ed34ac2e1ff1870ab083f2b87b",
        );
        GrumpkinCurve::create_point_from_affine(x, y).unwrap()
    }

    /*
    Sage script:
    p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    Fp = GF(p)
    a = Fp(0)
    b = Fp(-17)
    Grumpkin = EllipticCurve(Fp, [a, b])
    B = Grumpkin.random_point()
    B
    (3268464931655688768836539556233897014960714449731494149362780603936864794149 :
    21602298792738915452436924250014118513160691830629170285452329874861320517987 :
    1)
    hex(3268464931655688768836539556233897014960714449731494149362780603936864794149) = 0x739e2b6472d405bbc4263464866a15483858b3083004a4246834e8fbba0f625
    hex(21602298792738915452436924250014118513160691830629170285452329874861320517987) = 0x2fc277c38298f4cca2798dcf7f8cc4e943c3e913fff7f8adca03db5920073963
    */
    fn point_b() -> ShortWeierstrassProjectivePoint<GrumpkinCurve> {
        let x = FE::from_hex_unchecked(
            "0x739e2b6472d405bbc4263464866a15483858b3083004a4246834e8fbba0f625",
        );
        let y = FE::from_hex_unchecked(
            "0x2fc277c38298f4cca2798dcf7f8cc4e943c3e913fff7f8adca03db5920073963",
        );
        GrumpkinCurve::create_point_from_affine(x, y).unwrap()
    }

    /*
    Sage script:
    p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    Fp = GF(p)
    a = Fp(0)
    b = Fp(-17)
    Grumpkin = EllipticCurve(Fp, [a, b])
    A = Grumpkin(15485117031023686537706709698619053905755909389649581280762364787786480506330 :
    8998283053861550708725041915039948040873858194502192019982314435709819336827 :
    1)
    B = Grumpkin(
    3268464931655688768836539556233897014960714449731494149362780603936864794149 :
    21602298792738915452436924250014118513160691830629170285452329874861320517987 :
    1)
    C = A + B
    C
    (19589511723685515259809937322960156453150080799519189498903102061830421386151 :
    8780443621842377450927011336607258768303562920725643392690114380788691381924 :
    1)
    hex(19589511723685515259809937322960156453150080799519189498903102061830421386151) = 0x2b4f454dbe9c92fa321542229d214338d5459b71e9e2d995318d9358256093a7
    hex(8780443621842377450927011336607258768303562920725643392690114380788691381924) = 0x13698e12e1076132830116e982f293dd07271f34aa03ebfe23eb9301229c2ea4
    */
    fn point_c() -> ShortWeierstrassProjectivePoint<GrumpkinCurve> {
        let x = FE::from_hex_unchecked(
            "0x2b4f454dbe9c92fa321542229d214338d5459b71e9e2d995318d9358256093a7",
        );
        let y = FE::from_hex_unchecked(
            "0x13698e12e1076132830116e982f293dd07271f34aa03ebfe23eb9301229c2ea4",
        );
        GrumpkinCurve::create_point_from_affine(x, y).unwrap()
    }

    /*
    Sage script:
    p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    Fp = GF(p)
    a = Fp(0)
    b = Fp(-17)
    Grumpkin = EllipticCurve(Fp, [a, b])
    P = Grumpkin.random_point()
    P * 5
    (15046418650485865292177180299665505401798701105523584252220614421753423008361 : 17852720053004908540584849282553401192842244835354847668310708345588581105130 : 1)
    hex(15046418650485865292177180299665505401798701105523584252220614421753423008361) = 0x2143f89e0ac0942ed1a891a83b5e5b3d4ed46722c24f72dfbdd5fedad27d1269
    hex(17852720053004908540584849282553401192842244835354847668310708345588581105130) = 0x2778480e45647fbe25e497e995c2ac24b6e0411fb01c657460412c142d2f7dea
    */
    fn point_a_times_5() -> ShortWeierstrassProjectivePoint<GrumpkinCurve> {
        let x = FE::from_hex_unchecked(
            "0x2143f89e0ac0942ed1a891a83b5e5b3d4ed46722c24f72dfbdd5fedad27d1269",
        );
        let y = FE::from_hex_unchecked(
            "0x2778480e45647fbe25e497e995c2ac24b6e0411fb01c657460412c142d2f7dea",
        );
        GrumpkinCurve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn a_operate_with_b_is_c() {
        assert_eq!(point_a().operate_with(&point_b()), point_c())
    }

    #[test]
    fn adding_five_times_point_works() {
        let point = point_a();
        let point_times_5 = point_a_times_5();
        assert_eq!(point.operate_with_self(5_u16), point_times_5);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_a();
        assert_eq!(
            *p.x(),
            FE::new_base("0x223c44015b1ab0705802e079ad06dc25f608633c83192ed0720bd396ab3a55da")
        );
        assert_eq!(
            *p.y(),
            FE::new_base("0x13e4d9047d76f812c834a27f2bbaab6ca5fd62ed34ac2e1ff1870ab083f2b87b")
        );
        assert_eq!(*p.z(), FE::one());
    }

    #[test]
    fn addition_with_neutral_element_returns_same_element() {
        let p = point_a();
        assert_eq!(
            *p.x(),
            FE::new_base("0x223c44015b1ab0705802e079ad06dc25f608633c83192ed0720bd396ab3a55da")
        );
        assert_eq!(
            *p.y(),
            FE::new_base("0x13e4d9047d76f812c834a27f2bbaab6ca5fd62ed34ac2e1ff1870ab083f2b87b")
        );

        let neutral_element = ShortWeierstrassProjectivePoint::<GrumpkinCurve>::neutral_element();

        assert_eq!(p.operate_with(&neutral_element), p);
    }

    #[test]
    fn neutral_element_plus_neutral_element_is_neutral_element() {
        let neutral_element = ShortWeierstrassProjectivePoint::<GrumpkinCurve>::neutral_element();

        assert_eq!(
            neutral_element.operate_with(&neutral_element),
            neutral_element
        );
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            GrumpkinCurve::create_point_from_affine(FE::from(0), FE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = GrumpkinCurve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = GrumpkinCurve::generator();
        let g2 = g.operate_with_self(2_u64);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        // calculate both sides y² = 𝑥³ + 𝑎𝑥² + b
        // a = 0
        // b = -17
        let b = -FieldElement::from(17);
        let y_sq_0 = x.pow(3_u16) + b;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    // checks: P * O = O * P
    #[test]
    fn add_inf_to_point_should_not_modify_point() {
        // Pick an arbitrary point
        let point = point_a();
        // P * O
        let left = point.operate_with(&G::neutral_element());
        // O * P
        let right = G::neutral_element().operate_with(&point);
        assert_eq!(left, point);
        assert_eq!(right, point);
    }

    // P * -P = O
    #[test]
    fn add_opposite_of_a_point_to_itself_gives_neutral_element() {
        // Pick an arbitrary point
        let point = point_a();
        // P * O
        let neg_point = point.neg();
        let res = point.operate_with(&neg_point);
        assert_eq!(res, G::neutral_element());
    }

    //Scalar mul depends only on the scalar mod r
    #[test]
    fn scalar_mul_depends_on_scalar_mod_r() {
        let r = FrConfig::MODULUS;
        let gen = GrumpkinCurve::generator();
        let gen_neg = gen.neg();
        let g = gen.operate_with_self(r);

        let r_sub_one = r - U256::from(1u64);
        let op3 = gen.operate_with_self(r_sub_one);

        // random scalar value
        let s = U256::from(3u64);
        let blinded_scalar = (s * r) + s;
        let op1 = gen.operate_with_self(s);
        let op2 = gen.operate_with_self(blinded_scalar);

        assert_eq!(op1, op2);
        assert_eq!(g, G::neutral_element());
        assert_ne!(op1, G::neutral_element());
        assert_eq!(gen_neg, op3);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = GrumpkinCurve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }
}
