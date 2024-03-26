use super::field_extension::BLS12377PrimeField;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12377Curve;

impl IsEllipticCurve for BLS12377Curve {
    type BaseField = BLS12377PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new_base("8848defe740a67c8fc6225bf87ff5485951e2caa9d41bb188282c8bd37cb5cd5481512ffcd394eeab9b16eb21be9ef"),
            FieldElement::<Self::BaseField>::new_base("1914a69c5102eff1f674f5d30afeec4bd7fb348ca3e52d96d182ad44fb82305c2fe3d3634a9591afd82de55559c8ea6"),
            FieldElement::one()
        ])
    }
}

impl IsShortWeierstrass for BLS12377Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement,
    };

    use super::BLS12377Curve;

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BLS12377PrimeField>;
    type G = ShortWeierstrassProjectivePoint<BLS12377Curve>;

    /*
    Sage Script
    p = 0x01ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001
    K = GF(p)
    a = K(0x00)
    b = K(0x01)
    BLS12377 = EllipticCurve(K, (a, b))
    G = BLS12377(0x008848defe740a67c8fc6225bf87ff5485951e2caa9d41bb188282c8bd37cb5cd5481512ffcd394eeab9b16eb21be9ef, 0x01914a69c5102eff1f674f5d30afeec4bd7fb348ca3e52d96d182ad44fb82305c2fe3d3634a9591afd82de55559c8ea6)
    BLS12377.set_order(0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001 * 0x170b5d44300000000000000000000000)
    A = BLS12377.random_point()
    A
    (
        185715225421599677688496133374382711149570106831239566019550837184282390335799472770108146757505713919084294610666 :
        226835770810037734000988117489667915889151774823469923455001166091086436508552790841957035105033265332805473294242 :
        1
    )
    hex(185715225421599677688496133374382711149570106831239566019550837184282390335799472770108146757505713919084294610666) = 0x134e4cc122cb62a06767fb98e86f2d5f77e2a12fefe23bb0c4c31d1bd5348b88d6f5e5dee2b54db4a2146cc9f249eea
    hex(226835770810037734000988117489667915889151774823469923455001166091086436508552790841957035105033265332805473294242) = 0x17949c29effee7a9f13f69b1c28eccd78c1ed12b47068836473481ff818856594fd9c1935e3d9e621901a2d500257a2
    */
    fn point_a() -> ShortWeierstrassProjectivePoint<BLS12377Curve> {
        let x = FEE::new_base("134e4cc122cb62a06767fb98e86f2d5f77e2a12fefe23bb0c4c31d1bd5348b88d6f5e5dee2b54db4a2146cc9f249eea");
        let y = FEE::new_base("17949c29effee7a9f13f69b1c28eccd78c1ed12b47068836473481ff818856594fd9c1935e3d9e621901a2d500257a2");
        BLS12377Curve::create_point_from_affine(x, y).unwrap()
    }

    /*
    Sage Script
    p = 0x01ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001
    K = GF(p)
    a = K(0x00)
    b = K(0x01)
    BLS12377 = EllipticCurve(K, (a, b))
    G = BLS12377(0x008848defe740a67c8fc6225bf87ff5485951e2caa9d41bb188282c8bd37cb5cd5481512ffcd394eeab9b16eb21be9ef, 0x01914a69c5102eff1f674f5d30afeec4bd7fb348ca3e52d96d182ad44fb82305c2fe3d3634a9591afd82de55559c8ea6)
    BLS12377.set_order(0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001 * 0x170b5d44300000000000000000000000)
    B = BLS12377.random_point()
    B
    (
        167910251469625486980431977408648262447289657635704443470642748164203906484187619178645564374510526311475720833977 :
        233041263489949055107604421486755046495039707716704236453419951932994109989855281068314990316269556707377307191075 :
        1
    )
    hex(167910251469625486980431977408648262447289657635704443470642748164203906484187619178645564374510526311475720833977) = 0x1174782c0671cd2ae609b529978dae63f31c589cc996ebcaf792450e6b801bdf92ecb0d7865679747cf54341a854bb9
    hex(233041263489949055107604421486755046495039707716704236453419951932994109989855281068314990316269556707377307191075) = 0x1839c08bacf636e121f92f0742cda3ed6de87d20ece945ea88bf899b92e9ab6d1448ba77cd5dedaab14f63adbd00723
    */
    fn point_b() -> ShortWeierstrassProjectivePoint<BLS12377Curve> {
        let x = FEE::new_base("0x1174782c0671cd2ae609b529978dae63f31c589cc996ebcaf792450e6b801bdf92ecb0d7865679747cf54341a854bb9");
        let y = FEE::new_base("0x1839c08bacf636e121f92f0742cda3ed6de87d20ece945ea88bf899b92e9ab6d1448ba77cd5dedaab14f63adbd00723");
        BLS12377Curve::create_point_from_affine(x, y).unwrap()
    }

    /*
    Sage Script
    p = 0x01ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001
    K = GF(p)
    a = K(0x00)
    b = K(0x01)
    BLS12377 = EllipticCurve(K, (a, b))
    G = BLS12377(0x008848defe740a67c8fc6225bf87ff5485951e2caa9d41bb188282c8bd37cb5cd5481512ffcd394eeab9b16eb21be9ef, 0x01914a69c5102eff1f674f5d30afeec4bd7fb348ca3e52d96d182ad44fb82305c2fe3d3634a9591afd82de55559c8ea6)
    BLS12377.set_order(0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001 * 0x170b5d44300000000000000000000000)
    A = BLS12377(
        185715225421599677688496133374382711149570106831239566019550837184282390335799472770108146757505713919084294610666 :
        226835770810037734000988117489667915889151774823469923455001166091086436508552790841957035105033265332805473294242 :
        1
    )
    B = BLS12377(
        167910251469625486980431977408648262447289657635704443470642748164203906484187619178645564374510526311475720833977 :
        233041263489949055107604421486755046495039707716704236453419951932994109989855281068314990316269556707377307191075 :
        1
    )
    C = A + B
    C
    (
        105510848507580533682871489191558181735436434808435802972504642836660141416496473114260973143340460322786467521650 :
        123365990209709213735093352869543145829792441593791183105898411672710954937339866064025434786354433273692672296842 :
        1
    )
    hex(105510848507580533682871489191558181735436434808435802972504642836660141416496473114260973143340460322786467521650) = 0xaf7e1876d1de10b6770e4c903766c1792bbc38940936232062a947f789ae94bc556e13afea086e20203917629a2472
    hex(123365990209709213735093352869543145829792441593791183105898411672710954937339866064025434786354433273692672296842) = 0xcd30be41ef3055ebca9d96c9d391bd245d70c412b4340134df0dc1b18957da466ee8d45857363e4ee581f557c9078
    */
    fn point_c() -> ShortWeierstrassProjectivePoint<BLS12377Curve> {
        let x = FEE::new_base("0xaf7e1876d1de10b6770e4c903766c1792bbc38940936232062a947f789ae94bc556e13afea086e20203917629a2472");
        let y = FEE::new_base("0xcd30be41ef3055ebca9d96c9d391bd245d70c412b4340134df0dc1b18957da466ee8d45857363e4ee581f557c9078a");
        BLS12377Curve::create_point_from_affine(x, y).unwrap()
    }

    fn point_a_times_5() -> ShortWeierstrassProjectivePoint<BLS12377Curve> {
        let x = FEE::new_base("3c852d5aab73fbb51e57fbf5a0a8b5d6513ec922b2611b7547bfed74cba0dcdfc3ad2eac2733a4f55d198ec82b9964");
        let y = FEE::new_base("a71425e68e55299c64d7eada9ae9c3fb87a9626b941d17128b64685fc07d0e635f3c3a512903b4e0a43e464045967b");
        BLS12377Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn a_operate_with_b_is_c() {
        assert_eq!(point_a().operate_with(&point_b()), point_c())
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

    #[test]
    fn adding_five_times_point_a_works() {
        let point_a = point_a();
        let point_a_times_5 = point_a_times_5();
        assert_eq!(point_a.operate_with_self(5_u16), point_a_times_5);
    }

    #[test]
    fn add_point1_2point1_with_both_algorithms_matches() {
        let point_a = point_a();
        let point_2 = &point_a.operate_with(&point_a).to_affine();

        let first_algorithm_result = point_a.operate_with(point_2).to_affine();
        let second_algorithm_result = point_a.operate_with_affine(point_2).to_affine();

        assert_eq!(first_algorithm_result, second_algorithm_result);
    }

    #[test]
    fn add_point1_and_42424242point1_with_both_algorithms_matches() {
        let point_a = point_a();
        let point_2 = &point_a.operate_with_self(42424242u128).to_affine();

        let first_algorithm_result = point_a.operate_with(point_2).to_affine();
        let second_algorithm_result = point_a.operate_with_affine(point_2).to_affine();

        assert_eq!(first_algorithm_result, second_algorithm_result);
    }

    #[test]
    fn add_point1_with_point1_both_algorithms_matches() {
        let point_a = point_a().to_affine();

        let first_algorithm_result = point_a.operate_with(&point_a).to_affine();
        let second_algorithm_result = point_a.operate_with_affine(&point_a).to_affine();

        assert_eq!(first_algorithm_result, second_algorithm_result);
    }

    #[test]
    fn add_point2_with_point1_both_algorithms_matches() {
        let point_a = point_a().to_affine();

        // Create point 2
        let x = FEE::new_base("134e4cc122cb62a06767fb98e86f2d5f77e2a12fefe23bb0c4c31d1bd5348b88d6f5e5dee2b54db4a2146cc9f249eea") * FEE::from(2);
        let y = FEE::new_base("17949c29effee7a9f13f69b1c28eccd78c1ed12b47068836473481ff818856594fd9c1935e3d9e621901a2d500257a2") * FEE::from(2);
        let z = FEE::from(2);
        let point_2 = ShortWeierstrassProjectivePoint::<BLS12377Curve>::new([x, y, z]);

        let first_algorithm_result = point_2.operate_with(&point_a).to_affine();
        let second_algorithm_result = point_2.operate_with_affine(&point_a).to_affine();

        assert_eq!(first_algorithm_result, second_algorithm_result);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_a();
        assert_eq!(*p.x(), FEE::new_base("134e4cc122cb62a06767fb98e86f2d5f77e2a12fefe23bb0c4c31d1bd5348b88d6f5e5dee2b54db4a2146cc9f249eea"));
        assert_eq!(*p.y(), FEE::new_base("17949c29effee7a9f13f69b1c28eccd78c1ed12b47068836473481ff818856594fd9c1935e3d9e621901a2d500257a2"));
        assert_eq!(*p.z(), FEE::new_base("1"));
    }

    #[test]
    fn create_invalid_points_panics() {
        assert_eq!(
            BLS12377Curve::create_point_from_affine(FEE::from(1), FEE::from(1)).unwrap_err(),
            EllipticCurveError::InvalidPoint
        )
    }

    #[test]
    fn equality_works() {
        let g = BLS12377Curve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = BLS12377Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }
}
