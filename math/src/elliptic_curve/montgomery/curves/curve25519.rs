use crate::{
    elliptic_curve::{
        montgomery::{point::MontgomeryProjectivePoint, traits::IsMontgomery},
        traits::IsEllipticCurve,
    },
    field::{element::FieldElement, fields::curve25519_field::Curve25519PrimeField},
    unsigned_integer::element::U256,
};

//Implementation of the 255-bit prime field Montgomery Curve Curve25519
#[derive(Debug, Clone)]
pub struct Curve25519;

impl FieldElement<Curve25519PrimeField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U256::from(a_hex))
    }
}

impl IsEllipticCurve for Curve25519 {
    // p = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed
    type BaseField = Curve25519PrimeField;
    type PointRepresentation = MontgomeryProjectivePoint<Self>;

    // G = (0x09, 0x20ae19a1b8a086b4e01edd2c7748d14c923d4d7e6d7c61b229e9c5a27eced3d9)
    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new_base("0x09"),
            FieldElement::<Self::BaseField>::new_base(
                "0x20ae19a1b8a086b4e01edd2c7748d14c923d4d7e6d7c61b229e9c5a27eced3d9",
            ),
            FieldElement::<Self::BaseField>::one(),
        ])
    }
}

impl IsMontgomery for Curve25519 {
    fn a() -> FieldElement<Self::BaseField> {
        // a = 0x76d06
        FieldElement::from(486662)
    }

    fn b() -> FieldElement<Self::BaseField> {
        // b = 0x1
        FieldElement::one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement,
    };
    #[allow(clippy::upper_case_acronyms)]
    type FE = FieldElement<Curve25519PrimeField>;

    /*
       sage script
       p = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed
       Fp = GF(p)
       a = K(0x76d06)
       b = K(0x01)
       E = EllipticCurve(Fp, ((3 - a^2)/(3 * b^2), (2 * a^3 - 9 * a)/(27 * b^3)))
       def to_weierstrass(a, b, x, y):
           return (x/b + a/(3*b), y/b)
       def to_montgomery(a, b, u, v):
           return (b * (u - a/(3*b)), b*v)
       E.set_order(0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed * 0x08)

       P = E.random_point()
       P = (55982355253182784578671123550301779825433794365936117526811944583306035556801 : 7252143264526750761856340913670505401463091321304403727596372170490063892550 : 1)
       to_montgomery(a, b, K(55982355253182784578671123550301779825433794365936117526811944583306035556801), K(7252143264526750761856340913670505401463091321304403727596372170490063892550)) = (36683673713630085341409292715520461849888796921662690186902347248653847121264,
    7252143264526750761856340913670505401463091321304403727596372170490063892550)
       hex(36683673713630085341409292715520461849888796921662690186902347248653847121264) = 0x511a3939af1f568fadea95dd734fe90412af04a637afc3b11813d66741e24d70
       hex(7252143264526750761856340913670505401463091321304403727596372170490063892550) = 0x100891500644f6850e25d95925ab209aaf7e5e1fa5d8036f07725063e0543046
       */
    fn point() -> MontgomeryProjectivePoint<Curve25519> {
        let x = FE::new_base("0x511a3939af1f568fadea95dd734fe90412af04a637afc3b11813d66741e24d70");
        let y = FE::new_base("0x100891500644f6850e25d95925ab209aaf7e5e1fa5d8036f07725063e0543046");
        Curve25519::create_point_from_affine(x, y).unwrap()
    }

    /*
       sage script
       p = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed
       Fp = GF(p)
       a = K(0x76d06)
       b = K(0x01)
       E = EllipticCurve(Fp, ((3 - a^2)/(3 * b^2), (2 * a^3 - 9 * a)/(27 * b^3)))
       def to_weierstrass(a, b, x, y):
           return (x/b + a/(3*b), y/b)
       def to_montgomery(a, b, u, v):
           return (b * (u - a/(3*b)), b*v)
       G = E(*to_weierstrass(a, b, K(0x09), K(0x20ae19a1b8a086b4e01edd2c7748d14c923d4d7e6d7c61b229e9c5a27eced3d9)))
       E.set_order(0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed * 0x08)

       P = E.random_point()
       P * 5 = (43673264053381013043093028227158132460078739296636781789931124335052711458784 : 32008223164755653184586471723570822084978824108746555469179824380913641587957 : 1)
       to_montgomery(a, b, K(43673264053381013043093028227158132460078739296636781789931124335052711458784), K(32008223164755653184586471723570822084978824108746555469179824380913641587957)) = (24374582513828313805831197392376814484533741852363354450021527000400523023247,
    32008223164755653184586471723570822084978824108746555469179824380913641587957)
       hex(24374582513828313805831197392376814484533741852363354450021527000400523023247) = 0x35e38659ceab3f043e0f37a5872c1d8d484e6eca6014589b453bf375201dd38f
       hex(32008223164755653184586471723570822084978824108746555469179824380913641587957) = 0x46c403265a55dd5e2baffc0c0b64e18ab378d312d01768eed674a9fcbe4404f5
       */
    fn point_times_5() -> MontgomeryProjectivePoint<Curve25519> {
        let x = FE::new_base("0x35e38659ceab3f043e0f37a5872c1d8d484e6eca6014589b453bf375201dd38f");
        let y = FE::new_base("0x46c403265a55dd5e2baffc0c0b64e18ab378d312d01768eed674a9fcbe4404f5");
        Curve25519::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn adding_five_times_point_works() {
        let point = point();
        let point_times_5 = point_times_5();
        assert_eq!(point.operate_with_self(5_u16), point_times_5);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point();
        assert_eq!(
            *p.x(),
            FE::new_base("0x511a3939af1f568fadea95dd734fe90412af04a637afc3b11813d66741e24d70")
        );
        assert_eq!(
            *p.y(),
            FE::new_base("0x100891500644f6850e25d95925ab209aaf7e5e1fa5d8036f07725063e0543046")
        );
        assert_eq!(*p.z(), FE::new_base("1"));
    }

    #[test]
    fn addition_with_neutral_element_returns_same_element() {
        let p = point();
        assert_eq!(
            *p.x(),
            FE::new_base("0x511a3939af1f568fadea95dd734fe90412af04a637afc3b11813d66741e24d70")
        );
        assert_eq!(
            *p.y(),
            FE::new_base("0x100891500644f6850e25d95925ab209aaf7e5e1fa5d8036f07725063e0543046")
        );

        let neutral_element = ShortWeierstrassProjectivePoint::<Curve25519>::neutral_element();

        assert_eq!(p.operate_with(&neutral_element), p);
    }

    #[test]
    fn neutral_element_plus_neutral_element_is_neutral_element() {
        let neutral_element = ShortWeierstrassProjectivePoint::<BN254Curve>::neutral_element();

        assert_eq!(
            neutral_element.operate_with(&neutral_element),
            neutral_element
        );
    }


    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            Curve25519::create_point_from_affine(FE::from(0), FE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = Curve25519::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = Curve25519::generator();
        let g2 = g.operate_with_self(2_u64);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        //𝐵𝑦² = 𝑥³ + 𝐴𝑥² + 𝑥
        let a = FE::from(486662);
        let b = FieldElement::one();
        let y_sq_0 = x.pow(3_u16) + x.pow(2_u16) * a + x;
        let y_sq_1 = y.pow(2_u16) * b;

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works() {
        let g = Curve25519::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }
}
