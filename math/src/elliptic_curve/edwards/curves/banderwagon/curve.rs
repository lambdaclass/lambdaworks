pub use super::field::FqField;
use crate::elliptic_curve::edwards::point::EdwardsProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::{elliptic_curve::edwards::traits::IsEdwards, field::element::FieldElement};

pub type BaseBanderwagonFieldElement = FqField;

#[derive(Clone, Debug)]

pub struct BanderwagonCurve;

impl IsEllipticCurve for BanderwagonCurve {
    type BaseField = BaseBanderwagonFieldElement;
    type PointRepresentation = EdwardsProjectivePoint<Self>;

    // Values are from https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/edwards/curves/bandersnatch/curve.rs
    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new_base(
                "29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18",
            ),
            FieldElement::<Self::BaseField>::new_base(
                "2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166",
            ),
            FieldElement::one(),
        ])
    }
}

impl IsEdwards for BanderwagonCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base(
            "73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFEFFFFFFFC",
        )
    }

    fn d() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base(
            "6389C12633C267CBC66E3BF86BE3B6D8CB66677177E54F92B369F2F5188D58E7",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement, unsigned_integer::element::U256,
    };

    #[allow(clippy::upper_case_acronyms)]

    type FEE = FieldElement<BaseBanderwagonFieldElement>;

    fn point_1() -> EdwardsProjectivePoint<BanderwagonCurve> {
        /*
        Here the coordinates in sage are in Weierstrass form, so we need to convert them to Twisted Edwards form.
        The formula is taken from https://github.com/zhenfeizhang/bandersnatch/blob/main/bandersnatch/script/bandersnatch.sage
        sage: p=52435875175126190479447740508185965837690552500527637822603658699938581184513
        sage: Fp = GF(p)
        sage: ban = EllipticCurve(Fp, [-3763200000,-78675968000000])
        sage: G = ban(0xa76451786f95a802c0982bbd0abd68e41b92adc86c8859b4f44679b21658710,0x44d150c8b4bd14f79720d021a839e7b7eb4ee43844b30243126a72ac2375490a)
        sage: G
        (4732093294267640299242820317528400560681136891967543338160850811774078125840 : 31127102290931869693084292284935581507759552409643462510093198106308390504714 : 1)
        */

        let x = FEE::new_base("29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18");
        let y = FEE::new_base("2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166");

        BanderwagonCurve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn test_scalar_mul() {
        /*
        Here the coordinates in sage are in Weierstrass form, so we need to convert them to Edwards form.
        The formula is taken from https://github.com/zhenfeizhang/bandersnatch/blob/main/bandersnatch/script/bandersnatch.sage
        sage: p=52435875175126190479447740508185965837690552500527637822603658699938581184513
        sage: Fp = GF(p)
        sage: ban = EllipticCurve(Fp, [-3763200000,-78675968000000])
        sage: G = ban(0xa76451786f95a802c0982bbd0abd68e41b92adc86c8859b4f44679b21658710,0x44d150c8b4bd14f79720d021a839e7b7eb4ee43844b30243126a72ac2375490a)
        sage: G
        (4732093294267640299242820317528400560681136891967543338160850811774078125840 : 31127102290931869693084292284935581507759552409643462510093198106308390504714 : 1)
        sage: P = G*5
        sage: P
        (29008922875876791132439276322366220655270680966305110461148260308934935549949 : 5321989321016801167609012824343080578288974189043484058186624183847078047548 : 1)
        */

        let g = BanderwagonCurve::generator();
        let result1 = g.operate_with_self(5u16);

        assert_eq!(
            result1.x().clone(),
            FEE::new_base("68CBECE0B8FB55450410CBC058928A567EED293D168FAEF44BFDE25F943AABE0")
        );

        let scalar =
            U256::from_hex("1CFB69D4CA675F520CCE760202687600FF8F87007419047174FD06B52876E7E6")
                .unwrap();
        let result2 = g.operate_with_self(scalar);

        assert_eq!(
            result2.x().clone(),
            FEE::new_base("68CBECE0B8FB55450410CBC058928A567EED293D168FAEF44BFDE25F943AABE0")
        );
    }


    #[test]
    fn test_create_valid_point_works() {
        /*
        Here the coordinates in sage are in Weierstrass form, so we need to convert them to Twisted Edwards form.
        The formula is taken from https://github.com/zhenfeizhang/bandersnatch/blob/main/bandersnatch/script/bandersnatch.sage
        sage: p=52435875175126190479447740508185965837690552500527637822603658699938581184513
        sage: Fp = GF(p)
        sage: ban = EllipticCurve(Fp, [-3763200000,-78675968000000])
        sage: G = ban(0xa76451786f95a802c0982bbd0abd68e41b92adc86c8859b4f44679b21658710,0x44d150c8b4bd14f79720d021a839e7b7eb4ee43844b30243126a72ac2375490a)
        sage: G
        (4732093294267640299242820317528400560681136891967543338160850811774078125840 : 31127102290931869693084292284935581507759552409643462510093198106308390504714 : 1)
        */

        let p = BanderwagonCurve::generator();

        assert_eq!(p, p.clone());
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(
            *p.x(),
            FEE::new_base("29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18")
        );
        assert_eq!(
            *p.y(),
            FEE::new_base("2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166")
        );
        assert_eq!(*p.z(), FEE::new_base("1"));
    }

    #[test]
    fn create_invalid_points_panics() {
        assert_eq!(
            BanderwagonCurve::create_point_from_affine(FEE::from(1), FEE::from(1)).unwrap_err(),
            EllipticCurveError::InvalidPoint
        )
    }

    #[test]
    fn equality_works() {
        /*
        Here the coordinates in sage are in Weierstrass form, so we need to convert them to Twisted Edwards form.
        The formula is taken from https://github.com/zhenfeizhang/bandersnatch/blob/main/bandersnatch/script/bandersnatch.sage
        sage: p=52435875175126190479447740508185965837690552500527637822603658699938581184513
        sage: Fp = GF(p)
        sage: ban = EllipticCurve(Fp, [-3763200000,-78675968000000])
        sage: G = ban(0xa76451786f95a802c0982bbd0abd68e41b92adc86c8859b4f44679b21658710,0x44d150c8b4bd14f79720d021a839e7b7eb4ee43844b30243126a72ac2375490a)
        sage: G
        (4732093294267640299242820317528400560681136891967543338160850811774078125840 : 31127102290931869693084292284935581507759552409643462510093198106308390504714 : 1)
        sage: R = G*2
        sage: R
        (31951609130618743256244397091892446950585800848077116798961915903626008441884 : 44003096583616910216940410849480500633572695092183489174446167302668306571218 : 1)
        */
        
        let g = BanderwagonCurve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn operate_with_self_works_1() {
        /*
        Here the coordinates in sage are in Weierstrass form, so we need to convert them to Twisted Edwards form.
        The formula is taken from https://github.com/zhenfeizhang/bandersnatch/blob/main/bandersnatch/script/bandersnatch.sage
        sage: p=52435875175126190479447740508185965837690552500527637822603658699938581184513
        sage: Fp = GF(p)
        sage: ban = EllipticCurve(Fp, [-3763200000,-78675968000000])
        sage: G = ban(0xa76451786f95a802c0982bbd0abd68e41b92adc86c8859b4f44679b21658710,0x44d150c8b4bd14f79720d021a839e7b7eb4ee43844b30243126a72ac2375490a)
        sage: G
        (4732093294267640299242820317528400560681136891967543338160850811774078125840 : 31127102290931869693084292284935581507759552409643462510093198106308390504714 : 1)
        sage: Q = G*3
        sage: Q
        (31019807805656217859737735480197534678630587750605614610381590357187303463025 : 7443365003011992916088981749186597178295868666409019489786056142248668879353 : 1)
        */

        let g = BanderwagonCurve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }
}
