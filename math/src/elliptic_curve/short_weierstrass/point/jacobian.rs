//! Jacobian variant of projective
//! coordinates. It works in the similar way, but differs in how affine get divided by $z$ ($\frac{x}{z^2}, \frac{y}{z^3}$).

use crate::{
    cyclic_group::IsGroupElement, elliptic_curve::short_weierstrass::traits::IsShortWeierstrass,
    field::element::FieldElement,
};

use super::ShortWeierstrassProjectivePoint;

// feels like `crate::elliptic_curbe::point` could be redesigned to provide the trait to implement here
#[derive(Clone, Debug)]
pub struct JacobianPoint<E: IsShortWeierstrass> {
    x: FieldElement<E::BaseField>,
    y: FieldElement<E::BaseField>,
    z: FieldElement<E::BaseField>,
}

impl<E: IsShortWeierstrass> PartialEq for JacobianPoint<E> {
    fn eq(&self, other: &Self) -> bool {
        let (z_sq_self, z_sq_other) = (self.z.square(), other.z.square());
        (&self.x * &z_sq_self == &other.x * &z_sq_other)
            && (&self.y * &z_sq_self * &self.z == &other.y * z_sq_other * &other.z)
    }
}
impl<E: IsShortWeierstrass> Eq for JacobianPoint<E> {}

impl<E: IsShortWeierstrass> JacobianPoint<E> {
    pub fn double(&self) -> Self {
        if self.y == FieldElement::zero() {
            return Self::neutral_element();
        }
        let s = FieldElement::<E::BaseField>::from(4) * &self.x * &self.y.square();
        let m = FieldElement::<E::BaseField>::from(3) * &self.x.square()
            + E::a() * &self.z.square().square();

        let two = FieldElement::<E::BaseField>::from(2);

        let x = &m.square() - &two * &s;
        let y = m * (s - &x) - FieldElement::<E::BaseField>::from(8) * &self.y.square().square();
        let z = &two * &self.y * &self.z;
        Self { x, y, z }
    }
    /// Creates the same point in affine coordinates. That is,
    /// returns $[\frac{x}{z^2} : \frac{y}{z^3} : 1]$ where `self` is $[x: y: z]$.
    /// Panics if `self` is point at infinity.
    pub fn to_affine(&self) -> Self {
        let z_sq = self.z.square();
        Self {
            x: &self.x / &z_sq,
            y: &self.y / (&self.z * z_sq),
            z: FieldElement::one(),
        }
    }
    /// Creates an elliptic curve point giving the Jacobian projective $[x: y: 1]$ coordinates.
    pub const fn new(
        x: FieldElement<E::BaseField>,
        y: FieldElement<E::BaseField>,
        z: FieldElement<E::BaseField>,
    ) -> Self {
        Self { x, y, z }
    }
}

impl<E: IsShortWeierstrass> IsGroupElement for JacobianPoint<E> {
    fn neutral_element() -> Self {
        Self {
            x: FieldElement::zero(),
            y: FieldElement::one(),
            z: FieldElement::zero(),
        }
    }

    fn is_neutral_element(&self) -> bool {
        self.z == FieldElement::zero()
    }

    fn operate_with(&self, other: &Self) -> Self {
        let u1 = &other.y * &self.z;
        let u2 = &self.y * &other.z;
        let v1 = &other.x * &self.z;
        let v2 = &self.x * &other.z;

        if v1 == v2 {
            if u1 != u2 {
                return Self::neutral_element();
            } else {
                return self.double();
            }
        }

        let u = &u1 - &u2;
        let v = &v1 - &v2;
        let w = &self.z * &other.z;

        let v_squared = v.square();
        let v_cubed = &v * &v_squared;

        let a =
            u.square() * &w - &v_cubed - FieldElement::<E::BaseField>::from(2) * &v_squared * &v2;
        let x = &v * &a;
        let y = u * (v_squared * v2 - a) - &v_cubed * &u2;
        let z = v_cubed * w;

        Self { x, y, z }
    }

    fn neg(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: -self.y.clone(),
            z: self.z.clone(),
        }
    }
}

impl<E> From<ShortWeierstrassProjectivePoint<E>> for JacobianPoint<E>
where
    E: IsShortWeierstrass,
{
    fn from(point: ShortWeierstrassProjectivePoint<E>) -> Self {
        if let Ok(z_inv) = point.z().inv() {
            Self {
                x: point.x() * &z_inv,
                y: point.y() * z_inv.square(),
                z: point.z().to_owned(),
            }
        } else {
            Self::neutral_element()
        }
    }
}

#[cfg(test)]
mod tests {
    // `from` and `to_affine` are base function for these tests hence they are tested implicitly

    use super::{*, super::ProjectivePoint};
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
    use super::{ShortWeierstrassProjectivePoint, IsGroupElement};
    use crate::elliptic_curve::traits::IsEllipticCurve;

    /* TODO go with a random point when there's a method in `ProjectivePoint`
    until then the point from `super::tests` is just pasted here
    #naiveRandom */
    // not that naive random would use seeds for repeatable testing
    #[cfg(feature = "alloc")]
    fn point() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        use crate::elliptic_curve::{short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField, traits::IsEllipticCurve};

        let x = FieldElement::<BLS12381PrimeField>::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FieldElement::<BLS12381PrimeField>::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }
    // a random scalar would be cool too, let's just go with small numbers -- it's ok for these tests #naiveRandom
    const SMALLSCALAR_A: u16 = const_random::const_random!(u16);
    // fn helper_smallscalar() -> BLS12381Curve::ScalarField {
    //     BLS12381Curve::ScalarField:: from(1234567890123456789012345678901234567890)
    // }

    #[cfg(feature = "alloc")]
    #[test]
    fn double() {
        // let r: u128 = rand::random();
        let mut subj = JacobianPoint::from(point());
        let point_the = point();
        let mut proj = point_the.clone();

        for _ in 0..SMALLSCALAR_A {
            subj = subj.double();
            proj = proj.double();
        }

        let subj_aff = subj.to_affine();
        let proj_aff = proj.to_affine().coordinates().to_owned();
        assert_eq!(subj_aff.x, proj_aff[0]);
        assert_eq!(subj_aff.y, proj_aff[1]);

        let point_the_aff = point_the.to_affine().coordinates().to_owned();
        assert_ne!(subj_aff.x, point_the_aff[0]);
        assert_ne!(subj_aff.y, point_the_aff[1]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn eq_and_operate_with() {
        let subj = JacobianPoint::from(point());
        assert_eq!(subj.operate_with(&subj), subj.double());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn new() {
        let gen_aff = BLS12381Curve::generator().coordinates().to_owned();
        let subj = JacobianPoint::<BLS12381Curve>::new(
            gen_aff[0].clone(), gen_aff[1].clone(), FieldElement::one()
        );

        assert_eq!(subj.double().to_affine().x, BLS12381Curve::generator().double().to_affine().coordinates()[0]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn neutral_element() {
        // TODO when there will be improvement over unwrapping let back `.to_affine()` variant
        // let subj = JacobianPoint::<BLS12381Curve>::neutral_element().to_affine();
        // let proj = 
        //     <ShortWeierstrassProjectivePoint<BLS12381Curve> as IsGroupElement>::neutral_element().to_affine().coordinates().to_owned();
        let subj = JacobianPoint::<BLS12381Curve>::neutral_element();
        let proj = 
            <ShortWeierstrassProjectivePoint<BLS12381Curve> as IsGroupElement>::neutral_element().coordinates().to_owned();
        assert_eq!(subj.x, proj[0]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn neg_and_is_neutral_element() {
        let g = JacobianPoint::from(BLS12381Curve::generator());
        assert!(!g.is_neutral_element());
        assert!((g.operate_with(&g.neg())).is_neutral_element());
    }
}