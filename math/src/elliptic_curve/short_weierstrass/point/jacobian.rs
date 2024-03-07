use crate::{cyclic_group::IsGroupElement, elliptic_curve::{short_weierstrass::traits::IsShortWeierstrass, traits::IsEllipticCurve}, field::element::FieldElement};

use super::ShortWeierstrassProjectivePoint;

// feels like `crate::elliptic_curbe::point` could be redesigned to provide the trait to implement here
#[derive(Clone)]
pub struct JacobianPoint<E: IsShortWeierstrass> {
    x: FieldElement<E::BaseField>,
    y: FieldElement<E::BaseField>,
    z: FieldElement<E::BaseField>,
}

impl<E: IsShortWeierstrass> PartialEq for JacobianPoint<E> {
    fn eq(&self, other: &Self) -> bool {
        let (z_sq_self, z_sq_other) = 
            (self.z.square(), other.z.square());
        (&self.x * &z_sq_self == &other.x * &z_sq_other) 
            && (&self.y * &z_sq_self * &self.z == &other.y * z_sq_other * &other.z) 
    }
}
impl<E: IsShortWeierstrass> Eq for JacobianPoint<E> {}

impl<E: IsShortWeierstrass> JacobianPoint<E> {
    pub fn double(&self) -> Self {
        if &self.y == &FieldElement::zero() {return Self::neutral_element();}
        let s = FieldElement::<E::BaseField>::from(4) * &self.x * &self.y.square();
        let m = 
            FieldElement::<E::BaseField>::from(3) * &self.x.square() 
                + E::a() * &self.z.square().square();
        
        let two = FieldElement::<E::BaseField>::from(2);

        let x = 
            &m.square() - &two * &s;
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
        x: FieldElement<E::BaseField>, y: FieldElement<E::BaseField>, z: FieldElement<E::BaseField>
    ) -> Self {
        Self {
            x,
            y,
            z
        }
    }
}

impl<E: IsShortWeierstrass> IsGroupElement for JacobianPoint<E> {
    fn neutral_element() -> Self {
        Self { x: FieldElement::zero(),
        y: FieldElement::one(),
        z: FieldElement::zero(),}
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
        let y = 
            u * (v_squared * v2 - a) - &v_cubed * &u2;
        let z = v_cubed * w;

        Self { x, y, z }
    }


    fn neg(&self) -> Self {
        Self{
            x: self.x.clone(), 
            y: -self.y.clone(), 
            z: self.z.clone()
        }
    }
}

impl<E> From<ShortWeierstrassProjectivePoint<E>> for JacobianPoint<E>
where E: IsShortWeierstrass 
{
    fn from(point: ShortWeierstrassProjectivePoint<E>) -> Self {
        if let Ok(z_inv) = point.z().inv() {
            Self {
                x: point.x() * &z_inv,
                y: point.y() * z_inv.square(),
                z: point.z().to_owned()
            }
        }
        else {Self::neutral_element()}
    }
}
