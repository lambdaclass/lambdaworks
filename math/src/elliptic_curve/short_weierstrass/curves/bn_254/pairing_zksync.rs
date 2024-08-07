use super::{
    curve::BN254Curve,
    field_extension::{BN254PrimeField, Degree12ExtensionField, Degree2ExtensionField},
    twist::BN254TwistCurve,
};

use crate::elliptic_curve::traits::FromAffine;
use crate::{cyclic_group::IsGroup, elliptic_curve::traits::IsPairing, errors::PairingError};
use crate::{
    elliptic_curve::short_weierstrass::{
        curves::bn_254::field_extension::{Degree6ExtensionField, LevelTwoResidue},
        point::ShortWeierstrassProjectivePoint,
        traits::IsShortWeierstrass,
    },
    field::{element::FieldElement, extensions::cubic::HasCubicNonResidue},
    unsigned_integer::element::{UnsignedInteger, U256},
};
pub type BN254TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

type FpE = FieldElement<BN254PrimeField>;
type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp6E = FieldElement<Degree6ExtensionField>;
type Fp12E = FieldElement<Degree12ExtensionField>;

// Constants for the pairing
pub const X: u64 = 4965661367192848881;

/// @notice constant for the coeffitients of the sextic twist of the BN256 curve.
/// @dev E': y' ** 2 = x' ** 3 + 3 / (9 + u)
/// @dev the curve E' is defined over Fp2 elements.
/// @dev Constant form https://hackmd.io/@jpw/bn254#Twists.
pub const TWISTED_CURVE_COEFFS: Fp2E =
    Fp2E::const_from_raw([
        FieldElement::from_hex_unchecked(
            "2b149d40ceb8aaae81be18991be06ac3b5b4c5e559dbefa33267e6dc24a138e5",
        ),
        FieldElement::from_hex_unchecked(
            "9713b03af0fed4cd2cafadeed8fdf4a74fa084e52d1852e4a2bd0685c315d2",
        ),
    ]);

// t = 6 * x^2
// NAF calculated in python.
pub const T: [i32; 128] = [
    0, -1, 0, 1, 0, 0, 1, 0, 1, 0, -1, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, -1,
    0, 0, -1, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1, 0, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1,
    0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1, 0, 0, 0, -1, 0, 0, -1, 0, -1, 0,
    0, 0, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 1, 0, 1, 0,
    -1, 0, 0, 0, -1, 0, 0, 1,
];

fn double_step(
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
) -> (Fp12E, ShortWeierstrassProjectivePoint<BN254TwistCurve>) {
    let two_inv = FpE::from(2).inv().unwrap();
    let t0 = q.x() * q.y();
    let t1 = &two_inv * t0;
    let t2 = q.y().square();
    let t3 = q.z().square();
    let mut t4 = t3.double();
    t4 = t4 + &t3;
    let mut t5 = TWISTED_CURVE_COEFFS;
    t5 = &t5 * t4;
    let mut t6 = t5.double();
    t6 = t6 + &t5;
    let mut t7 = &t2 + &t6;
    t7 = two_inv * t7;
    let mut t8 = q.y() + q.z();
    t8 = t8.square();
    let t9 = t3 + &t2;
    t8 = &t8 - t9;
    let t10 = &t5 - &t2;
    let t11 = q.x().square();
    let t12 = t5.square();
    let mut t13 = t12.double();
    t13 = t13 + t12;

    let [x_p, y_p, _] = p.to_affine().coordinates().clone();

    let l = Fp12E::new([
        Fp6E::new([(-y_p) * t8.clone(), Fp2E::zero(), Fp2E::zero()]),
        Fp6E::new([x_p * (t11.double() + &t11), t10, Fp2E::zero()]),
    ]);

    let x_r = (&t2 - t6) * t1;
    let y_r = t7.square() - t13;
    let z_r = t2 * t8;
    let r = ShortWeierstrassProjectivePoint::new([x_r, y_r, z_r]);
    
    (l, r)
}

fn double_accumulate_line(
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
) -> (Fp12E, ShortWeierstrassProjectivePoint<BN254TwistCurve>) {
    // We convert the projective coordinates into jacobian coordinates.
    // projective (a, b, c) -> jacobian (a * c, b * c^2, c).
    let z_q = q.z();
    let x_q = q.x() * z_q;
    let y_q = q.y() * z_q.square();

    // @nicole: Is it ok p as a ProjectivPoint or do we need to have it in two coordinates?
    // let [x_p, y_p] = p.to_affine().coordintes();
    let [x_p, y_p, _] = p.to_affine().coordinates().clone();

    let mut tmp0 = x_q.square(); // 1
    let tmp1 = y_q.square(); // 2
    let tmp2 = &tmp1.square(); // 3
    let mut tmp3 = (&tmp1 + &x_q).square() - tmp0.clone() - tmp2; // 4
    tmp3 = tmp3.clone().double(); // 5) 2 * d
    let tmp4 = tmp0.clone().double() + &tmp0; // 6) 3 * a
    let mut tmp6 = x_q + &tmp4; // 7
    let tmp5 = tmp4.square(); // 8
    let mut x_t = &tmp5 - &tmp3 - &tmp3; // 9
    let z_t = (y_q + z_q).square() - &tmp1 - z_q.square(); // 10
    let mut y_t = (tmp3 - &x_t) * &tmp4 - &tmp2.double().double().double(); //11
    tmp3 = -((&tmp4 * z_q.square()).double()); //12
    tmp3 = x_p * tmp3; //13 . Watchout with the order of the operation  x_p*d .d * x_p is not defined
    tmp6 = tmp6.square() - tmp0 - tmp5 - tmp1.double().double(); //14
    tmp0 = (&z_t * z_q.square()).double(); //15
    tmp0 = y_p * tmp0; // 16

    // The zero coordinates don't seem right. (see line function).
    let a_0 = Fp6E::new([tmp0, Fp2E::zero(), Fp2E::zero()]); //18
    let a_1 = Fp6E::new([tmp3, tmp6, Fp2E::zero()]); //19 //a_1 = 0*v^2 + e*v + d
    let l = Fp12E::new([a_0, a_1]); //l = a_0 + a_1*w

    // convert t from jacobian to projective coordinates
    // jacob (a,b,c) -> projec (a/c, b/c^2, c)
    x_t = x_t * z_t.inv().unwrap();
    y_t = y_t * z_t.square().inv().unwrap();
    let t = ShortWeierstrassProjectivePoint::new([x_t, y_t, z_t]);

    // *accumulator = accumulator.square() * l;

    (l, t)
}

fn line(
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    t: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
) -> (Fp12E) {
    // We convert projective coordinates of p into affine coordinates.
    // Projective (a, b, c) -> Affine (a/c, b/b).
    let [x_p, y_p, _] = p.to_affine().coordinates().clone();

    // We convert the projective coordinates of q into jacobian coordinates.
    // projective (a, b, c) -> jacobian (a * c, b * c^2, c).
    let z_q = q.z().clone();
    let x_q = q.x() * z_q.clone();
    let y_q = q.y() * z_q.square();

    // We convert the projective coordinates of t into jacobian coordinates.
    let z_t = t.z().clone();
    let x_t = t.x() * z_t.clone();
    let y_t = t.y() * z_t.square();

    if q == t {
        let r = t.operate_with_self(2usize);
        // The projective and jacobian coordinates have the same z.
        let z_r = r.z();

        let l0 = Fp6E::new([
            &y_p * (z_r.double() * &z_t.square()), // 2 * z_r * (z_t)^2 * y_p
            (x_t.pow(3usize)).double().double() + (x_t.pow(3usize)).double()
                - y_t.square().double().double(), // 6 * (x_t)^3 - 4 * (y_t)^2
            Fp2E::zero(),
        ]);
        let l1 = Fp6E::new([
            -(&x_p * ((&x_t.square().double().double() + x_t.square().double()) * z_t.square())), // -6 * (x_t)^2 * (z_t)^2 * x_p
            Fp2E::zero(),
            Fp2E::zero(),
        ]);
        Fp12E::new([l0, l1])
    } else {
        let r = t.operate_with(q);
        let z_r = r.z();

        let l0 = Fp6E::new([
            &y_p * z_r.double(), // 2 * z_r * y_p
            x_q.double().double() * (&y_q * z_t.pow(3usize) * &x_q - &y_t) - y_q.double() * z_r, // 4 * x_q * (y_q * (z_t)^3 * x_q - y_t) - 2 * y_q * z_r
            Fp2E::zero(),
        ]);
        let l1 = Fp6E::new([
            -(x_p.double().double() * (&y_q * z_t.pow(3usize) + y_t)), // -4 * x_p * (y_q * (z_t)^3 + y_t)
            Fp2E::zero(),
            Fp2E::zero(),
        ]);
        Fp12E::new([l0, l1])
    }
}

fn line_2(
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    t: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
) -> (Fp12E) {
    /* TO DO?
    if p == FpE::zero() || q == Fp12E::zero() || t == Fp12E::zero() {
        ERROR
    } */

    let [x_p, y_p, _] = p.to_affine().coordinates().clone();
    let [x_q, y_q, _] = q.phi_inv().coordinates().clone();
    let [x_t, y_t, _] = t.phi_inv().coordinates().clone();

    //TODO: if p, q or t are inf return error.

    let mut l: Fp12E = Fp12E::from(1);

    // First case:
    if x_t != x_q {
        println!("t and q are different points");
        let a = y_p * (&x_q - &x_t).square();
        let b = x_p * (&y_t - &y_q).square();
        let c = (x_t * y_q - x_q * y_t).square();
        /*
        println!("a is: {:?}", a);
        println!("b is: {:?}", b);
        println!("c is: {:?}", c);
        */

        l = Fp12E::from_coefficients(&[
            "0",
            "0",
            &(a.value()[0].to_hex()),
            &(a.value()[1].to_hex()),
            "0",
            "0",
            "0",
            "0",
            &(b.value()[0].to_hex()),
            &(b.value()[1]).to_hex(),
            &(c.value()[0].to_hex()),
            &(c.value()[1]).to_hex(),
        ]);
    // Second case: t and q are the same points
    } else if y_t == y_q {
        println!("t and q are the same points");
        let a = Fp2E::new([FpE::from(9), FpE::one()])
            * (x_t.pow(3 as u32).double() + x_t.pow(3 as u32) - y_t.square().double()); //(9 + u) * (3 * (x_t)^3 - 2 * (y_t)^2)
        let b = (y_p * y_t).double(); // 2 * y_t * y_p
        let c = -((&x_p * x_t.square()).double() + (x_p * x_q.square())); // -3 * x_p * (x_t)^2
        l = Fp12E::from_coefficients(&[
            &(a.value()[0].to_hex()),
            &(a.value()[1].to_hex()),
            "0",
            "0",
            &(c.value()[0].to_hex()),
            &(c.value()[1]).to_hex(),
            "0",
            "0",
            &(b.value()[0].to_hex()),
            &(b.value()[1]).to_hex(),
            "0",
            "0",
        ]);
    }

    l
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve,
        unsigned_integer::element::U384,
    };

    use super::*;

    #[test]
    // works
    fn double_step_doubles_point_correctly() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let r = double_step(&p, &q).1;
        print!("{:?}", r);
        assert_eq!(r, q.operate_with_self(2usize));
    }

    #[test]
    fn double_step_line_equals_double_accumulate_line() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        assert_eq!(double_step(&p, &q).0, double_accumulate_line(&q, &p).0)
    }


    
    #[test]
    fn double_step_line_equals_line() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        assert_eq!(double_step(&p, &q).0, line(&p,&q,&q))
    }

    #[test]
    fn double_step_line_equals_line2() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        assert_eq!(double_step(&p, &q).0, line_2(&p,&q,&q))
    }
}