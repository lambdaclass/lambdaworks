use super::curve::MILLER_LOOP_CONSTANT;
// We defined MILLER_LOOP_CONSTANT in curve.rs
// see https://hackmd.io/@Wimet/ry7z1Xj-2
// @Juan is this the same parameter used in the NAF representation?
// t = 6x^2. Where x = 4965661367192848881

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

// We have to find the SUBGROUP_ORDER and see where it's used.
// In the implementation of zksync we have:
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
// 30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001 (fist I coverted it into hex)
/* pub const SUBGROUP_ORDER: U256 =
U256::from_hex_unchecked("TODO"); */

type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp6E = FieldElement<Degree6ExtensionField>;
type Fp12E = FieldElement<Degree12ExtensionField>;

// Need implementation of NAF representation

/// Millers loop uses to iterate the NAF representation of the MILLER_LOOP_CONSTANT.
/// A NAF representation uses values: -1, 0 and 1. https://en.wikipedia.org/wiki/Non-adjacent_form.
pub const MILLER_CONSTANT_NAF: [i32; 115] = [
    1, 0, -1, 0, 0, -1, 0, -1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0,
    -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, -1, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0, -1,
    0, -1, 0, 0, -1, 0, 1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0,
];

pub struct BN254AtePairing;
/*impl IsPairing for BN254AtePairing {
    type G1Point = ShortWeierstrassProjectivePoint<BN254Curve>;
    type G2Point = ShortWeierstrassProjectivePoint<BN254TwistCurve>;
    type OutputField = Degree12ExtensionField;


    // Computes the product of the ate pairings for a list of point pairs.
    // To optimize the pairing computation, we compute first all the miller
    // loops and multiply each other (so that we can then do the final exponentiation).
    fn compute_batch(
        pairs: &[(&Self::G1Point, &Self::G2Point)],
    ) -> Result<FieldElement<Self::OutputField>, PairingError> {
        let mut result = FieldElement::one();
        for (p, q) in pairs {
            // We think we can remove the condition !p.is_in_subgroup() because
            // the subgroup oF G1 is G1 (see BN254 for the rest of us).
            if !p.is_in_subgroup() || !q.is_in_subgroup() {
                return Err(PairingError::PointNotInSubgroup);
            }
            if !p.is_neutral_element() && !q.is_neutral_element() {
                let p = p.to_affine();
                let q = q.to_affine();
                result *= miller_naive(p, q);
            }
        }
        Ok(final_exponentiation(&result))
    }

}
*/

// Computes: accumulator <- (accumulator)^2 * l(p); t <- 2t,
// where l is the tangent line of t.
// See https://eprint.iacr.org/2010/354.pdf, page 28.
// to test something:
fn double_accumulate_line(
    t: &mut ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
    let [x_q, y_q, z_q] = t.coordinates();
    let [x_p, y_p, _] = p.coordinates();

    let mut tmp0 = x_q.square(); // 1. X_Q^2
    let tmp1 = y_q.square(); // 2. Y_Q^2
    let tmp2 = tmp1.square(); // 3. tmp1^2
    let mut tmp3 = &tmp1 * z_q.square(); // 4. tmp1 * Z_Q^2
    let tmp4 = tmp0.clone().double() + &tmp0; // 5. 3 * tmp0
    let tmp5 = tmp4.square(); // 6. tmp4^2
    let x_t = (tmp1.clone() + x_q).square() - tmp0.clone() - tmp2.clone(); // 7. (tmp1 + X_Q)^2 - tmp0 - tmp2
    let z_t = (y_q + z_q).square() - tmp1.clone() - z_q.square(); // 8. (Y_Q + Z_Q)^2 - tmp1 - Z_Q^2
    let y_t = (tmp3 - x_t.clone()) * tmp4.clone() - tmp2.clone().double().double().double(); // 9. (tmp3 - X_T) * tmp4 - 8 * tmp2

    tmp3 = tmp1 * z_q.double(); // 10. 2 * (tmp1 * Z_Q)
    tmp0 = y_p * tmp0; // 11. tmp0 * y_P

    // Crear los elementos para el cálculo de la línea
    let a_0 = Fp6E::new([Fp2E::zero(), Fp2E::zero(), tmp0]); // 12. l0 = tmp0 + 0v + 0v^2
    let a_1 = Fp6E::new([Fp2E::zero(), tmp3.clone(), tmp3 + (x_p * z_q) * tmp4]); // 13. l1 = tmp3 + tmp4 * X_P * Z_Q * v + 0v^2

    t.0.value = [x_t, y_t, z_t];
    let l = Fp12E::new([a_0, a_1]); // l = a_0 + a_1 * w

    *accumulator = accumulator.square() * l;
}

/*
fn double_accumulate_line(
    t: &mut ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
    let [x_q, y_q, z_q] = t.coordinates();
    // @nicole: Is it ok p as a ProjectivPoint or do we need to have it in two coordinates?
    // let [x_p, y_p] = p.to_affine().coordintes();
    let [x_p, y_p, _] = p.coordinates();

    let mut a = x_q.square(); // 1
    let b = y_q.square(); // 2
    let c = &b.square(); // 3
    let mut d = (&b + x_q).square() - a.clone() - c; // 4
    d = d.clone().double(); // 5) 2 * d
    let e = a.clone().double() + &a; // 6) 3 * a
    let mut f = x_q + &e; // 7
    let g = e.square(); // 8
    let x_t = &g - &d - &d; // 9
    let z_t = (y_q + z_q).square() - &b - z_q.square(); // 10
    let y_t = (d - &x_t) * &e - &c.double().double().double(); //11
    d = -((&e * z_q.square()).double()); //12
    d = x_p * d; //13 . Watchout with the order of the operation  x_p*d .d * x_p is not defined
    f = f.square() - a - g - b.double().double(); //14
    a = (&z_t * z_q.square()).double(); //15
    a = y_p * a; // 16
    let a_0 = Fp6E::new([Fp2E::zero(), Fp2E::zero(), a]); //18 TODO: check the coefficient's order
    let a_1 = Fp6E::new([Fp2E::zero(), e, d]); //19 //a_1 = 0*v^2 + e*v + d

    t.0.value = [x_t, y_t, z_t];
    let l = Fp12E::new([a_1, a_0]); //l = a_0 + a_1*w
    *accumulator = accumulator.square() * l;
}
*/

// A naive way to double a point without using operate_with_self().
fn double_naive(
    t: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
) -> ShortWeierstrassProjectivePoint<BN254TwistCurve> {
    // We convert the projective coordinates into jacobian coordinates.
    // projective (a, b, c) -> jacobian (a * c, b * c^2, c).
    let z_t = t.z();
    let x_t = t.x() * z_t;
    let y_t = t.y() * z_t.square();

    let mut x_r = x_t.square().square().double().double().double() + x_t.square().square()
        - (&x_t * y_t.square()).double().double().double();
    let mut y_r = (x_t.square().double() + x_t.square())
        * ((x_t * y_t.square()).double().double() - &x_r)
        - y_t.square().square().double().double().double();
    let z_r = (y_t * z_t).double();

    // convert from jacobian to projective coordinates
    // jacob (a,b,c) -> projec (a/c, b/c^2, c)
    x_r = x_r * z_r.inv().unwrap();
    y_r = y_r * z_r.square().inv().unwrap();

    ShortWeierstrassProjectivePoint::new([x_r, y_r, z_r])
}

// A naive way to add two points without using operate_with().
fn add_naive(
    t: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
) -> ShortWeierstrassProjectivePoint<BN254TwistCurve> {
    // We convert the projective coordinates into jacobian coordinates.
    // projective (a, b, c) -> jacobian (a * c, b * c^2, c).
    let z_t = t.z();
    let x_t = t.x() * z_t;
    let y_t = t.y() * z_t.square();
    println!("z_t in add_naive is {:?}", z_t);

    let z_q = q.z();
    let x_q = q.x() * z_q;
    let y_q = q.y() * z_q.square();

    let mut x_r = ((&y_q * z_t.square() * z_t).double() - y_t.double()).square()
        - ((&x_q * z_t.square() - &x_t).square() * (&x_q * z_t.square() - &x_t))
            .double()
            .double()
        - (&x_q * z_t.square() - &x_t)
            .square()
            .double()
            .double()
            .double()
            * &x_t;

    let mut y_r = ((y_q * z_t.square() * z_t).double() - y_t.double())
        * ((&x_q * z_t.square() - &x_t).square().double().double() * &x_t - &x_r)
        - y_t.double().double().double()
            * (&x_q * z_t.square() - &x_t).square()
            * (&x_q * z_t.square() - &x_t);

    let z_r = z_t.double() * (&x_q * z_t.square() - &x_t);
    println!("the factor zero: {:?} ", &x_q * z_t.square() == x_t);
    println!("z_r factor in add_naive: {:?}", x_q * z_t.square() - x_t);
    println!("z_r in add_naive is: {:?}", z_r);

    x_r = x_r * z_r.inv().unwrap();
    y_r = y_r * z_r.square().inv().unwrap();

    ShortWeierstrassProjectivePoint::new([x_r, y_r, z_r])
}

/*
// Computes: accumulator <- accumulator * l(P); t <- t + q,
// where l is the line between t and q.
fn add_accumulate_line(
    t: &mut ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
    let [x_q, y_q, z_q] = q.coordinates();
    let mut x_t = t.x();
    let mut y_t = t.y();
    let mut z_t = t.z();
    let [x_p, y_p, _] = p.coordinates();

    let t0 = x_q * z_t.square(); //1
    let mut t1 = (y_q + z_t).square() - y_q.square() - z_t.square(); //2
    let t1 = t1 * z_t.square(); // 3
    let t2 = t0 - x_t; // 4
    let t3 = t2.square(); //5
    let t4 = t3.double().double(); // 6
    let t5 = t4* t2; // 7
    let mut t6 = t1 - y_t.double(); // 8
    let mut t9 = t6 * x_q;//9
    let t7 = x_t * t4; //10
    x_t = &(t6.square() -&t5 - t7.double()); //11
    z_t = &((z_t + t2).square() - z_t.square()-t3); // 12
    let mut t10 = y_q + z_t; //13
    let t8 = (t1 - x_t) * t6; //14
    let t0 = (y_t * t5).double(); // 15
    let  y_t = t8 - t0; //16
    t10 = t10.square() - y_q.square() - z_t.square(); //17
    t9 = t9.double()-t10;  //18
    t10 = (y_p * z_t).double();//19
    t6 = -t6; //20
    t1 = (x_p * t6).double(); //21
    let l_0 = Fp6E::new([Fp2E::zero(), Fp2E::zero(), t10]); //23
    let l_1 = Fp6E::new([Fp2E::zero(), t9, t1]); //24
    t.0.value = [x_t.clone(), y_t, z_t.clone()];
    let l = Fp12E::new([l_1, l_0]); //26
    accumulator = accumulator * &l;
}
*/

// Function used in miller_naive().
// It computes de addition and line between two points.
fn add_and_line(
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    t: ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    f: FieldElement<Degree12ExtensionField>,
) -> (ShortWeierstrassProjectivePoint<BN254TwistCurve>, Fp12E) {
    // We convert projective coordinates of p into affine coordinates.
    // Projective (a, b, c) -> Affine (a/c, b/b).
    let x_p = p.x() * p.z().inv().unwrap();
    let y_p = p.y() * p.z().inv().unwrap();

    // We convert the projective coordinates of q into jacobian coordinates.
    // projective (a, b, c) -> jacobian (a * c, b * c^2, c).
    let z_q = q.z().clone();
    let x_q = q.x() * z_q.clone();
    let y_q = q.y() * z_q.square();

    // We convert the projective coordinates of the new t into jacobian coordinates.
    let z_t = t.z().clone();
    let x_t = t.x() * z_t.clone();
    let y_t = t.y() * z_t.square();

    let t_result = t.operate_with(&q);

    let l0 = Fp6E::new([
        &y_p * t_result.z().double(), // 2 * z_r * y_p
        &x_q.double().double() * (&y_q * z_t.square() * &z_t * &x_q - &y_t)
            - y_q.double() * t_result.z(), // 4 * x_q * (y_q * (z_t)^3 * x_q - y_t) - 2 * y_q * z_r
        Fp2E::zero(),
    ]);
    let l1 = -Fp6E::new([
        x_p.double().double() * (&y_q * z_t.square() * z_t + y_t), // - 4 * x_p * (y_q * (z_t)^3 + y_t)
        Fp2E::zero(),
        Fp2E::zero(),
    ]);
    let l = Fp12E::new([l0, l1]);
    let f_result = f * l;
    (t_result, f_result)
}

// Computes Miller loop using oprate_with() and operate_with_self instead of the previous algorithms.
fn miller_naive(
    p: ShortWeierstrassProjectivePoint<BN254Curve>,
    q: ShortWeierstrassProjectivePoint<BN254TwistCurve>,
) -> (ShortWeierstrassProjectivePoint<BN254TwistCurve>, Fp12E) {
    let mut t = q.clone();
    let mut f = Fp12E::from(1);
    let miller_lenght = MILLER_CONSTANT_NAF.len();

    // We convert projective coordinates of p into affine coordinates.
    // Projective (a, b, c) -> Affine (a/c, b/b).
    let x_p = p.x() * p.z().inv().unwrap();
    let y_p = p.y() * p.z().inv().unwrap();

    // We convert the projective coordinates of q into jacobian coordinates.
    // projective (a, b, c) -> jacobian (a * c, b * c^2, c).
    let z_q = q.z().clone();
    let x_q = q.x() * z_q.clone();
    let y_q = q.y() * z_q.square();

    for i in (0..miller_lenght - 1).rev() {
        // We convert the projective coordinates of t into jacobian coordinates.
        let mut z_t = t.z().clone();
        let mut x_t = t.x() * z_t.clone();
        let mut y_t = t.y() * z_t.square();

        t = t.operate_with_self(2usize);

        let l0 = Fp6E::new([
            &y_p * (t.z().double() * &z_t.square()), // 2 * z_r * (z_t)^2 * y_p
            (x_t.square() * &x_t).double().double() + (x_t.square() * &x_t).double()
                - y_t.square().double().double(), // 6 * (x_t)^3 - 4 * (y_t)^2
            Fp2E::zero(),
        ]);
        let l1 = -Fp6E::new([
            &x_p * ((&x_t.square().double().double() + x_t.square().double()) * z_t.square()), // 6 * (x_t)^2 * (z_t)^2 * x_p
            Fp2E::zero(),
            Fp2E::zero(),
        ]);
        let l = Fp12E::new([l0, l1]);
        f = f.square() * l;

        if MILLER_CONSTANT_NAF[i] == -1 {
            (t, f) = add_and_line(&p, &q.neg(), t, f);
        }
        if MILLER_CONSTANT_NAF[i] == 1 {
            (t, f) = add_and_line(&p, &q, t, f);
        }
    }
    (t, f)
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve,
        unsigned_integer::element::U384,
    };

    use super::*;

    #[test]
    fn test_double_accumulate_line_doubles_point_correctly() {
        let g1 = BN254Curve::generator();
        let g2 = BN254TwistCurve::generator();
        let mut r = g2.clone();
        let mut f = FieldElement::one();
        double_accumulate_line(&mut r, &g1, &mut f);
        assert_eq!(r, g2.operate_with(&g2));
    }

    #[test]
    fn test_double_naive() {
        let g2 = BN254TwistCurve::generator();
        let r = double_naive(&g2);
        println!("r in double is: {:?}", r);
        println!("g2 + g2 in double is: {:?}", g2.operate_with_self(2usize));
        assert_eq!(r, g2.operate_with_self(2usize))
    }

    // TODO: Write which are the x, y, real and complex parts.
    // Write where we found these points.
    #[test]
    fn test_double_naive_point_different_from_generator() {
        let q = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x1a031c43dfaa2dd04a2c5b2dd257b449ce088dfd6d8ca041f19365b94ae7ae0",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x19f3b18b9baad6dadea895c76728c461e7f188f1a3da94a697d90428f554f039",
                )),
            ]),
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x1467f6d823536b43c1d13f7ce580cc56ba88ad999b12e27e355c114d819bee81",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x2f992ff71d0d08f6271f2d40039924e831d53a43a4772e4322710ee41daed756",
                )),
            ]),
            FieldElement::one(),
        ]); // A twist curve point in projective coordinates
        let r = double_naive(&q);
        println!("r in double is: {:?}", r);
        println!("g2 + g2 in double is: {:?}", q.operate_with_self(2usize));
        assert_eq!(r, q.operate_with_self(2usize))
    }

    #[test]
    fn test_add_naive() {
        let q = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x1a031c43dfaa2dd04a2c5b2dd257b449ce088dfd6d8ca041f19365b94ae7ae0",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x19f3b18b9baad6dadea895c76728c461e7f188f1a3da94a697d90428f554f039",
                )),
            ]),
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x1467f6d823536b43c1d13f7ce580cc56ba88ad999b12e27e355c114d819bee81",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x2f992ff71d0d08f6271f2d40039924e831d53a43a4772e4322710ee41daed756",
                )),
            ]),
            FieldElement::one(),
        ]); // A twist curve point in projective coordinates.
        let t = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x1c0cdfa0b0eb6dcd93968a84ff1f5dfec3284b588e4bf72e6ed01052c23e1058",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x57182543d53551f8dc42e18396f78bec25a36dd07de7006308f90af112fa5d0",
                )),
            ]),
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x279b356bf1390a747e0d361cd896e31898f9d9705942c176c79aa904b5c89243",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x1f537cf9f716088e6a4bf2779cd7e66f83a29a498d2476a9e5a8d7eb9c21967d",
                )),
            ]),
            FieldElement::one(),
        ]);
        // A twist curve point different from q in projective coordinates.
        let r = add_naive(&t, &q);
        assert_eq!(r, q.operate_with(&t))
    }

    #[test]
    fn test_add_naive_values_zksync() {
        let q = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x3010c68cb50161b7d1d96bb71edfec9880171954e56871abf3d93cc94d745fa1",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x0476be093a6d2b4bbf907172049874af11e1b6267606e00804d3ff0037ec57fd",
                )),
            ]),
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x01b33461f39d9e887dbb100f170a2345dde3c07e256d1dfa2b657ba5cd030427",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x14c059d74e5b6c4ec14ae5864ebe23a71781d86c29fb8fb6cce94f70d3de7a21",
                )),
            ]),
            FieldElement::one(),
        ]); // A twist curve point in projective coordinates.
        let t = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x290158a80cd3d66530f74dc94c94adb88f5cdb481acca997b6e60071f08a115f",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x1a2c3013d2ea92e13c800cde68ef56a294b883f6ac35d25f587c09b1b3c635f7",
                )),
            ]),
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x29d1691530ca701b4a106054688728c9972c8512e9789e9567aae23e302ccd75",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x2f997f3dbd66a7afe07fe7862ce239edba9e05c5afff7f8a1259c9733b2dfbb9",
                )),
            ]),
            FieldElement::one(),
        ]); // A twist curve point different from q in projective coordinates.
        let r = add_naive(&t, &q);
        assert_eq!(r, q.operate_with(&t))
    }

    /* Q = (21740656624264531918905957436349160317178065932174634873434489096384118284193,
             2019050928575347605638490762886992026922085924959710776569383806797571971069 i ),
             (768940004759184688611731872359665907813921273999645987556749132562407031847,
              9386111668168143378799867099066976687163019156923561176364278935596535020065 i)
        T =((18547205523279150343907423763561667656905210989238825125041531932617195786591,
            11838207152290886924918158754888128755040578374158302623586388982144778515959i),
            (18914823083116165396144529068959135293951172901190760512481127643602037296501,
               21529909670613655107982420544538753603501207742700745772316790404883810220985 i))
        Po




        Points generated with the sage script:
        Q1 ∈ G2
       'Q1': {'x': {'c0': '0x1a031c43dfaa2dd04a2c5b2dd257b449ce088dfd6d8ca041f19365b94ae7ae0',
         'c1': '0x19f3b18b9baad6dadea895c76728c461e7f188f1a3da94a697d90428f554f039'},
        'y': {'c0': '0x1467f6d823536b43c1d13f7ce580cc56ba88ad999b12e27e355c114d819bee81',
         'c1': '0x2f992ff71d0d08f6271f2d40039924e831d53a43a4772e4322710ee41daed756'}}}

        Q2 ∈ G2
        'Q2': {'x': {'c0': '0x1c0cdfa0b0eb6dcd93968a84ff1f5dfec3284b588e4bf72e6ed01052c23e1058',
         'c1': '0x57182543d53551f8dc42e18396f78bec25a36dd07de7006308f90af112fa5d0'},
        'y': {'c0': '0x279b356bf1390a747e0d361cd896e31898f9d9705942c176c79aa904b5c89243',
         'c1': '0x1f537cf9f716088e6a4bf2779cd7e66f83a29a498d2476a9e5a8d7eb9c21967d'}}}]}


               FieldElement::new([FieldElement::new(U256::from(0x1a031c43dfaa2dd04a2c5b2dd257b449ce088dfd6d8ca041f19365b94ae7ae0)), FieldElement::new(U256::from(0x19f3b18b9baad6dadea895c76728c461e7f188f1a3da94a697d90428f554f039))]);

    let q1 = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
        FieldElement::<Degree2ExtensionField>::new([
            FieldElement::new(U256::from_hex_unchecked("0x1a031c43dfaa2dd04a2c5b2dd257b449ce088dfd6d8ca041f19365b94ae7ae0")),
            FieldElement::new(U256::from_hex_unchecked("0x19f3b18b9baad6dadea895c76728c461e7f188f1a3da94a697d90428f554f039"))
        ]),
        FieldElement::<Degree2ExtensionField>::new([
            FieldElement::new(U256::from_hex_unchecked("0x1467f6d823536b43c1d13f7ce580cc56ba88ad999b12e27e355c114d819bee81")),
            FieldElement::new(U256::from_hex_unchecked("0x2f992ff71d0d08f6271f2d40039924e831d53a43a4772e4322710ee41daed756"))
        ]),
        FieldElement::one()
    ]);

    */
}
