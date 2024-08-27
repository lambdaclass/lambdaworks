use super::{
    curve::BN254Curve,
    field_extension::{
        BN254PrimeField, Degree12ExtensionField, Degree2ExtensionField, Degree4ExtensionField, LevelTwoResidue, mul_fp2_by_nonresidue
    },
    twist::BN254TwistCurve,
};
use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{short_weierstrass::traits::IsShortWeierstrass, traits::IsPairing},
    errors::PairingError,
    field::extensions::cubic::HasCubicNonResidue,
};
use crate::{
    elliptic_curve::short_weierstrass::{
        curves::bn_254::field_extension::Degree6ExtensionField,
        point::ShortWeierstrassProjectivePoint,
    },
    field::element::FieldElement,
};
use std::time::{Duration, Instant};

use rayon::prelude::*;

type FpE = FieldElement<BN254PrimeField>;
type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp4E = FieldElement<Degree4ExtensionField>;
type Fp6E = FieldElement<Degree6ExtensionField>;
type Fp12E = FieldElement<Degree12ExtensionField>;
type G1Point = ShortWeierstrassProjectivePoint<BN254Curve>;
type G2Point = ShortWeierstrassProjectivePoint<BN254TwistCurve>;

////////////////// CONSTANTS //////////////////

/// x = 4965661367192848881.
/// A constant of the curve.
/// See https://hackmd.io/@jpw/bn254#Barreto-Naehrig-curves
pub const X: u64 = 0x44e992b44a6909f1;

// 100010011101001100100101011010001001010011010010000100111110001
pub const X_BINARY:[u32;63] = [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 
0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 
1, 1, 1, 1, 1, 0, 0, 0, 1];

/// Constant used in the Miller Loop.
/// MILLER_CONSTANT = 6x + 2 = 29793968203157093288.
/// Note that this is a representation using {1, -1, 0}, but it isn't a NAF representation
/// because it has non-zero values adjacent.
/// See the post https://hackmd.io/@Wimet/ry7z1Xj-2#The-Pairing.
/// See arkworks library https://github.com/arkworks-rs/algebra/blob/master/curves/bn254/src/curves/mod.rs#L21 (constant called ATE_LOOP_COUNT).
pub const MILLER_CONSTANT: [i32; 65] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, 1, -1, 0, 0, 1, 0, 0, 1, 1, 0, -1, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    1, 1, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 1, 1, 0, -1, 0,
    0, 1, 0, 1, 1,
];

/// GAMMA constants used to compute the Frobenius morphisms and G2 subgroup check.
/// We took these constants from https://github.com/hecmas/zkNotebook/blob/main/src/BN254/constants.ts#L48
/// GAMMA_1i = (9 + u)^{i(p-1) / 6} for all i = 1..5
pub const GAMMA_11: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("1284B71C2865A7DFE8B99FDD76E68B605C521E08292F2176D60B35DADCC9E470"),
    FpE::from_hex_unchecked("246996F3B4FAE7E6A6327CFE12150B8E747992778EEEC7E5CA5CF05F80F362AC"),
]);

pub const GAMMA_12: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("2FB347984F7911F74C0BEC3CF559B143B78CC310C2C3330C99E39557176F553D"),
    FpE::from_hex_unchecked("16C9E55061EBAE204BA4CC8BD75A079432AE2A1D0B7C9DCE1665D51C640FCBA2"),
]);

pub const GAMMA_13: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("63CF305489AF5DCDC5EC698B6E2F9B9DBAAE0EDA9C95998DC54014671A0135A"),
    FpE::from_hex_unchecked("7C03CBCAC41049A0704B5A7EC796F2B21807DC98FA25BD282D37F632623B0E3"),
]);

pub const GAMMA_14: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("5B54F5E64EEA80180F3C0B75A181E84D33365F7BE94EC72848A1F55921EA762"),
    FpE::from_hex_unchecked("2C145EDBE7FD8AEE9F3A80B03B0B1C923685D2EA1BDEC763C13B4711CD2B8126"),
]);

pub const GAMMA_15: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("183C1E74F798649E93A3661A4353FF4425C459B55AA1BD32EA2C810EAB7692F"),
    FpE::from_hex_unchecked("12ACF2CA76FD0675A27FB246C7729F7DB080CB99678E2AC024C6B8EE6E0C2C4B"),
]);

/// GAMMA_2i = GAMMA_1i * GAMMA_1i.conjugate()
pub const GAMMA_21: FpE =
    FpE::from_hex_unchecked("30644E72E131A0295E6DD9E7E0ACCCB0C28F069FBB966E3DE4BD44E5607CFD49");

pub const GAMMA_22: FpE =
    FpE::from_hex_unchecked("30644E72E131A0295E6DD9E7E0ACCCB0C28F069FBB966E3DE4BD44E5607CFD48");

pub const GAMMA_23: FpE =
    FpE::from_hex_unchecked("30644E72E131A029B85045B68181585D97816A916871CA8D3C208C16D87CFD46");

pub const GAMMA_24: FpE =
    FpE::from_hex_unchecked("59E26BCEA0D48BACD4F263F1ACDB5C4F5763473177FFFFFE");

pub const GAMMA_25: FpE =
    FpE::from_hex_unchecked("59E26BCEA0D48BACD4F263F1ACDB5C4F5763473177FFFFFF");

/// GAMMA_3i = GAMMA_1i * GAMMA_2i
pub const GAMMA_31: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("19DC81CFCC82E4BBEFE9608CD0ACAA90894CB38DBE55D24AE86F7D391ED4A67F"),
    FpE::from_hex_unchecked("ABF8B60BE77D7306CBEEE33576139D7F03A5E397D439EC7694AA2BF4C0C101"),
]);

pub const GAMMA_32: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("856E078B755EF0ABAFF1C77959F25AC805FFD3D5D6942D37B746EE87BDCFB6D"),
    FpE::from_hex_unchecked("4F1DE41B3D1766FA9F30E6DEC26094F0FDF31BF98FF2631380CAB2BAAA586DE"),
]);

pub const GAMMA_33: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("2A275B6D9896AA4CDBF17F1DCA9E5EA3BBD689A3BEA870F45FCC8AD066DCE9ED"),
    FpE::from_hex_unchecked("28A411B634F09B8FB14B900E9507E9327600ECC7D8CF6EBAB94D0CB3B2594C64"),
]);

pub const GAMMA_34: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("BC58C6611C08DAB19BEE0F7B5B2444EE633094575B06BCB0E1A92BC3CCBF066"),
    FpE::from_hex_unchecked("23D5E999E1910A12FEB0F6EF0CD21D04A44A9E08737F96E55FE3ED9D730C239F"),
]);

pub const GAMMA_35: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("13C49044952C0905711699FA3B4D3F692ED68098967C84A5EBDE847076261B43"),
    FpE::from_hex_unchecked("16DB366A59B1DD0B9FB1B2282A48633D3E2DDAEA200280211F25041384282499"),
]);

pub const TWO_INV: FpE =
    FpE::from_hex_unchecked("183227397098D014DC2822DB40C0AC2ECBC0B548B438E5469E10460B6C3E7EA4");

////////////////// PAIRING //////////////////

pub struct BN254AtePairing;
impl IsPairing for BN254AtePairing {
    type G1Point = ShortWeierstrassProjectivePoint<BN254Curve>;
    type G2Point = ShortWeierstrassProjectivePoint<BN254TwistCurve>;
    type OutputField = Degree12ExtensionField;

    /// Computes the product of the ate pairing for a list of point pairs.
    /// To optimize the pairing computation, we compute first all the miller loops
    /// and multiply each other (so that we can then do the final exponentiation).
    fn compute_batch(
        pairs: &[(&Self::G1Point, &Self::G2Point)],
    ) -> Result<FieldElement<Self::OutputField>, PairingError> {
        let mut result = Fp12E::one();
        for (p, q) in pairs {
            // We don't need to check if p is in the subgroup because the subgroup oF G1 is G1.
            // See https://hackmd.io/@jpw/bn254#Subgroup-checks.
            if !q.is_in_subgroup() {
                return Err(PairingError::PointNotInSubgroup);
            }
            if !p.is_neutral_element() && !q.is_neutral_element() {
                let p = p.to_affine();
                let q = q.to_affine();
                result *= miller_2(&p, &q);
            }
        }
        Ok(final_exponentiation_3(&result))
    }
}

/// Computes Miller loop using oprate_with(), operate_with_self() and line().
/// See https://eprint.iacr.org/2010/354.pdf (Page 4, Algorithm 1).
pub fn miller(p: &G1Point, q: &G2Point) -> Fp12E {
    let mut t = q.clone();
    let mut f = Fp12E::one();
    let q_neg = &q.neg();
    MILLER_CONSTANT.iter().rev().skip(1).for_each(|m| {
        f = f.square() * line(p, &t, &t);
        t = t.double();

        if *m == -1 {
            f *= line(p, &t, q_neg);
            t = t.operate_with_affine(q_neg);
        } else if *m == 1 {
            f *= line(p, &t, q);
            t = t.operate_with_affine(q);
        }
    });

    // q1 = ((x_q)^p, (y_q)^p, (z_q)^p)
    // See  https://hackmd.io/@Wimet/ry7z1Xj-2#The-Last-two-Lines
    let q1 = q.phi();
    f *= line(p, &t, &q1);
    t = t.operate_with(&q1);

    // q2 = ((x_q1)^p, (y_q1)^p, (z_q1)^p)
    let q2 = q1.phi();
    f *= line(p, &t, &q2.neg());

    f
}

pub fn miller_2(p: &G1Point, q: &G2Point) -> Fp12E {
    let mut t = q.clone();
    let mut f = Fp12E::one();
    let q_neg = &q.neg();
    MILLER_CONSTANT.iter().rev().skip(1).for_each(|m| {
        let (r, l) = line_2(p, &t, &t);
        f = f.square() * l;
        t = r;

        if *m == -1 {
            let (r, l) = line_2(p, &t, q_neg);
            f *= l;
            t = r;
        } else if *m == 1 {
            let (r, l) = line_2(p, &t, q);
            f *= l;
            t = r;
        }
    });

    // q1 = ((x_q)^p, (y_q)^p, (z_q)^p)
    // See  https://hackmd.io/@Wimet/ry7z1Xj-2#The-Last-two-Lines
    let q1 = q.phi();
    let (r, l) = line_2(p, &t, &q1);
    f *= l;
    t = r;

    // q2 = ((x_q1)^p, (y_q1)^p, (z_q1)^p)
    let q2 = q1.phi();
    f *= line_2(p, &t, &q2.neg()).1;

    f
}

fn line_2(p: &G1Point, t: &G2Point, q: &G2Point) -> (G2Point, Fp12E) {
    let [x_p, y_p, _] = p.coordinates();

    if t == q {
        let a = TWO_INV * t.x() * t.y();
        let b = t.y().square();
        let c = t.z().square();
        let e = BN254TwistCurve::b() * (c.double() + &c);
        let f = e.double() + &e;
        let g = TWO_INV * (&b + &f);
        let h = (t.y() + t.z()).square() - (&b + &c);
        let i = &e - &b;
        let j = t.x().square();
        let e_square = e.square();

        let x_r = a * (&b - f);
        let y_r = g.square() - (e_square.double() + e_square);
        let z_r = b * &h;

        let r = G2Point::new([x_r, y_r, z_r]);

        // We are transforming one representation of Fp12 into another:
        // If f in Fp12, then f = g + h * w = g0 + h0 * w + g1 * w^2 + h1 * w^3 + g2 * w^4 + h2 * w^5,
        // where g = g0 + g1 * v + g2 * v^2,
        // and h = h0 + h1 * v + h2 * v^2.
        // See https://hackmd.io/@Wimet/ry7z1Xj-2#Tower-of-Extension-Fields.
        let l = Fp12E::new([
            Fp6E::new([y_p * (-h), Fp2E::zero(), Fp2E::zero()]),
            Fp6E::new([x_p * (j.double() + &j), i, Fp2E::zero()]),
        ]);
        (r, l)
    } else {
        let [x_q, y_q, _] = q.coordinates();
        let [x_t, y_t, z_t] = t.coordinates();

        let a = y_q * z_t;
        let b = x_q * z_t;
        let theta = t.y() - (y_q * t.z());
        let lambda = t.x() - (x_q * t.z());
        let c = theta.square();
        let d = lambda.square();
        let e = &lambda * &d;
        let f = z_t * c;
        let g = x_t * d;
        let h = &e + f - g.double();
        let i = y_t * &e;
        let j = &theta * x_q - (&lambda * y_q);

        let x_r = &lambda * &h;
        let y_r = &theta * (g - h) - i;
        let z_r = z_t * e;

        let r = G2Point::new([x_r, y_r, z_r]);

        let l = Fp12E::new([
            Fp6E::new([y_p * lambda, Fp2E::zero(), Fp2E::zero()]),
            Fp6E::new([x_p * (-theta), j, Fp2E::zero()]),
        ]);
        (r, l)
    }
}

/// Computes the line between q and t and evaluates it in p.
/// Algorithm adapted from Arkowork's double_in_place and add_in_place.
/// See https://github.com/arkworks-rs/algebra/blob/master/ec/src/models/bn/g2.rs#L25.
/// See https://eprint.iacr.org/2013/722.pdf (Page 13, Equations 11 and 13).
fn line(p: &G1Point, t: &G2Point, q: &G2Point) -> Fp12E {
    let [x_p, y_p, _] = p.coordinates();

    if t == q {
        let b = t.y().square();
        let c = t.z().square();
        let e = BN254TwistCurve::b() * (c.double() + &c);
        let h = (t.y() + t.z()).square() - (&b + &c);
        let i = &e - &b;
        let j = t.x().square();

        // We are transforming one representation of Fp12 into another:
        // If f in Fp12, then f = g + h * w = g0 + h0 * w + g1 * w^2 + h1 * w^3 + g2 * w^4 + h2 * w^5,
        // where g = g0 + g1 * v + g2 * v^2,
        // and h = h0 + h1 * v + h2 * v^2.
        // See https://hackmd.io/@Wimet/ry7z1Xj-2#Tower-of-Extension-Fields.
        Fp12E::new([
            Fp6E::new([y_p * (-h), Fp2E::zero(), Fp2E::zero()]),
            Fp6E::new([x_p * (j.double() + &j), i, Fp2E::zero()]),
        ])
    } else {
        let [x_q, y_q, _] = q.coordinates();

        let theta = t.y() - (y_q * t.z());
        let lambda = t.x() - (x_q * t.z());
        let j = &theta * x_q - (&lambda * y_q);

        Fp12E::new([
            Fp6E::new([y_p * lambda, Fp2E::zero(), Fp2E::zero()]),
            Fp6E::new([x_p * (-theta), j, Fp2E::zero()]),
        ])
    }
}

/// Computes f ^ {(p^12 - 1) / r}
/// using that (p^12 - 1)/r = (p^6 - 1) * (p^2 + 1) * (p^4 - p^2 + 1)/r.
/// Algorithm taken from https://hackmd.io/@Wimet/ry7z1Xj-2#Final-Exponentiation.
pub fn final_exponentiation(
    f: &FieldElement<Degree12ExtensionField>,
) -> FieldElement<Degree12ExtensionField> {
    // Easy part:
    // Computes f ^ {(p^6 - 1) * (p^2 + 1)}
    let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1) because f^(p^6) = f.conjugate().
    let mut f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux; // (f^{p^6 - 1})^(p^2) * (f^{p^6 - 1}).

    /*
    // Hard part:
    // Computes f_easy ^ ((p^4 - p^2 + 1) / r)
    // See https://hackmd.io/@Wimet/ry7z1Xj-2#The-Hard-Part, where f_easy is called m.
    // We define different exponentiation of f_easy that we will use later.
    let mx = f_easy.pow(X);
    let mx2 = mx.pow(X);
    let mx3 = mx2.pow(X);
    let mp = frobenius(&f_easy);
    let mp2 = frobenius_square(&f_easy);
    let mp3 = frobenius_cube(&f_easy);
    let mxp = frobenius(&mx); // (m^x)^p
    let mx2p = frobenius(&mx2); // (m^{x^2})^p
    let mx3p = frobenius(&mx3); // (m^{x^3})^p
    let mx2p2 = frobenius_square(&mx2); // (m^{x^2})^p^2

    let y0 = mp * mp2 * mp3;
    let y1 = f_easy.conjugate();
    let y2 = mx2p2;
    let y3 = mxp.conjugate();
    let y4 = (mx * mx2p).conjugate();
    let y5 = mx2.conjugate();
    let y6 = (mx3 * mx3p).conjugate();

    y0 * y1.square()
        * y2.pow(6usize)
        * y3.pow(12usize)
        * y4.pow(18usize)
        * y5.pow(30usize)
        * y6.pow(36usize)
    */

    // Hard part following Arkworks library.
    let mut y0 = f_easy.pow(X);
    y0 = y0.inv().unwrap();
    // For degree 12 extensions, this can be computed faster than normal squaring.
    // See https://github.dev/arkworks-rs/algebra/blob/master/ec/src/models/bn/mod.rs (cyclotimic_square)
    let y1 = &y0.square();
    let y2 = &y1.square();
    let mut y3 = y2 * y1;
    let mut y4 = y3.pow(X);
    y4 = y4.inv().unwrap();
    let y5 = y4.square();
    let mut y6 = y5.pow(X);
    y6 = y6.inv().unwrap();
    // TODO: See if there is a faster way to take inverse.
    y3 = y3.inv().unwrap();
    y6 = y6.inv().unwrap();
    let y7 = y6 * &y4;
    let mut y8 = y7 * &y3;
    let y9 = &y8 * y1;
    let y10 = &y8 * y4;
    let y11 = y10 * &f_easy;
    let mut y12 = y9.clone();
    y12 = frobenius(&y12);
    let y13 = y12 * &y11;
    y8 = frobenius_square(&y8);
    let y14 = y8 * &y13;
    f_easy = f_easy.inv().unwrap();
    let mut y15 = f_easy * y9;
    y15 = frobenius_cube(&y15);
    let y16 = y15 * y14;
    y16

    /*
    // Optimal hard part from the post
    // https://hackmd.io/@Wimet/ry7z1Xj-2#The-Hard-Part
    let mx = cyclotomic_pow_x(f_easy.clone());
    let mx2 = cyclotomic_pow_x(mx.clone());
    let mx3 = cyclotomic_pow_x(mx2.clone());
    let mp = frobenius(&f_easy);
    let mp2 = frobenius_square(&f_easy);
    let mp3 = frobenius_cube(&f_easy);
    let mxp = frobenius(&mx); // (m^x)^p
    let mx2p = frobenius(&mx2); // (m^{x^2})^p
    let mx3p = frobenius(&mx3); // (m^{x^3})^p
    let mx2p2 = frobenius_square(&mx2); // (m^{x^2})^p^2

    let y0 = mp * mp2 * mp3;
    let y1 = f_easy.conjugate();
    let y2 = mx2p2;
    let y3 = mxp.conjugate();
    let y4 = (mx * mx2p).conjugate();
    let y5 = mx2.conjugate();
    let y6 = (mx3 * mx3p).conjugate();

    let t01 = cyclotimic_square(&y6) * y4 * &y5;
    let t11 = &t01 * y3 * y5;
    let t02 = t01 * y2;
    let t12 = cyclotimic_square(&t11) * t02;
    let t13 = cyclotimic_square(&t12);
    let t14 = &t13 * y0;
    let t03 = t13 * y1;
    let t04 = cyclotimic_square(&t03) * t14;

    t04
    */
}

pub fn final_exponentiation_2(
    f: &FieldElement<Degree12ExtensionField>,
) -> FieldElement<Degree12ExtensionField> {
    // Easy part:
    // Computes f ^ {(p^6 - 1) * (p^2 + 1)}
    let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1) because f^(p^6) = f.conjugate().
    let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux; // (f^{p^6 - 1})^(p^2) * (f^{p^6 - 1}).

    /*
    // Hard part:
    // Computes f_easy ^ ((p^4 - p^2 + 1) / r)
    // See https://hackmd.io/@Wimet/ry7z1Xj-2#The-Hard-Part, where f_easy is called m.
    // We define different exponentiation of f_easy that we will use later.
    let mx = f_easy.pow(X);
    let mx2 = mx.pow(X);
    let mx3 = mx2.pow(X);
    let mp = frobenius(&f_easy);
    let mp2 = frobenius_square(&f_easy);
    let mp3 = frobenius_cube(&f_easy);
    let mxp = frobenius(&mx); // (m^x)^p
    let mx2p = frobenius(&mx2); // (m^{x^2})^p
    let mx3p = frobenius(&mx3); // (m^{x^3})^p
    let mx2p2 = frobenius_square(&mx2); // (m^{x^2})^p^2

    let y0 = mp * mp2 * mp3;
    let y1 = f_easy.conjugate();
    let y2 = mx2p2;
    let y3 = mxp.conjugate();
    let y4 = (mx * mx2p).conjugate();
    let y5 = mx2.conjugate();
    let y6 = (mx3 * mx3p).conjugate();

    y0 * y1.square()
        * y2.pow(6usize)
        * y3.pow(12usize)
        * y4.pow(18usize)
        * y5.pow(30usize)
        * y6.pow(36usize)
    */

    /*
    // Hard part following Arkworks library.
    let mut y0 = f_easy.pow(X);
    y0 = y0.inv().unwrap();
    // For degree 12 extensions, this can be computed faster than normal squaring.
    // See https://github.dev/arkworks-rs/algebra/blob/master/ec/src/models/bn/mod.rs (cyclotimic_square)
    let y1 = &y0.square();
    let y2 = &y1.square();
    let mut y3 = y2 * y1;
    let mut y4 = y3.pow(X);
    y4 = y4.inv().unwrap();
    let y5 = y4.square();
    let mut y6 = y5.pow(X);
    y6 = y6.inv().unwrap();
    // TODO: See if there is a faster way to take inverse.
    y3 = y3.inv().unwrap();
    y6 = y6.inv().unwrap();
    let y7 = y6 * &y4;
    let mut y8 = y7 * &y3;
    let y9 = &y8 * y1;
    let y10 = &y8 * y4;
    let y11 = y10 * &f_easy;
    let mut y12 = y9.clone();
    y12 = frobenius(&y12);
    let y13 = y12 * &y11;
    y8 = frobenius_square(&y8);
    let y14 = y8 * &y13;
    f_easy = f_easy.inv().unwrap();
    let mut y15 = f_easy * y9;
    y15 = frobenius_cube(&y15);
    let y16 = y15 * y14;
    y16
    */

    // Optimal hard part from the post
    // https://hackmd.io/@Wimet/ry7z1Xj-2#The-Hard-Part
    let mx = cyclotomic_pow_x(&f_easy);
    let mx2 = cyclotomic_pow_x(&mx);
    let mx3 = cyclotomic_pow_x(&mx2);
    let mp = frobenius(&f_easy);
    let mp2 = frobenius_square(&f_easy);
    let mp3 = frobenius_cube(&f_easy);
    let mxp = frobenius(&mx); // (m^x)^p
    let mx2p = frobenius(&mx2); // (m^{x^2})^p
    let mx3p = frobenius(&mx3); // (m^{x^3})^p
    let mx2p2 = frobenius_square(&mx2); // (m^{x^2})^p^2

    let y0 = &mp * &mp2 * &mp3;
    let y1 = f_easy.conjugate();
    let y2 = mx2p2;
    let y3 = mxp.conjugate();
    let y4 = (mx * mx2p).conjugate();
    let y5 = mx2.conjugate();
    let y6 = (mx3 * mx3p).conjugate();

    let t01 = cyclotomic_square(&y6) * y4 * &y5;
    let t11 = &t01 * y3 * y5;
    let t02 = t01 * y2;
    let t12 = cyclotomic_square(&t11) * t02;
    let t13 = cyclotomic_square(&t12);
    let t14 = &t13 * y0;
    let t03 = t13 * y1;
    let t04 = cyclotomic_square(&t03) * t14;

    t04
}


pub fn final_exponentiation_3(
    f: &FieldElement<Degree12ExtensionField>,
) -> FieldElement<Degree12ExtensionField> {
    // Easy part:
    // Computes f ^ {(p^6 - 1) * (p^2 + 1)}
    let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1) because f^(p^6) = f.conjugate().
    let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux; // (f^{p^6 - 1})^(p^2) * (f^{p^6 - 1}).

    // Optimal Hard Part from the post
    // https://hackmd.io/@Wimet/ry7z1Xj-2#The-Hard-Part
    let mx = cyclotomic_pow_x_2(&f_easy);
    let mx2 = cyclotomic_pow_x_2(&mx);
    let mx3 = cyclotomic_pow_x_2(&mx2);
    let mp = frobenius(&f_easy);
    let mp2 = frobenius_square(&f_easy);
    let mp3 = frobenius_cube(&f_easy);
    let mxp = frobenius(&mx); // (m^x)^p
    let mx2p = frobenius(&mx2); // (m^{x^2})^p
    let mx3p = frobenius(&mx3); // (m^{x^3})^p
    let mx2p2 = frobenius_square(&mx2); // (m^{x^2})^p^2

    let y0 = &mp * &mp2 * &mp3;
    let y1 = f_easy.conjugate();
    let y2 = mx2p2;
    let y3 = mxp.conjugate();
    let y4 = (mx * mx2p).conjugate();
    let y5 = mx2.conjugate();
    let y6 = (mx3 * mx3p).conjugate();

    let t01 = cyclotomic_square_quad_over_cube(&y6) * y4 * &y5;
    let t11 = &t01 * y3 * y5;
    let t02 = t01 * y2;
    let t12 = cyclotomic_square_quad_over_cube(&t11) * t02;
    let t13 = cyclotomic_square_quad_over_cube(&t12);
    let t14 = &t13 * y0;
    let t03 = t13 * y1;
    let t04 = cyclotomic_square_quad_over_cube(&t03) * t14;

    t04
}

/// Computes the Frobenius morphism: f^p.
/// See https://hackmd.io/@Wimet/ry7z1Xj-2#Fp12-Arithmetic (First Frobenius Operator).
pub fn frobenius(f: &Fp12E) -> Fp12E {
    let [a, b] = f.value(); // f = a + bw, where a and b in Fp6.
    let [a0, a1, a2] = a.value(); // a = a0 + a1 * v + a2 * v^2, where a0, a1 and a2 in Fp2.
    let [b0, b1, b2] = b.value(); // b = b0 + b1 * v + b2 * v^2, where b0, b1 and b2 in Fp2.

    // c1 = a0.conjugate() + a1.conjugate() * GAMMA_12 * v + a2.conjugate() * GAMMA_14 * v^2
    let c1 = Fp6E::new([
        a0.conjugate(),
        a1.conjugate() * GAMMA_12,
        a2.conjugate() * GAMMA_14,
    ]);

    let c2 = Fp6E::new([
        b0.conjugate() * GAMMA_11,
        b1.conjugate() * GAMMA_13,
        b2.conjugate() * GAMMA_15,
    ]);

    Fp12E::new([c1, c2]) //c1 + c2 * w
}

/// Computes f^(p^2)
pub fn frobenius_square(
    f: &FieldElement<Degree12ExtensionField>,
) -> FieldElement<Degree12ExtensionField> {
    let [a, b] = f.value();
    let [a0, a1, a2] = a.value();
    let [b0, b1, b2] = b.value();

    let c1 = Fp6E::new([a0.clone(), GAMMA_22 * a1, GAMMA_24 * a2]);
    let c2 = Fp6E::new([GAMMA_21 * b0, GAMMA_23 * b1, GAMMA_25 * b2]);

    Fp12E::new([c1, c2])
}

/// Computes f^(p^3)
pub fn frobenius_cube(
    f: &FieldElement<Degree12ExtensionField>,
) -> FieldElement<Degree12ExtensionField> {
    let [a, b] = f.value();
    let [a0, a1, a2] = a.value();
    let [b0, b1, b2] = b.value();

    let c1 = Fp6E::new([
        a0.conjugate(),
        a1.conjugate() * GAMMA_32,
        a2.conjugate() * GAMMA_34,
    ]);

    let c2 = Fp6E::new([
        b0.conjugate() * GAMMA_31,
        b1.conjugate() * GAMMA_33,
        b2.conjugate() * GAMMA_35,
    ]);

    Fp12E::new([c1, c2])
}

////////////////// CYCLOTOMIC SUBGROUP OPERATIONS //////////////////

/// Computes a compression of an element of Fp12
/// if f = g + h * w where
/// g = g0 + g1 * v + g2 * v^2 and
/// h = h0 + h1 * v + h2 * v^2
/// then, compress(f) = [h0, g2, g1, h2]
/// Algorithm from https://hackmd.io/@Wimet/ry7z1Xj-2#Compression-and-Decompression
/// Recall that f = g0 + h0 * w + g1 * w^2 + h1 * w^3 + g2 * w^4 + h2 * w^5
fn compress(f: &Fp12E) -> Vec<Fp2E> {
    let h0 = f.value()[1].value()[0].clone();
    let g2 = f.value()[0].value()[2].clone();
    let g1 = f.value()[0].value()[1].clone();
    let h2 = f.value()[1].value()[2].clone();

    vec![h0, g2, g1, h2]
}

/// Computes the decompression to an element of Fp12.
/// decompress([h0, g2, g1, h2]) = f where
/// f = g + h * w
/// Algorithm from https://hackmd.io/@Wimet/ry7z1Xj-2#Compression-and-Decompression
/// See https://eprint.iacr.org/2010/542.pdf (Theorem 3.1)
fn decompress(c: &Vec<Fp2E>) -> Fp12E {
    let h0 = c[0].clone();
    let g2 = c[1].clone();
    let g1 = c[2].clone();
    let h2 = c[3].clone();
    let mut g0 = Fp2E::one();
    let mut h1 = Fp2E::one();

    if h0 != Fp2E::zero() {
        h1 = (mul_fp2_by_nonresidue(&h2.square()) + FpE::from(3) * g1.square() - g2.double())
            * h0.double().double().inv().unwrap();
        g0 = mul_fp2_by_nonresidue(&(h1.square().double() + &h0 * &h2 - FpE::from(3) * &g1 * &g2) )
            + Fp2E::one();
    } else {
        h1 = (&g1 * &h2).double() * g2.inv().unwrap();
        g0 = mul_fp2_by_nonresidue(&(h1.square().double() - FpE::from(3) * &g2 * &g1))  + Fp2E::one();
    }

    Fp12E::new([Fp6E::new([g0, g1, g2]), Fp6E::new([h0, h1, h2])])
}

pub fn cyclotomic_square(f: &Fp12E) -> Fp12E {
    if f == &Fp12E::one() {
        Fp12E::one()
    } else {
        decompress(&compressed_square(&compress(f)))
    }
}

fn compressed_square(c: &[Fp2E]) -> Vec<Fp2E> {
    let h0 = &c[0];
    let g2 = &c[1];
    let g1 = &c[2];
    let h2 = &c[3];
    let non_residue = Fp2E::new([FpE::from(9), FpE::one()]);
    let ten_plus_u = Fp2E::new([FpE::from(10), FpE::one()]);

    let h0_square = (h0 + FpE::from(3) * mul_fp2_by_nonresidue(&g1) * h2).double();
    let g2_square = FpE::from(3)
        * ((g1 +h2) * (g1 + mul_fp2_by_nonresidue(&h2)) - &ten_plus_u * g1 * h2)
        - g2.double();
    let g1_square = FpE::from(3)
        * ((h0 + g2) * (h0 + mul_fp2_by_nonresidue(&g2)) - ten_plus_u * h0 * g2)
        - g1.double();
    let h2_square = (h2 + FpE::from(3) * h0 * g2).double();

    let f_square = vec![h0_square, g2_square, g1_square, h2_square];
    f_square
}

/*
/// Computes the square of an element of Fp12E that belongs to the cyclotomic subgroup.
/// Algorithm from https://hackmd.io/@Wimet/ry7z1Xj-2#Compressed-Squaring
fn cyclotimic_square(f: &Fp12E) -> Fp12E {
    // If f = 1, we can't compress and decompress, but its square is one.
    if f == &Fp12E::one() {
        f.clone()
    } else {
        let c = compress(f);
        let h0 = c[0].clone();
        let g2 = c[1].clone();
        let g1 = c[2].clone();
        let h2 = c[3].clone();
        let non_residue = Fp2E::new([FpE::from(9), FpE::one()]);
        let ten_plus_u = Fp2E::new([FpE::from(10), FpE::one()]);

        let h0_square = (&h0 + FpE::from(3) * &non_residue * &g1 * &h2).double();
        let g2_square = FpE::from(3)
            * ((&g1 + &h2) * (&g1 + &non_residue * &h2) - &ten_plus_u * &g1 * &h2)
            - g2.double();
        let g1_square = FpE::from(3)
            * ((&h0 + &g2) * (&h0 + non_residue * &g2) - ten_plus_u * &h0 * &g2)
            - g1.double();
        let h2_square = (h2 + FpE::from(3) * h0 * g2).double();

        let f_square = vec![h0_square, g2_square, g1_square, h2_square];
        decompress(f_square)
    }
}
*/

/*
fn cyclotomic_pow_x(f: Fp12E) -> Fp12E {
    let mut g = f;
    let mut result = Fp12E::one();
    for i in 0..X_BINARY.len() {
        if X_BINARY[i] == 1 {
            result *= &g;
        }
        g = cyclotimic_square(&g);
    }
    result

}
*/

pub fn cyclotomic_pow_x(f: &Fp12E) -> Fp12E {
    if *f == Fp12E::one() {
        Fp12E::one()
    } else {
        let mut c = compress(&f);
        let mut result = Fp12E::one();
        let mut x = X;
        while x > 0 {
            if x & 1 == 1 {
                result *= &decompress(&c);
            }
            c = compressed_square(&c);
            x >>= 1;
        }
        result
    }
}

/*
fn cyclotomic_pow_x(f: Fp12E) -> Fp12E {
    let start = Instant::now();
    if f == Fp12E::one() {
        let duration = start.elapsed(); //
        println!("Duration for one-element case: {:?}", duration);
        Fp12E::one()
    } else {
        let mut c = compress(&f);
        let mut result = Fp12E::one();
        let mut x = X;
        while x > 0 {
            if x & 1 == 1 {
                result *= &decompress(&c);
            }
            c = compressed_square(c);
            x >>= 1;
        }

        let duration = start.elapsed();
        println!("Duration for computation: {:?}", duration);
        result
    }
}
*/

/*
fn cyclotomic_pow_x(f: Fp12E) -> Fp12E {
    let mut g = f;
    let mut result = Fp12E::one();

    let mut x = X;
    while x > 0 {
        if x & 1 == 1 {
            result *= &g;
        }

        g = cyclotimic_square(&g);
        x >>= 1;
    }

    result
}
*/

/*
// Algorithm from zksync to compute de square of an Fp12E in teh cyclotomic subgroup.
fn cyclotomic_square_fp12(a: &Fp12E) -> Fp12E {

    let a00  = a.value()[0].value()[0].square();
    let a01 = a.value()[0].value()[1].square();
    let a02 = a.value()[0].value()[2].square();
    let a10 = a.value()[1].value()[0].square();
    let a11 = a.value()[1].value()[1].square();
    let a12 = a.value()[1].value()[2].square();


    let mut t0 =a11.square();
    let t1 = a00.square();
    let mut t2 = &a11 + &a00;
    t2 = t2.square();
    t2 = t2 - &t0 ;
    t2 = t2 - &t1;

    let mut t3 =a02.square();
    let t4 = a10.square();
    let mut t5 = &a02 + &a10;
    t5 = t5.square();
    t5 = t5 - &t3 ;
    t5 = t5 - &t4;

    let mut t6 =a12.square();
    let t7 = a01.square();
    let mut t8 = &a12 + &a01;
    t8 = t8.square();
    t8 = t8 - &t6 ;
    t8 = t8 - &t7;

    let xi = Fp2E::new([FpE::from(9), FpE::one()]) ;
    t8 *= &xi;
    t0 *=&xi;
    t0 = t0 +t1;
    t3 = &xi*t3;
    t3 = t3 + t4;
    t6 *= xi;
    t6 +=t7;

    let mut c00 = &t0 - a00;
    c00= c00.double();
    c00 += t0;

    let mut c01 = &t3 - a01;
    c01= c01.double();
    c01 += t3;

    let mut c02 = &t6 - a02;
    c02= c02.double();
    c02 += t6;

    let mut c10 = &t8 + a10;
    c10 = c10.double();
    c10 += t8;

    let mut c11 = &t2 + a11;
    c11 = c11.double();
    c11 += t2;

    let mut c12 = &t5 + a12;
    c12 = c12.double();
    c12 += t5;

    Fp12E::new([
        Fp6E::new([c00, c01, c02]),
        Fp6E::new([c10, c11, c12]),
    ])
}
*/


/// Computes the cyclotomic square for Fp12
/// See same function in https://github.com/mratsim/constantine/blob/master/constantine/math/pairings/cyclotomic_subgroups.nim#L354
pub fn cyclotomic_square_quad_over_cube(a: &Fp12E) -> Fp12E {
    // a = g + h * w
    let b0 = &a.value()[0].value()[0]; // b0 = g0
    let b1 = &a.value()[0].value()[1]; // b1 = g1
    let b2 = &a.value()[0].value()[2]; // b2 = g2
    let b3 = &a.value()[1].value()[0]; // b3 = h0
    let b4 = &a.value()[1].value()[1];
    let b5 = &a.value()[1].value()[2];

    let v0 = Fp4E::new([b0.clone(), b4.clone()]).square();
    let v1 = Fp4E::new([b3.clone(), b2.clone()]).square();
    let v2 = Fp4E::new([b1.clone(), b5.clone()]).square();

    // r = r0 + r1 * w
    // r0 = r00 + r01 * v + r02 * v^2
    // r1 = r10 + r11 * v + r12 * v^2

    // r00 = 3v00 - 2b0
    let mut r00 =  &v0.value()[0] - b0;
    r00 = r00.double();
    r00 += v0.value()[0].clone();

    // r01 = 3v10 -2b1
    let mut r01 = &v1.value()[0] - b1;
    r01 = r01.double();
    r01 += v1.value()[0].clone();

    // r11 = 3v01 - 2b4
    let mut r11 = &v0.value()[1] + b4;
    r11 = r11.double();
    r11 += v0.value()[1].clone();

    // r12 = 3v11 - 2b5
    let mut r12 = &v1.value()[1] + b5;
    r12 = r12.double();
    r12 += v1.value()[1].clone(); 

    // 3 * (9 + u) * v21 + 2b3
    let v21 = mul_fp2_by_nonresidue(&v2.value()[1]);
    let mut r10 = &v21 + b3;
    r10 = r10.double(); 
    r10 += v21;

    // 3 * (9 + u) * v20 - 2b3
    let mut r02 = &v2.value()[0] - b2;
    r02 = r02.double();
    r02 += v2.value()[0].clone();

    Fp12E::new([
        Fp6E::new([r00, r01, r02,]),
        Fp6E::new([r10, r11, r12]),
    ])
}

// Algorithm from https://hackmd.io/@Wimet/ry7z1Xj-2#Exponentiation-in-the-Cyclotomic-Subgroup
pub fn cyclotomic_pow_x_2 (f:  &Fp12E) -> Fp12E {
    let mut result = Fp12E::one();
    for i in (0..63) {
        result = cyclotomic_square_quad_over_cube(&result);
        if X_BINARY[i] == 1 {
            result = &result * f;
        }
    }
    result
}


#[cfg(test)]
/// We took the G1 and G2 points from:
/// https://github.com/lambdaclass/zksync_era_precompiles/blob/4bdfebf831e21d58c5ba6945d4524763f1ef64d4/tests/tests/ecpairing_tests.rs
mod tests {
    use crate::elliptic_curve::traits::FromAffine;
    use crate::unsigned_integer::element::U256;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve,
        unsigned_integer::element::U384,
    };

    use super::*;

    #[test]
    // e(ap, bq) = e(abp, q) = e(p, abq) = e(bp, aq) = e(ap, q)^b
    fn batch_ate_pairing_bilinearity() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();

        let a = U384::from_u64(11);
        let b = U384::from_u64(93);

        let result_1 = BN254AtePairing::compute_batch(&[
            (&p.operate_with_self(a), &q.operate_with_self(b)),
            (&p.operate_with_self(a * b), &q.neg()),
        ])
        .unwrap();
        assert_eq!(result_1, Fp12E::one());

        let result_2 = BN254AtePairing::compute_batch(&[
            (&p.operate_with_self(a), &q.operate_with_self(b)),
            (&p.neg(), &q.operate_with_self(a * b)),
        ])
        .unwrap();
        assert_eq!(result_2, Fp12E::one());

        let result_3 = BN254AtePairing::compute_batch(&[
            (&p.operate_with_self(a), &q.operate_with_self(b)),
            (&p.operate_with_self(b), &q.operate_with_self(a).neg()),
        ])
        .unwrap();
        assert_eq!(result_3, Fp12E::one());

        let result_4 = BN254AtePairing::compute_batch(&[
            (&p.operate_with_self(a), &q.operate_with_self(b).neg()),
            (&p.operate_with_self(b), &q),
            (&p.operate_with_self(b), &q),
            (&p.operate_with_self(b), &q),
            (&p.operate_with_self(b), &q),
            (&p.operate_with_self(b), &q),
            (&p.operate_with_self(b), &q),
            (&p.operate_with_self(b), &q),
            (&p.operate_with_self(b), &q),
            (&p.operate_with_self(b), &q),
            (&p.operate_with_self(b), &q),
            (&p.operate_with_self(b), &q),
        ])
        .unwrap();
        assert_eq!(result_4, Fp12E::one());
    }

    #[test]
    fn ate_pairing_returns_one_when_one_element_is_the_neutral_element() {
        let p1 = BN254Curve::generator();
        let q1 = G2Point::neutral_element();
        let result_1 = BN254AtePairing::compute_batch(&[(&p1, &q1)]).unwrap();
        assert_eq!(result_1, Fp12E::one());

        let p2 = G1Point::neutral_element();
        let q2 = BN254TwistCurve::generator();
        let result_2 = BN254AtePairing::compute_batch(&[(&p2, &q2)]).unwrap();
        assert_eq!(result_2, Fp12E::one());

        let result_3 = BN254AtePairing::compute_batch(&[(&p2, &q1)]).unwrap();
        assert_eq!(result_3, Fp12E::one());
    }

    #[test]
    fn ate_pairing_errors_when_g2_element_is_not_in_subgroup() {
        let p = BN254Curve::generator();
        let q = G2Point::new([
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edaddde46bd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920daef312c20b9f1099ecefa8b45575d349b0a6f04c16d0d58af900",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "22376289c558493c1d6cc413a5f07dcb54526a964e4e687b65a881aa9752faa2",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "05a7a5759338c23ca603c1c4adf979e004c2f3e3c5bad6f07693c59a85d600a9",
                )),
            ]),
            Fp2E::one(),
        ]);
        let result = BN254AtePairing::compute_batch(&[(&p, &q)]);
        assert!(result.is_err())
    }

    #[test]
    fn apply_12_times_frobenius_is_identity() {
        let f = Fp12E::from_coefficients(&[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        ]);
        let mut result = frobenius(&f);
        for _ in 1..12 {
            result = frobenius(&result);
        }
        assert_eq!(f, result)
    }

    #[test]
    fn apply_6_times_frobenius_square_is_identity() {
        let f = Fp12E::from_coefficients(&[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        ]);
        let mut result = frobenius_square(&f);
        for _ in 1..6 {
            result = frobenius_square(&result);
        }
        assert_eq!(f, result)
    }

    #[test]
    fn apply_4_times_frobenius_cube_is_identity() {
        let f = Fp12E::from_coefficients(&[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        ]);
        let mut result = frobenius_cube(&f);
        for _ in 1..4 {
            result = frobenius_cube(&result);
        }
        assert_eq!(f, result)
    }

    #[test]
    fn two_pairs_of_points_match_1() {
        let p1 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q1 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
                )),
            ]),
        )
        .unwrap();

        let p2 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd45",
            )),
        )
        .unwrap();

        let q2 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
                )),
            ]),
        )
        .unwrap();

        let result = BN254AtePairing::compute_batch(&[(&p1, &q1), (&p2, &q2)]).unwrap();

        assert_eq!(result, Fp12E::one());
    }

    #[test]
    fn two_pairs_of_points_match_2() {
        let p1 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "105456a333e6d636854f987ea7bb713dfd0ae8371a72aea313ae0c32c0bf1016",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0cf031d41b41557f3e7e3ba0c51bebe5da8e6ecd855ec50fc87efcdeac168bcc",
            )),
        )
        .unwrap();

        let q1 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "3010c68cb50161b7d1d96bb71edfec9880171954e56871abf3d93cc94d745fa1",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "0476be093a6d2b4bbf907172049874af11e1b6267606e00804d3ff0037ec57fd",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "01b33461f39d9e887dbb100f170a2345dde3c07e256d1dfa2b657ba5cd030427",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "14c059d74e5b6c4ec14ae5864ebe23a71781d86c29fb8fb6cce94f70d3de7a21",
                )),
            ]),
        )
        .unwrap();

        let p2 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q2 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "290158a80cd3d66530f74dc94c94adb88f5cdb481acca997b6e60071f08a115f",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "1a2c3013d2ea92e13c800cde68ef56a294b883f6ac35d25f587c09b1b3c635f7",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "29d1691530ca701b4a106054688728c9972c8512e9789e9567aae23e302ccd75",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "2f997f3dbd66a7afe07fe7862ce239edba9e05c5afff7f8a1259c9733b2dfbb9",
                )),
            ]),
        )
        .unwrap();

        let result = BN254AtePairing::compute_batch(&[(&p1, &q1), (&p2, &q2)]).unwrap();
        assert_eq!(result, Fp12E::one());
    }

    #[test]
    fn two_pairs_of_points_fail() {
        let p1 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q1 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
                )),
            ]),
        )
        .unwrap();

        let p2 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q2 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
                )),
            ]),
        )
        .unwrap();

        let result = BN254AtePairing::compute_batch(&[(&p1, &q1), (&p2, &q2)]).unwrap();
        assert!(result != Fp12E::one());
    }

    #[test]
    fn three_pairs_of_points_fail() {
        let p1 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "105456a333e6d636854f987ea7bb713dfd0ae8371a72aea313ae0c32c0bf1016",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0cf031d41b41557f3e7e3ba0c51bebe5da8e6ecd855ec50fc87efcdeac168bcc",
            )),
        )
        .unwrap();

        let q1 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "3010c68cb50161b7d1d96bb71edfec9880171954e56871abf3d93cc94d745fa1",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "0476be093a6d2b4bbf907172049874af11e1b6267606e00804d3ff0037ec57fd",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "01b33461f39d9e887dbb100f170a2345dde3c07e256d1dfa2b657ba5cd030427",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "14c059d74e5b6c4ec14ae5864ebe23a71781d86c29fb8fb6cce94f70d3de7a21",
                )),
            ]),
        )
        .unwrap();

        let p2 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q2 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "290158a80cd3d66530f74dc94c94adb88f5cdb481acca997b6e60071f08a115f",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "1a2c3013d2ea92e13c800cde68ef56a294b883f6ac35d25f587c09b1b3c635f7",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "0692e55db067300e6e3fe56218fa2f940054e57e7ef92bf7d475a9d8a8502fd2",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "00cacf3523caf879d7d05e30549f1e6fdce364cbb8724b0329c6c2a39d4f018e",
                )),
            ]),
        )
        .unwrap();

        let p3 = G1Point::from_affine(
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000001",
            )),
            FpE::new(U256::from_hex_unchecked(
                "0000000000000000000000000000000000000000000000000000000000000002",
            )),
        )
        .unwrap();

        let q3 = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
                )),
            ]),
        )
        .unwrap();

        let result = BN254AtePairing::compute_batch(&[(&p1, &q1), (&p2, &q2), (&p3, &q3)]).unwrap();
        assert!(result != Fp12E::one());
    }

    const R: U256 = U256::from_hex_unchecked(
        "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
    );

    #[test]
    fn pairing_result_pow_r_is_one() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let pairing_result = BN254AtePairing::compute_batch(&[(&p, &q)]).unwrap();
        assert_eq!(pairing_result.pow(R), Fp12E::one());
    }

    #[test]
    fn pairing_is_non_degenerate() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let pairing_result = BN254AtePairing::compute_batch(&[(&p, &q)]).unwrap();
        assert_ne!(pairing_result, Fp12E::one());
    }

    #[test]
    fn compress_and_decompress_is_identity() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let f = miller(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1) because f^(p^6) = f.conjugate().
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux; // (f^{p^6 - 1})^(p^2) * (f^{p^6 - 1}).
        assert_eq!(decompress(&compress(&f_easy)), f_easy);
    }

    #[test]
    fn cyclotomic_square_equals_square() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let f = miller(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1) because f^(p^6) = f.conjugate().
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux; // (f^{p^6 - 1})^(p^2) * (f^{p^6 - 1}).
        assert_eq!(cyclotomic_square(&f_easy), f_easy.square());
    }

    #[test]
    fn cyclotomic_pow_x_equals_pow() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let f = miller(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1) because f^(p^6) = f.conjugate().
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux; // (f^{p^6 - 1})^(p^2) * (f^{p^6 - 1}).
        assert_eq!(cyclotomic_pow_x(&f_easy), f_easy.pow(X));
    }

    #[test]
    fn constant_two_inv_is_iwo_inverse() {
        assert_eq!(TWO_INV, FpE::from(2).inv().unwrap());
        assert_eq!(TWO_INV * FpE::from(2), FpE::one());
    }

    #[test]
    fn cyclotomic_square_quad_over_cube_equals_square() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let f = miller(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1) because f^(p^6) = f.conjugate().
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux; // (f^{p^6 - 1})^(p^2) * (f^{p^6 - 1}).
        assert_eq!(cyclotomic_square_quad_over_cube(&f_easy), f_easy.square());
    }

    #[test]
    fn cyclotomic_pow_x_2_equals_pow() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let f = miller(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1) because f^(p^6) = f.conjugate().
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux; // (f^{p^6 - 1})^(p^2) * (f^{p^6 - 1}).
        assert_eq!(cyclotomic_pow_x_2(&f_easy), f_easy.pow(X));
    }
}