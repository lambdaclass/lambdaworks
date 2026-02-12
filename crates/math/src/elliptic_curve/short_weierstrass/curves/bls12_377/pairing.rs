use super::{
    curve::BLS12377Curve,
    field_extension::{
        mul_fp2_by_nonresidue, sparse_fp12_mul, BLS12377PrimeField, Degree12ExtensionField,
        Degree2ExtensionField, Degree4ExtensionField,
    },
    twist::BLS12377TwistCurve,
};
use crate::{cyclic_group::IsGroup, elliptic_curve::traits::IsPairing, errors::PairingError};

use crate::{
    elliptic_curve::short_weierstrass::{
        curves::bls12_377::field_extension::Degree6ExtensionField,
        point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass,
    },
    field::element::FieldElement,
    unsigned_integer::element::U256,
};

type FpE = FieldElement<BLS12377PrimeField>;
type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp4E = FieldElement<Degree4ExtensionField>;
type Fp6E = FieldElement<Degree6ExtensionField>;
type Fp12E = FieldElement<Degree12ExtensionField>;
type G1Point = ShortWeierstrassProjectivePoint<BLS12377Curve>;
type G2Point = ShortWeierstrassProjectivePoint<BLS12377TwistCurve>;
pub const X: u64 = 0x8508c00000000001;

// X in binary = 1000010100001000110000000000000000000000000000000000000000000001s
pub const X_BINARY: &[bool] = &[
    true, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, true, true, false, false, false, true, false,
    false, false, false, true, false, true, false, false, false, false, true,
];

pub const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001");

// GAMMA constants used to compute the Frobenius morphisms
pub const GAMMA_11: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked(
        "0x9a9975399c019633c1e30682567f915c8a45e0f94ebc8ec681bf34a3aa559db57668e558eb0188e938a9d1104f2031",
    ),
    FpE::from_hex_unchecked("0"),
]);

pub const GAMMA_12: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked(
        "0x9b3af05dd14f6ec619aaf7d34594aabc5ed1347970dec00452217cc900000008508c00000000002",
    ),
    FpE::from_hex_unchecked("0"),
]);

pub const GAMMA_13: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked(
        "0x1680a40796537cac0c534db1a79beb1400398f50ad1dec1bce649cf436b0f6299588459bff27d8e6e76d5ecf1391c63",
    ),
    FpE::from_hex_unchecked("0"),
]);

pub const GAMMA_14: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked(
        "0x9b3af05dd14f6ec619aaf7d34594aabc5ed1347970dec00452217cc900000008508c00000000001",
    ),
    FpE::from_hex_unchecked("0"),
]);

pub const GAMMA_15: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked(
        "0xcd70cb3fc936348d0351d498233f1fe379531411832232f6648a9a9fc0b9c4e3e21b7467077c05853e2c1be0e9fc32",
    ),
    FpE::from_hex_unchecked("0"),
]);

/// GAMMA_2i = GAMMA_1i * GAMMA_1i.conjugate()
pub const GAMMA_21: FpE = FpE::from_hex_unchecked(
    "0x9b3af05dd14f6ec619aaf7d34594aabc5ed1347970dec00452217cc900000008508c00000000002",
);

pub const GAMMA_22: FpE = FpE::from_hex_unchecked(
    "0x9b3af05dd14f6ec619aaf7d34594aabc5ed1347970dec00452217cc900000008508c00000000001",
);

pub const GAMMA_23: FpE =
    FpE::from_hex_unchecked("0x1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000000");

pub const GAMMA_24: FpE =
    FpE::from_hex_unchecked("0x1ae3a4617c510eabc8756ba8f8c524eb8882a75cc9bc8e359064ee822fb5bffd1e945779fffffffffffffffffffffff");

pub const GAMMA_25: FpE =
    FpE::from_hex_unchecked("0x1ae3a4617c510eabc8756ba8f8c524eb8882a75cc9bc8e359064ee822fb5bffd1e94577a00000000000000000000000");

/// The inverse of two in Fp as a constant.
pub const TWO_INV: FpE =
    FpE::from_hex_unchecked("D71D230BE28875631D82E03650A49D8D116CF9807A89C78F79B117DD04A4000B85AEA2180000004284600000000001");

#[derive(Clone)]
pub struct BLS12377AtePairing;

impl IsPairing for BLS12377AtePairing {
    type G1Point = ShortWeierstrassProjectivePoint<BLS12377Curve>;
    type G2Point = ShortWeierstrassProjectivePoint<BLS12377TwistCurve>;
    type OutputField = Degree12ExtensionField;

    /// Compute the product of the ate pairings for a list of point pairs.
    fn compute_batch(
        pairs: &[(&Self::G1Point, &Self::G2Point)],
    ) -> Result<FieldElement<Self::OutputField>, PairingError> {
        let mut result = FieldElement::one();
        for (p, q) in pairs {
            if !p.is_in_subgroup() || !q.is_in_subgroup() {
                return Err(PairingError::PointNotInSubgroup);
            }
            if !p.is_neutral_element() && !q.is_neutral_element() {
                let p = p.to_affine();
                let q = q.to_affine();
                result *= miller(&p, &q);
            }
        }
        final_exponentiation(&result)
    }
}

pub fn miller(p: &G1Point, q: &G2Point) -> Fp12E {
    let mut t = q.clone();
    let mut f = Fp12E::one();

    X_BINARY.iter().rev().skip(1).for_each(|&bit| {
        let (r, l) = line(p, &t, &t);
        f = sparse_fp12_mul(&f.square(), &l);
        t = r;

        if bit {
            let (r, l) = line(p, &t, q);
            f = sparse_fp12_mul(&f, &l);
            t = r;
        }
    });
    f
}
fn line(p: &G1Point, t: &G2Point, q: &G2Point) -> (G2Point, Fp12E) {
    let [x_p, y_p, _] = p.coordinates();

    if core::ptr::eq(t, q) || t == q {
        let a = TWO_INV * t.x() * t.y();
        let b = t.y().square();
        let c = t.z().square();
        let e = BLS12377TwistCurve::b() * (c.double() + &c);
        let f = e.double() + &e;
        let g = TWO_INV * (&b + &f);
        let h = (t.y() + t.z()).square() - (&b + &c);
        let i = &e - &b;
        let j = t.x().square();
        let e_square = e.square();

        let x_r = a * (&b - f);
        let y_r = g.square() - (e_square.double() + e_square);
        let z_r = b * &h;

        debug_assert_eq!(
            BLS12377TwistCurve::defining_equation_projective(&x_r, &y_r, &z_r),
            Fp2E::zero()
        );
        // SAFETY: `unwrap()` is used here because we ensure that `x_r, y_r, z_r`
        // satisfy the curve equation. The previous assertion checks that this is indeed the case.
        let r = G2Point::new([x_r, y_r, z_r]);

        let l = Fp12E::new([
            Fp6E::new([y_p * (-h), Fp2E::zero(), Fp2E::zero()]),
            Fp6E::new([x_p * (j.double() + &j), i, Fp2E::zero()]),
        ]);
        (r.unwrap(), l)
    } else {
        let [x_q, y_q, _] = q.coordinates();
        let [x_t, y_t, z_t] = t.coordinates();

        let a = y_q * z_t;
        let b = x_q * z_t;
        let theta = t.y() - a;
        let lambda = t.x() - b;
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

        debug_assert_eq!(
            BLS12377TwistCurve::defining_equation_projective(&x_r, &y_r, &z_r),
            Fp2E::zero()
        );
        // SAFETY: The values `x_r, y_r, z_r` are computed correctly to be on the curve.
        // The assertion above verifies that the resulting point is valid.
        let r = G2Point::new([x_r, y_r, z_r]);

        let l = Fp12E::new([
            Fp6E::new([y_p * lambda, Fp2E::zero(), Fp2E::zero()]),
            Fp6E::new([x_p * (-theta), j, Fp2E::zero()]),
        ]);
        (r.unwrap(), l)
    }
}

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
fn frobenius_square(
    f: &FieldElement<Degree12ExtensionField>,
) -> FieldElement<Degree12ExtensionField> {
    let [a, b] = f.value();
    let [a0, a1, a2] = a.value();
    let [b0, b1, b2] = b.value();

    let c1 = Fp6E::new([a0.clone(), GAMMA_22 * a1, GAMMA_24 * a2]);
    let c2 = Fp6E::new([GAMMA_21 * b0, GAMMA_23 * b1, GAMMA_25 * b2]);

    Fp12E::new([c1, c2])
}

////////////////// CYCLOTOMIC SUBGROUP OPERATIONS //////////////////
// Since the result of the Easy Part of the Final Exponentiation belongs to the cyclotomic
// subgroup of Fp12, we can optimize the square and pow operations used in the Hard Part.
/// Computes the square of an element of a cyclotomic subgroup of Fp12.
/// Algorithm from Constantine's cyclotomic_square_quad_over_cube.
/// See <https://github.com/mratsim/constantine/blob/master/constantine/math/pairings/cyclotomic_subgroups.nim#L354>.
pub fn cyclotomic_square(a: &Fp12E) -> Fp12E {
    // a = g + h * w
    let [g, h] = a.value();
    let [b0, b1, b2] = g.value();
    let [b3, b4, b5] = h.value();

    let v0 = Fp4E::new([b0.clone(), b4.clone()]).square();
    let v1 = Fp4E::new([b3.clone(), b2.clone()]).square();
    let v2 = Fp4E::new([b1.clone(), b5.clone()]).square();

    // r = r0 + r1 * w
    // r0 = r00 + r01 * v + r02 * v^2
    // r1 = r10 + r11 * v + r12 * v^2

    // r00 = 3v00 - 2b0
    let mut r00 = &v0.value()[0] - b0;
    r00 = r00.double();
    r00 = &v0.value()[0] + r00;

    // r01 = 3v10 -2b1
    let mut r01 = &v1.value()[0] - b1;
    r01 = r01.double();
    r01 = &v1.value()[0] + r01;

    // r11 = 3v01 - 2b4
    let mut r11 = &v0.value()[1] + b4;
    r11 = r11.double();
    r11 = &v0.value()[1] + r11;
    // r12 = 3v11 - 2b5
    let mut r12 = &v1.value()[1] + b5;
    r12 = r12.double();
    r12 = &v1.value()[1] + r12;
    // r12 = 3v11 - 2b5

    let v21 = mul_fp2_by_nonresidue(&v2.value()[1]);
    let mut r10 = &v21 + b3;
    r10 = r10.double();
    r10 += v21;

    // 3 * ( u) * v20 - 2b3
    let mut r02 = &v2.value()[0] - b2;
    r02 = r02.double();
    r02 = &v2.value()[0] + r02;

    Fp12E::new([Fp6E::new([r00, r01, r02]), Fp6E::new([r10, r11, r12])])
}

// To understand more about how to reduce the final exponentiation
// read "Efficient Final Exponentiation via Cyclotomic Structure for
// Pairings over Families of Elliptic Curves" (https://eprint.iacr.org/2020/875.pdf)

pub fn final_exponentiation(f: &Fp12E) -> Result<Fp12E, PairingError> {
    let f_easy_aux = f.conjugate() * f.inv().map_err(|_| PairingError::DivisionByZero)?;
    let mut f_easy = frobenius_square(&f_easy_aux) * &f_easy_aux;

    let mut v2 = cyclotomic_square(&f_easy); // v2 = f²
    let mut v0 = cyclotomic_pow_x_compressed(&f_easy); //  v0 = f^x
    let mut v1 = f_easy.conjugate(); // v1 = f^-1

    //  (x−1)²
    v0 *= v1; // v0 = f^(x-1)
    v1 = cyclotomic_pow_x_compressed(&v0); // v1 = (f^(x-1))^(x)

    v0 = v0.conjugate(); // v0 = (f^(x-1))^(-1)
    v0 *= &v1; // v0 = (f^(x-1))^(-1) * (f^(x-1))^x = (f^(x-1))^(x-1) =  f^((x-1)²)

    // (x+p)
    v1 = cyclotomic_pow_x_compressed(&v0); // v1 = f^((x-1)².x)
    v0 = frobenius(&v0); // f^((x-1)².p)
    v0 *= &v1; // f^((x-1)².p + (x-1)².x) = f^((x-1)².(x+p))

    // + 3
    f_easy *= v2; // f^3

    // (x²+p²−1)
    v2 = cyclotomic_pow_x_compressed(&v0);
    v1 = cyclotomic_pow_x_compressed(&v2); // v1 = f^((x-1)².(x+p).x²)
    v2 = frobenius_square(&v0); // v2 = f^((x-1)².(x+p).p²)
    v0 = v0.conjugate(); // v0 = f^((x-1)².(x+p).-1)
    v0 *= &v1; // v0 = f^((x-1)².(x+p).(x²-1))
    v0 *= &v2; // v0 = f^((x-1)².(x+p).(x²+p²-1))

    f_easy *= &v0;
    Ok(f_easy)
}

#[cfg(test)]
pub fn cyclotomic_pow_x(f: &Fp12E) -> Fp12E {
    let mut result = Fp12E::one();
    X_BINARY.iter().rev().for_each(|&bit| {
        result = cyclotomic_square(&result);
        if bit {
            result = &result * f;
        }
    });
    result
}

////////////////// KARABINA COMPRESSION //////////////////
// Karabina's compressed representation for cyclotomic subgroup elements
// Based on "Squaring in Cyclotomic Subgroups" https://eprint.iacr.org/2010/542

/// Compressed representation of cyclotomic subgroup element
/// Stores (g1, g2, g3, g5) - skipping g0 and g4
#[derive(Clone, Debug)]
pub struct CompressedCyclotomic {
    pub g1: Fp2E,
    pub g2: Fp2E,
    pub g3: Fp2E,
    pub g5: Fp2E,
}

impl CompressedCyclotomic {
    /// Compress an Fp12 cyclotomic element to 4 Fp2 elements
    pub fn compress(f: &Fp12E) -> Self {
        let [c0, c1] = f.value();
        let [_g0, g1, g2] = c0.value();
        let [g3, _g4, g5] = c1.value();
        Self {
            g1: g1.clone(),
            g2: g2.clone(),
            g3: g3.clone(),
            g5: g5.clone(),
        }
    }

    /// Decompress back to Fp12 using cyclotomic subgroup constraints
    pub fn decompress(&self) -> Fp12E {
        let g1 = &self.g1;
        let g2 = &self.g2;
        let g3 = &self.g3;
        let g5 = &self.g5;

        // Recover g4 using cyclotomic constraint
        let g4 = if *g3 != Fp2E::zero() {
            let g5_sq = g5.square();
            let g1_sq = g1.square();
            let e_g5_sq = mul_fp2_by_nonresidue(&g5_sq);
            let three_g1_sq = &g1_sq + &g1_sq + &g1_sq;
            let two_g2 = g2.double();
            let num = &e_g5_sq + &three_g1_sq - &two_g2;
            let four_g3 = g3.double().double();
            &num * four_g3.inv().unwrap()
        } else if *g2 != Fp2E::zero() {
            let two_g1_g5 = (g1 * g5).double();
            &two_g1_g5 * g2.inv().unwrap()
        } else {
            Fp2E::zero()
        };

        // Recover g0
        let g4_sq = g4.square();
        let two_g4_sq = g4_sq.double();
        let g3_g5 = g3 * g5;
        let g2_g1 = g2 * g1;
        let three_g2_g1 = &g2_g1 + &g2_g1 + &g2_g1;
        let inner = &two_g4_sq + &g3_g5 - &three_g2_g1;
        let g0 = mul_fp2_by_nonresidue(&inner) + Fp2E::one();

        Fp12E::new([
            Fp6E::new([g0, g1.clone(), g2.clone()]),
            Fp6E::new([g3.clone(), g4, g5.clone()]),
        ])
    }

    /// Square in compressed form
    pub fn square(&self) -> Self {
        let g1 = &self.g1;
        let g2 = &self.g2;
        let g3 = &self.g3;
        let g5 = &self.g5;

        // Compute the Fp4 squares we need
        let v1 = Fp4E::new([g3.clone(), g2.clone()]).square();
        let [v1_0, v1_1] = v1.value();

        let v2 = Fp4E::new([g1.clone(), g5.clone()]).square();
        let [v2_0, v2_1] = v2.value();

        // g1' = 3*v1[0] - 2*g1
        let mut h1 = v1_0 - g1;
        h1 = h1.double();
        h1 = &h1 + v1_0;

        // g2' = 3*v2[0] - 2*g2
        let mut h2 = v2_0 - g2;
        h2 = h2.double();
        h2 = &h2 + v2_0;

        // g3' = 3*E*v2[1] + 2*g3
        let alpha_v2_1 = mul_fp2_by_nonresidue(v2_1);
        let mut h3 = &alpha_v2_1 + g3;
        h3 = h3.double();
        h3 = &h3 + &alpha_v2_1;

        // g5' = 3*v1[1] + 2*g5
        let mut h5 = v1_1 + g5;
        h5 = h5.double();
        h5 = &h5 + v1_1;

        Self {
            g1: h1,
            g2: h2,
            g3: h3,
            g5: h5,
        }
    }
}

/// Optimized cyclotomic_pow_x using Karabina compression
pub fn cyclotomic_pow_x_compressed(f: &Fp12E) -> Fp12E {
    let mut result = Fp12E::one();
    let mut squares_pending = 0u32;

    // Note: BLS12-377 iterates X_BINARY in reverse
    for &bit in X_BINARY.iter().rev() {
        squares_pending += 1;

        if bit {
            result = apply_compressed_squares(&result, squares_pending);
            squares_pending = 0;
            result = &result * f;
        }
    }

    if squares_pending > 0 {
        result = apply_compressed_squares(&result, squares_pending);
    }

    result
}

/// Apply n squarings using compressed form
fn apply_compressed_squares(f: &Fp12E, n: u32) -> Fp12E {
    if n == 0 {
        return f.clone();
    }
    if n == 1 {
        return cyclotomic_square(f);
    }

    let mut compressed = CompressedCyclotomic::compress(f);
    for _ in 0..n {
        compressed = compressed.square();
    }
    compressed.decompress()
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve,
        unsigned_integer::element::U384,
    };

    use super::*;
    #[test]
    fn test_line_optimized_doubles_point_correctly() {
        let g1 = BLS12377Curve::generator().to_affine();
        let g2 = BLS12377TwistCurve::generator().to_affine();

        let (r, _line) = line(&g1, &g2, &g2);

        let expected = g2.operate_with(&g2);
        assert_eq!(r, expected);
    }

    #[test]
    fn test_line_optimized_adds_points_correctly() {
        let g1 = BLS12377Curve::generator().to_affine();
        let g = BLS12377TwistCurve::generator();

        let a: u64 = 12;
        let b: u64 = 23;

        let g2 = g.operate_with_self(a).to_affine();
        let g3 = g.operate_with_self(b).to_affine();

        let expected = g.operate_with_self(a + b);
        let (r, _line) = line(&g1, &g2, &g3);
        assert_eq!(r, expected);
    }

    #[test]
    fn batch_ate_pairing_bilinearity() {
        let p = BLS12377Curve::generator();
        let q = BLS12377TwistCurve::generator();
        let a = U384::from_u64(11);
        let b = U384::from_u64(93);

        let result = BLS12377AtePairing::compute_batch(&[
            (
                &p.operate_with_self(a).to_affine(),
                &q.operate_with_self(b).to_affine(),
            ),
            (
                &p.operate_with_self(a * b).to_affine(),
                &q.neg().to_affine(),
            ),
        ])
        .unwrap();
        assert_eq!(result, FieldElement::one());
    }

    #[test]
    fn ate_pairing_returns_one_when_one_element_is_the_neutral_element() {
        let p = BLS12377Curve::generator().to_affine();
        let q = ShortWeierstrassProjectivePoint::neutral_element();
        let result = BLS12377AtePairing::compute_batch(&[(&p.to_affine(), &q)]).unwrap();
        assert_eq!(result, FieldElement::one());

        let p = ShortWeierstrassProjectivePoint::neutral_element();
        let q = BLS12377TwistCurve::generator();
        let result = BLS12377AtePairing::compute_batch(&[(&p, &q.to_affine())]).unwrap();
        assert_eq!(result, FieldElement::one());
    }

    #[test]
    fn ate_pairing_errors_when_one_element_is_not_in_subgroup() {
        // p = (0, 1, 1) is in the curve but not in the subgroup.
        // Recall that the BLS 12-377 curve equation is y^2 = x^3 + 1.
        let p = ShortWeierstrassProjectivePoint::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::one(),
        ])
        .unwrap();
        let q = ShortWeierstrassProjectivePoint::neutral_element();
        let result = BLS12377AtePairing::compute_batch(&[(&p.to_affine(), &q)]);
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
    fn cyclotomic_square_equals_square() {
        let p = BLS12377Curve::generator();
        let q = BLS12377TwistCurve::generator();
        let f = miller(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1) because f^(p^6) = f.conjugate().
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux; // (f^{p^6 - 1})^(p^2) * (f^{p^6 - 1}).
        assert_eq!(cyclotomic_square(&f_easy), f_easy.square());
    }

    #[test]
    fn cyclotomic_pow_x_equals_pow() {
        let p = BLS12377Curve::generator();
        let q = BLS12377TwistCurve::generator();
        let f = miller(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1)
        let f_easy = frobenius_square(&f_easy_aux) * &f_easy_aux; // f^{(p^2)(p^6 - 1)}
        assert_eq!(cyclotomic_pow_x(&f_easy), f_easy.pow(X));
    }

    #[test]
    fn karabina_compress_decompress_roundtrip() {
        let p = BLS12377Curve::generator();
        let q = BLS12377TwistCurve::generator();
        let f = miller(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap();
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux;

        let compressed = CompressedCyclotomic::compress(&f_easy);
        let decompressed = compressed.decompress();
        assert_eq!(f_easy, decompressed);
    }

    #[test]
    fn karabina_compressed_square_equals_cyclotomic_square() {
        let p = BLS12377Curve::generator();
        let q = BLS12377TwistCurve::generator();
        let f = miller(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap();
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux;

        let normal_sq = cyclotomic_square(&f_easy);
        let compressed = CompressedCyclotomic::compress(&f_easy);
        let compressed_sq = compressed.square();
        let decompressed_sq = compressed_sq.decompress();

        assert_eq!(normal_sq, decompressed_sq);
    }

    #[test]
    fn cyclotomic_pow_x_compressed_equals_original() {
        let p = BLS12377Curve::generator();
        let q = BLS12377TwistCurve::generator();
        let f = miller(&p, &q);
        let f_easy_aux = f.conjugate() * f.inv().unwrap();
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux;

        let original = cyclotomic_pow_x(&f_easy);
        let compressed = cyclotomic_pow_x_compressed(&f_easy);
        assert_eq!(original, compressed);
    }
}
