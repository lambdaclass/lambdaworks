use super::{
    curve::BLS12381Curve,
    field_extension::{
        mul_fp2_by_nonresidue, BLS12381PrimeField, Degree12ExtensionField, Degree2ExtensionField,
        Degree4ExtensionField,
    },
    twist::BLS12381TwistCurve,
};
use crate::{cyclic_group::IsGroup, elliptic_curve::traits::IsPairing, errors::PairingError};
use crate::{
    elliptic_curve::short_weierstrass::{
        curves::bls12_381::field_extension::Degree6ExtensionField,
        point::ShortWeierstrassJacobianPoint,
        traits::IsShortWeierstrass,
    },
    field::element::FieldElement,
    unsigned_integer::element::U256,
};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

type FpE = FieldElement<BLS12381PrimeField>;
type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp4E = FieldElement<Degree4ExtensionField>;
type Fp6E = FieldElement<Degree6ExtensionField>;
type Fp12E = FieldElement<Degree12ExtensionField>;

pub const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

/// Precomputed line coefficients for Miller loop optimization.
/// Stores (b0, b2, b3) coefficients from line functions during the Miller loop.
/// This allows faster repeated pairings with the same G2 point.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct G2Prepared {
    /// Precomputed coefficients for each Miller loop iteration.
    /// Each entry is (b0, b2, b3) for line function multiplication.
    pub coefficients: Vec<(Fp2E, Fp2E, Fp2E)>,
    /// Whether this is the point at infinity
    pub infinity: bool,
}

#[cfg(feature = "alloc")]
impl G2Prepared {
    /// Precompute Miller loop coefficients for a G2 point.
    /// This allows faster pairing computation when the same G2 point is used multiple times.
    pub fn from_g2_affine(q: &ShortWeierstrassJacobianPoint<BLS12381TwistCurve>) -> Self {
        if q.is_neutral_element() {
            return Self {
                coefficients: Vec::new(),
                infinity: true,
            };
        }

        let q_affine = q.to_affine();
        let mut coefficients = Vec::with_capacity(X_BINARY.len());
        let mut r = q_affine.clone();

        // Precompute coefficients for each bit of X_BINARY (skip first bit)
        for bit in X_BINARY.iter().skip(1) {
            // Double step - compute line coefficients
            let double_coeffs = precompute_double_line(&mut r);
            coefficients.push(double_coeffs);

            if *bit {
                // Add step - compute line coefficients
                let add_coeffs = precompute_add_line(&mut r, &q_affine);
                coefficients.push(add_coeffs);
            }
        }

        Self {
            coefficients,
            infinity: false,
        }
    }
}

/// Precompute doubling line coefficients without the P-dependent parts.
/// Returns (b0_base, b2_scale, b3_scale) where:
/// - b0 = b0_base (independent of P)
/// - b2 = b2_scale * px (scaled by P.x)
/// - b3 = b3_scale * py (scaled by P.y)
fn precompute_double_line(
    t: &mut ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
) -> (Fp2E, Fp2E, Fp2E) {
    let [x1, y1, z1] = t.coordinates();

    let a = &TWO_INV * x1 * y1;
    let b = y1.square();
    let c = z1.square();
    let d = triple_fp2(&c);
    let e = BLS12381TwistCurve::b() * &d;
    let f = triple_fp2(&e);
    let g = &TWO_INV * (&b + &f);
    let h = (y1 + z1).square() - (&b + &c);

    let x3 = &a * (&b - &f);
    let e_sq = e.square();
    let y3 = g.square() - triple_fp2(&e_sq);
    let z3 = &b * &h;

    // Compute line coefficients (P-independent parts)
    let x1_sq = x1.square();
    let x1_sq_3 = triple_fp2(&x1_sq);

    let b0 = &e - &b; // Fully computed
    let b2_scale = x1_sq_3; // Will be multiplied by px
    let b3_scale = -&h; // Will be multiplied by py

    // Update T
    t.set_unchecked([x3, y3, z3]);

    (b0, b2_scale, b3_scale)
}

/// Precompute addition line coefficients.
fn precompute_add_line(
    t: &mut ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    q: &ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
) -> (Fp2E, Fp2E, Fp2E) {
    let [x1, y1, z1] = t.coordinates();
    let [x2, y2, _] = q.coordinates();

    let a = y2 * z1;
    let b = x2 * z1;
    let theta = y1 - &a;
    let lambda = x1 - &b;
    let c = theta.square();
    let d = lambda.square();
    let e = &lambda * &d;
    let f = z1 * &c;
    let g = x1 * &d;
    let h = &e + &f - g.double();
    let i = y1 * &e;

    let x3 = &lambda * &h;
    let y3 = &theta * (&g - &h) - &i;
    let z3 = z1 * &e;

    // Compute line coefficients
    let b0 = -&lambda * y2 + &theta * x2;
    let b2_scale = -&theta;
    let b3_scale = lambda;

    // Update T
    t.set_unchecked([x3, y3, z3]);

    (b0, b2_scale, b3_scale)
}

/// Miller loop using precomputed G2 coefficients.
/// This is faster than the standard miller() when pairing with the same G2 point multiple times.
#[cfg(feature = "alloc")]
pub fn miller_with_prepared(
    q_prepared: &G2Prepared,
    p: &ShortWeierstrassJacobianPoint<BLS12381Curve>,
) -> FieldElement<Degree12ExtensionField> {
    if q_prepared.infinity {
        return FieldElement::one();
    }

    let [px, py, _] = p.coordinates();
    let mut f = FieldElement::<Degree12ExtensionField>::one();
    let mut coeff_idx = 0;

    for bit in X_BINARY.iter().skip(1) {
        // Use precomputed doubling coefficients
        let (b0, b2_scale, b3_scale) = &q_prepared.coefficients[coeff_idx];
        coeff_idx += 1;

        // Compute b2 and b3 using P coordinates
        let [b2_0, b2_1] = b2_scale.value();
        let [b3_0, b3_1] = b3_scale.value();
        let b2 = Fp2E::new([b2_0 * px, b2_1 * px]);
        let b3 = Fp2E::new([b3_0 * py, b3_1 * py]);

        f = sparse_fp12_mul_by_line(&f.square(), b0, &b2, &b3);

        if *bit {
            // Use precomputed addition coefficients
            let (b0, b2_scale, b3_scale) = &q_prepared.coefficients[coeff_idx];
            coeff_idx += 1;

            let [b2_0, b2_1] = b2_scale.value();
            let [b3_0, b3_1] = b3_scale.value();
            let b2 = Fp2E::new([b2_0 * px, b2_1 * px]);
            let b3 = Fp2E::new([b3_0 * py, b3_1 * py]);

            f = sparse_fp12_mul_by_line(&f, b0, &b2, &b3);
        }
    }

    f.conjugate()
}

// value of x in binary

// We use |x|, then if needed we apply the minus sign in the final exponentiation by applying the inverse (in this case the conjugate because the element is in the cyclotomic subgroup)
pub const X: u64 = 0xd201000000010000;

// X = 1101001000000001000000000000000000000000000000010000000000000000
pub const X_BINARY: &[bool] = &[
    true, true, false, true, false, false, true, false, false, false, false, false, false, false,
    false, true, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, true, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
];

// GAMMA constants used to compute the Frobenius morphisms
// We took these constants from https://github.com/hecmas/zkNotebook/blob/main/src/BLS12381/constants.ts
pub const GAMMA_11: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("1904D3BF02BB0667C231BEB4202C0D1F0FD603FD3CBD5F4F7B2443D784BAB9C4F67EA53D63E7813D8D0775ED92235FB8"),
    FpE::from_hex_unchecked("FC3E2B36C4E03288E9E902231F9FB854A14787B6C7B36FEC0C8EC971F63C5F282D5AC14D6C7EC22CF78A126DDC4AF3"),
]);

pub const GAMMA_12: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("0"),
    FpE::from_hex_unchecked("1A0111EA397FE699EC02408663D4DE85AA0D857D89759AD4897D29650FB85F9B409427EB4F49FFFD8BFD00000000AAAC"),
]);

pub const GAMMA_13: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("6AF0E0437FF400B6831E36D6BD17FFE48395DABC2D3435E77F76E17009241C5EE67992F72EC05F4C81084FBEDE3CC09"),
    FpE::from_hex_unchecked("6AF0E0437FF400B6831E36D6BD17FFE48395DABC2D3435E77F76E17009241C5EE67992F72EC05F4C81084FBEDE3CC09"),
]);

pub const GAMMA_14: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("1A0111EA397FE699EC02408663D4DE85AA0D857D89759AD4897D29650FB85F9B409427EB4F49FFFD8BFD00000000AAAD"),
    FpE::from_hex_unchecked("0"),
]);

pub const GAMMA_15: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("5B2CFD9013A5FD8DF47FA6B48B1E045F39816240C0B8FEE8BEADF4D8E9C0566C63A3E6E257F87329B18FAE980078116"),
    FpE::from_hex_unchecked("144E4211384586C16BD3AD4AFA99CC9170DF3560E77982D0DB45F3536814F0BD5871C1908BD478CD1EE605167FF82995"),
]);

/// GAMMA_2i = GAMMA_1i * GAMMA_1i.conjugate()
pub const GAMMA_21: FpE = FpE::from_hex_unchecked(
    "5F19672FDF76CE51BA69C6076A0F77EADDB3A93BE6F89688DE17D813620A00022E01FFFFFFFEFFFF",
);

pub const GAMMA_22: FpE = FpE::from_hex_unchecked(
    "5F19672FDF76CE51BA69C6076A0F77EADDB3A93BE6F89688DE17D813620A00022E01FFFFFFFEFFFE",
);

pub const GAMMA_23: FpE =
    FpE::from_hex_unchecked("1A0111EA397FE69A4B1BA7B6434BACD764774B84F38512BF6730D2A0F6B0F6241EABFFFEB153FFFFB9FEFFFFFFFFAAAA");

pub const GAMMA_24: FpE =
    FpE::from_hex_unchecked("1A0111EA397FE699EC02408663D4DE85AA0D857D89759AD4897D29650FB85F9B409427EB4F49FFFD8BFD00000000AAAC");

pub const GAMMA_25: FpE =
    FpE::from_hex_unchecked("1A0111EA397FE699EC02408663D4DE85AA0D857D89759AD4897D29650FB85F9B409427EB4F49FFFD8BFD00000000AAAD");

#[derive(Clone)]
pub struct BLS12381AtePairing;

impl IsPairing for BLS12381AtePairing {
    type G1Point = ShortWeierstrassJacobianPoint<BLS12381Curve>;
    type G2Point = ShortWeierstrassJacobianPoint<BLS12381TwistCurve>;
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
                result *= miller(&q, &p);
            }
        }
        final_exponentiation(&result)
    }
}

/// Implements the miller loop for the ate pairing of the BLS12 381 curve.
/// Based on algorithm 9.2, page 212 of the book
/// "Topics in computational number theory" by W. Bons and K. Lenstra
#[allow(unused)]
pub fn miller(
    q: &ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassJacobianPoint<BLS12381Curve>,
) -> FieldElement<Degree12ExtensionField> {
    // Convert to affine to ensure Z=1, which makes the line function formulas work correctly
    // The line formulas were designed for projective coordinates where (X,Y,Z) = (X/Z, Y/Z)
    // With Z=1, this is consistent with Jacobian (X,Y,1) = (X/1², Y/1³) = (X, Y)
    let q_affine = q.to_affine();
    let p_affine = p.to_affine();
    let mut r = q_affine.clone();
    let mut f = FieldElement::<Degree12ExtensionField>::one();
    X_BINARY.iter().skip(1).for_each(|bit| {
        double_accumulate_line(&mut r, &p_affine, &mut f);
        if *bit {
            add_accumulate_line(&mut r, &q_affine, &p_affine, &mut f);
        }
    });

    f.conjugate()
}

/// The multiplicative inverse of 2 in Fp2.
const TWO_INV: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd556"),
    FpE::from_hex_unchecked("0"),
]);

/// Multiplies an Fp2 element by 3 using addition chain: 3x = 2x + x
#[inline]
fn triple_fp2(x: &Fp2E) -> Fp2E {
    x.double() + x
}

/// Sparse Fp12 multiplication by a line evaluation: f * (b0 + b2*w^2 + b3*w^3).
/// This is specific to BLS12-381 where the line operates at the Fp2 level with
/// `mul_fp2_by_nonresidue` (different from BN254/BLS12-377's Fp6-level `sparse_fp12_mul`).
#[inline]
fn sparse_fp12_mul_by_line(f: &Fp12E, b0: &Fp2E, b2: &Fp2E, b3: &Fp2E) -> Fp12E {
    let [x, y] = f.value();
    let [a0, a2, a4] = x.value();
    let [a1, a3, a5] = y.value();

    let a3b3 = a3 * b3;
    let a4b2 = a4 * b2;
    let a5b3 = a5 * b3;
    let a4b3 = a4 * b3;
    let a5b2 = a5 * b2;

    Fp12E::new([
        Fp6E::new([
            a0 * b0 + mul_fp2_by_nonresidue(&(&a3b3 + &a4b2)),
            a2 * b0 + mul_fp2_by_nonresidue(&a5b3) + a0 * b2,
            a4 * b0 + a1 * b3 + a2 * b2,
        ]),
        Fp6E::new([
            a1 * b0 + mul_fp2_by_nonresidue(&(&a4b3 + &a5b2)),
            a3 * b0 + a0 * b3 + a1 * b2,
            a5 * b0 + a2 * b3 + a3 * b2,
        ]),
    ])
}

fn double_accumulate_line(
    t: &mut ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassJacobianPoint<BLS12381Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
    let [x1, y1, z1] = t.coordinates();
    let [px, py, _] = p.coordinates();

    let a = &TWO_INV * x1 * y1;
    let b = y1.square();
    let c = z1.square();
    let d = triple_fp2(&c); // 3c via addition chain
    let e = BLS12381TwistCurve::b() * d;
    let f = triple_fp2(&e); // 3e via addition chain
    let g = &TWO_INV * (&b + &f);
    let h = (y1 + z1).square() - (&b + &c);

    let x3 = &a * (&b - &f);
    let e_sq = e.square();
    let y3 = g.square() - triple_fp2(&e_sq); // 3*e^2 via addition chain
    let z3 = &b * &h;

    let [h0, h1] = h.value();
    let x1_sq = x1.square();
    let x1_sq_3 = triple_fp2(&x1_sq); // 3*x1^2 via addition chain
    let [x1_sq_30, x1_sq_31] = x1_sq_3.value();

    // SAFETY: The values `x_3, y_3, z_3` are computed correctly to be on the curve.
    t.set_unchecked([x3, y3, z3]);

    let b0 = e - b;
    let b2 = Fp2E::new([x1_sq_30 * px, x1_sq_31 * px]);
    let b3 = Fp2E::new([-h0 * py, -h1 * py]);

    *accumulator = sparse_fp12_mul_by_line(&accumulator.square(), &b0, &b2, &b3);
}

fn add_accumulate_line(
    t: &mut ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    q: &ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassJacobianPoint<BLS12381Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
    let [x1, y1, z1] = t.coordinates();
    let [x2, y2, _] = q.coordinates();
    let [px, py, _] = p.coordinates();

    let a = y2 * z1;
    let b = x2 * z1;
    let theta = y1 - a;
    let lambda = x1 - b;
    let c = theta.square();
    let d = lambda.square();
    let e = &lambda * &d;
    let f = z1 * c;
    let g = x1 * d;
    let h = &e + f - g.double(); // 2*g via addition chain
    let i = y1 * &e;

    let x3 = &lambda * &h;
    let y3 = &theta * (g - h) - i;
    let z3 = z1 * e;

    // SAFETY: The values `x_3, y_3, z_3` are computed correctly to be on the curve.
    t.set_unchecked([x3, y3, z3]);

    let [lambda0, lambda1] = lambda.value();
    let [theta0, theta1] = theta.value();

    let b0 = -&lambda * y2 + &theta * x2;
    let b2 = Fp2E::new([-theta0 * px, -theta1 * px]);
    let b3 = Fp2E::new([lambda0 * py, lambda1 * py]);

    *accumulator = sparse_fp12_mul_by_line(accumulator, &b0, &b2, &b3);
}

// To understand more about how to reduce the final exponentiation
// read "Efficient Final Exponentiation via Cyclotomic Structure for
// Pairings over Families of Elliptic Curves" (https://eprint.iacr.org/2020/875.pdf)
// Hard part from https://eprint.iacr.org/2020/875.pdf p14
// Same implementation as in Constantine https://github.com/mratsim/constantine/blob/master/constantine/math/pairings/pairings_bls12.nim
pub fn final_exponentiation(f: &Fp12E) -> Result<Fp12E, PairingError> {
    let f_easy_aux = f.conjugate() * f.inv().map_err(|_| PairingError::DivisionByZero)?;
    let mut f_easy = frobenius_square(&f_easy_aux) * &f_easy_aux;

    let mut v2 = cyclotomic_square(&f_easy); // v2 = f²
    let mut v0 = cyclotomic_pow_x_compressed(&f_easy).conjugate(); //  v0 = f^x
    let mut v1 = f_easy.conjugate(); // v1 = f^-1

    //  (x−1)²
    v0 *= v1; // v0 = f^(x-1)
    v1 = cyclotomic_pow_x_compressed(&v0).conjugate(); // v1 = (f^(x-1))^(x)

    v0 = v0.conjugate(); // v0 = (f^(x-1))^(-1)
    v0 *= &v1; // v0 = (f^(x-1))^(-1) * (f^(x-1))^x = (f^(x-1))^(x-1) =  f^((x-1)²)

    // (x+p)
    v1 = cyclotomic_pow_x_compressed(&v0).conjugate(); // v1 = f^((x-1)².x)
    v0 = frobenius(&v0); // f^((x-1)².p)
    v0 *= &v1; // f^((x-1)².p + (x-1)².x) = f^((x-1)².(x+p))

    // + 3
    f_easy *= v2; // f^3

    // (x²+p²−1)
    v2 = cyclotomic_pow_x_compressed(&v0).conjugate();
    v1 = cyclotomic_pow_x_compressed(&v2).conjugate(); // v1 = f^((x-1)².(x+p).x²)
    v2 = frobenius_square(&v0); // v2 = f^((x-1)².(x+p).p²)
    v0 = v0.conjugate(); // v0 = f^((x-1)².(x+p).-1)
    v0 *= &v1; // v0 = f^((x-1)².(x+p).(x²-1))
    v0 *= &v2; // v0 = f^((x-1)².(x+p).(x²+p²-1))

    f_easy *= &v0;
    Ok(f_easy)
}

////////////////// FROBENIUS MORPHISIMS //////////////////
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

fn frobenius_square(f: &Fp12E) -> Fp12E {
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

    Fp12E::new([Fp6E::new([r00, r01, r02]), Fp6E::new([r10, r11, r12])])
}

pub fn cyclotomic_pow_x(f: &Fp12E) -> Fp12E {
    let mut result = Fp12E::one();
    X_BINARY.iter().for_each(|&bit| {
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
//
// An Fp12 element f has 6 Fp2 coefficients: (g0, g1, g2, g3, g4, g5)
// where f = (g0 + g1*v + g2*v²) + (g3 + g4*v + g5*v²)*w
//
// Compressed form stores only (g1, g2, g3, g5) - 4 Fp2 elements
// g0 and g4 can be recovered using cyclotomic subgroup constraints

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

    /// Decompress to full Fp12 element
    /// Uses the cyclotomic constraint to recover g0 and g4
    pub fn decompress(&self) -> Fp12E {
        let g1 = &self.g1;
        let g2 = &self.g2;
        let g3 = &self.g3;
        let g5 = &self.g5;

        // E = non-residue (1 + u) in Fp2
        // Recover g4:
        // If g3 ≠ 0: g4 = (E * g5² + 3 * g1² - 2 * g2) / (4 * g3)
        // If g3 = 0: g4 = (2 * g1 * g5) / g2
        let g4 = if *g3 != Fp2E::zero() {
            let g5_sq = g5.square();
            let g1_sq = g1.square();
            // E * g5² where E = (1+u)
            let e_g5_sq = mul_fp2_by_nonresidue(&g5_sq);
            // 3 * g1²
            let three_g1_sq = triple_fp2(&g1_sq);
            // 2 * g2
            let two_g2 = g2.double();
            // numerator = E * g5² + 3 * g1² - 2 * g2
            let num = &e_g5_sq + &three_g1_sq - &two_g2;
            // denominator = 4 * g3
            let four_g3 = g3.double().double();
            &num * four_g3.inv().unwrap()
        } else if *g2 != Fp2E::zero() {
            // g4 = 2 * g1 * g5 / g2
            let two_g1_g5 = (g1 * g5).double();
            &two_g1_g5 * g2.inv().unwrap()
        } else {
            // Special case: return identity's g4 component
            Fp2E::zero()
        };

        // Recover g0:
        // g0 = E * (2 * g4² + g3 * g5 - 3 * g2 * g1) + 1
        let g4_sq = g4.square();
        let two_g4_sq = g4_sq.double();
        let g3_g5 = g3 * g5;
        let g2_g1 = g2 * g1;
        let three_g2_g1 = triple_fp2(&g2_g1);
        let inner = &two_g4_sq + &g3_g5 - &three_g2_g1;
        let g0 = mul_fp2_by_nonresidue(&inner) + Fp2E::one();

        Fp12E::new([
            Fp6E::new([g0, g1.clone(), g2.clone()]),
            Fp6E::new([g3.clone(), g4, g5.clone()]),
        ])
    }

    /// Square in compressed form (Karabina's algorithm)
    /// Computes squared representation without computing g0, g4
    ///
    /// From the cyclotomic_square formulas with Fp4 squares:
    /// - v0 = Fp4(g0, g4).square() = [g0² + α*g4², 2*g0*g4]
    /// - v1 = Fp4(g3, g2).square() = [g3² + α*g2², 2*g3*g2]
    /// - v2 = Fp4(g1, g5).square() = [g1² + α*g5², 2*g1*g5]
    ///
    /// Output formulas:
    /// - g0' = 3*v0[0] - 2*g0  (needs g0, g4 - SKIP)
    /// - g1' = 3*v1[0] - 2*g1  (only needs g3, g2, g1) ✓
    /// - g2' = 3*v2[0] - 2*g2  (only needs g1, g5, g2) ✓
    /// - g3' = 3*α*v2[1] + 2*g3  (only needs g1, g5, g3) ✓
    /// - g4' = 3*v0[1] + 2*g4  (needs g0, g4 - SKIP)
    /// - g5' = 3*v1[1] + 2*g5  (only needs g3, g2, g5) ✓
    pub fn square(&self) -> Self {
        let g1 = &self.g1;
        let g2 = &self.g2;
        let g3 = &self.g3;
        let g5 = &self.g5;

        // Compute the Fp4 squares we need (only v1 and v2)
        // v1 = Fp4(g3, g2).square() = [g3² + α*g2², 2*g3*g2]
        let v1 = Fp4E::new([g3.clone(), g2.clone()]).square();
        let [v1_0, v1_1] = v1.value();

        // v2 = Fp4(g1, g5).square() = [g1² + α*g5², 2*g1*g5]
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

        // g3' = 3*α*v2[1] + 2*g3
        // Note: v2[1] = 2*g1*g5, so α*v2[1] = mul_fp2_by_nonresidue(2*g1*g5)
        let alpha_v2_1 = mul_fp2_by_nonresidue(v2_1);
        let mut h3 = &alpha_v2_1 + g3;
        h3 = h3.double();
        h3 = &h3 + &alpha_v2_1;

        // g5' = 3*v1[1] + 2*g5
        // Note: v1[1] = 2*g3*g2
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

    /// Multiply compressed element by full Fp12 element, returning full Fp12
    pub fn mul_by_fp12(&self, other: &Fp12E) -> Fp12E {
        self.decompress() * other
    }
}

/// Computes f^X using Karabina compression for efficient cyclotomic squaring.
///
/// X = 0xd201000000010000 has only 6 set bits, allowing us to batch consecutive
/// squarings in compressed form and only decompress when multiplication is needed.
pub fn cyclotomic_pow_x_compressed(f: &Fp12E) -> Fp12E {
    let mut result = f.clone();

    // Bit 1: f -> f^2 -> f^3
    result = cyclotomic_square(&result);
    result = &result * f;

    // Bit 3: f^3 -> f^12 -> f^13
    result = apply_compressed_squares(&result, 2);
    result = &result * f;

    // Bit 6: f^13 -> f^104 -> f^105
    result = apply_compressed_squares(&result, 3);
    result = &result * f;

    // Bit 15: f^105 -> f^53760 -> f^53761
    result = apply_compressed_squares(&result, 9);
    result = &result * f;

    // Bit 47: 32 squares then multiply
    result = apply_compressed_squares(&result, 32);
    result = &result * f;

    // Final 16 trailing zeros
    apply_compressed_squares(&result, 16)
}

/// Applies n consecutive cyclotomic squarings, using Karabina compression for long runs.
///
/// For runs of 10+ squares, compression amortizes the decompression cost.
/// For shorter runs, regular squaring avoids the inversion overhead.
fn apply_compressed_squares(f: &Fp12E, n: u32) -> Fp12E {
    const COMPRESSION_THRESHOLD: u32 = 10;

    if n < COMPRESSION_THRESHOLD {
        let mut result = f.clone();
        for _ in 0..n {
            result = cyclotomic_square(&result);
        }
        result
    } else {
        let mut compressed = CompressedCyclotomic::compress(f);
        for _ in 0..n {
            compressed = compressed.square();
        }
        compressed.decompress()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve,
        unsigned_integer::element::U384,
    };

    use super::*;

    // Note: The line function tests (double_accumulate_line, add_accumulate_line)
    // were removed because they test internal formulas that differ between coordinate
    // systems. The pairing correctness is verified by the bilinearity tests below.

    #[test]
    fn batch_ate_pairing_bilinearity() {
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator();
        let a = U384::from_u64(11);
        let b = U384::from_u64(93);

        let result = BLS12381AtePairing::compute_batch(&[
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
        let p = BLS12381Curve::generator().to_affine();
        let q = ShortWeierstrassJacobianPoint::neutral_element();
        let result = BLS12381AtePairing::compute_batch(&[(&p.to_affine(), &q)]).unwrap();
        assert_eq!(result, FieldElement::one());

        let p = ShortWeierstrassJacobianPoint::neutral_element();
        let q = BLS12381TwistCurve::generator();
        let result = BLS12381AtePairing::compute_batch(&[(&p, &q.to_affine())]).unwrap();
        assert_eq!(result, FieldElement::one());
    }

    #[test]
    fn ate_pairing_errors_when_one_element_is_not_in_subgroup() {
        // p = (0, 2, 1) is in the curve but not in the subgroup.
        // Recall that the BLS 12-381 curve equation is y^2 = x^3 + 4.
        let p = ShortWeierstrassJacobianPoint::new([
            FieldElement::zero(),
            FieldElement::from(2),
            FieldElement::one(),
        ])
        .unwrap();
        let q = ShortWeierstrassJacobianPoint::neutral_element();
        let result = BLS12381AtePairing::compute_batch(&[(&p.to_affine(), &q)]);
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
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator();
        let f = miller(&q, &p);
        let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1) because f^(p^6) = f.conjugate().
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux; // (f^{p^6 - 1})^(p^2) * (f^{p^6 - 1}).
        assert_eq!(cyclotomic_square(&f_easy), f_easy.square());
    }

    #[test]
    fn cyclotomic_pow_x_equals_pow() {
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator();
        let f = miller(&q, &p);
        let f_easy_aux = f.conjugate() * f.inv().unwrap(); // f ^ (p^6 - 1) because f^(p^6) = f.conjugate().
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux; // (f^{p^6 - 1})^(p^2) * (f^{p^6 - 1}).
        assert_eq!(cyclotomic_pow_x(&f_easy), f_easy.pow(X));
    }

    #[test]
    fn karabina_compress_decompress_roundtrip() {
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator();
        let f = miller(&q, &p);
        // Get element in cyclotomic subgroup
        let f_easy_aux = f.conjugate() * f.inv().unwrap();
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux;

        let compressed = CompressedCyclotomic::compress(&f_easy);
        let decompressed = compressed.decompress();
        assert_eq!(f_easy, decompressed);
    }

    #[test]
    fn karabina_compressed_square_equals_cyclotomic_square() {
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator();
        let f = miller(&q, &p);
        let f_easy_aux = f.conjugate() * f.inv().unwrap();
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux;

        // Normal cyclotomic square
        let normal_sq = cyclotomic_square(&f_easy);

        // Compressed square: compress -> square -> decompress
        let compressed = CompressedCyclotomic::compress(&f_easy);
        let compressed_sq = compressed.square();
        let decompressed_sq = compressed_sq.decompress();

        assert_eq!(normal_sq, decompressed_sq);
    }

    #[test]
    fn cyclotomic_pow_x_compressed_equals_original() {
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator();
        let f = miller(&q, &p);
        let f_easy_aux = f.conjugate() * f.inv().unwrap();
        let f_easy = &frobenius_square(&f_easy_aux) * f_easy_aux;

        let original = cyclotomic_pow_x(&f_easy);
        let compressed = cyclotomic_pow_x_compressed(&f_easy);
        assert_eq!(original, compressed);
    }

    #[test]
    fn prepared_miller_equals_standard_miller() {
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator();

        // Standard miller loop
        let standard_result = miller(&q.to_affine(), &p.to_affine());

        // Prepared miller loop
        let q_prepared = G2Prepared::from_g2_affine(&q);
        let prepared_result = miller_with_prepared(&q_prepared, &p.to_affine());

        assert_eq!(standard_result, prepared_result);
    }

    #[test]
    fn prepared_pairing_bilinearity() {
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator();
        let a: u64 = 17;
        let b: u64 = 31;

        // Compute e(aP, bQ) using prepared
        let ap = p.operate_with_self(a).to_affine();
        let bq = q.operate_with_self(b);
        let bq_prepared = G2Prepared::from_g2_affine(&bq);
        let f1 = miller_with_prepared(&bq_prepared, &ap);
        let result1 = final_exponentiation(&f1).unwrap();

        // Compute e(P, Q)^(ab) using standard
        let f2 = miller(&q.to_affine(), &p.to_affine());
        let result2 = final_exponentiation(&f2).unwrap().pow(a * b);

        assert_eq!(result1, result2);
    }
}
