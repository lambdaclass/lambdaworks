use super::{
    curve::BLS12381Curve,
    field_extension::{
        mul_fp2_by_nonresidue, BLS12381PrimeField, Degree12ExtensionField, Degree2ExtensionField,
    },
    twist::BLS12381TwistCurve,
};
use crate::{cyclic_group::IsGroup, elliptic_curve::traits::IsPairing, errors::PairingError};
use crate::{
    elliptic_curve::short_weierstrass::{
        curves::bls12_381::field_extension::Degree6ExtensionField,
        point::ShortWeierstrassJacobianPoint,
    },
    field::element::FieldElement,
    unsigned_integer::element::U256,
};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

type FpE = FieldElement<BLS12381PrimeField>;
type Fp2E = FieldElement<Degree2ExtensionField>;
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

    let a = mul_fp2_by_fp(&TWO_INV_FP, &(x1 * y1));
    let b = y1.square();
    let c = z1.square();
    let d = triple_fp2(&c);
    let e = mul_fp2_by_twist_b(&d); // b'·d using additions only
    let f = triple_fp2(&e);
    let g = mul_fp2_by_fp(&TWO_INV_FP, &(&b + &f));
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
        let (db0, db2_scale, db3_scale) = &q_prepared.coefficients[coeff_idx];
        coeff_idx += 1;

        let [db2_0, db2_1] = db2_scale.value();
        let [db3_0, db3_1] = db3_scale.value();
        let db2 = Fp2E::new([db2_0 * px, db2_1 * px]);
        let db3 = Fp2E::new([db3_0 * py, db3_1 * py]);

        if *bit {
            // Line combining: combine double and add lines, then multiply once
            let (ab0, ab2_scale, ab3_scale) = &q_prepared.coefficients[coeff_idx];
            coeff_idx += 1;

            let [ab2_0, ab2_1] = ab2_scale.value();
            let [ab3_0, ab3_1] = ab3_scale.value();
            let ab2 = Fp2E::new([ab2_0 * px, ab2_1 * px]);
            let ab3 = Fp2E::new([ab3_0 * py, ab3_1 * py]);

            let line_line = mul_line_by_line(db0, &db2, &db3, ab0, &ab2, &ab3);
            f = mul_fp12_by_line_line(&f.square(), &line_line);
        } else {
            f = sparse_fp12_mul_by_line(&f.square(), db0, &db2, &db3);
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
    ///
    /// With the `parallel` feature, runs independent Miller loops in parallel
    /// across CPU cores, then multiplies the results and applies a single final
    /// exponentiation. Without `parallel`, uses a shared-square multi-Miller
    /// loop that saves (n-1)×63 Fp12 squares.
    fn compute_batch(
        pairs: &[(&Self::G1Point, &Self::G2Point)],
    ) -> Result<FieldElement<Self::OutputField>, PairingError> {
        // Validate subgroup membership and filter neutral elements
        let mut valid_pairs = Vec::with_capacity(pairs.len());
        for (p, q) in pairs {
            if !p.is_in_subgroup() || !q.is_in_subgroup() {
                return Err(PairingError::PointNotInSubgroup);
            }
            if !p.is_neutral_element() && !q.is_neutral_element() {
                valid_pairs.push((p.to_affine(), q.to_affine()));
            }
        }

        if valid_pairs.is_empty() {
            return Ok(FieldElement::one());
        }

        // Single pair: use direct miller loop (no allocation overhead)
        if valid_pairs.len() == 1 {
            let result = miller(&valid_pairs[0].1, &valid_pairs[0].0);
            return final_exponentiation(&result);
        }

        // Multi-pair: parallel independent Miller loops (when available)
        // or shared-square Miller loop (sequential fallback)
        #[cfg(feature = "parallel")]
        let result = {
            use rayon::prelude::*;
            valid_pairs
                .par_iter()
                .map(|(p, q)| miller(q, p))
                .reduce(Fp12E::one, |acc, f| acc * f)
        };

        #[cfg(not(feature = "parallel"))]
        let result = multi_miller(&valid_pairs);

        final_exponentiation(&result)
    }
}

/// Implements the miller loop for the ate pairing of the BLS12 381 curve.
/// Based on algorithm 9.2, page 212 of the book
/// "Topics in computational number theory" by W. Bons and K. Lenstra
pub fn miller(
    q: &ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassJacobianPoint<BLS12381Curve>,
) -> FieldElement<Degree12ExtensionField> {
    let q_affine = q.to_affine();
    let p_affine = p.to_affine();
    let mut r = q_affine.clone();
    let mut f = FieldElement::<Degree12ExtensionField>::one();

    for bit in X_BINARY.iter().skip(1) {
        let (db0, db2, db3) = double_step(&mut r, &p_affine);

        if *bit {
            // Line combining: combine double and add lines, then multiply once
            let (ab0, ab2, ab3) = add_step(&mut r, &q_affine, &p_affine);
            let line_line = mul_line_by_line(&db0, &db2, &db3, &ab0, &ab2, &ab3);
            f = mul_fp12_by_line_line(&f.square(), &line_line);
        } else {
            f = sparse_fp12_mul_by_line(&f.square(), &db0, &db2, &db3);
        }
    }

    f.conjugate()
}

/// Multi-pair Miller loop with shared squaring.
///
/// Instead of computing n independent Miller loops and multiplying the results,
/// this function shares the Fp12 squaring across all pairs. Each iteration:
/// 1. Square the shared accumulator once (not n times)
/// 2. For each pair, compute line evaluations and multiply into the accumulator
///
/// Saves (n-1) × 63 Fp12 squares for n-pair batches, which is the dominant
/// cost reduction for batch pairings.
#[cfg(all(feature = "alloc", not(feature = "parallel")))]
fn multi_miller(
    pairs: &[(
        ShortWeierstrassJacobianPoint<BLS12381Curve>,
        ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    )],
) -> Fp12E {
    let n = pairs.len();

    // Initialize R points (running G2 accumulators) and P/Q points
    let mut rs: Vec<ShortWeierstrassJacobianPoint<BLS12381TwistCurve>> =
        pairs.iter().map(|(_, q)| q.to_affine()).collect();
    let qs: Vec<ShortWeierstrassJacobianPoint<BLS12381TwistCurve>> = rs.clone();
    let ps: Vec<ShortWeierstrassJacobianPoint<BLS12381Curve>> =
        pairs.iter().map(|(p, _)| p.to_affine()).collect();

    // Process first bit (MSB = 1) separately to avoid squaring 1
    // X_BINARY[0] = true, so first iteration is always double+add
    let mut f = {
        // First pair: compute lines, assign to f directly (no mul needed)
        let (db0, db2, db3) = double_step(&mut rs[0], &ps[0]);
        let (ab0, ab2, ab3) = add_step(&mut rs[0], &qs[0], &ps[0]);
        let mut acc = mul_line_by_line(&db0, &db2, &db3, &ab0, &ab2, &ab3);

        // Remaining pairs: compute lines, multiply into accumulator
        for k in 1..n {
            let (db0, db2, db3) = double_step(&mut rs[k], &ps[k]);
            let (ab0, ab2, ab3) = add_step(&mut rs[k], &qs[k], &ps[k]);
            let ll = mul_line_by_line(&db0, &db2, &db3, &ab0, &ab2, &ab3);
            acc = &acc * &ll;
        }
        acc
    };

    // Process remaining bits
    for bit in X_BINARY.iter().skip(2) {
        // ONE shared square
        f = f.square();

        for k in 0..n {
            let (db0, db2, db3) = double_step(&mut rs[k], &ps[k]);

            if *bit {
                let (ab0, ab2, ab3) = add_step(&mut rs[k], &qs[k], &ps[k]);
                let ll = mul_line_by_line(&db0, &db2, &db3, &ab0, &ab2, &ab3);
                f = mul_fp12_by_line_line(&f, &ll);
            } else {
                f = sparse_fp12_mul_by_line(&f, &db0, &db2, &db3);
            }
        }
    }

    f.conjugate()
}

/// The multiplicative inverse of 2 in the base field Fp.
/// Used for halving Fp2 elements component-wise (2 Fp muls instead of 3 via Karatsuba).
const TWO_INV_FP: FpE = FpE::from_hex_unchecked("d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd556");

/// Multiplies an Fp2 element by 3 using addition chain: 3x = 2x + x
#[inline]
fn triple_fp2(x: &Fp2E) -> Fp2E {
    x.double() + x
}

/// Multiply Fp2 element by a base field (Fp) scalar.
/// Cost: 2 Fp multiplications (vs 3 for generic Fp2 × Fp2 Karatsuba when one operand is (c, 0)).
#[inline]
fn mul_fp2_by_fp(scalar: &FpE, a: &Fp2E) -> Fp2E {
    let [a0, a1] = a.value();
    Fp2E::new([scalar * a0, scalar * a1])
}

/// Multiply Fp2 element by the twist curve parameter b' = (4 + 4i).
/// Uses addition chains only: (a₀+a₁i)(4+4i) = 4(a₀-a₁) + 4(a₀+a₁)i.
/// Cost: ~0 Fp multiplications (only Fp additions/subtractions),
/// vs 3 Fp muls for generic Fp2 Karatsuba.
#[inline]
fn mul_fp2_by_twist_b(a: &Fp2E) -> Fp2E {
    let [a0, a1] = a.value();
    let diff = a0 - a1;
    let sum = a0 + a1;
    Fp2E::new([diff.double().double(), sum.double().double()])
}

/// Specialized Fp4 squaring for BLS12-381 where the quadratic non-residue is β = 1+u.
///
/// Uses the complex squaring trick in Fp4 = Fp2[t]/(t² - β):
///   (a + b·t)² = c0 + c1·t where:
///     c0 = (a+b)(a+β·b) - (1+β)·v,  with v = a·b
///     c1 = 2·v
///
/// Since β = 1+u, mul by β is free (just 2 Fp additions via mul_fp2_by_nonresidue).
/// (1+β) = (2+u), so (1+β)·v = (2v₀-v₁, v₀+2v₁) — also just Fp additions.
///
/// Cost: 2 Fp2 muls (= 6 Fp muls), vs generic Fp4::square = 2 Fp2 squares + 2 Fp2 muls (= 10 Fp muls).
#[inline]
fn fp4_square(a: &Fp2E, b: &Fp2E) -> (Fp2E, Fp2E) {
    let v = a * b;
    let beta_b = mul_fp2_by_nonresidue(b);
    let t = (a + b) * (a + &beta_b);
    // (1+β)·v = (2+u)·(v₀ + v₁·u) = (2v₀ - v₁) + (v₀ + 2v₁)·u
    let [v0, v1] = v.value();
    let one_plus_beta_v = Fp2E::new([v0.double() - v1, v0 + v1.double()]);
    let c0 = t - &one_plus_beta_v;
    let c1 = v.double();
    (c0, c1)
}

/// Sparse Fp12 multiplication by a line evaluation using Karatsuba.
///
/// A line has the form b0 + b2·v + b3·v·w, which in the Fp6[w]/(w²-v) tower gives:
/// L_X = (b0, b2, 0), L_Y = (0, b3, 0).
///
/// Uses Fp12-level Karatsuba: f·L = (X·L_X + v·(Y·L_Y), (X+Y)·(L_X+L_Y) - X·L_X - Y·L_Y)
///
/// Cost: 13 Fp2 multiplications (vs 18 without Karatsuba).
#[inline]
fn sparse_fp12_mul_by_line(f: &Fp12E, b0: &Fp2E, b2: &Fp2E, b3: &Fp2E) -> Fp12E {
    let [x, y] = f.value();
    let [x0, x1, x2] = x.value();
    let [y0, y1, y2] = y.value();

    // A = X · L_X where L_X = (b0, b2, 0), cost: 5 Fp2 muls
    let v0 = x0 * b0;
    let v1 = x1 * b2;
    // v2 = x2 · 0 = 0
    let a0 = &v0 + mul_fp2_by_nonresidue(&((x1 + x2) * b2 - &v1));
    let a1 = (x0 + x1) * (b0 + b2) - &v0 - &v1;
    let a2 = (x0 + x2) * b0 - &v0 + &v1;

    // B = Y · L_Y where L_Y = (0, b3, 0), cost: 3 Fp2 muls
    // Fp6 product: (y0,y1,y2)·(0,b3,0) = (β·y2·b3, y0·b3, y1·b3)
    let y0b3 = y0 * b3;
    let y1b3 = y1 * b3;
    let y2b3 = y2 * b3;
    let bb0 = mul_fp2_by_nonresidue(&y2b3); // β·(y2·b3)

    // v·B = (β·B[2], B[0], B[1]) = (β·(y1·b3), β·(y2·b3), y0·b3)
    let vb0 = mul_fp2_by_nonresidue(&y1b3);
    // R_X = A + v·B
    let rx0 = &a0 + &vb0; // a0 + β·(y1·b3)
    let rx1 = &a1 + &bb0; // a1 + β·(y2·b3)
    let rx2 = &a2 + &y0b3; // a2 + y0·b3

    // C = (X+Y) · (L_X+L_Y) where L_X+L_Y = (b0, b2+b3, 0), cost: 5 Fp2 muls
    let sx0 = x0 + y0;
    let sx1 = x1 + y1;
    let sx2 = x2 + y2;
    let sb1 = b2 + b3;
    let u0 = &sx0 * b0;
    let u1 = &sx1 * &sb1;
    let b0_plus_sb1 = b0 + &sb1;
    let c0 = &u0 + mul_fp2_by_nonresidue(&((&sx1 + &sx2) * &sb1 - &u1));
    let c1 = (&sx0 + &sx1) * &b0_plus_sb1 - &u0 - &u1;
    let c2 = (&sx0 + &sx2) * b0 - &u0 + &u1;

    // R_Y = C - A - B
    let ry0 = c0 - a0 - bb0;
    let ry1 = c1 - a1 - y0b3;
    let ry2 = c2 - a2 - y1b3;

    Fp12E::new([Fp6E::new([rx0, rx1, rx2]), Fp6E::new([ry0, ry1, ry2])])
}

/// Multiply two sparse line evaluation results.
///
/// Both lines have the form: L_X = (a0, a1, 0), L_Y = (0, a3, 0) in Fp6[w]/(w²-v).
/// Result has sparsity: R_X = (d0, d1, d2) dense, R_Y = (0, d4, d5) with position 0 zero.
///
/// Cost: 6 Fp2 multiplications.
#[inline]
fn mul_line_by_line(a0: &Fp2E, a1: &Fp2E, a3: &Fp2E, c0: &Fp2E, c1: &Fp2E, c3: &Fp2E) -> Fp12E {
    let t0 = a0 * c0;
    let t1 = a1 * c1;
    let t2 = (a0 + a1) * (c0 + c1);
    let t3 = a3 * c3;
    let t4 = (a1 + a3) * (c1 + c3);
    let t5 = (a0 + a1 + a3) * (c0 + c1 + c3);

    // L1_X·L2_X = (t0, t2-t0-t1, t1)
    // L1_Y·L2_Y = (0,b3,0)·(0,c3,0) = (0, 0, t3) in Fp6
    // v·(0,0,t3) = (β·t3, 0, 0)
    // R_X = L1_X·L2_X + v·(L1_Y·L2_Y)
    let rx0 = &t0 + mul_fp2_by_nonresidue(&t3);
    let rx1 = &t2 - &t0 - &t1;
    let rx2 = t1.clone();

    // Cross: (L1_X+L1_Y)·(L2_X+L2_Y) = (a0,a1+a3,0)·(c0,c1+c3,0) = (t0, t5-t0-t4, t4)
    // R_Y = Cross - L1_X·L2_X - L1_Y·L2_Y = (0, t5-t4-t2+t1, t4-t1-t3)
    let ry1 = &t5 - &t4 - &t2 + &t1;
    let ry2 = &t4 - &t1 - &t3;

    Fp12E::new([
        Fp6E::new([rx0, rx1, rx2]),
        Fp6E::new([Fp2E::zero(), ry1, ry2]),
    ])
}

/// Multiply full Fp12 element by the result of mul_line_by_line.
///
/// The sparse element S has the form: S_X = (s0, s1, s2) fully dense,
/// S_Y = (0, s4, s5) with position 0 zero.
///
/// Cost: 17 Fp2 multiplications (vs 18 for full Fp12 mul).
#[inline]
fn mul_fp12_by_line_line(f: &Fp12E, sparse: &Fp12E) -> Fp12E {
    let [x, y] = f.value();
    let [x0, x1, x2] = x.value();
    let [y0, y1, y2] = y.value();

    let [sx, sy] = sparse.value();
    let [s0, s1, s2] = sx.value();
    let [_s3, s4, s5] = sy.value(); // _s3 is zero

    // A = X · S_X (full Fp6 × full Fp6, Karatsuba): 6 Fp2 muls
    let v0 = x0 * s0;
    let v1 = x1 * s1;
    let v2 = x2 * s2;
    let a0 = &v0 + mul_fp2_by_nonresidue(&((x1 + x2) * (s1 + s2) - &v1 - &v2));
    let a1 = (x0 + x1) * (s0 + s1) - &v0 - &v1 + mul_fp2_by_nonresidue(&v2);
    let a2 = (x0 + x2) * (s0 + s2) - &v0 + &v1 - &v2;

    // B = Y · S_Y where S_Y = (0, s4, s5): 5 Fp2 muls
    // Fp6 product: (y0,y1,y2)·(0,s4,s5) = (β·(y1·s5+y2·s4), y0·s4+β·y2·s5, y0·s5+y1·s4)
    let p1 = y1 * s4;
    let p2 = y2 * s5;
    let p3 = (y1 + y2) * (s4 + s5);
    let p4 = y0 * s4;
    let p5 = y0 * s5;
    let b0 = mul_fp2_by_nonresidue(&(&p3 - &p1 - &p2));
    let b1 = &p4 + mul_fp2_by_nonresidue(&p2);
    let b2 = &p5 + &p1;

    // v·B = (β·b2, b0, b1)
    let vb0 = mul_fp2_by_nonresidue(&b2);
    // R_X = A + v·B
    let rx0 = &a0 + &vb0;
    let rx1 = &a1 + &b0;
    let rx2 = &a2 + &b1;

    // C = (X+Y) · (S_X+S_Y): S_X+S_Y = (s0, s1+s4, s2+s5), full Fp6: 6 Fp2 muls
    let ux0 = x0 + y0;
    let ux1 = x1 + y1;
    let ux2 = x2 + y2;
    let us1 = s1 + s4;
    let us2 = s2 + s5;
    let w0 = &ux0 * s0;
    let w1 = &ux1 * &us1;
    let w2 = &ux2 * &us2;
    let us1_plus_us2 = &us1 + &us2;
    let s0_plus_us1 = s0 + &us1;
    let s0_plus_us2 = s0 + &us2;
    let c0 = &w0 + mul_fp2_by_nonresidue(&((&ux1 + &ux2) * &us1_plus_us2 - &w1 - &w2));
    let c1 = (&ux0 + &ux1) * &s0_plus_us1 - &w0 - &w1 + mul_fp2_by_nonresidue(&w2);
    let c2 = (&ux0 + &ux2) * &s0_plus_us2 - &w0 + &w1 - &w2;

    // R_Y = C - A - B
    let ry0 = c0 - a0 - b0;
    let ry1 = c1 - a1 - b1;
    let ry2 = c2 - a2 - b2;

    Fp12E::new([Fp6E::new([rx0, rx1, rx2]), Fp6E::new([ry0, ry1, ry2])])
}

/// Compute doubling step: updates T ← 2T and returns line coefficients (b0, b2, b3).
fn double_step(
    t: &mut ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassJacobianPoint<BLS12381Curve>,
) -> (Fp2E, Fp2E, Fp2E) {
    let [x1, y1, z1] = t.coordinates();
    let [px, py, _] = p.coordinates();

    let a = mul_fp2_by_fp(&TWO_INV_FP, &(x1 * y1));
    let b = y1.square();
    let c = z1.square();
    let d = triple_fp2(&c); // 3c via addition chain
    let e = mul_fp2_by_twist_b(&d); // b'·d using additions only
    let f = triple_fp2(&e); // 3e via addition chain
    let g = mul_fp2_by_fp(&TWO_INV_FP, &(&b + &f));
    let h = (y1 + z1).square() - (&b + &c);

    let x3 = &a * (&b - &f);
    let e_sq = e.square();
    let y3 = g.square() - triple_fp2(&e_sq); // 3*e^2 via addition chain
    let z3 = &b * &h;

    let [h0, h1] = h.value();
    let x1_sq = x1.square();
    let x1_sq_3 = triple_fp2(&x1_sq); // 3*x1^2 via addition chain
    let [x1_sq_30, x1_sq_31] = x1_sq_3.value();

    t.set_unchecked([x3, y3, z3]);

    let b0 = e - b;
    let b2 = Fp2E::new([x1_sq_30 * px, x1_sq_31 * px]);
    let b3 = Fp2E::new([-h0 * py, -h1 * py]);

    (b0, b2, b3)
}

/// Compute addition step: updates T ← T + Q and returns line coefficients (b0, b2, b3).
fn add_step(
    t: &mut ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    q: &ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassJacobianPoint<BLS12381Curve>,
) -> (Fp2E, Fp2E, Fp2E) {
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

    t.set_unchecked([x3, y3, z3]);

    let [lambda0, lambda1] = lambda.value();
    let [theta0, theta1] = theta.value();

    let b0 = -&lambda * y2 + &theta * x2;
    let b2 = Fp2E::new([-theta0 * px, -theta1 * px]);
    let b3 = Fp2E::new([lambda0 * py, lambda1 * py]);

    (b0, b2, b3)
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

    // Specialized Fp4 squaring (2 Fp2 muls each instead of 2S+2M in generic)
    let (v0_0, v0_1) = fp4_square(b0, b4);
    let (v1_0, v1_1) = fp4_square(b3, b2);
    let (v2_0, v2_1) = fp4_square(b1, b5);

    // r = r0 + r1 * w
    // r0 = r00 + r01 * v + r02 * v^2
    // r1 = r10 + r11 * v + r12 * v^2

    // r00 = 3v00 - 2b0
    let mut r00 = &v0_0 - b0;
    r00 = r00.double();
    r00 += v0_0;

    // r01 = 3v10 - 2b1
    let mut r01 = &v1_0 - b1;
    r01 = r01.double();
    r01 += v1_0;

    // r11 = 3v01 + 2b4
    let mut r11 = &v0_1 + b4;
    r11 = r11.double();
    r11 += v0_1;

    // r12 = 3v11 + 2b5
    let mut r12 = &v1_1 + b5;
    r12 = r12.double();
    r12 += v1_1;

    // 3 * β * v21 + 2b3
    let beta_v2_1 = mul_fp2_by_nonresidue(&v2_1);
    let mut r10 = &beta_v2_1 + b3;
    r10 = r10.double();
    r10 += beta_v2_1;

    // r02 = 3v20 - 2b2
    let mut r02 = &v2_0 - b2;
    r02 = r02.double();
    r02 += v2_0;

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

        // Specialized Fp4 squaring (2 Fp2 muls each instead of 2S+2M in generic)
        let (v1_0, v1_1) = fp4_square(g3, g2);
        let (v2_0, v2_1) = fp4_square(g1, g5);

        // g1' = 3*v1_0 - 2*g1
        let mut h1 = &v1_0 - g1;
        h1 = h1.double();
        h1 = &h1 + &v1_0;

        // g2' = 3*v2_0 - 2*g2
        let mut h2 = &v2_0 - g2;
        h2 = h2.double();
        h2 = &h2 + &v2_0;

        // g3' = 3*β*v2_1 + 2*g3
        let beta_v2_1 = mul_fp2_by_nonresidue(&v2_1);
        let mut h3 = &beta_v2_1 + g3;
        h3 = h3.double();
        h3 = &h3 + &beta_v2_1;

        // g5' = 3*v1_1 + 2*g5
        let mut h5 = &v1_1 + g5;
        h5 = h5.double();
        h5 = &h5 + &v1_1;

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

    /// Decompress a compressed cyclotomic element using a precomputed inverse of 4*g3.
    ///
    /// This avoids the per-element Fp2 inversion, enabling batch decompression.
    fn decompress_with_inverse(&self, inv_four_g3: &Fp2E) -> Fp12E {
        let g1 = &self.g1;
        let g2 = &self.g2;
        let g3 = &self.g3;
        let g5 = &self.g5;

        // Recover g4 = (β*g5² + 3*g1² - 2*g2) / (4*g3)
        let g5_sq = g5.square();
        let g1_sq = g1.square();
        let e_g5_sq = mul_fp2_by_nonresidue(&g5_sq);
        let three_g1_sq = triple_fp2(&g1_sq);
        let two_g2 = g2.double();
        let num = &e_g5_sq + &three_g1_sq - &two_g2;
        let g4 = &num * inv_four_g3;

        // Recover g0 = β*(2*g4² + g3*g5 - 3*g2*g1) + 1
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
}

/// Batch-decompress multiple Karabina-compressed cyclotomic elements using
/// a single Fp2 batch inversion (Montgomery's trick).
///
/// Cost: 1 Fp2 inversion + 3*(n-1) Fp2 muls for the batch,
/// vs n individual inversions without batching. Saves (n-1) Fp2 inversions.
#[cfg(feature = "alloc")]
fn batch_decompress_karabina(elements: &[CompressedCyclotomic]) -> Vec<Fp12E> {
    if elements.is_empty() {
        return Vec::new();
    }

    // Collect denominators: 4*g3 for each element
    let mut denominators: Vec<Fp2E> = elements.iter().map(|c| c.g3.double().double()).collect();

    // Batch invert all denominators
    Fp2E::inplace_batch_inverse(&mut denominators)
        .expect("g3 should be nonzero for cyclotomic subgroup elements");

    // Decompress each element using the pre-computed inverse
    elements
        .iter()
        .zip(denominators.iter())
        .map(|(c, inv)| c.decompress_with_inverse(inv))
        .collect()
}

/// Computes f^|X| using a right-to-left decomposition with all-compressed squarings.
///
/// |X| = 0xd201000000010000 = 2^16 + 2^48 + 2^57 + 2^60 + 2^62 + 2^63
///
/// Instead of MSB-to-LSB scanning (which requires regular cyclotomic squares between
/// multiplications), we compute all 63 squarings as a single chain of cheap Karabina
/// compressed squares, saving checkpoints at each set-bit position. At the end, we
/// batch-decompress all checkpoints using a single Fp2 inversion, then multiply.
///
/// Cost: 63 Karabina compressed squares + 1 batch decompression (1 Fp2 inv for 6 elements)
///       + 5 Fp12 multiplications.
pub fn cyclotomic_pow_x_compressed(f: &Fp12E) -> Fp12E {
    // Early return for identity (all compressed components are zero)
    if *f == Fp12E::one() {
        return Fp12E::one();
    }

    // |x| = 2^16 + 2^48 + 2^57 + 2^60 + 2^62 + 2^63
    let mut compressed = CompressedCyclotomic::compress(f);

    // Square 16 times → f^(2^16)
    for _ in 0..16 {
        compressed = compressed.square();
    }
    let c16 = compressed.clone();

    // Square 32 more → f^(2^48)
    for _ in 0..32 {
        compressed = compressed.square();
    }
    let c48 = compressed.clone();

    // Square 9 more → f^(2^57)
    for _ in 0..9 {
        compressed = compressed.square();
    }
    let c57 = compressed.clone();

    // Square 3 more → f^(2^60)
    for _ in 0..3 {
        compressed = compressed.square();
    }
    let c60 = compressed.clone();

    // Square 2 more → f^(2^62)
    for _ in 0..2 {
        compressed = compressed.square();
    }
    let c62 = compressed.clone();

    // Square 1 more → f^(2^63)
    compressed = compressed.square();
    let c63 = compressed;

    // Batch decompress all 6 checkpoints (1 Fp2 inversion total)
    let decompressed = batch_decompress_karabina(&[c16, c48, c57, c60, c62, c63]);

    // Multiply all checkpoints: f^(2^16) * f^(2^48) * f^(2^57) * f^(2^60) * f^(2^62) * f^(2^63)
    let mut result = &decompressed[0] * &decompressed[1];
    result = &result * &decompressed[2];
    result = &result * &decompressed[3];
    result = &result * &decompressed[4];
    &result * &decompressed[5]
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
