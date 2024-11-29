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
        curves::bls12_381::field_extension::{Degree6ExtensionField, LevelTwoResidue},
        point::ShortWeierstrassProjectivePoint,
        traits::IsShortWeierstrass,
    },
    field::{element::FieldElement, extensions::cubic::HasCubicNonResidue},
    unsigned_integer::element::U256,
};

type FpE = FieldElement<BLS12381PrimeField>;
type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp4E = FieldElement<Degree4ExtensionField>;
type Fp6E = FieldElement<Degree6ExtensionField>;
type Fp12E = FieldElement<Degree12ExtensionField>;

pub const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
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
    type G1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;
    type G2Point = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;
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
        Ok(final_exponentiation(&result))
    }
}

/// Implements the miller loop for the ate pairing of the BLS12 381 curve.
/// Based on algorithm 9.2, page 212 of the book
/// "Topics in computational number theory" by W. Bons and K. Lenstra
#[allow(unused)]
pub fn miller(
    q: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
) -> FieldElement<Degree12ExtensionField> {
    let mut r = q.clone();
    let mut f = FieldElement::<Degree12ExtensionField>::one();
    X_BINARY.iter().skip(1).for_each(|bit| {
        double_accumulate_line(&mut r, p, &mut f);
        if *bit {
            add_accumulate_line(&mut r, q, p, &mut f);
        }
    });

    f.conjugate()
}

fn double_accumulate_line(
    t: &mut ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
    let [x1, y1, z1] = t.coordinates();
    let [px, py, _] = p.coordinates();
    let residue = LevelTwoResidue::residue();
    let two_inv = FieldElement::<Degree2ExtensionField>::new_base("d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd556");
    let three = FieldElement::<BLS12381PrimeField>::from(3);

    let a = &two_inv * x1 * y1;
    let b = y1.square();
    let c = z1.square();
    let d = &three * &c;
    let e = BLS12381TwistCurve::b() * d;
    let f = &three * &e;
    let g = two_inv * (&b + &f);
    let h = (y1 + z1).square() - (&b + &c);

    let x3 = &a * (&b - &f);
    let y3 = g.square() - (&three * e.square());
    let z3 = &b * &h;

    let [h0, h1] = h.value();
    let x1_sq_3 = three * x1.square();
    let [x1_sq_30, x1_sq_31] = x1_sq_3.value();

    t.0.value = [x3, y3, z3];

    // (a0 + a2w2 + a4w4 + a1w + a3w3 + a5w5) * (b0 + b2 w2 + b3 w3) =
    // (a0b0 + r (a3b3 + a4b2)) w0 + (a1b0 + r (a4b3 + a5b2)) w
    // (a2b0 + r  a5b3 + a0b2 ) w2 + (a3b0 + a0b3 + a1b2    ) w3
    // (a4b0 +    a1b3 + a2b2 ) w4 + (a5b0 + a2b3 + a3b2    ) w5
    let accumulator_sq = accumulator.square();
    let [x, y] = accumulator_sq.value();
    let [a0, a2, a4] = x.value();
    let [a1, a3, a5] = y.value();
    let b0 = e - b;
    let b2 = FieldElement::new([x1_sq_30 * px, x1_sq_31 * px]);
    let b3 = FieldElement::<Degree2ExtensionField>::new([-h0 * py, -h1 * py]);
    *accumulator = FieldElement::new([
        FieldElement::new([
            a0 * &b0 + &residue * (a3 * &b3 + a4 * &b2), // w0
            a2 * &b0 + &residue * a5 * &b3 + a0 * &b2,   // w2
            a4 * &b0 + a1 * &b3 + a2 * &b2,              // w4
        ]),
        FieldElement::new([
            a1 * &b0 + &residue * (a4 * &b3 + a5 * &b2), // w1
            a3 * &b0 + a0 * &b3 + a1 * &b2,              // w3
            a5 * &b0 + a2 * &b3 + a3 * &b2,              // w5
        ]),
    ]);
}

fn add_accumulate_line(
    t: &mut ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    q: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
    let [x1, y1, z1] = t.coordinates();
    let [x2, y2, _] = q.coordinates();
    let [px, py, _] = p.coordinates();
    let residue = LevelTwoResidue::residue();

    let a = y2 * z1;
    let b = x2 * z1;
    let theta = y1 - a;
    let lambda = x1 - b;
    let c = theta.square();
    let d = lambda.square();
    let e = &lambda * &d;
    let f = z1 * c;
    let g = x1 * d;
    let h = &e + f - FieldElement::<BLS12381PrimeField>::from(2) * &g;
    let i = y1 * &e;

    let x3 = &lambda * &h;
    let y3 = &theta * (g - h) - i;
    let z3 = z1 * e;

    t.0.value = [x3, y3, z3];

    let [lambda0, lambda1] = lambda.value();
    let [theta0, theta1] = theta.value();

    let [x, y] = accumulator.value();
    let [a0, a2, a4] = x.value();
    let [a1, a3, a5] = y.value();
    let b0 = -lambda.clone() * y2 + theta.clone() * x2;
    let b2 = FieldElement::new([-theta0 * px, -theta1 * px]);
    let b3 = FieldElement::<Degree2ExtensionField>::new([lambda0 * py, lambda1 * py]);
    *accumulator = FieldElement::new([
        FieldElement::new([
            a0 * &b0 + &residue * (a3 * &b3 + a4 * &b2), // w0
            a2 * &b0 + &residue * a5 * &b3 + a0 * &b2,   // w2
            a4 * &b0 + a1 * &b3 + a2 * &b2,              // w4
        ]),
        FieldElement::new([
            a1 * &b0 + &residue * (a4 * &b3 + a5 * &b2), // w1
            a3 * &b0 + a0 * &b3 + a1 * &b2,              // w3
            a5 * &b0 + a2 * &b3 + a3 * &b2,              // w5
        ]),
    ]);
}

// To understand more about how to reduce the final exponentiation
// read "Efficient Final Exponentiation via Cyclotomic Structure for
// Pairings over Families of Elliptic Curves" (https://eprint.iacr.org/2020/875.pdf)
// Hard part from https://eprint.iacr.org/2020/875.pdf p14
// Same implementation as in Constantine https://github.com/mratsim/constantine/blob/master/constantine/math/pairings/pairings_bls12.nim
pub fn final_exponentiation(f: &Fp12E) -> Fp12E {
    let f_easy_aux = f.conjugate() * f.inv().unwrap();
    let mut f_easy = frobenius_square(&f_easy_aux) * &f_easy_aux;

    let mut v2 = cyclotomic_square(&f_easy); // v2 = f²
    let mut v0 = cyclotomic_pow_x(&f_easy).conjugate(); //  v0 = f^x
    let mut v1 = f_easy.conjugate(); // v1 = f^-1

    //  (x−1)²
    v0 *= v1; // v0 = f^(x-1)
    v1 = cyclotomic_pow_x(&v0).conjugate(); // v1 = (f^(x-1))^(x)

    v0 = v0.conjugate(); // v0 = (f^(x-1))^(-1)
    v0 *= &v1; // v0 = (f^(x-1))^(-1) * (f^(x-1))^x = (f^(x-1))^(x-1) =  f^((x-1)²)

    // (x+p)
    v1 = cyclotomic_pow_x(&v0).conjugate(); // v1 = f^((x-1)².x)
    v0 = frobenius(&v0); // f^((x-1)².p)
    v0 *= &v1; // f^((x-1)².p + (x-1)².x) = f^((x-1)².(x+p))

    // + 3
    f_easy *= v2; // f^3

    // (x²+p²−1)
    v2 = cyclotomic_pow_x(&v0).conjugate();
    v1 = cyclotomic_pow_x(&v2).conjugate(); // v1 = f^((x-1)².(x+p).x²)
    v2 = frobenius_square(&v0); // v2 = f^((x-1)².(x+p).p²)
    v0 = v0.conjugate(); // v0 = f^((x-1)².(x+p).-1)
    v0 *= &v1; // v0 = f^((x-1)².(x+p).(x²-1))
    v0 *= &v2; // v0 = f^((x-1)².(x+p).(x²+p²-1))

    f_easy *= &v0;
    f_easy
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
/// Algorithm from Constantine's cyclotomic_square_quad_over_cube
/// https://github.com/mratsim/constantine/blob/master/constantine/math/pairings/cyclotomic_subgroups.nim#L354
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

#[allow(clippy::needless_range_loop)]
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
#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve,
        unsigned_integer::element::U384,
    };

    use super::*;

    #[test]
    fn test_double_accumulate_line_doubles_point_correctly() {
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let mut r = g2.clone();
        let mut f = FieldElement::one();
        double_accumulate_line(&mut r, &g1, &mut f);
        assert_eq!(r, g2.operate_with(&g2));
    }

    #[test]
    fn test_add_accumulate_line_adds_points_correctly() {
        let g1 = BLS12381Curve::generator();
        let g = BLS12381TwistCurve::generator();
        let a: u64 = 12;
        let b: u64 = 23;
        let g2 = g.operate_with_self(a).to_affine();
        let g3 = g.operate_with_self(b).to_affine();
        let expected = g.operate_with_self(a + b);
        let mut r = g2;
        let mut f = FieldElement::one();
        add_accumulate_line(&mut r, &g3, &g1, &mut f);
        assert_eq!(r, expected);
    }

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
        let q = ShortWeierstrassProjectivePoint::neutral_element();
        let result = BLS12381AtePairing::compute_batch(&[(&p.to_affine(), &q)]).unwrap();
        assert_eq!(result, FieldElement::one());

        let p = ShortWeierstrassProjectivePoint::neutral_element();
        let q = BLS12381TwistCurve::generator();
        let result = BLS12381AtePairing::compute_batch(&[(&p, &q.to_affine())]).unwrap();
        assert_eq!(result, FieldElement::one());
    }

    #[test]
    fn ate_pairing_errors_when_one_element_is_not_in_subgroup() {
        let p = ShortWeierstrassProjectivePoint::new([
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::one(),
        ]);
        let q = ShortWeierstrassProjectivePoint::neutral_element();
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
    fn test_double_accumulate_line_doubles_point_correctl_2() {
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let mut r = g2.clone();
        let mut f = FieldElement::one();
        double_accumulate_line(&mut r, &g1, &mut f);
        let expected_r = g2.operate_with(&g2);
        assert_eq!(r.to_affine(), expected_r.to_affine());
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
}
