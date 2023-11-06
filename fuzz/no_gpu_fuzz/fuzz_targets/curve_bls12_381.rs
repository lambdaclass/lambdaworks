#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::{
    field::element::FieldElement,
    cyclic_group::IsGroup,
    elliptic_curve::{
        traits::{IsEllipticCurve, EllipticCurveError, IsPairing},
        short_weierstrass::{
            curves::bls12_381::{
                curve::{BLS12381FieldElement, BLS12381TwistCurveFieldElement, BLS12381Curve}, 
                twist::BLS12381TwistCurve,
                pairing::BLS12381AtePairing,
                field_extension::Degree12ExtensionField,
            },
            point::ShortWeierstrassProjectivePoint,
        }
    },
    unsigned_integer::element::U384,
    traits::ByteConversion,
};

use ark_bls12_381::{Fr, Fq, G1Projective as G1, G2Projective as G2};
use ark_ec::{CurveGroup, short_weierstrass::Affine};
use ark_ff::{PrimeField, QuadExtField, BigInteger};


type FeG1 = BLS12381FieldElement;
type FeG2 = BLS12381TwistCurveFieldElement;
type LambdaG1 = ShortWeierstrassProjectivePoint<BLS12381Curve>;
type LambdaG2 = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;

fn create_g1_points(points: (u64, u64, u64, u64)) -> Result<(LambdaG1, LambdaG1), EllipticCurveError> {
    let (val_a_x, val_a_y, val_b_x, val_b_y) = points;

    let a_x =  FeG1::from(val_a_x);
    let a_y =  FeG1::from(val_a_y);

    let b_x =  FeG1::from(val_b_x);
    let b_y =  FeG1::from(val_b_y);

    // Create G1 points
    let a = BLS12381Curve::create_point_from_affine(a_x, a_y)?;
    let b = BLS12381Curve::create_point_from_affine(b_x, b_y)?;
    Ok((a, b))
}

fn create_g2_points(points: (u64, u64, u64, u64)) -> Result<(LambdaG2, LambdaG2), EllipticCurveError> {
    let (val_a_x, val_a_y, val_b_x, val_b_y) = points;

    let a_x =  FeG2::from(val_a_x);
    let a_y =  FeG2::from(val_a_y);

    let b_x =  FeG2::from(val_b_x);
    let b_y =  FeG2::from(val_b_y);

    // Create G2 points
    let a = BLS12381TwistCurve::create_point_from_affine(a_x, a_y)?;
    let b = BLS12381TwistCurve::create_point_from_affine(b_x, b_y)?;
    Ok((a, b))
}

fn create_ark_g1_points(points: (u64, u64, u64, u64)) -> (G1, G1) {
    let (val_a_x, val_a_y, val_b_x, val_b_y) = points;

    let a = G1::from(
        Affine::new(
            Fq::from(val_a_x),
            Fq::from(val_a_y)
        )
    );
    let b = G1::from(
        Affine::new(
            Fq::from(val_b_x),
            Fq::from(val_b_y)
        )
    );

    (a, b)
}

fn create_ark_g2_points(points: (u64, u64, u64, u64)) -> (G2, G2) {
    let (val_a_x, val_a_y, val_b_x, val_b_y) = points;

    let a = G2::from(
        Affine::new(
            QuadExtField::from(val_a_x),
            QuadExtField::from(val_a_y)
        )
    );
    let b = G2::from(
        Affine::new(
            QuadExtField::from(val_b_x),
            QuadExtField::from(val_b_y)
        )
    );

    (a, b)
}

fn g1_equal(lambda_g1: &LambdaG1, ark_g1: &G1) {
    assert_eq!(lambda_g1.x().to_bytes_be(), ark_g1.x.into_bigint().to_bytes_be());
    assert_eq!(lambda_g1.y().to_bytes_be(), ark_g1.y.into_bigint().to_bytes_be());
    assert_eq!(lambda_g1.z().to_bytes_be(), ark_g1.z.into_bigint().to_bytes_be());
}

fn g2_equal(lambda_g2: &LambdaG2, ark_g2: &G2) {
    // https://github.com/arkworks-rs/algebra/blob/master/ff/src/fields/models/quadratic_extension.rs#L106
    assert_eq!(lambda_g2.x().value()[0].to_bytes_be(), ark_g2.x.c0.into_bigint().to_bytes_be());
    assert_eq!(lambda_g2.x().value()[1].to_bytes_be(), ark_g2.x.c1.into_bigint().to_bytes_be());
    assert_eq!(lambda_g2.y().value()[0].to_bytes_be(), ark_g2.y.c0.into_bigint().to_bytes_be());
    assert_eq!(lambda_g2.y().value()[1].to_bytes_be(), ark_g2.y.c1.into_bigint().to_bytes_be());
    assert_eq!(lambda_g2.z().value()[0].to_bytes_be(), ark_g2.z.c0.into_bigint().to_bytes_be());
    assert_eq!(lambda_g2.z().value()[1].to_bytes_be(), ark_g2.z.c1.into_bigint().to_bytes_be());
}

//TODO: derive arbitrary for Affine and Projective or change this to use &[u8] as input to cover more cases.
//TODO: use more advanced options to generate values over curve specifically given most inputs will fail curve check.
//NOTE: Ark serialize for QuadExtField compresses the first coeff.
fuzz_target!(|values: ((u64, u64, u64, u64), (u64, u64, u64, u64))| {

    let (g1_points, g2_points) = values;

    // Create G1 Lambdaworks points
    let (a_g1, b_g1) = match create_g1_points(g1_points) {
        Ok(v) => v,
        Err(_) => return
    };

    // Create G2 Lambdaworks points
    let (a_g2, b_g2) = match create_g2_points(g2_points) {
        Ok(v) => v,
        Err(_) => return
    };

    // Create G1 Arkworks points
    let (a_expected_g1, b_expected_g1) = create_ark_g1_points(g1_points);

    // Create G2 Arkworks points
    let (a_expected_g2, b_expected_g2) = create_ark_g2_points(g2_points);

    // ***OPERATION SOUNDNESS***
    // op_with
    let op_with_g1 = a_g1.operate_with(&a_g1);
    let op_with_g2 = a_g2.operate_with(&a_g2);
    let operate_with_g1 = a_expected_g1 + b_expected_g1;
    let operate_with_g2 = a_expected_g2 + b_expected_g2;
    // op_with G1
    g1_equal(&op_with_g1, &operate_with_g1);
    // op_with() G2
    g2_equal(&op_with_g2, &operate_with_g2);


    // Neg()
    let neg_g1 = a_g1.neg();
    let neg_g2 = a_g2.neg();
    let negative_g1 = -a_expected_g1;
    let negative_g2 = -a_expected_g2;
    // Neg() G1
    g1_equal(&neg_g1, &negative_g1);
    // Neg() G2
    g2_equal(&neg_g2, &negative_g2);
    
    // -
    let sub_g1 = a_g1.operate_with(&b_g1.neg());
    let sub_g2 = a_g2.operate_with(&b_g2.neg());
    let subtraction_g1 = &a_expected_g1 - &b_expected_g1;
    let subtraction_g2 = &a_expected_g2 - &b_expected_g2;
    // - G1
    g1_equal(&sub_g1, &subtraction_g1);
    // - G2
    g2_equal(&sub_g2, &subtraction_g2);


    // operate_with_self
    let op_with_self_g1 = a_g1.operate_with_self(g1_points.0);
    let op_with_self_g2 = a_g2.operate_with_self(g2_points.0);
    let operate_with_self_g1 = a_expected_g1 * Fr::from(g1_points.0);
    let operate_with_self_g2 = a_expected_g2 * Fr::from(g2_points.0);
    // operate_with_self G1
    g1_equal(&op_with_self_g1, &operate_with_self_g1);
    // operate_with_self G2
    g2_equal(&op_with_self_g2, &operate_with_self_g2);

    //to_affine()
    let to_aff_g1 = a_g1.to_affine();
    let to_aff_g2 = a_g2.to_affine();
    let to_affine_g1 = a_expected_g1.into_affine();
    let to_affine_g2 = a_expected_g2.into_affine();

    // to_affine() G1
    assert_eq!(to_aff_g1.x().to_bytes_be(), to_affine_g1.x.into_bigint().to_bytes_be());
    assert_eq!(to_aff_g1.y().to_bytes_be(), to_affine_g1.y.into_bigint().to_bytes_be());

    // to_affine() G2
    assert_eq!(to_aff_g2.x().value()[0].to_bytes_be(), to_affine_g2.x.c0.into_bigint().to_bytes_be());
    assert_eq!(to_aff_g2.x().value()[1].to_bytes_be(), to_affine_g2.x.c1.into_bigint().to_bytes_be());
    assert_eq!(to_aff_g2.y().value()[0].to_bytes_be(), to_affine_g2.y.c0.into_bigint().to_bytes_be());
    assert_eq!(to_aff_g2.y().value()[1].to_bytes_be(), to_affine_g2.y.c1.into_bigint().to_bytes_be());

    // ***AXIOM SOUNDNESS***
    let g1_zero = LambdaG1::neutral_element();

    let g2_zero = LambdaG2::neutral_element();

    // G1
    // -O = O
    assert_eq!(g1_zero.neg(), g1_zero, "Neutral mul element a failed");

    // P * O = O
    assert_eq!(a_g1.operate_with(&g1_zero), a_g1, "Neutral operate_with element a failed");
    assert_eq!(b_g1.operate_with(&g1_zero), b_g1, "Neutral operate_with element b failed");

    // P * Q = Q * P
    assert_eq!(a_g1.operate_with(&b_g1), b_g1.operate_with(&a_g1), "Commutative add property failed");

    // (P * Q) * R = Q * (P * R)
    let c_g1 = a_g1.operate_with(&b_g1);
    assert_eq!((a_g1.operate_with(&b_g1)).operate_with(&c_g1), a_g1.operate_with(&b_g1.operate_with(&c_g1)), "Associative operate_with property failed");

    // P * -P = O
    assert_eq!(a_g1.operate_with(&a_g1.neg()), g1_zero, "Inverse add a failed");
    assert_eq!(b_g1.operate_with(&b_g1.neg()), g1_zero, "Inverse add b failed");

    // G2
    // -O = O
    assert_eq!(g2_zero.neg(), g2_zero, "Neutral mul element a failed");

    // P * O = O
    assert_eq!(a_g2.operate_with(&g2_zero), a_g2, "Neutral operate_with element a failed");
    assert_eq!(b_g2.operate_with(&g2_zero), b_g2, "Neutral operate_with element b failed");

    // P * Q = Q * P
    assert_eq!(a_g2.operate_with(&b_g2), b_g2.operate_with(&a_g2), "Commutative add property failed");

    // (P * Q) * R = Q * (P * R)
    let c_g2 = a_g2.operate_with(&b_g2);
    assert_eq!((a_g2.operate_with(&b_g2)).operate_with(&c_g2), a_g2.operate_with(&b_g2.operate_with(&c_g2)), "Associative operate_with property failed");

    // P * -P = O
    assert_eq!(a_g2.operate_with(&a_g2.neg()), g2_zero, "Inverse add a failed");
    assert_eq!(b_g2.operate_with(&b_g2.neg()), g2_zero, "Inverse add b failed");

    // Pairing Bilinearity
    let a = U384::from_u64(11);
    let b = U384::from_u64(93);
    let result = BLS12381AtePairing::compute_batch(&[
        (
            &a_g1.operate_with_self(a).to_affine(),
            &a_g2.operate_with_self(b).to_affine(),
        ),
        (
            &a_g1.operate_with_self(a * b).to_affine(),
            &a_g2.neg().to_affine(),
        ),
    ]);
    assert_eq!(result, FieldElement::<Degree12ExtensionField>::one());

    // Ate Pairing returns one with one element is neutral element
    let result = BLS12381AtePairing::compute_batch(&[(&a_g1.to_affine(), &LambdaG2::neutral_element())]);
    assert_eq!(result, FieldElement::<Degree12ExtensionField>::one());

    let result = BLS12381AtePairing::compute_batch(&[(&LambdaG1::neutral_element(), &a_g2.to_affine())]);
    assert_eq!(result, FieldElement::<Degree12ExtensionField>::one());
});
