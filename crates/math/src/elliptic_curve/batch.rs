//! Batch operations for elliptic curves using Montgomery's trick.
//!
//! This module provides efficient batch operations that significantly reduce the number of
//! field inversions required when processing multiple elliptic curve points.
//!
//! # Montgomery's Batch Inversion Algorithm
//!
//! Given elements `[a_1, a_2, ..., a_n]`, compute `[1/a_1, 1/a_2, ..., 1/a_n]` with only ONE
//! field inversion (instead of `n` inversions):
//!
//! 1. Compute prefix products: `p_i = a_1 * a_2 * ... * a_i`
//! 2. Invert the final product: `inv = 1/(a_1 * a_2 * ... * a_n)`
//! 3. Compute inverses backwards using the identity: `1/a_i = inv * p_{i-1}`
//!
//! ## Complexity
//! - Standard approach: `n` inversions + 0 multiplications = `O(n * I)` where `I` is inversion cost
//! - Montgomery's trick: 1 inversion + `3(n-1)` multiplications = `O(I + 3n * M)` where `M` is multiplication cost
//!
//! Since inversion typically costs 50-100x more than multiplication, this provides ~5x speedup
//! for large batches.
//!
//! # Usage
//!
//! ```ignore
//! use lambdaworks_math::elliptic_curve::batch::batch_normalize;
//! use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
//!
//! // Convert many projective points to affine efficiently
//! let affine_points = batch_normalize(&projective_points);
//! ```
//!
//! # References
//!
//! - Montgomery, P. L. "Speeding the Pollard and elliptic curve methods of factorization"
//! - Cohen, H. "A Course in Computational Algebraic Number Theory", Algorithm 10.3.4

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;
use crate::field::element::FieldElement;

/// Batch normalize (convert to affine) multiple Short Weierstrass projective points.
///
/// This function uses Montgomery's batch inversion trick to convert many projective
/// points to affine representation using only a single field inversion.
///
/// # Algorithm
///
/// For projective points `(X_i : Y_i : Z_i)`, affine coordinates are `(X_i/Z_i, Y_i/Z_i)`.
/// Instead of computing `n` inversions of `Z_i`, we:
/// 1. Collect all non-zero `Z` coordinates
/// 2. Batch invert them using Montgomery's trick
/// 3. Apply the inversions to get affine coordinates
///
/// # Arguments
///
/// * `points` - A slice of projective points to normalize
///
/// # Returns
///
/// A vector of affine points (with Z=1). Points at infinity remain unchanged.
///
/// # Performance
///
/// - Speedup: ~3-5x faster than individual `to_affine()` calls for large batches
/// - Memory: Allocates O(n) temporary storage for Z coordinates
///
/// # Example
///
/// ```ignore
/// let projective_points: Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>> = ...;
/// let affine_points = batch_normalize(&projective_points);
/// assert!(affine_points.iter().all(|p| p.z() == &FieldElement::one() || p.is_neutral_element()));
/// ```
#[cfg(feature = "alloc")]
pub fn batch_normalize_sw<E: IsShortWeierstrass>(
    points: &[ShortWeierstrassProjectivePoint<E>],
) -> Vec<ShortWeierstrassProjectivePoint<E>> {
    ShortWeierstrassProjectivePoint::batch_to_affine(points)
}

/// Efficiently add multiple affine points to a single accumulator.
///
/// This function accumulates a sequence of points into a single result using
/// mixed addition (affine + projective), which is faster than general projective addition.
///
/// # Algorithm
///
/// Uses iterative mixed addition:
/// ```text
/// result = O (neutral element)
/// for each point p in points:
///     result = result + p  (using mixed addition)
/// ```
///
/// Mixed addition (affine + projective) costs ~11M vs ~16M for projective + projective,
/// providing roughly 30% speedup per addition.
///
/// # Arguments
///
/// * `points` - A slice of affine points (Z coordinate should be 1)
///
/// # Returns
///
/// The sum of all input points as a projective point.
///
/// # Performance
///
/// For `n` points:
/// - This function: `O(11n * M)` field operations
/// - Naive approach with projective addition: `O(16n * M)` field operations
///
/// # Example
///
/// ```ignore
/// let affine_points: Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>> = ...;
/// let sum = batch_add_sw(&affine_points);
/// ```
#[cfg(feature = "alloc")]
pub fn batch_add_sw<E: IsShortWeierstrass>(
    points: &[ShortWeierstrassProjectivePoint<E>],
) -> ShortWeierstrassProjectivePoint<E> {
    if points.is_empty() {
        return ShortWeierstrassProjectivePoint::neutral_element();
    }

    let mut accumulator = ShortWeierstrassProjectivePoint::neutral_element();
    for point in points {
        // Use operate_with_affine for mixed addition (faster when point is affine)
        accumulator = accumulator.operate_with_affine(point);
    }
    accumulator
}

/// Batch normalize Jacobian points to affine representation.
///
/// For Jacobian coordinates `(X : Y : Z)`, affine coordinates are `(X/Z^2, Y/Z^3)`.
/// This function efficiently computes these conversions using batch inversion.
///
/// # Algorithm
///
/// 1. Compute `Z^2` and `Z^3` for each point
/// 2. Batch invert all `Z^3` values (we only need Z^3 since Z^2 = Z^3 * Z^{-1})
/// 3. Apply inversions:
///    - `x_affine = X * (Z^3)^{-1} * Z = X * Z^{-2}`
///    - `y_affine = Y * (Z^3)^{-1}`
///
/// # Arguments
///
/// * `points` - A slice of Jacobian points
///
/// # Returns
///
/// A vector of affine points (Z=1). Points at infinity remain unchanged.
#[cfg(feature = "alloc")]
pub fn batch_normalize_jacobian<E: IsShortWeierstrass>(
    points: &[crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint<E>],
) -> Vec<crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint<E>> {
    use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;

    if points.is_empty() {
        return Vec::new();
    }

    // Collect Z^3 values for non-infinity points
    let mut z_cubes: Vec<FieldElement<E::BaseField>> = Vec::with_capacity(points.len());
    let mut z_values: Vec<FieldElement<E::BaseField>> = Vec::with_capacity(points.len());
    let mut indices: Vec<usize> = Vec::with_capacity(points.len());

    for (i, point) in points.iter().enumerate() {
        if !point.is_neutral_element() {
            let z = point.z();
            let z_sq = z.square();
            let z_cu = &z_sq * z;
            z_cubes.push(z_cu);
            z_values.push(z.clone());
            indices.push(i);
        }
    }

    // Batch invert all Z^3 values
    if FieldElement::<E::BaseField>::inplace_batch_inverse(&mut z_cubes).is_err() {
        // Fall back to individual conversion
        return points.iter().map(|p| p.to_affine()).collect();
    }

    // Build result vector
    let mut result: Vec<ShortWeierstrassJacobianPoint<E>> = Vec::with_capacity(points.len());
    let mut inv_idx = 0;

    for point in points.iter() {
        if point.is_neutral_element() {
            result.push(ShortWeierstrassJacobianPoint::neutral_element());
        } else {
            // z_inv_cubed = 1/Z^3
            // z_inv_squared = z_inv_cubed * Z = 1/Z^2
            let z_inv_cubed = &z_cubes[inv_idx];
            let z_inv_squared = z_inv_cubed * &z_values[inv_idx];

            let [x, y, _z] = point.coordinates();
            // x_affine = X * Z^{-2}
            let x_affine = x * &z_inv_squared;
            // y_affine = Y * Z^{-3}
            let y_affine = y * z_inv_cubed;

            result.push(ShortWeierstrassJacobianPoint::new_unchecked([
                x_affine,
                y_affine,
                FieldElement::one(),
            ]));
            inv_idx += 1;
        }
    }

    result
}

/// Batch add for Jacobian points using mixed addition.
///
/// Efficiently accumulates affine points into a Jacobian result using mixed addition,
/// which is significantly faster than pure Jacobian addition.
#[cfg(feature = "alloc")]
pub fn batch_add_jacobian<E: IsShortWeierstrass>(
    points: &[crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint<E>],
) -> crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint<E> {
    use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;

    if points.is_empty() {
        return ShortWeierstrassJacobianPoint::neutral_element();
    }

    let mut accumulator = ShortWeierstrassJacobianPoint::neutral_element();
    for point in points {
        accumulator = accumulator.operate_with_affine(point);
    }
    accumulator
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cyclic_group::IsGroup;
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;
    use crate::elliptic_curve::short_weierstrass::point::{
        ShortWeierstrassJacobianPoint, ShortWeierstrassProjectivePoint,
    };
    use crate::elliptic_curve::traits::FromAffine;
    use crate::field::element::FieldElement;

    type FE = FieldElement<BLS12381PrimeField>;
    type G1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;
    type G1JacobianPoint = ShortWeierstrassJacobianPoint<BLS12381Curve>;

    fn sample_point() -> G1Point {
        let x = FE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        G1Point::from_affine(x, y).expect("sample_point: hardcoded coordinates must be valid")
    }

    fn sample_jacobian_point() -> G1JacobianPoint {
        let x = FE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        G1JacobianPoint::from_affine(x, y)
            .expect("sample_jacobian_point: hardcoded coordinates must be valid")
    }

    // ==================== batch_normalize_sw tests ====================

    #[test]
    fn test_batch_normalize_sw_empty() {
        let points: Vec<G1Point> = vec![];
        let result = batch_normalize_sw(&points);
        assert!(result.is_empty());
    }

    #[test]
    fn test_batch_normalize_sw_single_point() {
        let p = sample_point().operate_with_self(3_u16);
        let points = vec![p.clone()];
        let result = batch_normalize_sw(&points);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], p.to_affine());
        assert_eq!(result[0].z(), &FE::one());
    }

    #[test]
    fn test_batch_normalize_sw_multiple_points() {
        let p = sample_point();
        let points: Vec<G1Point> = (1..=10).map(|i| p.operate_with_self(i as u16)).collect();

        let batch_result = batch_normalize_sw(&points);
        let individual_result: Vec<G1Point> = points.iter().map(|p| p.to_affine()).collect();

        assert_eq!(batch_result.len(), individual_result.len());
        for (batch, individual) in batch_result.iter().zip(individual_result.iter()) {
            assert_eq!(batch, individual);
            assert_eq!(batch.z(), &FE::one());
        }
    }

    #[test]
    fn test_batch_normalize_sw_with_neutral_element() {
        let p = sample_point();
        let neutral = G1Point::neutral_element();

        let points = vec![
            p.clone(),
            neutral.clone(),
            p.operate_with_self(2_u16),
            neutral.clone(),
            p.operate_with_self(3_u16),
        ];

        let result = batch_normalize_sw(&points);

        assert_eq!(result.len(), 5);
        assert_eq!(result[0], points[0].to_affine());
        assert!(result[1].is_neutral_element());
        assert_eq!(result[2], points[2].to_affine());
        assert!(result[3].is_neutral_element());
        assert_eq!(result[4], points[4].to_affine());
    }

    #[test]
    fn test_batch_normalize_sw_all_neutral_elements() {
        let neutral = G1Point::neutral_element();
        let points = vec![neutral.clone(), neutral.clone(), neutral.clone()];

        let result = batch_normalize_sw(&points);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|p| p.is_neutral_element()));
    }

    // ==================== batch_add_sw tests ====================

    #[test]
    fn test_batch_add_sw_empty() {
        let points: Vec<G1Point> = vec![];
        let result = batch_add_sw(&points);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn test_batch_add_sw_single_point() {
        let p = sample_point().to_affine();
        let points = vec![p.clone()];
        let result = batch_add_sw(&points);
        assert_eq!(result.to_affine(), p);
    }

    #[test]
    fn test_batch_add_sw_multiple_points() {
        let p = sample_point().to_affine();
        let affine_points: Vec<G1Point> = (1..=5)
            .map(|i| p.operate_with_self(i as u16).to_affine())
            .collect();

        let batch_result = batch_add_sw(&affine_points);

        // Compute expected result individually
        let mut expected = G1Point::neutral_element();
        for point in &affine_points {
            expected = expected.operate_with(point);
        }

        assert_eq!(batch_result.to_affine(), expected.to_affine());
    }

    #[test]
    fn test_batch_add_sw_with_neutral_elements() {
        let p = sample_point().to_affine();
        let neutral = G1Point::neutral_element();

        let points = vec![p.clone(), neutral.clone(), p.clone()];

        let result = batch_add_sw(&points);

        // p + 0 + p = 2p
        let expected = p.operate_with_self(2_u16);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    // ==================== batch_normalize_jacobian tests ====================

    #[test]
    fn test_batch_normalize_jacobian_empty() {
        let points: Vec<G1JacobianPoint> = vec![];
        let result = batch_normalize_jacobian(&points);
        assert!(result.is_empty());
    }

    #[test]
    fn test_batch_normalize_jacobian_single_point() {
        let p = sample_jacobian_point().operate_with_self(3_u16);
        let points = vec![p.clone()];
        let result = batch_normalize_jacobian(&points);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], p.to_affine());
        assert_eq!(result[0].z(), &FE::one());
    }

    #[test]
    fn test_batch_normalize_jacobian_multiple_points() {
        let p = sample_jacobian_point();
        let points: Vec<G1JacobianPoint> =
            (1..=10).map(|i| p.operate_with_self(i as u16)).collect();

        let batch_result = batch_normalize_jacobian(&points);
        let individual_result: Vec<G1JacobianPoint> =
            points.iter().map(|p| p.to_affine()).collect();

        assert_eq!(batch_result.len(), individual_result.len());
        for (batch, individual) in batch_result.iter().zip(individual_result.iter()) {
            assert_eq!(batch, individual);
            assert_eq!(batch.z(), &FE::one());
        }
    }

    #[test]
    fn test_batch_normalize_jacobian_with_neutral_element() {
        let p = sample_jacobian_point();
        let neutral = G1JacobianPoint::neutral_element();

        let points = vec![
            p.clone(),
            neutral.clone(),
            p.operate_with_self(2_u16),
            neutral.clone(),
        ];

        let result = batch_normalize_jacobian(&points);

        assert_eq!(result.len(), 4);
        assert_eq!(result[0], points[0].to_affine());
        assert!(result[1].is_neutral_element());
        assert_eq!(result[2], points[2].to_affine());
        assert!(result[3].is_neutral_element());
    }

    // ==================== batch_add_jacobian tests ====================

    #[test]
    fn test_batch_add_jacobian_empty() {
        let points: Vec<G1JacobianPoint> = vec![];
        let result = batch_add_jacobian(&points);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn test_batch_add_jacobian_single_point() {
        let p = sample_jacobian_point().to_affine();
        let points = vec![p.clone()];
        let result = batch_add_jacobian(&points);
        assert_eq!(result.to_affine(), p);
    }

    #[test]
    fn test_batch_add_jacobian_multiple_points() {
        let p = sample_jacobian_point().to_affine();
        let affine_points: Vec<G1JacobianPoint> = (1..=5)
            .map(|i| p.operate_with_self(i as u16).to_affine())
            .collect();

        let batch_result = batch_add_jacobian(&affine_points);

        // Compute expected result individually
        let mut expected = G1JacobianPoint::neutral_element();
        for point in &affine_points {
            expected = expected.operate_with(point);
        }

        assert_eq!(batch_result.to_affine(), expected.to_affine());
    }

    // ==================== Correctness verification tests ====================

    #[test]
    fn test_batch_operations_preserve_curve_membership() {
        let p = sample_point();
        let points: Vec<G1Point> = (1..=5).map(|i| p.operate_with_self(i as u16)).collect();

        let normalized = batch_normalize_sw(&points);
        let sum = batch_add_sw(&normalized);

        // Verify the sum point is valid by checking it's in the curve
        // (The point construction would fail otherwise)
        let affine_sum = sum.to_affine();
        let [x, y, _z] = affine_sum.coordinates();
        assert_eq!(
            BLS12381Curve::defining_equation(x, y),
            FE::zero(),
            "Result should be on the curve"
        );
    }

    #[test]
    fn test_batch_add_associativity() {
        let p = sample_point().to_affine();
        let points: Vec<G1Point> = (1..=4)
            .map(|i| p.operate_with_self(i as u16).to_affine())
            .collect();

        // (a + b) + (c + d)
        let left_half = batch_add_sw(&points[..2]);
        let right_half = batch_add_sw(&points[2..]);
        let result1 = left_half.operate_with(&right_half);

        // ((a + b + c) + d)
        let result2 = batch_add_sw(&points);

        assert_eq!(result1.to_affine(), result2.to_affine());
    }

    // ==================== Large batch tests ====================

    #[test]
    fn test_batch_normalize_large_batch() {
        let p = sample_point();
        let points: Vec<G1Point> = (1..=100).map(|i| p.operate_with_self(i as u16)).collect();

        let batch_result = batch_normalize_sw(&points);

        // Verify a few random points
        assert_eq!(batch_result[0], points[0].to_affine());
        assert_eq!(batch_result[49], points[49].to_affine());
        assert_eq!(batch_result[99], points[99].to_affine());
    }

    #[test]
    fn test_batch_sum_equals_scalar_multiplication() {
        let p = sample_point().to_affine();

        // Sum p + p + p + p + p = 5p
        let points: Vec<G1Point> = (0..5).map(|_| p.clone()).collect();
        let batch_sum = batch_add_sw(&points);

        let scalar_mult = p.operate_with_self(5_u16);

        assert_eq!(batch_sum.to_affine(), scalar_mult.to_affine());
    }
}
