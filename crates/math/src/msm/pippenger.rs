use crate::{cyclic_group::IsGroup, unsigned_integer::element::UnsignedInteger};

use super::naive::MSMError;

use alloc::vec;
use alloc::vec::Vec;

/// This function computes the multiscalar multiplication (MSM).
///
/// Assume a group G of order r is given.
/// Let `points = [g_1, ..., g_n]` be a tuple of group points in G and
/// let `cs = [k_1, ..., k_n]` be a tuple of scalars in the Galois field GF(r).
///
/// Then, with additive notation, `msm(cs, points)` computes k_1 * g_1 + .... + k_n * g_n.
///
/// If `points` and `cs` are empty, then `msm` returns the zero element of the group.
///
/// Panics if `cs` and `points` have different lengths.
pub fn msm<const NUM_LIMBS: usize, G>(
    cs: &[UnsignedInteger<NUM_LIMBS>],
    points: &[G],
) -> Result<G, MSMError>
where
    G: IsGroup,
{
    if cs.len() != points.len() {
        return Err(MSMError::LengthMismatch(cs.len(), points.len()));
    }

    let window_size = optimum_window_size(cs.len());

    Ok(msm_with(cs, points, window_size))
}

/// Select optimal window size based on number of scalars.
/// Uses empirically-tuned values based on arkworks benchmarks.
/// The window size balances bucket count (2^c) against number of windows (bits/c).
fn optimum_window_size(num_scalars: usize) -> usize {
    match num_scalars {
        0..=4 => 2,
        5..=32 => 3,
        33..=128 => 4,
        129..=500 => 6,
        501..=2048 => 8,
        2049..=8192 => 10,
        8193..=32768 => 12,
        32769..=131072 => 14,
        _ => 16,
    }
}

pub fn msm_with<const NUM_LIMBS: usize, G>(
    cs: &[UnsignedInteger<NUM_LIMBS>],
    points: &[G],
    window_size: usize,
) -> G
where
    G: IsGroup,
{
    const MIN_WINDOW_SIZE: usize = 2;
    const MAX_WINDOW_SIZE: usize = 32;

    let window_size = window_size.clamp(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE);
    let num_windows = (64 * NUM_LIMBS - 1) / window_size + 1;

    let n_buckets = (1 << window_size) - 1;
    let mut buckets = vec![G::neutral_element(); n_buckets];

    (0..num_windows)
        .rev()
        .map(|window_idx| {
            cs.iter().zip(points).for_each(|(k, p)| {
                // Truncate to least significant limb (safe because window_size < usize::BITS).
                let window_unmasked = (k >> (window_idx * window_size)).limbs[NUM_LIMBS - 1];
                let m_ij = window_unmasked & n_buckets as u64;
                if m_ij != 0 {
                    buckets[(m_ij - 1) as usize] = buckets[(m_ij - 1) as usize].operate_with(p);
                }
            });

            // Bucket reduction: sum weighted buckets so bucket_i gets weight (i+1).
            buckets
                .iter_mut()
                .rev()
                .scan(G::neutral_element(), |m, b| {
                    *m = m.operate_with(b);
                    *b = G::neutral_element();
                    Some(m.clone())
                })
                .reduce(|g, m| g.operate_with(&m))
                .unwrap_or_else(G::neutral_element)
        })
        .reduce(|t, g| t.operate_with_self(1_u64 << window_size).operate_with(&g))
        .unwrap_or_else(G::neutral_element)
}

/// Recode scalars to signed digits for Pippenger's algorithm.
/// Uses signed representation to halve the bucket count from 2^c - 1 to 2^(c-1).
/// Returns a flat vector: digits for scalar i at `[i * total_windows .. (i+1) * total_windows)`.
fn recode_scalars_signed<const NUM_LIMBS: usize>(
    scalars: &[UnsignedInteger<NUM_LIMBS>],
    window_size: usize,
    num_windows: usize,
    total_windows: usize,
) -> Vec<i64> {
    let half_bucket = 1i64 << (window_size - 1);
    let full_bucket = 1i64 << window_size;
    let mask = (1u64 << window_size) - 1;
    let n_scalars = scalars.len();

    let mut digits = vec![0i64; n_scalars * total_windows];

    for (i, scalar) in scalars.iter().enumerate() {
        let mut carry = 0i64;
        let base_idx = i * total_windows;

        for window_idx in 0..num_windows {
            let shift = window_idx * window_size;
            let raw_val = if shift < 64 * NUM_LIMBS {
                (scalar >> shift).limbs[NUM_LIMBS - 1] & mask
            } else {
                0
            };
            let window_val = raw_val as i64 + carry;

            if window_val >= half_bucket {
                digits[base_idx + window_idx] = window_val - full_bucket;
                carry = 1;
            } else {
                digits[base_idx + window_idx] = window_val;
                carry = 0;
            }
        }
        digits[base_idx + num_windows] = carry;
    }

    digits
}

#[inline(always)]
fn get_digit(
    flat_digits: &[i64],
    total_windows: usize,
    scalar_idx: usize,
    window_idx: usize,
) -> i64 {
    flat_digits[scalar_idx * total_windows + window_idx]
}

/// MSM using signed bucket recoding (Pippenger with signed digits).
/// This uses 2^(c-1) buckets instead of 2^c - 1, providing ~10-20% speedup.
pub fn msm_with_signed<const NUM_LIMBS: usize, G>(
    cs: &[UnsignedInteger<NUM_LIMBS>],
    points: &[G],
    window_size: usize,
) -> G
where
    G: IsGroup,
{
    const MIN_WINDOW_SIZE: usize = 2;
    const MAX_WINDOW_SIZE: usize = 32;

    let window_size = window_size.clamp(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE);
    let num_windows = (64 * NUM_LIMBS - 1) / window_size + 1;
    // +1 to handle potential carry from signed recoding
    let total_windows = num_windows + 1;

    let n_buckets = (1 << (window_size - 1)) as usize;
    let signed_digits = recode_scalars_signed(cs, window_size, num_windows, total_windows);

    (0..total_windows)
        .rev()
        .map(|window_idx| {
            let mut buckets = vec![G::neutral_element(); n_buckets];

            for (scalar_idx, p) in points.iter().take(cs.len()).enumerate() {
                let digit = get_digit(&signed_digits, total_windows, scalar_idx, window_idx);
                if digit > 0 {
                    let idx = digit as usize - 1;
                    buckets[idx] = buckets[idx].operate_with(p);
                } else if digit < 0 {
                    let idx = (-digit) as usize - 1;
                    buckets[idx] = buckets[idx].operate_with(&p.neg());
                }
            }

            let mut m = G::neutral_element();
            buckets
                .into_iter()
                .rev()
                .map(|b| {
                    m = m.operate_with(&b);
                    m.clone()
                })
                .reduce(|g, m| g.operate_with(&m))
                .unwrap_or_else(G::neutral_element)
        })
        .reduce(|t, g| t.operate_with_self(1_u64 << window_size).operate_with(&g))
        .unwrap_or_else(G::neutral_element)
}

#[cfg(feature = "parallel")]
/// Parallel MSM using signed bucket recoding.
/// Combines parallel window processing with signed digit optimization.
pub fn parallel_msm_with_signed<const NUM_LIMBS: usize, G>(
    cs: &[UnsignedInteger<NUM_LIMBS>],
    points: &[G],
    window_size: usize,
) -> G
where
    G: IsGroup + Send + Sync,
{
    use rayon::prelude::*;

    const MIN_WINDOW_SIZE: usize = 2;
    const MAX_WINDOW_SIZE: usize = 32;

    let window_size = window_size.clamp(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE);
    let num_windows = (64 * NUM_LIMBS - 1) / window_size + 1;
    // +1 to handle potential carry from signed recoding
    let total_windows = num_windows + 1;
    let n_buckets = (1 << (window_size - 1)) as usize;

    let signed_digits = recode_scalars_signed(cs, window_size, num_windows, total_windows);

    (0..total_windows)
        .into_par_iter()
        .map(|window_idx| {
            let mut buckets = vec![G::neutral_element(); n_buckets];

            for (scalar_idx, p) in points.iter().take(cs.len()).enumerate() {
                let digit = get_digit(&signed_digits, total_windows, scalar_idx, window_idx);
                if digit > 0 {
                    let idx = digit as usize - 1;
                    buckets[idx] = buckets[idx].operate_with(p);
                } else if digit < 0 {
                    let idx = (-digit) as usize - 1;
                    buckets[idx] = buckets[idx].operate_with(&p.neg());
                }
            }

            let mut m = G::neutral_element();
            let window_item = buckets
                .into_iter()
                .rev()
                .map(|b| {
                    m = m.operate_with(&b);
                    m.clone()
                })
                .reduce(|g, m| g.operate_with(&m))
                .unwrap_or_else(G::neutral_element);

            let shift = window_idx * window_size;
            if shift < 64 * NUM_LIMBS {
                window_item.operate_with_self(UnsignedInteger::<NUM_LIMBS>::from_u64(1) << shift)
            } else {
                // For shifts that overflow UnsignedInteger representation,
                // use repeated doubling: P * 2^shift = double(P) shift times
                (0..shift).fold(window_item, |acc, _| acc.operate_with(&acc))
            }
        })
        .reduce(G::neutral_element, |a, b| a.operate_with(&b))
}

#[cfg(feature = "parallel")]
pub fn parallel_msm_with<const NUM_LIMBS: usize, G>(
    cs: &[UnsignedInteger<NUM_LIMBS>],
    points: &[G],
    window_size: usize,
) -> G
where
    G: IsGroup + Send + Sync,
{
    use rayon::prelude::*;

    assert!(window_size < usize::BITS as usize);

    let num_windows = (64 * NUM_LIMBS - 1) / window_size + 1;
    let n_buckets = (1 << window_size) - 1;

    (0..num_windows)
        .into_par_iter()
        .map(|window_idx| {
            let mut buckets = vec![G::neutral_element(); n_buckets];
            let shift = window_idx * window_size;
            cs.iter().zip(points).for_each(|(k, p)| {
                let window_unmasked = (k >> shift).limbs[NUM_LIMBS - 1];
                let m_ij = window_unmasked & n_buckets as u64;
                if m_ij != 0 {
                    buckets[(m_ij - 1) as usize] = buckets[(m_ij - 1) as usize].operate_with(p);
                }
            });

            let mut m = G::neutral_element();
            let window_item = buckets
                .into_iter()
                .rev()
                .map(|b| {
                    m = m.operate_with(&b);
                    m.clone()
                })
                .reduce(|g, m| g.operate_with(&m))
                .unwrap_or_else(G::neutral_element);

            window_item.operate_with_self(UnsignedInteger::<NUM_LIMBS>::from_u64(1) << shift)
        })
        .reduce(G::neutral_element, |a, b| a.operate_with(&b))
}

#[cfg(test)]
mod tests {
    use crate::cyclic_group::IsGroup;
    use crate::msm::{naive, pippenger};
    use crate::{
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::curve::BLS12381Curve, traits::IsEllipticCurve,
        },
        unsigned_integer::element::UnsignedInteger,
    };
    use alloc::vec::Vec;
    use proptest::{collection, prelude::*, prop_assert_eq, prop_compose, proptest};

    const _CASES: u32 = 20;
    const _MAX_WSIZE: usize = 8;
    const _MAX_LEN: usize = 30;

    prop_compose! {
        fn unsigned_integer()(limbs: [u64; 6]) -> UnsignedInteger<6> {
            UnsignedInteger::from_limbs(limbs)
        }
    }

    prop_compose! {
        fn unsigned_integer_vec()(vec in collection::vec(unsigned_integer(), 0.._MAX_LEN)) -> Vec<UnsignedInteger<6>> {
            vec
        }
    }

    prop_compose! {
        fn point()(power: u128) -> <BLS12381Curve as IsEllipticCurve>::PointRepresentation {
            BLS12381Curve::generator().operate_with_self(power)
        }
    }

    prop_compose! {
        fn points_vec()(vec in collection::vec(point(), 0.._MAX_LEN)) -> Vec<<BLS12381Curve as IsEllipticCurve>::PointRepresentation> {
            vec
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: _CASES, .. ProptestConfig::default()
          })]
        // Property-based test that ensures `pippenger::msm` gives same result as `naive::msm`.
        #[test]
        fn test_pippenger_matches_naive_msm(window_size in 1.._MAX_WSIZE, cs in unsigned_integer_vec(), points in points_vec()) {
            let min_len = cs.len().min(points.len());
            let cs = cs[..min_len].to_vec();
            let points = points[..min_len].to_vec();

            let pippenger = pippenger::msm_with(&cs, &points, window_size);
            let naive = naive::msm(&cs, &points).unwrap();

            prop_assert_eq!(naive, pippenger);
        }

        // Property-based test that ensures `pippenger::msm_with` gives same result as `pippenger::parallel_msm_with`.
        #[test]
        #[cfg(feature = "parallel")]
        fn test_parallel_pippenger_matches_sequential(window_size in 1.._MAX_WSIZE, cs in unsigned_integer_vec(), points in points_vec()) {
            let min_len = cs.len().min(points.len());
            let cs = cs[..min_len].to_vec();
            let points = points[..min_len].to_vec();

            let sequential = pippenger::msm_with(&cs, &points, window_size);
            let parallel = pippenger::parallel_msm_with(&cs, &points, window_size);

            prop_assert_eq!(parallel, sequential);
        }

        // Property-based test that ensures signed MSM matches unsigned MSM.
        #[test]
        fn test_signed_pippenger_matches_unsigned(window_size in 2.._MAX_WSIZE, cs in unsigned_integer_vec(), points in points_vec()) {
            let min_len = cs.len().min(points.len());
            let cs = cs[..min_len].to_vec();
            let points = points[..min_len].to_vec();

            let unsigned = pippenger::msm_with(&cs, &points, window_size);
            let signed = pippenger::msm_with_signed(&cs, &points, window_size);

            prop_assert_eq!(unsigned, signed);
        }

        // Property-based test for parallel signed MSM.
        #[test]
        #[cfg(feature = "parallel")]
        fn test_parallel_signed_pippenger_matches_sequential(window_size in 2.._MAX_WSIZE, cs in unsigned_integer_vec(), points in points_vec()) {
            let min_len = cs.len().min(points.len());
            let cs = cs[..min_len].to_vec();
            let points = points[..min_len].to_vec();

            let sequential = pippenger::msm_with_signed(&cs, &points, window_size);
            let parallel = pippenger::parallel_msm_with_signed(&cs, &points, window_size);

            prop_assert_eq!(parallel, sequential);
        }
    }

    // Regression test: ensure signed MSM handles points.len() > cs.len() without panic
    #[test]
    fn test_signed_msm_with_more_points_than_scalars() {
        let cs: Vec<UnsignedInteger<6>> =
            Vec::from([UnsignedInteger::from_u64(1), UnsignedInteger::from_u64(2)]);
        let points: Vec<_> = (0..5)
            .map(|i| BLS12381Curve::generator().operate_with_self(i as u64 + 1))
            .collect();

        // Should not panic, uses only first cs.len() points
        let result = pippenger::msm_with_signed(&cs, &points, 4);

        // Verify correctness: 1*G + 2*2G = G + 4G = 5G
        let expected = BLS12381Curve::generator().operate_with_self(5u64);
        assert_eq!(result, expected);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_signed_msm_with_more_points_than_scalars() {
        let cs: Vec<UnsignedInteger<6>> =
            Vec::from([UnsignedInteger::from_u64(1), UnsignedInteger::from_u64(2)]);
        let points: Vec<_> = (0..5)
            .map(|i| BLS12381Curve::generator().operate_with_self(i as u64 + 1))
            .collect();

        let result = pippenger::parallel_msm_with_signed(&cs, &points, 4);

        let expected = BLS12381Curve::generator().operate_with_self(5u64);
        assert_eq!(result, expected);
    }
}
