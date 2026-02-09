extern crate alloc;
#[cfg(feature = "alloc")]
use crate::circle::cfft::order_icfft_input_in_place;
#[cfg(feature = "alloc")]
use crate::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
type FE = FieldElement<Mersenne31Field>;

/// Inverse of 2 in Mersenne31: 2^{-1} mod (2^31 - 1) = 2^30 = 1073741824.
#[cfg(feature = "alloc")]
const INV_TWO: u32 = 1_073_741_824;

/// Folds evaluations in butterfly order using one layer of inverse twiddles
/// and a random challenge.
///
/// Evaluations must be in butterfly order: index `i` and index `i + half`
/// form a conjugate pair (y-fold) or antipodal pair (x-fold).
/// This is the same order used internally by the CFFT.
///
/// The formula computes:
///   result[i] = (evals[i] + evals[i+half]) / 2
///             + alpha * (evals[i] - evals[i+half]) * inv_twiddle[i] / 2
///
/// # Arguments
/// * `evaluations` - Evaluations in butterfly order (size 2m)
/// * `inv_twiddles` - Inverse twiddle factors for this layer (size m).
///   For the y-fold: 1/y_i. For x-folds: 1/x_i.
/// * `alpha` - Random folding challenge
#[cfg(feature = "alloc")]
pub fn fold(evaluations: &[FE], inv_twiddles: &[FE], alpha: &FE) -> Vec<FE> {
    let half = evaluations.len() / 2;
    debug_assert_eq!(evaluations.len(), 2 * half);
    debug_assert_eq!(inv_twiddles.len(), half);

    let inv_two = FE::from(&INV_TWO);

    let mut result = Vec::with_capacity(half);
    for i in 0..half {
        let f_hi = &evaluations[i];
        let f_lo = &evaluations[i + half];
        let sum = f_hi + f_lo;
        let diff = (f_hi - f_lo) * inv_twiddles[i];
        result.push((sum + alpha * diff) * inv_two);
    }
    result
}

/// Folds a single pair of evaluations.
///
/// This is the single-pair version of `fold`, used by the FRI verifier to
/// recompute the fold for a specific query position.
///
/// # Arguments
/// * `f_hi` - Evaluation in the first half (index < half)
/// * `f_lo` - Evaluation in the second half (index >= half)
/// * `inv_twiddle` - Inverse twiddle factor for this pair (1/y_i or 1/x_i)
/// * `alpha` - Random folding challenge
#[cfg(feature = "alloc")]
pub fn fold_pair(f_hi: &FE, f_lo: &FE, inv_twiddle: &FE, alpha: &FE) -> FE {
    let inv_two = FE::from(&INV_TWO);
    let sum = f_hi + f_lo;
    let diff = (f_hi - f_lo) * inv_twiddle;
    (sum + alpha * diff) * inv_two
}

/// Converts a natural-order index to its position in butterfly (CFFT) order.
///
/// In butterfly order:
///   - Even natural indices go to the first half: natural 2k → butterfly k
///   - Odd natural indices go to the second half (reversed): natural 2k+1 → butterfly n-1-k
pub fn natural_to_butterfly(natural_idx: usize, n: usize) -> usize {
    if natural_idx.is_multiple_of(2) {
        natural_idx / 2
    } else {
        n - 1 - natural_idx / 2
    }
}

/// Reorders evaluations from natural coset order to butterfly (CFFT) order.
///
/// In natural order, the i-th element is the evaluation at the i-th coset point.
/// In butterfly order, conjugate pairs are at distance n/2, which is the layout
/// needed by `fold`.
///
/// The butterfly order places:
///   - Even-indexed evaluations first (ascending): eval[0], eval[2], eval[4], ...
///   - Odd-indexed evaluations second (descending): ..., eval[3], eval[1]
#[cfg(feature = "alloc")]
pub fn reorder_natural_to_butterfly(evaluations: &mut [FE]) {
    order_icfft_input_in_place(evaluations);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circle::cosets::Coset;
    use crate::circle::polynomial::{evaluate_cfft, interpolate_cfft};
    use crate::circle::twiddles::{get_twiddles, TwiddlesConfig};
    use alloc::vec;

    /// Helper: get interpolation twiddles for a standard domain of given log_2_size.
    fn get_fold_twiddles(log_2_size: u32) -> Vec<Vec<FE>> {
        let coset = Coset::new_standard(log_2_size);
        get_twiddles(coset, TwiddlesConfig::Interpolation)
    }

    #[test]
    fn fold_of_constant_polynomial_is_constant() {
        let c = FE::from(42u64);
        let n = 8;
        let mut evals = vec![c; n];

        let twiddles = get_fold_twiddles(3);
        reorder_natural_to_butterfly(&mut evals);

        let alpha = FE::from(7u64);
        let result = fold(&evals, &twiddles[0], &alpha);

        assert_eq!(result.len(), 4);
        for val in &result {
            assert_eq!(*val, c);
        }
    }

    #[test]
    fn fold_reduces_evaluation_count() {
        let coeffs: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let mut evals = evaluate_cfft(coeffs);
        assert_eq!(evals.len(), 8);

        let twiddles = get_fold_twiddles(3);
        reorder_natural_to_butterfly(&mut evals);

        let alpha = FE::from(3u64);
        let result = fold(&evals, &twiddles[0], &alpha);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn full_fold_chain_reduces_to_single_value() {
        // A complete fold chain: y-fold then x-folds until 1 value
        let coeffs: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let mut evals = evaluate_cfft(coeffs);

        let twiddles = get_fold_twiddles(3); // 3 layers for 8 points
        reorder_natural_to_butterfly(&mut evals);

        // Y-fold: 8 -> 4
        let alpha = FE::from(5u64);
        let folded1 = fold(&evals, &twiddles[0], &alpha);
        assert_eq!(folded1.len(), 4);

        // X-fold: 4 -> 2
        let beta = FE::from(7u64);
        let folded2 = fold(&folded1, &twiddles[1], &beta);
        assert_eq!(folded2.len(), 2);

        // X-fold: 2 -> 1
        let gamma = FE::from(11u64);
        let folded3 = fold(&folded2, &twiddles[2], &gamma);
        assert_eq!(folded3.len(), 1);
    }

    #[test]
    fn fold_y_with_alpha_zero_gives_even_part() {
        // With alpha=0, fold takes only the even part: (f(P) + f(conj(P))) / 2
        // For a constant polynomial, even part = constant.
        let coeffs: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let mut evals = evaluate_cfft(coeffs);

        let twiddles = get_fold_twiddles(3);
        reorder_natural_to_butterfly(&mut evals);

        let alpha = FE::from(0u64);
        let folded = fold(&evals, &twiddles[0], &alpha);
        assert_eq!(folded.len(), 4);

        // The folded values should be a valid polynomial evaluation
        // (can be interpolated and re-evaluated consistently).
        let reinterp = interpolate_cfft(folded.clone());
        let reevals = evaluate_cfft(reinterp);
        assert_eq!(reevals, folded);
    }

    #[test]
    fn fold_y_with_alpha_one_on_linear() {
        // f(x,y) = y has coefficients [0, 1, 0, 0].
        // f_even(x) = 0, f_odd(x) = 1 for all x.
        // fold = f_even + 1 * f_odd = 0 + 1 = 1
        let coeffs = vec![
            FE::from(0u64),
            FE::from(1u64),
            FE::from(0u64),
            FE::from(0u64),
        ];
        let mut evals = evaluate_cfft(coeffs);

        let twiddles = get_fold_twiddles(2);
        reorder_natural_to_butterfly(&mut evals);

        let alpha = FE::from(1u64);
        let folded = fold(&evals, &twiddles[0], &alpha);
        assert_eq!(folded.len(), 2);

        for val in &folded {
            assert_eq!(*val, FE::one());
        }
    }

    #[test]
    fn fold_chain_16_points() {
        // Test with 16-point domain (log_2_size=4)
        let coeffs: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
        let mut evals = evaluate_cfft(coeffs);

        let twiddles = get_fold_twiddles(4);
        reorder_natural_to_butterfly(&mut evals);

        let challenges: Vec<FE> = (0..4).map(|i| FE::from((i * 3 + 1) as u64)).collect();

        let mut current = evals.to_vec();
        for (layer, challenge) in challenges.iter().enumerate() {
            current = fold(&current, &twiddles[layer], challenge);
        }

        assert_eq!(current.len(), 1);
    }
}
