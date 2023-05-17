use crate::{cyclic_group::IsGroup, unsigned_integer::element::UnsignedInteger};

use super::naive::MSMError;

/// This function computes the multiscalar multiplication (MSM).
///
/// Assume a group G of order r is given.
/// Let `hidings = [g_1, ..., g_n]` be a tuple of group points in G and
/// let `cs = [k_1, ..., k_n]` be a tuple of scalars in the Galois field GF(r).
///
/// Then, with additive notation, `msm(cs, hidings)` computes k_1 * g_1 + .... + k_n * g_n.
///
/// If `hidings` and `cs` are empty, then `msm` returns the zero element of the group.
///
/// Panics if `cs` and `hidings` have different lengths.
pub fn msm<const NUM_LIMBS: usize, G>(
    cs: &[UnsignedInteger<NUM_LIMBS>],
    hidings: &[G],
) -> Result<G, MSMError>
where
    G: IsGroup,
{
    if cs.len() != hidings.len() {
        return Err(MSMError::LengthMismatch(cs.len(), hidings.len()));
    }
    // When input is small enough, windows of length 2 seem faster than 1.
    const MIN_WINDOWS: usize = 2;
    const SCALE_FACTORS: (usize, usize) = (4, 5);

    // We approximate the optimum window size with: f(n) = k * log2(n), where k is a scaling factor
    let window_size =
        ((usize::BITS - cs.len().leading_zeros() - 1) as usize * SCALE_FACTORS.0) / SCALE_FACTORS.1;
    Ok(msm_with(cs, hidings, MIN_WINDOWS.min(window_size)))
}

pub fn msm_with<const NUM_LIMBS: usize, G>(
    cs: &[UnsignedInteger<NUM_LIMBS>],
    hidings: &[G],
    window_size: usize,
) -> G
where
    G: IsGroup,
{
    assert!(window_size < usize::BITS as usize); // Program would go OOM anyways

    // The number of windows of size `s` is ceil(lambda/s).
    let num_windows = (64 * NUM_LIMBS - 1) / window_size + 1;

    // We define `buckets` outside of the loop so we only have to allocate once, and reuse it.
    //
    // This line forces a heap allocation which might be undesired. We can define this buckets
    // variable in the Pippenger struct to only allocate once, but use a bit of extra memory.
    // If we accept a const window_size, we could make it an array instaed of a vector
    // avoiding the heap allocation. We should be aware if that might be too agressive for
    // the stack and cause a potential stack overflow.
    let mut buckets = vec![G::neutral_element(); (1 << window_size) - 1];

    (0..num_windows)
        .rev()
        .map(|window_idx| {
            // Put in the right bucket the corresponding ps[i] for the current window.
            cs.iter().zip(hidings).for_each(|(k, p)| {
                // We truncate the number to the least significative limb.
                // This is ok because window_size < usize::BITS.
                let window_unmasked = (k >> (window_idx * window_size)).limbs[NUM_LIMBS - 1];
                let m_ij = window_unmasked & ((1 << window_size) - 1);
                if m_ij != 0 {
                    let idx = (m_ij - 1) as usize;
                    buckets[idx] = buckets[idx].operate_with(p);
                }
            });

            // Do the reduction step for the buckets.
            buckets
                .iter_mut()
                // This first part iterates buckets in descending order, generating an iterator with the sum of
                // each bucket and all that came before as its items; i.e: (b_n, b_n + b_n-1, ..., b_n + ... + b_0)
                .rev()
                .scan(G::neutral_element(), |m, b| {
                    *m = m.operate_with(b); // Reduction step.
                    *b = G::neutral_element(); // Cleanup bucket slot to reuse in the next window.
                    Some(m.clone())
                })
                // This next part sums all elements of the iterator: (b_n) + (b_n + b_n-1) + ...
                // This results in: (n + 1) * b_n + n * b_n-1 + ... + b_0
                .reduce(|g, m| g.operate_with(&m))
                .unwrap_or_else(G::neutral_element)
        })
        .reduce(|t, g| t.operate_with_self(1_u64 << window_size).operate_with(&g))
        .unwrap_or_else(G::neutral_element)
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
        fn test_pippenger_matches_naive_msm(window_size in 1.._MAX_WSIZE, cs in unsigned_integer_vec(), hidings in points_vec()) {
            let min_len = cs.len().min(hidings.len());
            let cs = cs[..min_len].to_vec();
            let hidings = hidings[..min_len].to_vec();

            let pippenger = pippenger::msm_with(&cs, &hidings, window_size);
            let naive = naive::msm(&cs, &hidings).unwrap();

            prop_assert_eq!(naive, pippenger);
        }
    }
}
