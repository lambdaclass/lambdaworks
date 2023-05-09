use crate::{
    cyclic_group::IsGroup,
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::UnsignedInteger,
};
use std::fmt::Debug;

pub fn msm<M, G, const NUM_LIMBS: usize>(
    cs: &[FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>],
    hidings: &[G],
    window_size: usize,
) -> G
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
    G: IsGroup,
{
    debug_assert_eq!(
        cs.len(),
        hidings.len(),
        "Slices `cs` and `hidings` must be of the same length to compute `msm`."
    );
    assert!(window_size < usize::BITS as usize);
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
                let window_unmasked = (dbg!(k.representative())
                    >> (dbg!(window_idx) * dbg!(window_size)))
                .limbs[NUM_LIMBS - 1];
                let m_ij = window_unmasked & ((1 << window_size) - 1);
                if dbg!(m_ij) != 0 {
                    let idx = (m_ij - 1) as usize;
                    buckets[idx] = buckets[idx].operate_with(p);
                }
            });

            // Do the reduction step for the buckets.
            buckets
                .iter_mut()
                .rev()
                .scan(G::neutral_element(), |m, b| {
                    *m = m.operate_with(b); // Reduction step.
                    *b = G::neutral_element(); // Cleanup bucket slot to reuse in the next window.
                    Some(m.clone())
                })
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
            short_weierstrass::curves::bls12_381::{
                curve::BLS12381Curve, field_extension::BLS12381PrimeField,
            },
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        unsigned_integer::element::UnsignedInteger,
    };
    use proptest::{collection, prelude::*, prop_assert_eq, prop_compose, proptest};

    const MAX_WSIZE: usize = 16;

    prop_compose! {
        fn bls12381_element()(limbs: [u64; 6]) -> FieldElement::<BLS12381PrimeField> {
            FieldElement::<BLS12381PrimeField>::new(UnsignedInteger::from_limbs(limbs))
        }
    }

    prop_compose! {
        fn bls12381_element_vec()(vec in collection::vec(bls12381_element(), 0..100)) -> Vec<FieldElement::<BLS12381PrimeField>> {
            vec
        }
    }

    prop_compose! {
        fn point()(power: u128) -> <BLS12381Curve as IsEllipticCurve>::PointRepresentation {
            BLS12381Curve::generator().operate_with_self(power)
        }
    }

    prop_compose! {
        fn points_vec()(vec in collection::vec(point(), 0..100)) -> Vec<<BLS12381Curve as IsEllipticCurve>::PointRepresentation> {
            vec
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 1, .. ProptestConfig::default()
          })]
        // Property-based test that ensures FFT eval. gives same result as a naive polynomial evaluation.
        #[test]
        fn test_pippenger_matches_naive_msm(window_size in 1..MAX_WSIZE, cs in bls12381_element_vec(), hidings in points_vec()) {
            let min_len = cs.len().min(hidings.len());
            let cs = cs[..min_len].to_vec();
            let hidings = hidings[..min_len].to_vec();

            let pippenger = pippenger::msm(&cs, &hidings, window_size);

            let cs: Vec<_> = cs.into_iter().map(|x| x.representative()).collect();
            let naive = naive::msm(&cs, &hidings);

            prop_assert_eq!(naive, pippenger);
        }
    }
}
