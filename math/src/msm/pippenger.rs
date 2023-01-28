use crate::{
    cyclic_group::IsCyclicBilinearGroup,
    field::{element::FieldElement, traits::IsLinearField},
};

use super::MSM;

pub struct Pippenger {
    window_size: usize,
}

/// Calculates the multiscalar multiplication using the Pippenger algorithm
/// as described in section 2.3 of the paper "cuZK: Accelerating Zero-Knowledge
/// Proof with A Faster Parallel Multi-Scalar Multiplication Algorithm on GPUs".
/// Reference: https://eprint.iacr.org/2022/1321.pdf
impl Pippenger {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
}

impl<F, G> MSM<F, G> for Pippenger
where
    G: IsCyclicBilinearGroup,
    F: IsLinearField,
{
    fn msm(&self, ks: &[FieldElement<F>], ps: &[G]) -> G {
        let s = self.window_size;
        assert!(s > 0, "window size should be greater than zero");
        assert_eq!(
            ks.len(),
            ps.len(),
            "the scalar and elliptic curve point list don't have the same length"
        );

        // The number of windows of size `s` is ceil(lambda/s)
        let num_windows = (F::bit_size() + s - 1) / s;

        // We define `buckets` outside of the loop so we only have to allocate once, and reuse it
        let mut buckets = vec![G::neutral_element(); (1 << s) - 1];

        (0..num_windows)
            .rev()
            .map(|window_idx| {
                // Put in the right bucket the corresponding ps[i] for the current window.
                ks.iter().zip(ps).for_each(|(k, p)| {
                    let m_ij = (k >> (window_idx * s)) & ((1 << s) - 1);
                    if m_ij != 0 {
                        buckets[m_ij - 1] = buckets[m_ij - 1].operate_with(p);
                    }
                });

                // Do the reduction step for the buckets.
                buckets
                    .iter_mut()
                    .rev()
                    .scan(G::neutral_element(), |m, b| {
                        *m = m.operate_with(b); // Reduction step.
                        *b = G::neutral_element(); // Cleanup bucket slot to reuse in the next window
                        Some(m.clone())
                    })
                    .reduce(|g, m| g.operate_with(&m))
                    .unwrap_or_else(G::neutral_element)
            })
            .reduce(|t, g| t.operate_with_self(1 << s).operate_with(&g))
            .unwrap_or_else(G::neutral_element)
    }
}
