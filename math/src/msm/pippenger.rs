use thiserror::Error;

use crate::{
    cyclic_group::IsCyclicBilinearGroup,
    field::{
        element::FieldElement,
        traits::{HasFieldOperations, IsLinearField},
    },
};

use super::MSM;

pub struct Pippenger {
    window_size: usize,
}

#[derive(Error, Debug, Clone)]
pub enum PippengerError {
    #[error("window size must be greater than zero")]
    InvalidWindowSize,
}

/// Calculates the multiscalar multiplication using the Pippenger algorithm
/// as described in section 2.3 of the paper "cuZK: Accelerating Zero-Knowledge
/// Proof with A Faster Parallel Multi-Scalar Multiplication Algorithm on GPUs".
/// Reference: https://eprint.iacr.org/2022/1321.pdf
impl Pippenger {
    pub fn new(window_size: usize) -> Result<Self, PippengerError> {
        if window_size == 0 {
            Err(PippengerError::InvalidWindowSize)
        } else {
            Ok(Self { window_size })
        }
    }
}

impl<F, G> MSM<F, G> for Pippenger
where
    G: IsCyclicBilinearGroup,
    F: HasFieldOperations + IsLinearField<BaseType = <F as HasFieldOperations>::BaseType>,
{
    fn msm(&self, ks: &[FieldElement<F>], ps: &[G]) -> G {
        assert_eq!(
            ks.len(),
            ps.len(),
            "the scalar and elliptic curve point list don't have the same length"
        );

        // The number of windows of size `s` is ceil(lambda/s).
        let num_windows = (F::num_bits() + self.window_size - 1) / self.window_size;

        // We define `buckets` outside of the loop so we only have to allocate once, and reuse it.
        //
        // This line forces a heap allocation which might be undesired. We can define this buckets
        // variable in the Pippenger struct to only allocate once, but use a bit of extra memory.
        // If we accept a const window_size, we could make it an array instaed of a vector
        // avoiding the heap allocation. We should be aware if that might be too agressive for
        // the stack and cause a potential stack overflow.
        let mut buckets = vec![G::neutral_element(); (1 << self.window_size) - 1];

        (0..num_windows)
            .rev()
            .map(|window_idx| {
                // Put in the right bucket the corresponding ps[i] for the current window.
                ks.iter().zip(ps).for_each(|(k, p)| {
                    let m_ij =
                        (k >> (window_idx * self.window_size)) & ((1 << self.window_size) - 1);
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
                        *b = G::neutral_element(); // Cleanup bucket slot to reuse in the next window.
                        Some(m.clone())
                    })
                    .reduce(|g, m| g.operate_with(&m))
                    .unwrap_or_else(G::neutral_element)
            })
            .reduce(|t, g| t.operate_with_self(1 << self.window_size).operate_with(&g))
            .unwrap_or_else(G::neutral_element)
    }
}
