extern crate alloc;
#[cfg(feature = "alloc")]
use crate::{
    circle::{cosets::Coset, traits::IsCircleFriField},
    field::element::FieldElement,
};
#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwiddlesConfig {
    Evaluation,
    Interpolation,
}
#[cfg(feature = "alloc")]
pub fn get_twiddles<F: IsCircleFriField>(
    domain: Coset<F>,
    config: TwiddlesConfig,
) -> Vec<Vec<FieldElement<F>>> {
    // We first take the half coset.
    let half_domain_points = domain.half_coset().get_coset_points();

    // The first set of twiddles are all the y coordinates of the half coset.
    let mut twiddles: Vec<Vec<FieldElement<F>>> =
        vec![half_domain_points.iter().map(|p| p.y.clone()).collect()];

    if domain.log_2_size >= 2 {
        // The second set of twiddles are the x coordinates of the first half of the half coset.
        twiddles.push(
            half_domain_points
                .iter()
                .take(half_domain_points.len() / 2)
                .map(|p| p.x.clone())
                .collect(),
        );
        for _ in 0..(domain.log_2_size - 2) {
            // The rest of the sets of twiddles are the "square" of the x coordinates of the first half of the previous set.
            let prev = twiddles
                .last()
                .expect("twiddles vector is non-empty at this point");
            let cur = prev
                .iter()
                .take(prev.len() / 2)
                .map(|x| x.square().double() - FieldElement::<F>::one())
                .collect();
            twiddles.push(cur);
        }
    }

    match config {
        TwiddlesConfig::Interpolation => {
            // For the interpolation, we need to take the inverse element of each twiddle in the default order.
            // The twiddles are coordinates of elements of the coset (or their squares) so they can't be zero.
            twiddles.iter_mut().for_each(|x| {
                FieldElement::<F>::inplace_batch_inverse(x)
                    .expect("twiddle batch inverse failed: coset coordinates are non-zero");
            });
        }
        TwiddlesConfig::Evaluation => {
            // For the evaluation, we need reverse the order of the vector of twiddles.
            twiddles.reverse();
        }
    }
    twiddles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::mersenne31::field::Mersenne31Field;

    type TestCoset = Coset<Mersenne31Field>;

    #[test]
    fn evaluation_twiddles_vectors_length_is_correct() {
        let domain = TestCoset::new_standard(20);
        let twiddles = get_twiddles(domain, TwiddlesConfig::Evaluation);
        for w in twiddles.windows(2) {
            assert_eq!(2 * w[0].len(), w[1].len());
        }
    }

    #[test]
    fn interpolation_twiddles_vectors_length_is_correct() {
        let domain = TestCoset::new_standard(20);
        let twiddles = get_twiddles(domain, TwiddlesConfig::Interpolation);
        for w in twiddles.windows(2) {
            assert_eq!(w[0].len(), 2 * w[1].len());
        }
    }
}
