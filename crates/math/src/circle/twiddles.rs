extern crate alloc;
#[cfg(feature = "alloc")]
use crate::{
    circle::{cosets::Coset, traits::IsCircleFriField},
    field::element::FieldElement,
};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[derive(PartialEq)]
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
    let half_domain_points = Coset::get_coset_points(&Coset::half_coset(domain.clone()));

    // The first set of twiddles are all the y coordinates of the half coset.
    let mut twiddles: Vec<Vec<FieldElement<F>>> = Vec::new();
    twiddles.push(half_domain_points.iter().map(|p| p.y.clone()).collect());

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

    if config == TwiddlesConfig::Interpolation {
        // For the interpolation, we need to take the inverse element of each twiddle in the default order.
        // The twiddles are coordinates of elements of the coset (or their squares) so they can't be zero.
        twiddles.iter_mut().for_each(|x| {
            FieldElement::<F>::inplace_batch_inverse(x)
                .expect("twiddle batch inverse failed: coset coordinates are non-zero");
        });
    } else {
        // For the evaluation, we need reverse the order of the vector of twiddles.
        twiddles.reverse();
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
        let config = TwiddlesConfig::Evaluation;
        let twiddles = get_twiddles(domain, config);
        for i in 0..twiddles.len() - 1 {
            assert_eq!(2 * twiddles[i].len(), twiddles[i + 1].len())
        }
    }

    #[test]
    fn interpolation_twiddles_vectors_length_is_correct() {
        let domain = TestCoset::new_standard(20);
        let config = TwiddlesConfig::Interpolation;
        let twiddles = get_twiddles(domain, config);
        for i in 0..twiddles.len() - 1 {
            assert_eq!(twiddles[i].len(), 2 * twiddles[i + 1].len())
        }
    }
}
