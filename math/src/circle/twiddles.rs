extern crate alloc;
use crate::{
    circle::cosets::Coset,
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field},
};
use alloc::vec::Vec;

#[derive(PartialEq)]
pub enum TwiddlesConfig {
    Evaluation,
    Interpolation,
}
#[cfg(feature = "alloc")]
pub fn get_twiddles(
    domain: Coset,
    config: TwiddlesConfig,
) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
    // We first take the half coset.
    let half_domain_points = Coset::get_coset_points(&Coset::half_coset(domain.clone()));

    // The first set of twiddles are all the y coordinates of the half coset.
    let mut twiddles: Vec<Vec<FieldElement<Mersenne31Field>>> = Vec::new();
    twiddles.push(half_domain_points.iter().map(|p| p.y).collect());

    if domain.log_2_size >= 2 {
        // The second set of twiddles are the x coordinates of the first half of the half coset.
        twiddles.push(
            half_domain_points
                .iter()
                .take(half_domain_points.len() / 2)
                .map(|p| p.x)
                .collect(),
        );
        for _ in 0..(domain.log_2_size - 2) {
            // The rest of the sets of twiddles are the "square" of the x coordinates of the first half of the previous set.
            let prev = twiddles.last().unwrap();
            let cur = prev
                .iter()
                .take(prev.len() / 2)
                .map(|x| x.square().double() - FieldElement::<Mersenne31Field>::one())
                .collect();
            twiddles.push(cur);
        }
    }

    if config == TwiddlesConfig::Interpolation {
        // For the interpolation, we need to take the inverse element of each twiddle in the default order.
        twiddles.iter_mut().for_each(|x| {
            FieldElement::<Mersenne31Field>::inplace_batch_inverse(x).unwrap();
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

    #[test]
    fn evaluation_twiddles_vectors_length_is_correct() {
        let domain = Coset::new_standard(3);
        let config = TwiddlesConfig::Evaluation;
        let twiddles = get_twiddles(domain, config);
        for i in 0..twiddles.len() - 1 {
            assert_eq!(2 * twiddles[i].len(), twiddles[i + 1].len())
        }
    }

    #[test]
    fn interpolation_twiddles_vectors_length_is_correct() {
        let domain = Coset::new_standard(3);
        let config = TwiddlesConfig::Interpolation;
        let twiddles = get_twiddles(domain, config);
        for i in 0..twiddles.len() - 1 {
            assert_eq!(twiddles[i].len(), 2 * twiddles[i + 1].len())
        }
    }
}
