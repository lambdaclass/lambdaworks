extern crate alloc;
use crate::{
    circle::{cosets::Coset, point::CirclePoint},
    fft::cpu::bit_reversing::in_place_bit_reverse_permute,
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
    let mut half_domain_points = Coset::get_coset_points(&Coset::half_coset(domain.clone()));
    if config == TwiddlesConfig::Evaluation {
        in_place_bit_reverse_permute::<CirclePoint<Mersenne31Field>>(&mut half_domain_points[..]);
    }

    let mut twiddles: Vec<Vec<FieldElement<Mersenne31Field>>> =
        vec![half_domain_points.iter().map(|p| p.y).collect()];

    if domain.log_2_size >= 2 {
        twiddles.push(half_domain_points.iter().step_by(2).map(|p| p.x).collect());
        for _ in 0..(domain.log_2_size - 2) {
            let prev = twiddles.last().unwrap();
            let cur = prev
                .iter()
                .step_by(2)
                .map(|x| x.square().double() - FieldElement::<Mersenne31Field>::one())
                .collect();
            twiddles.push(cur);
        }
    }
    twiddles.reverse();

    if config == TwiddlesConfig::Interpolation {
        twiddles.iter_mut().for_each(|x| {
            FieldElement::<Mersenne31Field>::inplace_batch_inverse(x).unwrap();
        });
    }
    twiddles
}

pub fn get_twiddles_itnerpolation_4(domain: Coset) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
    let half_domain_points = Coset::get_coset_points(&Coset::half_coset(domain.clone()));
    let mut twiddles: Vec<Vec<FieldElement<Mersenne31Field>>> =
        vec![half_domain_points.iter().map(|p| p.y).collect()];
    twiddles.push(half_domain_points.iter().take(1).map(|p| p.x).collect());
    twiddles.iter_mut().for_each(|x| {
        FieldElement::<Mersenne31Field>::inplace_batch_inverse(x).unwrap();
    });
    twiddles
}

pub fn get_twiddles_itnerpolation_8(domain: Coset) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
    let half_domain_points = Coset::get_coset_points(&Coset::half_coset(domain.clone()));
    let mut twiddles: Vec<Vec<FieldElement<Mersenne31Field>>> =
        vec![half_domain_points.iter().map(|p| p.y).collect()];
    twiddles.push(half_domain_points.iter().take(2).map(|p| p.x).collect());
    twiddles.push(vec![
        half_domain_points[0].x.square().double() - FieldElement::<Mersenne31Field>::one(),
    ]);
    twiddles.iter_mut().for_each(|x| {
        FieldElement::<Mersenne31Field>::inplace_batch_inverse(x).unwrap();
    });
    twiddles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twiddles_vectors_lenght() {
        let domain = Coset::new_standard(3);
        let config = TwiddlesConfig::Evaluation;
        let twiddles = get_twiddles(domain, config);
        for i in 0..twiddles.len() - 1 {
            assert_eq!(twiddles[i].len(), 2 * twiddles[i + 1].len())
        }
    }
}
