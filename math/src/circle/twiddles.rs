use super::{cosets::Coset, point::CirclePoint};
use crate::{fft::cpu::bit_reversing::in_place_bit_reverse_permute, field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field}};

pub fn get_twiddles(domain: Coset) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
   let mut half_domain_points = Coset::get_coset_points(&Coset::half_coset(domain.clone()));
   in_place_bit_reverse_permute::<CirclePoint::<Mersenne31Field>>(&mut half_domain_points[..]);

   let mut twiddles: Vec<Vec<FieldElement<Mersenne31Field>>> = vec![half_domain_points.iter().map(|p| p.y).collect()];

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
    twiddles
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn twiddles_vectors_lenght() {
    //     let domain = Coset::new_standard(3);
    //     let twiddles = get_twiddles(domain);
    //     for i in 0..twiddles.len() - 1 {
    //         assert_eq!(twiddles[i].len(), 2 * twiddles[i+1].len())
    //     }
    // }

    #[test]
    fn twiddles_test() {
        let domain = Coset::new_standard(3);g
        let _twiddles = get_twiddles(domain.clone());
        // println!("DOMAIN: {:?}", Coset::get_coset_points(&domain));
        // println!("----------------------");
        // println!("TWIDDLES: {:?}", twiddles);

        assert_eq!(FieldElement::<Mersenne31Field>::from(&32768), 
            FieldElement::<Mersenne31Field>::from(&590768354).square().double() - FieldElement::<Mersenne31Field>::one()
        );
        assert_eq!(-FieldElement::<Mersenne31Field>::from(&32768), 
        FieldElement::<Mersenne31Field>::from(&978592373).square().double() - FieldElement::<Mersenne31Field>::one()
    )
    }
}