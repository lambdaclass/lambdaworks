use crate::circle::{cosets::Coset, point::CirclePoint};
use crate::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field};


pub fn cfft(input: &mut [FieldElement<Mersenne31Field>], twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>)
{
    // divide input in groups, starting with 1, duplicating the number of groups in each stage.
    let mut group_count = 1;
    let mut group_size = input.len();

    // for each group, there'll be group_size / 2 butterflies.
    // a butterfly is the atomic operation of a FFT, e.g: (a, b) = (a + wb, a - wb).
    // The 0.5 factor is what gives FFT its performance, it recursively halves the problem size
    // (group size).

    while group_count < input.len() {
        #[allow(clippy::needless_range_loop)] // the suggestion would obfuscate a bit the algorithm
        for group in 0..group_count {
            let first_in_group = group * group_size;
            let first_in_next_group = first_in_group + group_size / 2;

            let w = &twiddles[group]; // a twiddle factor is used per group

            for i in first_in_group..first_in_next_group {
                let wi = w[i] * &input[i + group_size / 2];

                let y0 = &input[i] + &wi;
                let y1 = &input[i] - &wi;

                input[i] = y0;
                input[i + group_size / 2] = y1;
            }
        }
        group_count *= 2;
        group_size /= 2;
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::circle::twiddles;

//     use super::*;

//     #[test]
//     fn cfft() {
//         let coset = Coset::new_standard(3);
//         let twiddles = 
//         assert_eq!(1 << coset.log_2_size, points.len())
//     }
// }
