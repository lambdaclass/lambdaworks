extern crate alloc;
use crate::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field};

#[cfg(feature = "alloc")]
pub fn cfft(
    input: &mut [FieldElement<Mersenne31Field>],
    twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>,
) {
    let log_2_size = input.len().trailing_zeros();
    
    (0..log_2_size).for_each(|i| {
        let chunk_size = 1 << i + 1;
        let half_chunk_size = 1 << i;
        input.chunks_mut(chunk_size).for_each(|chunk| {
            let (hi_part, low_part) = chunk.split_at_mut(half_chunk_size);
            hi_part.into_iter().zip(low_part).enumerate().for_each( |(j, (hi, low))| {
                let temp = *low * twiddles[i as usize][j];
                *low = *hi - temp;
                *hi = *hi + temp;    
            });
        });
    });
} 


#[cfg(feature = "alloc")]
pub fn icfft(
    input: &mut [FieldElement<Mersenne31Field>],
    twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>,
) {
    let log_2_size = input.len().trailing_zeros();
    
    println!("{:?}", twiddles);
    
    (0..log_2_size).for_each(|i| {
        let chunk_size = 1 << log_2_size - i;
        let half_chunk_size = chunk_size >> 1;
        input.chunks_mut(chunk_size).for_each(|chunk| {
            let (hi_part, low_part) = chunk.split_at_mut(half_chunk_size);
            hi_part.into_iter().zip(low_part).enumerate().for_each( |(j, (hi, low))| {
                let temp = *hi + *low;
                *low = (*hi - *low) * twiddles[i as usize][j];
                *hi = temp;
            });
        });
    });    
} 

// From [0, 2, 4, 6, 7, 5, 3, 1] to [0, 1, 2, 3, 4, 5, 6, 7]
pub fn order_cfft_result_naive(input: &mut [FieldElement<Mersenne31Field>]) -> Vec<FieldElement<Mersenne31Field>> {
    let mut result = Vec::new();
    let length = input.len();
    for i in (0..length/2) {
        result.push(input[i]);
        result.push(input[length - i - 1]);
    }
    result
}

// From [0, 1, 2, 3, 4, 5, 6, 7] to [0, 2, 4, 6, 7, 5, 3, 1]
pub fn order_icfft_input_naive(input: &mut [FieldElement<Mersenne31Field>]) -> Vec<FieldElement<Mersenne31Field>> {
    let mut result = Vec::new();
    (0..input.len()).step_by(2).for_each( |i| {
        result.push(input[i]);
    });
    (1..input.len()).step_by(2).rev().for_each( |i| {
        result.push(input[i]);
    });
    result
}

pub fn reverse_cfft_index(index: usize, length: usize) -> usize {
    if index < (length >> 1) { // index < length / 2
        index << 1 // index * 2
    } else {
        (((length - 1) - index) << 1) + 1
    }
}


pub fn cfft_4(
    input: &mut [FieldElement<Mersenne31Field>],
    twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>,
) -> Vec<FieldElement<Mersenne31Field>> {
    let mut stage1: Vec<FieldElement<Mersenne31Field>> = Vec::with_capacity(4);

    stage1.push(input[0] + input[1]);
    stage1.push((input[0] - input[1]) * twiddles[0][0]);

    stage1.push(input[2] + input[3]);
    stage1.push((input[2] - input[3]) * twiddles[0][1]);

    let mut stage2: Vec<FieldElement<Mersenne31Field>> = Vec::with_capacity(4);

    stage2.push(stage1[0] + stage1[2]);
    stage2.push(stage1[1] + stage1[3]);

    stage2.push((stage1[0] - stage1[2]) * twiddles[1][0]);
    stage2.push((stage1[1] - stage1[3]) * twiddles[1][0]);

    let f = FieldElement::<Mersenne31Field>::from(4).inv().unwrap();
    stage2.into_iter().map(|elem| elem * f).collect()
}

pub fn cfft_8(
    input: &mut [FieldElement<Mersenne31Field>],
    twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>,
) -> Vec<FieldElement<Mersenne31Field>> {
    let mut stage1: Vec<FieldElement<Mersenne31Field>> = Vec::with_capacity(8);

    stage1.push(input[0] + input[4]);
    stage1.push(input[1] + input[5]);
    stage1.push(input[2] + input[6]);
    stage1.push(input[3] + input[7]);
    stage1.push((input[0] - input[4]) * twiddles[0][0]);
    stage1.push((input[1] - input[5]) * twiddles[0][1]);
    stage1.push((input[2] - input[6]) * twiddles[0][2]);
    stage1.push((input[3] - input[7]) * twiddles[0][3]);

    let mut stage2: Vec<FieldElement<Mersenne31Field>> = Vec::with_capacity(8);

    stage2.push(stage1[0] + stage1[2]);
    stage2.push(stage1[1] + stage1[3]);
    stage2.push((stage1[0] - stage1[2]) * twiddles[1][0]);
    stage2.push((stage1[1] - stage1[3]) * twiddles[1][1]);

    stage2.push(stage1[4] + stage1[6]);
    stage2.push(stage1[5] + stage1[7]);
    stage2.push((stage1[4] - stage1[6]) * twiddles[1][0]);
    stage2.push((stage1[5] - stage1[7]) * twiddles[1][1]);

    let mut stage3: Vec<FieldElement<Mersenne31Field>> = Vec::with_capacity(8);

    stage3.push(stage2[0] + stage2[1]);
    stage3.push((stage2[0] - stage2[1]) * twiddles[2][0]);

    stage3.push(stage2[2] + stage2[3]);
    stage3.push((stage2[2] - stage2[3]) * twiddles[2][0]);

    stage3.push(stage2[4] + stage2[5]);
    stage3.push((stage2[4] - stage2[5]) * twiddles[2][0]);

    stage3.push(stage2[6] + stage2[7]);
    stage3.push((stage2[6] - stage2[7]) * twiddles[2][0]);

    let f = FieldElement::<Mersenne31Field>::from(8).inv().unwrap();
    stage3.into_iter().map(|elem| elem * f).collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    type FE = FieldElement<Mersenne31Field>;

    #[test]
    fn ordering_4() {
        let expected_slice = [
            FE::from(0),
            FE::from(1),
            FE::from(2),
            FE::from(3),
        ];

        let mut slice = [
            FE::from(0),
            FE::from(2),
            FE::from(3),
            FE::from(1),
        ];

        let res = order_cfft_result_naive(&mut slice);

        assert_eq!(res, expected_slice)
    }

    #[test]
    fn ordering() {
        let expected_slice = [
            FE::from(0),
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
            FE::from(9),
            FE::from(10),
            FE::from(11),
            FE::from(12),
            FE::from(13),
            FE::from(14),
            FE::from(15),
        ];

        let mut slice = [
            FE::from(0),
            FE::from(2),
            FE::from(4),
            FE::from(6),
            FE::from(8),
            FE::from(10),
            FE::from(12),
            FE::from(14),
            FE::from(15),
            FE::from(13),
            FE::from(11),
            FE::from(9),
            FE::from(7),
            FE::from(5),
            FE::from(3),
            FE::from(1),
        ];

        let res = order_cfft_result_naive(&mut slice);

        assert_eq!(res, expected_slice)
    }

    #[test]
    fn reverse_cfft_index_works() {
        let mut reversed: Vec<usize> = Vec::with_capacity(16);
        for i in 0..reversed.capacity() {
            reversed.push(reverse_cfft_index(i, reversed.capacity()));
        }
        assert_eq!(
            reversed[..],
            [0, 2, 4, 6, 8, 10, 12, 14, 15, 13, 11, 9, 7, 5, 3, 1]
        );
    }

    #[test]
    fn from_natural_to_icfft_input_order() {
        let mut slice = [
            FE::from(0),
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
            FE::from(9),
            FE::from(10),
            FE::from(11),
            FE::from(12),
            FE::from(13),
            FE::from(14),
            FE::from(15),
        ];

        let expected_slice = [
            FE::from(0),
            FE::from(2),
            FE::from(4),
            FE::from(6),
            FE::from(8),
            FE::from(10),
            FE::from(12),
            FE::from(14),
            FE::from(15),
            FE::from(13),
            FE::from(11),
            FE::from(9),
            FE::from(7),
            FE::from(5),
            FE::from(3),
            FE::from(1),
        ];

        let res = order_icfft_input_naive(&mut slice);

        assert_eq!(res, expected_slice)
    }


}
