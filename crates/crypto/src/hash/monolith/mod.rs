use alloc::vec::Vec;
use lambdaworks_math::field::{
    fields::mersenne31::field::MERSENNE_31_PRIME_FIELD_ORDER, traits::IsField,
};
use sha3::{
    digest::{ExtendableOutput, Update},
    Shake128, Shake128Reader,
};

mod utils;
use utils::*;

// Ported from https://github.com/Plonky3/Plonky3/blob/main/monolith

pub const NUM_BARS: usize = 8;
const MATRIX_CIRC_MDS_16_MERSENNE31_MONOLITH: [u32; 16] = [
    61402, 17845, 26798, 59689, 12021, 40901, 41351, 27521, 56951, 12034, 53865, 43244, 7454,
    33823, 28750, 1108,
];

pub struct MonolithMersenne31<const WIDTH: usize, const NUM_FULL_ROUNDS: usize> {
    round_constants: Vec<Vec<u32>>,
    lookup1: Vec<u16>,
    lookup2: Vec<u16>,
}

impl<const WIDTH: usize, const NUM_FULL_ROUNDS: usize> Default
    for MonolithMersenne31<WIDTH, NUM_FULL_ROUNDS>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const WIDTH: usize, const NUM_FULL_ROUNDS: usize> MonolithMersenne31<WIDTH, NUM_FULL_ROUNDS> {
    pub fn new() -> Self {
        assert!(WIDTH >= 8);
        assert!(WIDTH <= 24);
        assert!(WIDTH.is_multiple_of(4));
        Self {
            round_constants: Self::instantiate_round_constants(),
            lookup1: Self::instantiate_lookup1(),
            lookup2: Self::instantiate_lookup2(),
        }
    }

    fn instantiate_round_constants() -> Vec<Vec<u32>> {
        let mut shake = Shake128::default();
        shake.update("Monolith".as_bytes());
        shake.update(&[WIDTH as u8, (NUM_FULL_ROUNDS + 1) as u8]);
        shake.update(&MERSENNE_31_PRIME_FIELD_ORDER.to_le_bytes());
        shake.update(&[8, 8, 8, 7]);
        let mut shake_finalized = shake.finalize_xof();
        random_matrix(&mut shake_finalized, NUM_FULL_ROUNDS, WIDTH)
    }

    fn instantiate_lookup1() -> Vec<u16> {
        (0..=u16::MAX)
            .map(|i| {
                let hi = (i >> 8) as u8;
                let lo = i as u8;
                ((Self::s_box(hi) as u16) << 8) | Self::s_box(lo) as u16
            })
            .collect()
    }

    fn instantiate_lookup2() -> Vec<u16> {
        (0..(1 << 15))
            .map(|i| {
                let hi = (i >> 8) as u8;
                let lo: u8 = i as u8;
                ((Self::final_s_box(hi) as u16) << 8) | Self::s_box(lo) as u16
            })
            .collect()
    }

    fn s_box(y: u8) -> u8 {
        (y ^ !y.rotate_left(1) & y.rotate_left(2) & y.rotate_left(3)).rotate_left(1)
    }

    fn final_s_box(y: u8) -> u8 {
        debug_assert_eq!(y >> 7, 0);

        let y_rot_1 = (y >> 6) | (y << 1);
        let y_rot_2 = (y >> 5) | (y << 2);

        let tmp = (y ^ !y_rot_1 & y_rot_2) & 0x7F;
        ((tmp >> 6) | (tmp << 1)) & 0x7F
    }

    pub fn permutation(&self, state: &mut [u32]) {
        self.concrete(state);
        for round in 0..NUM_FULL_ROUNDS {
            self.bars(state);
            Self::bricks(state);
            self.concrete(state);
            Self::add_round_constants(state, &self.round_constants[round]);
        }
        self.bars(state);
        Self::bricks(state);
        self.concrete(state);
    }

    // MDS matrix
    fn concrete(&self, state: &mut [u32]) {
        let new_state = if WIDTH == 16 {
            Self::apply_circulant(&mut MATRIX_CIRC_MDS_16_MERSENNE31_MONOLITH.clone(), state)
        } else {
            let mut shake = Shake128::default();
            shake.update("Monolith".as_bytes());
            shake.update(&[WIDTH as u8, (NUM_FULL_ROUNDS + 1) as u8]);
            shake.update(&MERSENNE_31_PRIME_FIELD_ORDER.to_le_bytes());
            shake.update(&[16, 15]);
            shake.update("MDS".as_bytes());
            let mut shake_finalized = shake.finalize_xof();
            Self::apply_cauchy_mds_matrix(&mut shake_finalized, state)
        };
        state.copy_from_slice(&new_state);
    }

    // S-box lookups
    fn bars(&self, state: &mut [u32]) {
        for state in state.iter_mut().take(NUM_BARS) {
            *state = ((self.lookup2[(*state >> 16) as u16 as usize] as u32) << 16)
                | (self.lookup1[*state as u16 as usize] as u32);
        }
    }

    // (x_{n+1})² = (x_n)² + x_{n+1}
    fn bricks(state: &mut [u32]) {
        for i in (0..state.len() - 1).rev() {
            state[i + 1] = F::add(&state[i + 1], &F::square(&state[i]));
        }
    }

    fn add_round_constants(state: &mut [u32], round_constants: &[u32]) {
        for (x, rc) in state.iter_mut().zip(round_constants) {
            *x = F::add(x, rc);
        }
    }

    // O(n²)
    fn apply_circulant(circ_matrix: &mut [u32], input: &[u32]) -> Vec<u32> {
        let mut output = vec![F::zero(); WIDTH];
        for out_i in output.iter_mut().take(WIDTH - 1) {
            *out_i = dot_product(circ_matrix, input);
            circ_matrix.rotate_right(1);
        }
        output[WIDTH - 1] = dot_product(circ_matrix, input);
        output
    }

    fn apply_cauchy_mds_matrix(shake: &mut Shake128Reader, to_multiply: &[u32]) -> Vec<u32> {
        let mut output = vec![F::zero(); WIDTH];

        let bits: u32 = u64::BITS
            - (MERSENNE_31_PRIME_FIELD_ORDER as u64)
                .saturating_sub(1)
                .leading_zeros();

        let x_mask = (1 << (bits - 9)) - 1;
        let y_mask = ((1 << bits) - 1) >> 2;

        let y = get_random_y_i(shake, WIDTH, x_mask, y_mask);
        let mut x = y.clone();
        x.iter_mut().for_each(|x_i| *x_i &= x_mask);

        for (i, x_i) in x.iter().enumerate() {
            for (j, yj) in y.iter().enumerate() {
                output[i] = F::add(
                    &output[i],
                    // We are using that x_i + yj != 0 in Mersenne31 because they are both much smaller than the modulus.
                    &F::div(&to_multiply[j], &F::add(x_i, yj))
                        .expect("x_i + yj != 0 in Mersenne31 because both are limited by bit masks ensuring they're < modulus"),
                );
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_input(width: usize) -> Vec<u32> {
        (0..width).map(|i| F::from_base_type(i as u32)).collect()
    }

    #[test]
    fn from_plonky3_concrete_width_16() {
        let mut input = get_test_input(16);
        MonolithMersenne31::<16, 5>::new().concrete(&mut input);
        assert_eq!(
            input,
            [
                3470365, 3977394, 4042151, 4025740, 4431233, 4264086, 3927003, 4259216, 3872757,
                3957178, 3820319, 3690660, 4023081, 3592814, 3688803, 3928040
            ]
        );
    }

    #[test]
    fn from_plonky3_concrete_width_12() {
        let mut input = get_test_input(12);
        MonolithMersenne31::<12, 5>::new().concrete(&mut input);
        assert_eq!(
            input,
            [
                365726249, 1885122147, 379836542, 860204337, 889139350, 1052715727, 151617411,
                700047874, 925910152, 339398001, 721459023, 464532407
            ]
        );
    }

    #[test]
    fn from_plonky3_width_16() {
        let mut input = get_test_input(16);
        MonolithMersenne31::<16, 5>::new().permutation(&mut input);
        assert_eq!(
            input,
            [
                609156607, 290107110, 1900746598, 1734707571, 2050994835, 1648553244, 1307647296,
                1941164548, 1707113065, 1477714255, 1170160793, 93800695, 769879348, 375548503,
                1989726444, 1349325635
            ]
        );
    }
}
