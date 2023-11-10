use crate::hash::rescue::parameters::Parameters;
use lambdaworks_math::field::{
    traits::{IsField, IsPrimeField}, element::FieldElement,
};


#[allow(dead_code)] //TODO: Remove when finalized
#[derive(Clone)]
pub struct Rescue<F: IsField, const WIDTH: usize, const CAPACITY: usize, const SECURITY_LEVEL: usize> {
    params: Parameters<F, WIDTH, CAPACITY, SECURITY_LEVEL>,
}

pub trait RescueTrait<F: IsPrimeField, const WIDTH: usize>: Permutation<[FieldElement<F>;WIDTH]> + SBoxLayer<F, WIDTH> + MDS<F, WIDTH> {
    fn hash(&self, input_sequence: [FieldElement<F>; WIDTH]) -> [FieldElement<F>; WIDTH];
    fn permutation(&self, state: &mut [FieldElement<F>; WIDTH]);
}

impl<F: IsPrimeField + Copy, const WIDTH: usize, const CAPACITY: usize, const SECURITY_LEVEL: usize> 
RescueTrait<F,WIDTH> for Rescue<F, WIDTH, CAPACITY, SECURITY_LEVEL>
{
    fn hash(&self, mut input_sequence: [FieldElement<F>; WIDTH]) -> [FieldElement<F>; WIDTH] {
        self.permutation( &mut input_sequence);
        input_sequence
    }

    fn permutation(&self,state: &mut [FieldElement<F>; WIDTH]){
        for round in 0..self.params.n {

            self.sbox_layer(state);
            self.mds_permute(state);
            
            //  Add round constants
            for (state_item, round_constant) in state
                .iter_mut()
                .zip(&self.params.round_constants[round * WIDTH * 2..])
            {
                *state_item += round_constant.clone();
            }

            self.inv_sbox_layer(state);
            self.mds_permute(state);
            
            // Add round constants
            for (state_item, round_constant) in state
                .iter_mut()
                .zip(&self.params.round_constants[round * WIDTH * 2..])
            {
                *state_item += round_constant.clone();
            }

        }
    }
}

pub trait Permutation<T: Clone>: Clone {
    fn permute(&self, mut state: T) -> T {
        self.permute_mut(&mut state);
        state
    }
    fn permute_mut(&self, state: &mut T);
}

impl<F: IsPrimeField, const WIDTH: usize, const CAPACITY: usize, const SECURITY_LEVEL: usize> Permutation<[FieldElement<F>;WIDTH]> for Rescue<F, WIDTH, CAPACITY, SECURITY_LEVEL>
{
    fn permute(&self, mut input: [FieldElement<F>;WIDTH]) -> [FieldElement<F>;WIDTH] {
        self.permute_mut(&mut input);
        input
    }

    fn permute_mut(&self, state: &mut [FieldElement<F>;WIDTH]) {
            for round in 0..self.params.n {
            // S-box
            self.sbox_layer(state);

            // MDS
            self.permute_mut(state);

            // Constants
            for (state_item, round_constant) in state
                .iter_mut()
                .zip(&self.params.round_constants[round * WIDTH * 2..])
            {
                *state_item += round_constant.clone();
            }

            // Inverse S-box
            self.inv_sbox_layer(state);

            // MDS
            self.permute_mut(state);

            // Constants
            for (state_item, round_constant) in state
                .iter_mut()
                .zip(&self.params.round_constants[round * WIDTH * 2 + WIDTH..])
            {
                *state_item += round_constant.clone();
            }
    }}

}

pub trait SBoxLayer<F: IsField, const WIDTH: usize> {
    fn sbox_layer(&self, state: &mut[FieldElement<F>;WIDTH]);
    fn inv_sbox_layer(&self, state: &mut[FieldElement<F>;WIDTH]);
}

impl<F: IsField, const WIDTH: usize, const CAPACITY: usize, const SECURITY_LEVEL: usize> SBoxLayer<F,WIDTH> for Rescue<F, WIDTH, CAPACITY, SECURITY_LEVEL>
{
    fn sbox_layer(&self, state: &mut [FieldElement<F>; WIDTH]) {
        for x in state.iter_mut() {
            *x = x.pow(self.params.alpha);
        }
    }

    fn inv_sbox_layer(&self, state: &mut [FieldElement<F>; WIDTH]) {
        for x in state.iter_mut() {
            *x = x.pow(self.params.alpha_inv);
        }
    }
}

pub trait MDS<F: IsField, const WIDTH: usize> {
    fn mds_permute(&self, state: &mut [FieldElement<F>; WIDTH]);
}


impl<F: IsField + Copy, const WIDTH: usize, const CAPACITY: usize, const SECURITY_LEVEL: usize> MDS<F,WIDTH> for Rescue<F, WIDTH, CAPACITY, SECURITY_LEVEL> 
{
    fn mds_permute(&self, state: &mut [FieldElement<F>; WIDTH]) {
        fn linear_combination_u64<T : IsField>(u: &[u64], v: &[FieldElement<T>]) -> FieldElement<T> {
            assert_eq!(u.len(), v.len(), "The lengths of u and v must be the same.");
        
            let mut result = FieldElement::<T>::zero();
        
            for (ui, vi) in u.iter().zip(v.iter()) {
                // Perform the field multiplication and addition
                result = result + FieldElement::<T>::from(*ui) * vi;
            }      
        
            result
        }

        fn apply_circulant_12_sml<F: IsField + Copy>(state: &mut [FieldElement<F>]) where <F as lambdaworks_math::field::traits::IsField>::BaseType: std::marker::Copy {
        // Check that the state has the correct length to apply the MDS matrix.
            assert_eq!(state.len(), 12, "State must be of length 12");

            let mut new_state = [FieldElement::<F>::zero(); 12];

            for i in 0..12 {
                // Generate the i-th row of the circulant matrix by rotating the first row
                let rotated_matrix_row = rotate_right(MATRIX_CIRC_MDS_12_SML, i);

                // Compute the linear combination of the state with the i-th row of the MDS matrix
                new_state[i] = linear_combination_u64(
                    &rotated_matrix_row,
                    state,
                );
            }

            for (s, &new_s) in state.iter_mut().zip(new_state.iter()) {
                *s = new_s;
            }
        }

        // Helper function to rotate an array to the right.
        fn rotate_right<const N: usize>(input: [u64; N], offset: usize) -> [u64; N] {
            let mut output = [0u64; N];
            let offset = offset % N; // Ensure the offset is within the bounds of the array size
            for (i, item) in input.iter().enumerate() {
                output[(i + offset) % N] = *item;
            }
            output
        }

        const MATRIX_CIRC_MDS_12_SML: [u64; 12] = [9, 7, 4, 1, 16, 2, 256, 128, 3, 32, 1, 1];
        apply_circulant_12_sml(state)
    }

}

// const PRIME: u64 = 2147483647;
// type Mersenne31 = U64PrimeField<PRIME>;
// impl MDS<Mersenne31,12> for Rescue<Mersenne31, 12, 4, 200> {
//     fn mds_permute(&self, state: &mut [FieldElement<Mersenne31>; 4]) {
//         todo!() // state
//     }
// }

// impl Default for Rescue<Stark252PrimeField> {
//     fn default() -> Self {
//         Self {
//             params: Parameters::<Stark252PrimeField>::new().expect("Error loading parameters"),
//         }
//     }
// }
