mod parameters;
mod cairo_poseidon_constants;
use self::parameters::PermutationParameters;

use lambdaworks_math::field::{element::FieldElement, traits::{IsField, IsPrimeField}};
use std::ops::{Add, Mul};


pub struct Poseidon<F: IsPrimeField> {
    params: PermutationParameters<F>,
    // Suggestion: Add the state here
}

impl<F: IsPrimeField> Poseidon<F>
{
    pub fn new_with_params(params: PermutationParameters<F>) -> Self {
        Poseidon { params }
    }

    pub fn ark(&self, state: &mut [FieldElement<F>], round_number: usize) {
        for i in 0..state.len() {
            state[i] = &state[i] + &self.params.add_round_constants[round_number][i]
        }
    }

    pub fn sbox(&self, state: &mut [FieldElement<F>], round_number: usize) {
        let is_full_round = round_number < self.params.n_full_rounds / 2
            || round_number >= self.params.n_full_rounds / 2 + self.params.n_partial_rounds;

        if is_full_round {
            // full s-box
            for current_state in state.iter_mut() {
                *current_state = current_state.pow(self.params.alpha);
            }
        } else {
            // partial s-box
            let last_state_index = state.len() - 1;
            state[last_state_index] = state[last_state_index].pow(self.params.alpha);
        }
    }

    pub fn mix(&self, state: &mut [FieldElement<F>]) {
        let mut new_state: Vec<FieldElement<F>> = Vec::with_capacity(state.len());
        for i in 0..state.len() {
            new_state.push(FieldElement::zero());
            for (j, current_state) in state.iter().enumerate() {
                let mut mij = self.params.mds_matrix[i][j].clone();
                mij = mij.mul(current_state);
                new_state[i] = new_state[i].clone().add(&mij);
            }
        }
        state.clone_from_slice(&new_state[0..state.len()]);
    }

    fn permute(&self, state: &mut [FieldElement<F>]) {
        for i in 0..(self.params.n_full_rounds + self.params.n_partial_rounds) {
            self.ark(state, i);
            self.sbox(state, i);
            self.mix(state);
        }
    }
}
