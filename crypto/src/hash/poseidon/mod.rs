mod parameters;
mod cairo_poseidon_constants;
use self::parameters::Parameters;

use lambdaworks_math::field::{element::FieldElement, traits::{IsField, IsPrimeField}};
use std::ops::{Add, Mul};


pub struct Poseidon<F: IsPrimeField> {
    params: Parameters<F>,
}

impl<F: IsPrimeField> Poseidon<F>
{
    pub fn new_with_params(params: Parameters<F>) -> Self {
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

    fn ensure_permuted(&self, state: &mut [FieldElement<F>], offset: &mut usize) {
        // offset should be <= rate, so really testing for equality
        if *offset >= self.params.rate {
            self.permute(state);
            *offset = 0;
        }
    }

    pub fn hash(&self, inputs: &[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, String>
    where
        F: IsField,
    {
        let t = self.params.rate + self.params.capacity;
        if inputs.is_empty() || inputs.len() >= self.params.n_partial_rounds - 1 {
            return Err("Wrong input length".to_string());
        }

        let mut state = vec![FieldElement::zero(); t];
        let mut offset: usize = 0;

        let n_remaining = inputs.len() % self.params.rate;
        if n_remaining != 0 {
            return Err(format!(
                "Input length {} must be a multiple of the hash rate {}",
                inputs.len(),
                self.params.rate
            ));
        }

        // absorb
        for input in inputs {
            self.ensure_permuted(&mut state, &mut offset);
            state[offset] += input.clone();
            offset += 1;
        }

        // squeeze
        let mut result = vec![FieldElement::zero(); self.params.rate];
        for result_element in result.iter_mut().take(self.params.rate) {
            self.ensure_permuted(&mut state, &mut offset);
            *result_element = state[offset].clone();
            offset += 1;
        }

        Ok(result)
    }
}

