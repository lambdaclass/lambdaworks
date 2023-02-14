/// Poseidon implementation for curve BLS12381
use self::parameters::Parameters;

use super::traits::IsCryptoHash;
use lambdaworks_math::{
    elliptic_curve::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::{element::FieldElement, traits::IsField},
};
use std::{ops::{Add, Mul}};

mod parameters;
mod test_helper;

pub struct Poseidon<F: IsField> {
    pub params: Parameters<F>,
    pub state: Vec<FieldElement<F>>,
    pub offset: usize,
}

/// Implements hashing for BLS 12381's field.
/// Alpha = 5 and parameters are predefined for secure implementations
impl<F:IsField> IsCryptoHash<F> for Poseidon<F> {
    fn hash_one(
        input: FieldElement<F>,
    ) -> Result<FieldElement<F>, String> {
        self.hash(vec![input], &params)
    }

    fn hash_two(
        left: FieldElement<BLS12381PrimeField>,
        right: FieldElement<BLS12381PrimeField>,
    ) -> Result<FieldElement<BLS12381PrimeField>, String> {
        Self::hash(vec![left, right], &params)
    }
}

impl<F> Poseidon<F>
where
    F: IsField,
{
    pub fn new(params: Parameters<F>) -> Self  {
        Poseidon {
            offset: 0,
            params,
            state: vec![FieldElement::zero(); params.rate + params.capacity],
        }
    }

    fn ark(state: &mut Vec<FieldElement<F>>, round_constants: &[FieldElement<F>], round: usize) {
        for i in 0..state.len() {
            state[i] += round_constants[round + i].clone();
        }
    }

    fn sbox(
        n_rounds_f: usize,
        n_rounds_p: usize,
        state: &mut [FieldElement<F>],
        i: usize,
        alpha: u32,
    ) {
        if i < n_rounds_f / 2 || i >= n_rounds_f / 2 + n_rounds_p {
            for current_state in state.iter_mut() {
                *current_state = current_state.pow(alpha);
            }
        } else {
            state[0] = state[0].pow(alpha);
        }
    }

    fn mix(state: &Vec<FieldElement<F>>, mds: &[Vec<FieldElement<F>>]) -> Vec<FieldElement<F>> {
        let mut new_state: Vec<FieldElement<F>> = Vec::new();
        for i in 0..state.len() {
            new_state.push(FieldElement::zero());
            for (j, current_state) in state.iter().enumerate() {
                let mut mij = mds[i][j].clone();
                mij = mij.mul(current_state);
                new_state[i] = new_state[i].clone().add(&mij);
            }
        }
        new_state.clone()
    }

    fn permute(
        &self,
        inp: Vec<FieldElement<F>>,
    ) -> Result<FieldElement<F>, String> {
        let t = inp.len() + 1;
        let params = self.params;
        if inp.is_empty() || inp.len() >= params.n_partial_rounds - 1 {
            return Err("Wrong inputs length".to_string());
        }

        let mut state = vec![FieldElement::zero(); t];
        state[1..].clone_from_slice(&inp);

        for i in 0..(params.n_full_rounds + params.n_partial_rounds) {
            Self::ark(&mut state, &params.round_constants, i * t);
            Self::sbox(
                params.n_full_rounds,
                params.n_partial_rounds,
                &mut state,
                i,
                params.alpha,
            );
            state = Self::mix(&state, &params.mds_matrix);
        }

        Ok(state[0].clone())
    }

    fn ensure_permuted(&mut self) {
        // offset should be <= rate, so really testing for equality
        if self.offset >= self.params.rate {
            self.permute(self.state);
            self.offset = 0;
        }
    }

    pub fn absorb(&mut self, input: &FieldElement<F>) {
        self.ensure_permuted();
        self.state[self.offset] += input.clone();
        self.offset += 1;
    }

    pub fn squeeze(&mut self) -> FieldElement<F> {
        self.ensure_permuted();
        let result = self.state[self.offset];
        self.offset += 1;
        result
    }

    pub fn hash(&self, inputs: &[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, String>
    where
        F: IsField,
    {
        if inputs.len() == 0 {
            return Err("Empty inputs".to_string());
        }
        let n_remaining = inputs.len() % self.params.rate;
        if n_remaining != 0 {
            return Err(format!(
                "Input length {} must be a multiple of the hash rate {}",
                inputs.len(),
                self.params.rate
            )
            .to_string());
        }
    
        for input in inputs {
            self.absorb(input);
        }

        let mut result = vec![FieldElement::zero(); self.params.rate];
        
        for i in 0..self.params.rate {
            result[i] = self.squeeze();
        }
        Ok(result)
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poseidon_bls() {
        let a = FieldElement::<BLS12381PrimeField>::new_base("1");
        let b = FieldElement::<BLS12381PrimeField>::new_base("2");

        //assert_eq!(
        //    "0000000000000000000000000000000000000000000000000000000000000001",
        //    a
        //);

        let v = Poseidon::hash_one(a.clone()).unwrap();
        let v = Poseidon::hash_one(b.clone()).unwrap();

        let v = Poseidon::hash_two(a, b).unwrap();
    }
}
