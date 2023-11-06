use self::parameters::Parameters;
use lambdaworks_math::field::{traits::IsField, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField};

mod parameters;
#[allow(dead_code)]  //TODO: Remove when finalized
pub struct Rescue<F: IsField> {
    params: Parameters<F>,
}

pub trait Permutation<T: Clone> : Clone {
    fn permute(&self, mut input: T) -> T {
        self.permute_mut(&mut input);
        input
    }
    fn permute_mut(&self, input: &mut T);
}

pub trait SBoxLayer {
    fn sbox_layer() {}
    fn inv_sbox_layer() {}
}

impl Default for Rescue<Stark252PrimeField> {
    fn default() -> Self {
        Self { params: Parameters::<Stark252PrimeField>::new().expect("Error loading parameters") }
    }
}