use alloc::vec::Vec;
use lambdaworks_math::field::{
    fields::mersenne31::field::{Mersenne31Field, MERSENNE_31_PRIME_FIELD_ORDER},
    traits::IsField,
};
use sha3::{digest::XofReader, Shake128Reader};

// Ported from https://github.com/Plonky3/Plonky3/blob/main/monolith

pub type F = Mersenne31Field;

pub fn random_matrix(shake: &mut Shake128Reader, n: usize, m: usize) -> Vec<Vec<u32>> {
    (0..n)
        .map(|_| (0..m).map(|_| random_field_element(shake)).collect())
        .collect()
}

fn random_field_element(shake: &mut Shake128Reader) -> u32 {
    let mut val = shake_random_u32(shake);
    while val >= MERSENNE_31_PRIME_FIELD_ORDER {
        val = shake_random_u32(shake);
    }
    F::from_base_type(val)
}

pub fn dot_product(u: &[u32], v: &[u32]) -> u32 {
    Mersenne31Field::sum(u.iter().zip(v).map(|(x, y)| F::mul(x, y)))
}

pub fn get_random_y_i(
    shake: &mut Shake128Reader,
    width: usize,
    x_mask: u32,
    y_mask: u32,
) -> Vec<u32> {
    let mut res = vec![0; width];

    for i in 0..width {
        let mut y_i = shake_random_u32(shake) & y_mask;
        let mut x_i = y_i & x_mask;
        while res.iter().take(i).any(|r| r & x_mask == x_i) {
            y_i = shake_random_u32(shake) & y_mask;
            x_i = y_i & x_mask;
        }
        res[i] = y_i;
    }

    res
}

fn shake_random_u32(shake: &mut Shake128Reader) -> u32 {
    let mut rand = [0u8; 4];
    shake.read(&mut rand);
    u32::from_le_bytes(rand)
}
