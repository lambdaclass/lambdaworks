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

// O(n²)
pub fn apply_circulant(circ_matrix: &mut Vec<u32>, input: &Vec<u32>) -> Vec<u32> {
    let width = input.len();
    let mut output = vec![F::zero(); width];
    for out_i in output.iter_mut().take(width - 1) {
        *out_i = dot_product(&circ_matrix, &input);
        circ_matrix.rotate_right(1);
    }
    output[width - 1] = dot_product(&circ_matrix, &input);
    output
}

pub fn apply_cauchy_mds_matrix(shake: &mut Shake128Reader, to_multiply: &Vec<u32>) -> Vec<u32> {
    let width = to_multiply.len();
    let mut output = vec![F::zero(); width];

    let bits: u32 = u64::BITS
        - (MERSENNE_31_PRIME_FIELD_ORDER as u64)
            .saturating_sub(1)
            .leading_zeros();

    let x_mask = (1 << (bits - 9)) - 1;
    let y_mask = ((1 << bits) - 1) >> 2;

    let y = get_random_y_i(shake, width, x_mask, y_mask);
    let mut x = y.clone();
    x.iter_mut().for_each(|x_i| *x_i &= x_mask);

    for (i, x_i) in x.iter().enumerate() {
        for (j, yj) in y.iter().enumerate() {
            output[i] = F::add(&output[i], &F::div(&to_multiply[j], &F::add(&x_i, &yj)));
        }
    }

    output
}

fn random_field_element(shake: &mut Shake128Reader) -> u32 {
    let mut val = shake_random_u32(shake);
    while val >= MERSENNE_31_PRIME_FIELD_ORDER {
        val = shake_random_u32(shake);
    }
    F::from_base_type(val)
}

fn dot_product(u: &Vec<u32>, v: &Vec<u32>) -> u32 {
    u.iter()
        .zip(v)
        .map(|(x, y)| F::mul(x, y))
        .reduce(|a, b| F::add(&a, &b))
        .unwrap()
}

fn get_random_y_i(shake: &mut Shake128Reader, width: usize, x_mask: u32, y_mask: u32) -> Vec<u32> {
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
