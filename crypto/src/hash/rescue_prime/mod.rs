// Need to implement a Field over the prime field p
// p = 2^64 - 2^32 + 1

// I have GoldilockPrimeField

// MiniGoldilockPrimeField
// 2^32 - 2^16 + 1

use std::io::Read;

use lambdaworks_math::field::fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField;
use lambdaworks_math::{field::element::FieldElement, unsigned_integer::element::U64};

use sha3::{
    digest::{ExtendableOutput, Update},
    Shake256,
};
const P: U64 = U64::from_u64(18446744069414584321); // p = 2^64 - 2^32 + 1
const NUM_FULL_ROUNDS: usize = 7;
const ALPHA: u64 = 7;
const ALPHA_INV: u64 = 10540996611094048183;

type Fp = FieldElement<U64GoldilocksPrimeField>;

fn get_round_constants(p: u64, m: usize, c: usize, lambda: usize, num_rounds: usize) -> Vec<Fp> {
    let seed_string = format!("RPO({},{},{},{})", p, m, c, lambda);
    let mut shake = Shake256::default();
    shake.update(seed_string.as_bytes());

    let num_constants = 2 * m * num_rounds;
    let mut shake_output = shake.finalize_xof();
    let mut round_constants = Vec::new();

    for _ in 0..num_constants {
        let mut bytes = [0u8; 8];
        shake_output.read_exact(&mut bytes).unwrap();
        let constant = Fp::from(u64::from_le_bytes(bytes)); // Convert to field element
        round_constants.push(constant);
    }
    round_constants
}
fn get_mds(m: usize) -> Vec<u64> {
    match m {
        12 => vec![7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8],
        16 => vec![
            256, 2, 1073741824, 2048, 16777216, 128, 8, 16, 524288, 4194304, 1, 268435456, 1, 1024,
            2, 8192,
        ],
        _ => panic!("Unsupported state size"), // Avoid using panic
    }
}

fn ntt(state: &[u64], omega: u64, order: usize, p: u64) -> Vec<u64> {
    let mut result = vec![0u64; order];
    for i in 0..order {
        let mut sum = 0u64;
        for (j, &val) in state.iter().enumerate() {
            sum = (sum + val * mod_exp(omega, (i * j) as u64, p)) % p;
        }
        result[i] = sum;
    }
    result
}

fn intt(state: &[u64], omega: u64, order: usize, p: u64) -> Vec<u64> {
    let inv_order = mod_inv(order as u64, p);
    let mut result = ntt(state, mod_inv(omega, p), order, p);
    for res in result.iter_mut() {
        *res = (*res * inv_order) % p;
    }
    result
}

fn mod_exp(base: u64, exp: u64, modulus: u64) -> u64 {
    let mut result = 1;
    let mut base = base % modulus;
    let mut exp = exp;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exp /= 2;
    }
    result
}
fn mod_inv(a: Fp) -> Fp {
    Fp::inv(&a) // Use Lambdaworks' field inversion
}

fn apply_sbox(state: &mut [Fp], alpha: u64) {
    for x in state.iter_mut() {
        *x = x.pow(alpha); // Using the pow method of FieldElement
    }
}

fn apply_inverse_sbox(state: &mut [Fp], alpha_inv: u64) {
    for x in state.iter_mut() {
        *x = x.pow(alpha_inv); // Inverse S-Box using pow
    }
}
fn rescue_prime_optimized(
    state: &mut [Fp],
    round_constants: &[Fp],
    mds_matrix: &[Fp],
    alpha: u64,
    alpha_inv: u64,
    num_rounds: usize,
) {
    let m = state.len();

    for round in 0..num_rounds {
        // Apply MDS matrix
        *state = mds_matrix_vector_multiplication(mds_matrix, state);

        // Add round constants
        for j in 0..m {
            state[j] = state[j].add(&round_constants[round * 2 * m + j]);
        }

        // Apply S-Box
        apply_sbox(state, alpha);

        // Apply MDS again
        *state = mds_matrix_vector_multiplication(mds_matrix, state);

        // Add round constants again
        for j in 0..m {
            state[j] = state[j].add(&round_constants[round * 2 * m + m + j]);
        }

        // Apply Inverse S-Box
        apply_inverse_sbox(state, alpha_inv);
    }
}

fn mds_matrix_vector_multiplication(mds: &[Fp], state: &[Fp]) -> Vec<Fp> {
    let mut new_state = vec![Fp::zero(); state.len()];
    let m = state.len();

    for i in 0..m {
        for j in 0..m {
            new_state[i] = new_state[i].add(&mds[(i + j) % m].mul(&state[j]));
        }
    }
    new_state
}

fn rpo_hash(security_level: usize, input_sequence: Vec<Fp>) -> Vec<Fp> {
    let (m, capacity) = if security_level == 128 {
        (12, 4)
    } else {
        (16, 6)
    };
    let p = P;
    let rate = m - capacity;

    // Get MDS matrix, round constants, alpha and inverse alpha
    let mds = get_mds(m);
    let round_constants = get_round_constants(p, m, capacity, security_level, NUM_FULL_ROUNDS);
    let (alpha, alpha_inv) = (ALPHA, ALPHA_INV);

    let mut state = vec![Fp::zero(); m];
    let mut padded_input = input_sequence.clone();

    // Padding
    if input_sequence.len() % rate != 0 {
        padded_input.push(Fp::one());
        while padded_input.len() % rate != 0 {
            padded_input.push(Fp::zero());
        }
        state[0] = Fp::one(); // Domain separation
    }

    // Absorb the input
    for chunk in padded_input.chunks(rate) {
        for i in 0..rate {
            state[capacity + i] = chunk[i];
        }
        rescue_prime_optimized(
            &mut state,
            &round_constants,
            &mds,
            alpha,
            alpha_inv,
            NUM_FULL_ROUNDS,
        );
    }

    // Return squeezed output
    state[capacity..capacity + rate / 2].to_vec()
}
