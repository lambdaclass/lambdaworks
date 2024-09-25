use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use std::io::Read;

use sha3::{
    digest::{ExtendableOutput, Update},
    Shake256,
};

pub const ALPHA: u64 = 7;
pub const ALPHA_INV: u64 = 10540996611094048183;
//pub const P: u64 = 18446744069414584321; // p = 2^64 - 2^32 + 1

type Fp = FieldElement<Goldilocks64Field>;

#[derive(Clone)]
#[allow(dead_code)] // Is this needed?
enum MdsMethod {
    MatrixMultiplication,
    Ntt,
    Karatsuba,
}

pub struct RescuePrimeOptimized<const SECURITY_LEVEL: usize, const NUM_FULL_ROUNDS: usize> {
    m: usize,
    capacity: usize,
    rate: usize,
    round_constants: Vec<Fp>,
    mds_matrix: Vec<Vec<Fp>>,
    mds_vector: Vec<Fp>,
    alpha: u64,
    alpha_inv: u64,
    mds_method: MdsMethod,
}

impl<const SECURITY_LEVEL: usize, const NUM_FULL_ROUNDS: usize> Default
    for RescuePrimeOptimized<SECURITY_LEVEL, NUM_FULL_ROUNDS>
{
    fn default() -> Self {
        Self::new(MdsMethod::MatrixMultiplication)
    }
}

impl<const SECURITY_LEVEL: usize, const NUM_FULL_ROUNDS: usize>
    RescuePrimeOptimized<SECURITY_LEVEL, NUM_FULL_ROUNDS>
{
    const P: u64 = 18446744069414584321; // p = 2^64 - 2^32 + 1
    const ALPHA: u64 = 7;
    const ALPHA_INV: u64 = 10540996611094048183;

    fn new(mds_method: MdsMethod) -> Self {
        assert!(SECURITY_LEVEL == 128 || SECURITY_LEVEL == 160);

        let (m, capacity) = if SECURITY_LEVEL == 128 {
            (12, 4)
        } else {
            (16, 6)
        };
        let rate = m - capacity;

        let mds_vector = Self::get_mds_vector(m);
        let mds_matrix = Self::generate_circulant_matrix(&mds_vector);

        let round_constants = Self::instantiate_round_constants(
            Self::P,
            m,
            capacity,
            SECURITY_LEVEL,
            NUM_FULL_ROUNDS,
        );

        Self {
            m,
            capacity,
            rate,
            round_constants,
            mds_matrix,
            mds_vector,
            alpha: Self::ALPHA,
            alpha_inv: Self::ALPHA_INV,
            mds_method,
        }
    }

    pub fn apply_inverse_sbox(state: &mut [Fp]) {
        // Compute x^(ALPHA_INV) using an optimized method
        // ALPHA_INV = 10540996611094048183
        // Binary representation: 1001001001001001001001001001000110110110110110110110110110110111

        // Precompute necessary powers
        let mut t1 = state.to_vec(); // x^2
        for x in t1.iter_mut() {
            *x = x.square();
        }

        let mut t2 = t1.clone(); // x^4
        for x in t2.iter_mut() {
            *x = x.square();
        }

        let t3 = Self::exp_acc(&t2, &t2, 3); // x^100100

        let t4 = Self::exp_acc(&t3, &t3, 6); // x^100100100100

        let t5 = Self::exp_acc(&t4, &t4, 12); // x^100100100100100100100100

        let t6 = Self::exp_acc(&t5, &t3, 6); // x^100100100100100100100100100100

        let t7 = Self::exp_acc(&t6, &t6, 31); // x^1001001001001001001001001001000100100100100100100100100100100

        for i in 0..state.len() {
            let a = (t7[i].square() * t6[i].clone()).square().square();
            let b = t1[i].clone() * t2[i].clone() * state[i].clone();
            state[i] = a * b;
        }
    }
    #[inline(always)]
    fn exp_acc(base: &[Fp], tail: &[Fp], num_squarings: usize) -> Vec<Fp> {
        let mut result = base.to_vec();
        for x in result.iter_mut() {
            for _ in 0..num_squarings {
                *x = x.square();
            }
        }
        result
            .iter_mut()
            .zip(tail.iter())
            .for_each(|(r, t)| *r = *r * t.clone());
        result
    }
    fn get_mds_vector(m: usize) -> Vec<Fp> {
        match m {
            12 => vec![7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8]
                .into_iter()
                .map(Fp::from)
                .collect(),
            16 => vec![
                256, 2, 1073741824, 2048, 16777216, 128, 8, 16, 524288, 4194304, 1, 268435456, 1,
                1024, 2, 8192,
            ]
            .into_iter()
            .map(Fp::from)
            .collect(),
            _ => panic!("Unsupported state size"),
        }
    }

    fn generate_circulant_matrix(mds_vector: &[Fp]) -> Vec<Vec<Fp>> {
        let m = mds_vector.len();
        let mut mds_matrix = vec![vec![Fp::zero(); m]; m];
        for i in 0..m {
            for j in 0..m {
                mds_matrix[i][j] = mds_vector[(j + m - i) % m].clone();
            }
        }
        mds_matrix
    }
    /*
    fn instantiate_round_constants(
        p: u64,
        m: usize,
        capacity: usize,
        security_level: usize,
        num_rounds: usize,
    ) -> Vec<Fp> {
        let seed_string = format!("RPO({},{},{},{})", p, m, capacity, security_level);
        let mut shake = Shake256::default();
        shake.update(seed_string.as_bytes());

        let num_constants = 2 * m * num_rounds;
        let mut shake_output = shake.finalize_xof();
        let mut round_constants = Vec::new();

        for _ in 0..num_constants {
            let mut bytes = [0u8; 8];
            shake_output.read_exact(&mut bytes).unwrap();
            let constant = Fp::from(u64::from_le_bytes(bytes));
            round_constants.push(constant);
        }
        round_constants
    }
    */
    /*pub fn instantiate_round_constants(
        p: u64,
        m: usize,
        capacity: usize,
        security_level: usize,
        num_rounds: usize,
    ) -> Vec<Fp> {
        let seed_string = format!("RPO({},{},{},{})", p, m, capacity, security_level);
        let mut shake = Shake256::default();
        shake.update(seed_string.as_bytes());

        let num_constants = 2 * m * num_rounds;
        let bytes_per_int = ((p as f64).log2() / 8.0).ceil() as usize + 1; // Should be 9
        let num_bytes = bytes_per_int * num_constants;

        let mut shake_output = shake.finalize_xof();
        let mut test_bytes = vec![0u8; num_bytes];
        shake_output.read_exact(&mut test_bytes).unwrap();

        // Output the first few bytes for comparison
        println!("SHAKE256 output (Rust): {:?}", &test_bytes[..64]);

        let mut round_constants = Vec::new();

        for i in 0..num_constants {
            let start = i * bytes_per_int;
            let end = start + bytes_per_int;
            let bytes = &test_bytes[start..end];
            let integer = bytes
                .iter()
                .enumerate()
                .fold(0u128, |acc, (i, &b)| acc + (b as u128) << (8 * i));
            let reduced = integer % (p as u128);
            let constant = Fp::from(reduced as u64);
            round_constants.push(constant);
        }
        round_constants
    }*/
    fn instantiate_round_constants(
        p: u64,
        m: usize,
        capacity: usize,
        security_level: usize,
        num_rounds: usize,
    ) -> Vec<Fp> {
        let seed_string = format!("RPO({},{},{},{})", p, m, capacity, security_level);
        let mut shake = Shake256::default();
        shake.update(seed_string.as_bytes());

        let num_constants = 2 * m * num_rounds;
        let bytes_per_int = 8; // Use 8 bytes per integer
        let num_bytes = bytes_per_int * num_constants;

        let mut shake_output = shake.finalize_xof();
        let mut test_bytes = vec![0u8; num_bytes];
        shake_output.read_exact(&mut test_bytes).unwrap();

        // Output the first few bytes for comparison
        println!("SHAKE256 output (Rust): {:?}", &test_bytes[..64]);

        let mut round_constants = Vec::new();

        for i in 0..num_constants {
            let start = i * bytes_per_int;
            let end = start + bytes_per_int;
            let bytes = &test_bytes[start..end];

            // Convert bytes to integer in **little-endian** order
            let integer = u64::from_le_bytes(bytes.try_into().unwrap());
            let constant = Fp::from(integer);
            round_constants.push(constant);
        }
        round_constants
    }

    pub fn apply_sbox(state: &mut [Fp]) {
        for x in state.iter_mut() {
            *x = x.pow(Self::ALPHA);
        }
    }

    /*
        pub fn apply_inverse_sbox(state: &mut [Fp], alpha_inv: u64) {
            for x in state.iter_mut() {
                *x = x.pow(alpha_inv);
            }
        }
    */

    fn mds_matrix_vector_multiplication(&self, state: &[Fp]) -> Vec<Fp> {
        let m = state.len();
        let mut new_state = vec![Fp::zero(); m];
        for i in 0..m {
            for j in 0..m {
                new_state[i] = new_state[i] + self.mds_matrix[i][j] * state[j];
            }
        }
        new_state
    }

    fn mds_ntt(&self, state: &[Fp]) -> Vec<Fp> {
        let m = state.len();
        let omega = match m {
            12 => Fp::from(281474976645120u64),
            16 => Fp::from(17293822564807737345u64),
            _ => panic!("Unsupported state size for NTT"),
        };

        // NTT of MDS vector and state
        let mds_ntt = ntt(&self.mds_vector, omega);
        let state_rev: Vec<Fp> = std::iter::once(state[0].clone())
            .chain(state[1..].iter().rev().cloned())
            .collect();
        let state_ntt = ntt(&state_rev, omega);

        // Point-wise multiplication
        let mut product_ntt = vec![Fp::zero(); m];
        for i in 0..m {
            product_ntt[i] = mds_ntt[i] * state_ntt[i];
        }

        // Inverse NTT
        let omega_inv = omega.inv().unwrap();
        let result = intt(&product_ntt, omega_inv);

        // Adjust the result
        std::iter::once(result[0].clone())
            .chain(result[1..].iter().rev().cloned())
            .collect()
    }

    fn mds_karatsuba(&self, state: &[Fp]) -> Vec<Fp> {
        let m = state.len();
        let mds_rev: Vec<Fp> = std::iter::once(self.mds_vector[0].clone())
            .chain(self.mds_vector[1..].iter().rev().cloned())
            .collect();

        let conv = karatsuba(&mds_rev, state);

        let mut result = vec![Fp::zero(); m];
        for i in 0..m {
            result[i] = conv[i].clone();
        }
        for i in m..conv.len() {
            result[i - m] = result[i - m] + conv[i].clone();
        }

        result
    }

    fn apply_mds(&self, state: &mut [Fp]) {
        let new_state = match self.mds_method {
            MdsMethod::MatrixMultiplication => self.mds_matrix_vector_multiplication(state),
            MdsMethod::Ntt => self.mds_ntt(state),
            MdsMethod::Karatsuba => self.mds_karatsuba(state),
        };
        state.copy_from_slice(&new_state);
    }

    fn add_round_constants(&self, state: &mut [Fp], round: usize) {
        let m = self.m;
        let round_constants = &self.round_constants;

        for j in 0..m {
            state[j] = state[j] + round_constants[round * 2 * m + j];
        }
    }

    fn add_round_constants_second(&self, state: &mut [Fp], round: usize) {
        let m = self.m;
        let round_constants = &self.round_constants;

        for j in 0..m {
            state[j] = state[j] + round_constants[round * 2 * m + m + j];
        }
    }

    pub fn permutation(&self, state: &mut [Fp]) {
        let num_rounds = NUM_FULL_ROUNDS;
        let alpha = self.alpha;
        let alpha_inv = self.alpha_inv;

        for round in 0..num_rounds {
            self.apply_mds(state);
            self.add_round_constants(state, round);
            Self::apply_sbox(state);
            self.apply_mds(state);
            self.add_round_constants_second(state, round);
            Self::apply_inverse_sbox(state);
        }
    }

    pub fn hash(&self, input_sequence: &[Fp]) -> Vec<Fp> {
        let m = self.m;
        let capacity = self.capacity;
        let rate = self.rate;

        let mut state = vec![Fp::zero(); m];
        let mut padded_input = input_sequence.to_vec();

        // Check if padding is needed
        let needs_padding = input_sequence.len() % rate != 0;

        if needs_padding {
            // Set domain separation flag
            state[0] = Fp::one();
        }

        // Absorb full chunks
        let full_chunks = padded_input.len() / rate;
        for i in 0..full_chunks {
            let chunk = &padded_input[i * rate..(i + 1) * rate];
            for j in 0..rate {
                state[capacity + j] += chunk[j];
            }
            self.permutation(&mut state);
        }

        // Handle the last chunk if it's incomplete
        let remaining = padded_input.len() % rate;
        if remaining != 0 {
            let start = full_chunks * rate;
            let chunk = &padded_input[start..];

            // Absorb the remaining elements
            for j in 0..chunk.len() {
                state[capacity + j] += chunk[j];
            }

            // Add padding
            state[capacity + chunk.len()] += Fp::one();

            // Perform permutation
            self.permutation(&mut state);
        }

        // Return squeezed output
        state[capacity..capacity + rate / 2].to_vec()
    }

    pub fn hash_bytes(&self, input: &[u8]) -> Vec<Fp> {
        let field_elements = bytes_to_field_elements(input);
        self.hash(&field_elements)
    }
}

fn bytes_to_field_elements(input: &[u8]) -> Vec<Fp> {
    let mut elements = Vec::new();

    let chunk_size = 7; // Use 7 bytes per chunk
    let mut buf = [0u8; 8]; // Buffer for 8 bytes, last byte for padding

    let mut chunks = input.chunks(chunk_size).peekable();

    while let Some(chunk) = chunks.next() {
        buf.fill(0);
        buf[..chunk.len()].copy_from_slice(chunk);
        if chunk.len() < chunk_size {
            // Add a 1 after the last byte if the chunk is not full
            buf[chunk.len()] = 1;
        }
        let value = u64::from_le_bytes(buf);
        elements.push(Fp::from(value));
    }

    elements
}

// Implement NTT and INTT functions
fn ntt(input: &[Fp], omega: Fp) -> Vec<Fp> {
    let n = input.len();
    let mut output = vec![Fp::zero(); n];
    for i in 0..n {
        let mut sum = Fp::zero();
        for (j, val) in input.iter().enumerate() {
            sum = sum + *val * omega.pow((i * j) as u64);
        }
        output[i] = sum;
    }
    output
}

fn intt(input: &[Fp], omega_inv: Fp) -> Vec<Fp> {
    let n = input.len();
    let inv_n = Fp::from(n as u64).inv().unwrap();
    let mut output = ntt(input, omega_inv);
    for val in output.iter_mut() {
        *val = *val * inv_n;
    }
    output
}

// Implement Karatsuba multiplication
fn karatsuba(lhs: &[Fp], rhs: &[Fp]) -> Vec<Fp> {
    let n = lhs.len();
    if n <= 32 {
        // For small n, use the standard multiplication
        let mut result = vec![Fp::zero(); 2 * n - 1];
        for i in 0..n {
            for j in 0..n {
                result[i + j] = result[i + j] + lhs[i] * rhs[j];
            }
        }
        return result;
    }

    let half = n / 2;

    let lhs_low = &lhs[..half];
    let lhs_high = &lhs[half..];
    let rhs_low = &rhs[..half];
    let rhs_high = &rhs[half..];

    let z0 = karatsuba(lhs_low, rhs_low);
    let z2 = karatsuba(lhs_high, rhs_high);

    let lhs_sum: Vec<Fp> = lhs_low
        .iter()
        .zip(lhs_high.iter())
        .map(|(a, b)| *a + *b)
        .collect();

    let rhs_sum: Vec<Fp> = rhs_low
        .iter()
        .zip(rhs_high.iter())
        .map(|(a, b)| *a + *b)
        .collect();

    let z1 = karatsuba(&lhs_sum, &rhs_sum);

    let mut result = vec![Fp::zero(); 2 * n - 1];

    for i in 0..z0.len() {
        result[i] = result[i] + z0[i];
    }

    for i in 0..z1.len() {
        result[i + half] = result[i + half] + z1[i]
            - z0.get(i).cloned().unwrap_or(Fp::zero())
            - z2.get(i).cloned().unwrap_or(Fp::zero());
    }

    for i in 0..z2.len() {
        result[i + 2 * half] = result[i + 2 * half] + z2[i];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::rescue_prime::*;
    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    //const ALPHA: u64 = 7;
    //const ALPHA_INV: u64 = 10540996611094048183;

    fn rand_field_element<R: Rng>(rng: &mut R) -> Fp {
        Fp::from(rng.gen::<u64>())
    }

    #[test]
    fn test_alphas() {
        let mut rng = StdRng::seed_from_u64(0);
        let e = rand_field_element(&mut rng);
        let e_exp = e.pow(ALPHA);
        assert_eq!(e, e_exp.pow(ALPHA_INV));
    }

    #[test]
    fn test_sbox() {
        let mut rng = StdRng::seed_from_u64(1);
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let mut expected = state.clone();
        expected.iter_mut().for_each(|v| *v = v.pow(ALPHA));

        RescuePrimeOptimized::<128, 7>::apply_sbox(&mut state);
        assert_eq!(expected, state);
    }

    #[test]
    fn test_inv_sbox() {
        let mut rng = StdRng::seed_from_u64(2);
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let mut expected = state.clone();
        expected.iter_mut().for_each(|v| *v = v.pow(ALPHA_INV));

        RescuePrimeOptimized::<128, 7>::apply_inverse_sbox(&mut state);
        assert_eq!(expected, state);
    }

    #[test]
    fn hash_padding() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);

        let input1 = vec![1u8, 2, 3];
        let input2 = vec![1u8, 2, 3, 0];

        let hash1 = rescue.hash_bytes(&input1);
        let hash2 = rescue.hash_bytes(&input2);

        assert_ne!(hash1, hash2);

        let input1 = vec![1_u8, 2, 3, 4, 5, 6];
        let input2 = vec![1_u8, 2, 3, 4, 5, 6, 0];

        let hash1 = rescue.hash_bytes(&input1);
        let hash2 = rescue.hash_bytes(&input2);
        assert_ne!(hash1, hash2);

        let input1 = vec![1_u8, 2, 3, 4, 5, 6, 7, 0, 0];
        let input2 = vec![1_u8, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0];

        let hash1 = rescue.hash_bytes(&input1);
        let hash2 = rescue.hash_bytes(&input2);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn sponge_zeroes_collision() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);

        let mut zeroes = Vec::new();
        let mut hashes = std::collections::HashSet::new();

        for _ in 0..255 {
            let hash = rescue.hash(&zeroes);
            assert!(hashes.insert(hash));
            zeroes.push(Fp::zero());
        }
    }

    #[test]
    fn test_mds_methods_consistency() {
        let rescue_matrix = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let rescue_ntt = RescuePrimeOptimized::<128, 7>::new(MdsMethod::Ntt);
        let rescue_karatsuba = RescuePrimeOptimized::<128, 7>::new(MdsMethod::Karatsuba);

        let input = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
            Fp::from(5u64),
            Fp::from(6u64),
            Fp::from(7u64),
            Fp::from(8u64),
            Fp::from(9u64),
        ];

        let hash_matrix = rescue_matrix.hash(&input);
        let hash_ntt = rescue_ntt.hash(&input);
        let hash_karatsuba = rescue_karatsuba.hash(&input);

        assert_eq!(hash_matrix, hash_ntt);
        assert_eq!(hash_ntt, hash_karatsuba);
    }

    // Creaate a test function that generates test vectors, maybe it would be better to
    // generate them using proptest or in a separate file
    #[test]
    fn generate_test_vectors() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let elements = vec![
            Fp::from(0u64),
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
            Fp::from(5u64),
            Fp::from(6u64),
            Fp::from(7u64),
            Fp::from(8u64),
            Fp::from(9u64),
            Fp::from(10u64),
            Fp::from(11u64),
            Fp::from(12u64),
            Fp::from(13u64),
            Fp::from(14u64),
            Fp::from(15u64),
            Fp::from(16u64),
            Fp::from(17u64),
            Fp::from(18u64),
        ];

        println!("let expected_hashes = vec![");
        for i in 0..elements.len() {
            let input = &elements[..=i]; // Take prefix up to i
            let hash_output = rescue.hash(input);

            print!("    vec![");
            for value in &hash_output {
                print!("Fp::from({}u64), ", value.value());
            }
            println!("],");
        }
        println!("];");
    }

    // This test is wrong, it should be the same that the values from the paper
    #[test]
    fn hash_test_vectors() {
        let elements = vec![
            Fp::new(0u64),
            Fp::new(1u64),
            Fp::new(2u64),
            Fp::new(3u64),
            Fp::new(4u64),
            Fp::new(5u64),
            Fp::new(6u64),
            Fp::new(7u64),
            Fp::new(8u64),
            Fp::new(9u64),
            Fp::new(10u64),
            Fp::new(11u64),
            Fp::new(12u64),
            Fp::new(13u64),
            Fp::new(14u64),
            Fp::new(15u64),
            Fp::new(16u64),
            Fp::new(17u64),
            Fp::new(18u64),
        ];

        let expected_hashes = vec![
            vec![
                Fp::from(17254761148825111576u64),
                Fp::from(5748658173944016543u64),
                Fp::from(15507216263332757191u64),
                Fp::from(3983249929669394122u64),
            ],
            vec![
                Fp::from(17712221936341625086u64),
                Fp::from(2943298861192409284u64),
                Fp::from(2494572860652577379u64),
                Fp::from(2378199810979427322u64),
            ],
            vec![
                Fp::from(3876314992495866678u64),
                Fp::from(17611388455687538623u64),
                Fp::from(3911042865754506040u64),
                Fp::from(16776766772018109848u64),
            ],
            vec![
                Fp::from(6056716262596077506u64),
                Fp::from(16158290354505703086u64),
                Fp::from(17447029989314528820u64),
                Fp::from(1567470650296395962u64),
            ],
            vec![
                Fp::from(13380585531133108000u64),
                Fp::from(3137417240495852984u64),
                Fp::from(3098660641723081460u64),
                Fp::from(5150917506181658097u64),
            ],
            vec![
                Fp::from(7024209367141755347u64),
                Fp::from(16246734205622419915u64),
                Fp::from(7503077358698812671u64),
                Fp::from(12133031123118477720u64),
            ],
            vec![
                Fp::from(8402140550106217856u64),
                Fp::from(7956967817259077006u64),
                Fp::from(7462144441524583670u64),
                Fp::from(16871123896451099924u64),
            ],
            vec![
                Fp::from(3306774948807044313u64),
                Fp::from(9076368178691092936u64),
                Fp::from(2759350540710171864u64),
                Fp::from(3210614416697826413u64),
            ],
            vec![
                Fp::from(9674626283660829003u64),
                Fp::from(7912357911043410654u64),
                Fp::from(11533209507830100464u64),
                Fp::from(10170478333989115619u64),
            ],
            vec![
                Fp::from(10555879534445000475u64),
                Fp::from(7964878826308278072u64),
                Fp::from(9043911582507760001u64),
                Fp::from(17545606895180851135u64),
            ],
            vec![
                Fp::from(11622987728342977762u64),
                Fp::from(6359923222102476051u64),
                Fp::from(13174910429968985338u64),
                Fp::from(15940765068206503257u64),
            ],
            vec![
                Fp::from(9317736379809246942u64),
                Fp::from(1131339481094147037u64),
                Fp::from(7694193714172738957u64),
                Fp::from(3021948153504618318u64),
            ],
            vec![
                Fp::from(15452378822943328512u64),
                Fp::from(8092427130361699902u64),
                Fp::from(4291324087873637870u64),
                Fp::from(7948600115971845347u64),
            ],
            vec![
                Fp::from(7316340811681411604u64),
                Fp::from(12607667321921907916u64),
                Fp::from(1038271716522639176u64),
                Fp::from(14471693297474155620u64),
            ],
            vec![
                Fp::from(12773488141023465780u64),
                Fp::from(6221729007489926125u64),
                Fp::from(1486696495069601684u64),
                Fp::from(5967114319922573516u64),
            ],
            vec![
                Fp::from(12321222925119640558u64),
                Fp::from(3287873714193039570u64),
                Fp::from(14642702142995841541u64),
                Fp::from(15416139920349778032u64),
            ],
            vec![
                Fp::from(1397847368400388990u64),
                Fp::from(10132081287571009963u64),
                Fp::from(8992380008215239242u64),
                Fp::from(13825336864150598095u64),
            ],
            vec![
                Fp::from(11938166169298599670u64),
                Fp::from(6941295435497807075u64),
                Fp::from(1474794246787649407u64),
                Fp::from(13435514261569247470u64),
            ],
            vec![
                Fp::from(488103042505954048u64),
                Fp::from(953948736820844501u64),
                Fp::from(18197062251142516718u64),
                Fp::from(459513752465468917u64),
            ],
        ];

        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);

        for i in 0..elements.len() {
            let input = &elements[..=i]; // Take prefix up to index i
            let hash_output = rescue.hash(input);

            let expected_hash = &expected_hashes[i];
            assert_eq!(hash_output, *expected_hash, "Hash mismatch at index {}", i);
        }
    }
    // This repo https://github.com/jonathanxuu/RescuePrimeOptimiezd/tree/main
    // uses the crate proptest to generate random inputs and compare the results
    // should we do the same?

    proptest! {
        #[test]
        fn rescue_hash_wont_panic_with_arbitrary_input(input in any::<Vec<u8>>()) {
            let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
            let _ = rescue.hash_bytes(&input);
        }
    }

    #[test]
    fn test_instantiate_round_constants() {
        let p = 18446744069414584321u64; // 2^64 - 2^32 + 1
        let m = 12;
        let capacity = 4;
        let security_level = 128;
        let num_rounds = 7;

        // Call the function to instantiate round constants
        let round_constants = RescuePrimeOptimized::<128, 7>::instantiate_round_constants(
            p,
            m,
            capacity,
            security_level,
            num_rounds,
        );

        // Print the first few round constants for inspection
        println!("First few round constants:");
        for (i, constant) in round_constants.iter().enumerate() {
            println!("Constant {}: {:?}", i, constant);
        }
    }
    #[test]
    fn test_hash_outputs() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::Ntt);

        for i in 1..6 {
            let input_sequence: Vec<Fp> = (0..i as u64).map(Fp::from).collect();
            let hash_output = rescue.hash(&input_sequence);

            println!("Input {}: {:?}", i, input_sequence);
            println!("Hash {}: {:?}", i, hash_output);
        }
    }
}
