use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
//use lazy_static::lazy_static;
//use std::io::Read;

//use sha3::{
//    digest::{ExtendableOutput, Update},
//    Shake256,
//};

mod parameters;
mod utils;

use parameters::*;
use utils::*;

type Fp = FieldElement<Goldilocks64Field>;

#[derive(Clone)]
#[allow(dead_code)]
enum MdsMethod {
    MatrixMultiplication,
    Ntt,
    Karatsuba,
}
#[warn(dead_code)]
pub struct RescuePrimeOptimized<const SECURITY_LEVEL: usize, const NUM_FULL_ROUNDS: usize> {
    m: usize,
    capacity: usize,
    rate: usize,
    round_constants: [Fp; 168],
    mds_matrix: Vec<Vec<Fp>>,
    mds_vector: Vec<Fp>,
    //alpha: u64,
    //alpha_inv: u64,
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
        let round_constants = ROUND_CONSTANTS;

        Self {
            m,
            capacity,
            rate,
            round_constants,
            mds_matrix,
            mds_vector,
            mds_method,
        }
    }

    pub fn apply_inverse_sbox(state: &mut [Fp]) {
        for x in state.iter_mut() {
            *x = x.pow(ALPHA_INV);
        }
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
    /*
        fn generate_circulant_matrix(mds_vector: &[Fp]) -> Vec<Vec<Fp>> {
            let m = mds_vector.len();
            let mut mds_matrix = vec![vec![Fp::zero(); m]; m];
            for i in 0..m {
                for j in 0..m {
                    mds_matrix[i][j] = mds_vector[(j + m - i) % m];
                }
            }
            mds_matrix
        }
    */
    fn generate_circulant_matrix(mds_vector: &[Fp]) -> Vec<Vec<Fp>> {
        let m = mds_vector.len();
        (0..m)
            .map(|i| {
                mds_vector
                    .iter()
                    .cycle()
                    .skip(m - i)
                    .take(m)
                    .cloned()
                    .collect::<Vec<Fp>>()
            })
            .collect()
    }

    pub fn apply_sbox(state: &mut [Fp]) {
        for x in state.iter_mut() {
            *x = x.pow(ALPHA);
        }
    }

    fn mds_matrix_vector_multiplication(&self, state: &[Fp]) -> Vec<Fp> {
        let m = state.len();
        let mut new_state = vec![Fp::zero(); m];

        for (i, new_value) in new_state.iter_mut().enumerate() {
            for (j, state_value) in state.iter().enumerate() {
                *new_value += self.mds_matrix[i][j] * state_value;
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

        let mds_ntt = ntt(&self.mds_vector, omega);
        let state_rev: Vec<Fp> = std::iter::once(state[0])
            .chain(state[1..].iter().rev().cloned())
            .collect();
        let state_ntt = ntt(&state_rev, omega);

        let mut product_ntt = vec![Fp::zero(); m];
        for i in 0..m {
            product_ntt[i] = mds_ntt[i] * state_ntt[i];
        }

        let omega_inv = omega.inv().unwrap();
        let result = intt(&product_ntt, omega_inv);

        std::iter::once(result[0])
            .chain(result[1..].iter().rev().cloned())
            .collect()
    }

    fn mds_karatsuba(&self, state: &[Fp]) -> Vec<Fp> {
        let m = state.len();
        let mds_rev: Vec<Fp> = std::iter::once(self.mds_vector[0])
            .chain(self.mds_vector[1..].iter().rev().cloned())
            .collect();

        let conv = karatsuba(&mds_rev, state);

        let mut result = vec![Fp::zero(); m];
        result[..m].copy_from_slice(&conv[..m]);
        //for i in 0..m {
        //    result[i] = conv[i].clone();
        //}

        for i in m..conv.len() {
            result[i - m] += conv[i];
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
            state[j] += round_constants[round * 2 * m + j];
        }
    }

    fn add_round_constants_second(&self, state: &mut [Fp], round: usize) {
        let m = self.m;
        let round_constants = &self.round_constants;

        for j in 0..m {
            state[j] += round_constants[round * 2 * m + m + j]
        }
    }

    pub fn permutation(&self, state: &mut [Fp]) {
        let num_rounds = NUM_FULL_ROUNDS;

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
        let input_len = input_sequence.len();

        if input_len % rate != 0 {
            state[0] = Fp::one();
        }

        let num_full_chunks = input_len / rate;

        for i in 0..num_full_chunks {
            let chunk = &input_sequence[i * rate..(i + 1) * rate];
            state[capacity..(rate + capacity)].copy_from_slice(&chunk[..rate]);
            //for j in 0..rate {
            //    state[capacity + j] = chunk[j];
            //}
            self.permutation(&mut state);
        }

        let last_chunk_size = input_len % rate;

        if last_chunk_size != 0 {
            let mut last_chunk = vec![Fp::zero(); rate];
            for j in 0..last_chunk_size {
                last_chunk[j] = input_sequence[num_full_chunks * rate + j];
            }
            last_chunk[last_chunk_size] = Fp::one();

            state[capacity..(rate + capacity)].copy_from_slice(&last_chunk[..rate]);
            //for j in 0..rate {
            //    state[capacity + j] = last_chunk[j];
            //}

            self.permutation(&mut state);
        }

        state[capacity..capacity + rate / 2].to_vec()
    }

    pub fn hash_bytes(&self, input: &[u8]) -> Vec<Fp> {
        let field_elements = bytes_to_field_elements(input);
        self.hash(&field_elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    pub const EXPECTED: [[Fp; 4]; 19] = [
        [
            Fp::const_from_raw(1502364727743950833u64),
            Fp::const_from_raw(5880949717274681448u64),
            Fp::const_from_raw(162790463902224431u64),
            Fp::const_from_raw(6901340476773664264u64),
        ],
        [
            Fp::const_from_raw(7478710183745780580u64),
            Fp::const_from_raw(3308077307559720969u64),
            Fp::const_from_raw(3383561985796182409u64),
            Fp::const_from_raw(17205078494700259815u64),
        ],
        [
            Fp::const_from_raw(17439912364295172999u64),
            Fp::const_from_raw(17979156346142712171u64),
            Fp::const_from_raw(8280795511427637894u64),
            Fp::const_from_raw(9349844417834368814u64),
        ],
        [
            Fp::const_from_raw(5105868198472766874u64),
            Fp::const_from_raw(13090564195691924742u64),
            Fp::const_from_raw(1058904296915798891u64),
            Fp::const_from_raw(18379501748825152268u64),
        ],
        [
            Fp::const_from_raw(9133662113608941286u64),
            Fp::const_from_raw(12096627591905525991u64),
            Fp::const_from_raw(14963426595993304047u64),
            Fp::const_from_raw(13290205840019973377u64),
        ],
        [
            Fp::const_from_raw(3134262397541159485u64),
            Fp::const_from_raw(10106105871979362399u64),
            Fp::const_from_raw(138768814855329459u64),
            Fp::const_from_raw(15044809212457404677u64),
        ],
        [
            Fp::const_from_raw(162696376578462826u64),
            Fp::const_from_raw(4991300494838863586u64),
            Fp::const_from_raw(660346084748120605u64),
            Fp::const_from_raw(13179389528641752698u64),
        ],
        [
            Fp::const_from_raw(2242391899857912644u64),
            Fp::const_from_raw(12689382052053305418u64),
            Fp::const_from_raw(235236990017815546u64),
            Fp::const_from_raw(5046143039268215739u64),
        ],
        [
            Fp::const_from_raw(9585630502158073976u64),
            Fp::const_from_raw(1310051013427303477u64),
            Fp::const_from_raw(7491921222636097758u64),
            Fp::const_from_raw(9417501558995216762u64),
        ],
        [
            Fp::const_from_raw(1994394001720334744u64),
            Fp::const_from_raw(10866209900885216467u64),
            Fp::const_from_raw(13836092831163031683u64),
            Fp::const_from_raw(10814636682252756697u64),
        ],
        [
            Fp::const_from_raw(17486854790732826405u64),
            Fp::const_from_raw(17376549265955727562u64),
            Fp::const_from_raw(2371059831956435003u64),
            Fp::const_from_raw(17585704935858006533u64),
        ],
        [
            Fp::const_from_raw(11368277489137713825u64),
            Fp::const_from_raw(3906270146963049287u64),
            Fp::const_from_raw(10236262408213059745u64),
            Fp::const_from_raw(78552867005814007u64),
        ],
        [
            Fp::const_from_raw(17899847381280262181u64),
            Fp::const_from_raw(14717912805498651446u64),
            Fp::const_from_raw(10769146203951775298u64),
            Fp::const_from_raw(2774289833490417856u64),
        ],
        [
            Fp::const_from_raw(3794717687462954368u64),
            Fp::const_from_raw(4386865643074822822u64),
            Fp::const_from_raw(8854162840275334305u64),
            Fp::const_from_raw(7129983987107225269u64),
        ],
        [
            Fp::const_from_raw(7244773535611633983u64),
            Fp::const_from_raw(19359923075859320u64),
            Fp::const_from_raw(10898655967774994333u64),
            Fp::const_from_raw(9319339563065736480u64),
        ],
        [
            Fp::const_from_raw(4935426252518736883u64),
            Fp::const_from_raw(12584230452580950419u64),
            Fp::const_from_raw(8762518969632303998u64),
            Fp::const_from_raw(18159875708229758073u64),
        ],
        [
            Fp::const_from_raw(14871230873837295931u64),
            Fp::const_from_raw(11225255908868362971u64),
            Fp::const_from_raw(18100987641405432308u64),
            Fp::const_from_raw(1559244340089644233u64),
        ],
        [
            Fp::const_from_raw(8348203744950016968u64),
            Fp::const_from_raw(4041411241960726733u64),
            Fp::const_from_raw(17584743399305468057u64),
            Fp::const_from_raw(16836952610803537051u64),
        ],
        [
            Fp::const_from_raw(16139797453633030050u64),
            Fp::const_from_raw(1090233424040889412u64),
            Fp::const_from_raw(10770255347785669036u64),
            Fp::const_from_raw(16982398877290254028u64),
        ],
    ];

    fn rand_field_element<R: Rng>(rng: &mut R) -> Fp {
        Fp::from(rng.gen::<u64>())
    }

    #[test]
    fn test_apply_sbox() {
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
    fn test_apply_inverse_sbox() {
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
    fn test_mds_matrix_multiplication() {
        let mut rng = StdRng::seed_from_u64(3);
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let state: Vec<Fp> = (0..rescue.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let expected_state = rescue.mds_matrix_vector_multiplication(&state);
        let mut computed_state = state.clone();
        rescue.apply_mds(&mut computed_state);

        assert_eq!(expected_state, computed_state);
    }

    #[test]
    fn test_mds_ntt() {
        let mut rng = StdRng::seed_from_u64(4);
        let rescue_ntt = RescuePrimeOptimized::<128, 7>::new(MdsMethod::Ntt);
        let state: Vec<Fp> = (0..rescue_ntt.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let expected_state = rescue_ntt.mds_ntt(&state);
        let mut computed_state = state.clone();
        rescue_ntt.apply_mds(&mut computed_state);

        assert_eq!(expected_state, computed_state);
    }

    #[test]
    fn test_mds_karatsuba() {
        let mut rng = StdRng::seed_from_u64(5);
        let rescue_karatsuba = RescuePrimeOptimized::<128, 7>::new(MdsMethod::Karatsuba);
        let state: Vec<Fp> = (0..rescue_karatsuba.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let expected_state = rescue_karatsuba.mds_karatsuba(&state);
        let mut computed_state = state.clone();
        rescue_karatsuba.apply_mds(&mut computed_state);

        assert_eq!(expected_state, computed_state);
    }

    #[test]
    fn test_add_round_constants() {
        let mut rng = StdRng::seed_from_u64(6);
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let round = 0;
        let expected_state = state
            .iter()
            .enumerate()
            .map(|(i, &x)| x + rescue.round_constants[round * 2 * rescue.m + i])
            .collect::<Vec<_>>();

        rescue.add_round_constants(&mut state, round);

        assert_eq!(expected_state, state);
    }

    #[test]
    fn test_permutation() {
        let mut rng = StdRng::seed_from_u64(7);
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let expected_state = {
            let mut temp_state = state.clone();
            for round in 0..7 {
                rescue.apply_mds(&mut temp_state);
                rescue.add_round_constants(&mut temp_state, round);
                RescuePrimeOptimized::<128, 7>::apply_sbox(&mut temp_state);
                rescue.apply_mds(&mut temp_state);
                rescue.add_round_constants_second(&mut temp_state, round);
                RescuePrimeOptimized::<128, 7>::apply_inverse_sbox(&mut temp_state);
            }
            temp_state
        };

        rescue.permutation(&mut state);

        assert_eq!(expected_state, state);
    }

    #[test]
    fn test_hash_single_chunk() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..8).map(Fp::from).collect();
        let hash_output = rescue.hash(&input_sequence);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_hash_multiple_chunks() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..16).map(Fp::from).collect(); // Two chunks of size 8
        let hash_output = rescue.hash(&input_sequence);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_hash_with_padding() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..5).map(Fp::from).collect();
        let hash_output = rescue.hash(&input_sequence);

        assert_eq!(hash_output.len(), 4);
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
    fn test_hash_bytes() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let input_bytes = b"Rescue Prime Optimized";
        let hash_output = rescue.hash_bytes(input_bytes);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_instantiate_round_constants() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);

        let round_constants = &rescue.round_constants;
        assert_eq!(round_constants.len(), 2 * 12 * 7);
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
    /*
    Test used to generate the expected hashes for the test vectors (they are the same as in the Polygon Miden implementation)
        #[test]
        fn generate_test_vectors() {
            let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
            let elements: Vec<Fp> = (0..19).map(Fp::from).collect();

            println!("let expected_hashes = vec![");
            elements.iter().enumerate().for_each(|(i, _)| {
                let input = elements.iter().take(i + 1); // Tomar el prefijo hasta i + 1
                let hash_output = rescue.hash(input.cloned().collect::<Vec<_>>().as_slice());

                print!("    vec![");
                hash_output.iter().for_each(|value| {
                    print!("Fp::from({}u64), ", value.value());
                });
                println!("],");
            });
            println!("];");
        }
        */
    #[test]
    fn test_print_round_constants() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);

        println!("Round constants:");
        for (i, constant) in rescue.round_constants.iter().enumerate() {
            println!("Constant {}: Fp::from({}u64)", i, constant.value());
        }

        assert_eq!(rescue.round_constants.len(), 2 * rescue.m * 7);
    }
    /*
        #[test]
        fn test_hash_vectors() {
            let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
            let elements: Vec<Fp> = (0..19).map(Fp::from).collect();

            for (i, expected) in EXPECTED.iter().enumerate() {
                let input = &elements[..=i]; // Tomar el prefijo hasta i
                let hash_output = rescue.hash(input);

                assert_eq!(
                    hash_output,
                    *expected,
                    "Hash mismatch for input length {}",
                    i + 1
                );
            }
        }
    */
    #[test]
    fn test_hash_vectors() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let elements: Vec<Fp> = (0..19).map(Fp::from).collect();

        EXPECTED.iter().enumerate().for_each(|(i, expected)| {
            let input = elements.iter().take(i + 1);
            let hash_output = rescue.hash(input.cloned().collect::<Vec<_>>().as_slice());

            assert_eq!(
                hash_output,
                *expected,
                "Hash mismatch for input length {}",
                i + 1
            );
        });
    }
    #[test]
    fn test_hash_example_and_print() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::Ntt);

        let input = b"Hello there";

        let hash_result = rescue.hash_bytes(input);

        println!("Input: {:?}", input);
        println!("Hash result:");
        for (i, value) in hash_result.iter().enumerate() {
            println!("  {}: {}", i, value);
        }

        println!("Hash as u64 values:");
        for value in hash_result.iter() {
            print!("{}, ", value.value());
        }
        println!();
        assert_eq!(hash_result.len(), 4);
    }
    // this gives the same in Polygon Miden
}
