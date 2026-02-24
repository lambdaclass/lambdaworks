use super::parameters::*;
use super::rescue_core::RescueCore;
use super::Fp;
use crate::alloc::vec::Vec;

// Implementation of the Rescue Prime Optimized hash function.
// https://eprint.iacr.org/2022/1577
// https://github.com/ASDiscreteMathematics/rpo/tree/master/reference_implementation
// It supports two security levels: 128-bit and 160-bit. Depending on the security level chosen
// the integer parameters are set accordingly.

// For the Security level (λ) of 128 bits we have:
// Number of rounds (N): 7
// State size (m): 12
// Rate (r): 8
// Capacity (c): 4

// For the Security level (λ) of 160 bits we have:
// Number of rounds (N): 7
// State size (m): 16
// Rate (r): 10
// Capacity (c): 6

// In the paper, the authors use a number of rounds equal to 7 as a trade-off between security and performance.
// The number of rounds can be increased to 8 or 9 to achieve a higher level of security at the cost of performance.
const NUM_FULL_ROUNDS: usize = 7;

pub struct Rpo256 {
    core: RescueCore,
}

impl Default for Rpo256 {
    fn default() -> Self {
        Self::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication)
    }
}

impl Rpo256 {
    /// Creates a new instance of `Rpo256` with corresponding Security level and the specified MDS method.
    pub fn new(security_level: SecurityLevel, mds_method: MdsMethod) -> Self {
        Self {
            core: RescueCore::new(security_level, mds_method),
        }
    }

    /// Performs the full permutation on the state.
    pub fn permutation(&self, state: &mut [Fp]) {
        debug_assert_eq!(
            state.len(),
            self.core.m,
            "State size must match state width"
        );
        for round in 0..NUM_FULL_ROUNDS {
            self.core.apply_mds(state);
            self.core.add_round_constants(state, round, 0);
            RescueCore::apply_sbox(state);
            self.core.apply_mds(state);
            self.core.add_round_constants(state, round, 1);
            RescueCore::apply_inverse_sbox(state);
        }
    }

    /// Hashes an input sequence of field elements.
    pub fn hash(&self, input_sequence: &[Fp]) -> Vec<Fp> {
        self.core.hash(input_sequence, |s| self.permutation(s))
    }

    /// Hashes an input sequence of bytes.
    pub fn hash_bytes(&self, input: &[u8]) -> Vec<Fp> {
        self.core.hash_bytes(input, |s| self.permutation(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // Values obtained from the Sage implemenstation in
    // https://github.com/ASDiscreteMathematics/rpo/tree/master/reference_implementation
    pub const EXPECTED_128: [[Fp; 4]; 19] = [
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

    pub const EXPECTED_160: [[Fp; 5]; 19] = [
        [
            Fp::const_from_raw(4766737105427868572),
            Fp::const_from_raw(7538777753317835226),
            Fp::const_from_raw(13644171984579649606),
            Fp::const_from_raw(6748107971891460622),
            Fp::const_from_raw(3480072938342119934),
        ],
        [
            Fp::const_from_raw(6277287777617382937),
            Fp::const_from_raw(5688033921803605355),
            Fp::const_from_raw(1104978478612014217),
            Fp::const_from_raw(973672476085279574),
            Fp::const_from_raw(7883652116413797779),
        ],
        [
            Fp::const_from_raw(3071553803427093579),
            Fp::const_from_raw(12239501990998925662),
            Fp::const_from_raw(14411295652479845526),
            Fp::const_from_raw(5735407824213194294),
            Fp::const_from_raw(6714816738691504270),
        ],
        [
            Fp::const_from_raw(4455998568145007624),
            Fp::const_from_raw(18218360213084301612),
            Fp::const_from_raw(8963555484142424669),
            Fp::const_from_raw(13451196299356019287),
            Fp::const_from_raw(660967320761434775),
        ],
        [
            Fp::const_from_raw(7894041400531553560),
            Fp::const_from_raw(3138084719322472990),
            Fp::const_from_raw(15017675162298246509),
            Fp::const_from_raw(12340633143623038238),
            Fp::const_from_raw(3710158928968726190),
        ],
        [
            Fp::const_from_raw(18345924309197503617),
            Fp::const_from_raw(6448668044176965096),
            Fp::const_from_raw(5891298758878861437),
            Fp::const_from_raw(18404292940273103487),
            Fp::const_from_raw(399715742058360811),
        ],
        [
            Fp::const_from_raw(4293522863608749708),
            Fp::const_from_raw(11352999694211746044),
            Fp::const_from_raw(15850245073570756600),
            Fp::const_from_raw(1206950096837096206),
            Fp::const_from_raw(6945598368659615878),
        ],
        [
            Fp::const_from_raw(1339949574743034442),
            Fp::const_from_raw(5967452101017112419),
            Fp::const_from_raw(824612579975542151),
            Fp::const_from_raw(3327557828938393394),
            Fp::const_from_raw(14113149399665697150),
        ],
        [
            Fp::const_from_raw(3540904694808418824),
            Fp::const_from_raw(5951416386790014715),
            Fp::const_from_raw(13859113410786779774),
            Fp::const_from_raw(17205554479494520251),
            Fp::const_from_raw(7359323608260195110),
        ],
        [
            Fp::const_from_raw(7504301802792161339),
            Fp::const_from_raw(12879743137663115497),
            Fp::const_from_raw(17245986604042562042),
            Fp::const_from_raw(8175050867418132561),
            Fp::const_from_raw(1063965910664731268),
        ],
        [
            Fp::const_from_raw(18267475461736255602),
            Fp::const_from_raw(4481864641736940956),
            Fp::const_from_raw(11260039501101148638),
            Fp::const_from_raw(7529970948767692955),
            Fp::const_from_raw(4177810888704753150),
        ],
        [
            Fp::const_from_raw(16604116128892623566),
            Fp::const_from_raw(1520851983040290492),
            Fp::const_from_raw(9361704524730297620),
            Fp::const_from_raw(7447748879766268839),
            Fp::const_from_raw(10834422028571028806),
        ],
        [
            Fp::const_from_raw(243957224918814907),
            Fp::const_from_raw(9966149007214472697),
            Fp::const_from_raw(18130816682404489504),
            Fp::const_from_raw(3814760895598122151),
            Fp::const_from_raw(862573500652233787),
        ],
        [
            Fp::const_from_raw(13414343823130474877),
            Fp::const_from_raw(1002887112060795246),
            Fp::const_from_raw(16685735965176892618),
            Fp::const_from_raw(16172309857128312555),
            Fp::const_from_raw(5158081519803147178),
        ],
        [
            Fp::const_from_raw(14614132925482133961),
            Fp::const_from_raw(7618082792229868740),
            Fp::const_from_raw(1881720834768448253),
            Fp::const_from_raw(11508391877383996679),
            Fp::const_from_raw(5348386073072413261),
        ],
        [
            Fp::const_from_raw(6268111131988518030),
            Fp::const_from_raw(17920308297240232909),
            Fp::const_from_raw(17719152474870950965),
            Fp::const_from_raw(14857432101092580778),
            Fp::const_from_raw(5708937553833180778),
        ],
        [
            Fp::const_from_raw(11597726741964198121),
            Fp::const_from_raw(1568026444559423552),
            Fp::const_from_raw(3233218961458461983),
            Fp::const_from_raw(9700509409081014876),
            Fp::const_from_raw(7989061413164577390),
        ],
        [
            Fp::const_from_raw(11180580619692834182),
            Fp::const_from_raw(16871004730930134181),
            Fp::const_from_raw(17810700669516829599),
            Fp::const_from_raw(13679692060051982328),
            Fp::const_from_raw(10386085719330760064),
        ],
        [
            Fp::const_from_raw(6222872143719551583),
            Fp::const_from_raw(3842704143974291265),
            Fp::const_from_raw(18311432727968603639),
            Fp::const_from_raw(12278517700025439333),
            Fp::const_from_raw(7011953052853282225),
        ],
    ];
    fn rand_field_element<R: Rng>(rng: &mut R) -> Fp {
        Fp::from(rng.gen::<u64>())
    }

    #[test]
    fn test_apply_sbox() {
        let mut rng = StdRng::seed_from_u64(1);
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.core.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let mut expected = state.clone();
        expected.iter_mut().for_each(|v| *v = v.pow(ALPHA));

        RescueCore::apply_sbox(&mut state);
        assert_eq!(expected, state);
    }

    #[test]
    fn test_apply_inverse_sbox() {
        let mut rng = StdRng::seed_from_u64(2);
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.core.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let mut expected = state.clone();
        expected.iter_mut().for_each(|v| *v = v.pow(ALPHA_INV));

        RescueCore::apply_inverse_sbox(&mut state);
        assert_eq!(expected, state);
    }

    #[test]
    fn test_mds_matrix_multiplication() {
        let mut rng = StdRng::seed_from_u64(3);
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let state: Vec<Fp> = (0..rescue.core.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let original = state.clone();
        let mut computed_state = state;
        rescue.core.apply_mds(&mut computed_state);

        // Verify against direct matrix multiplication
        let rescue2 = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let mut expected = original;
        rescue2.core.apply_mds(&mut expected);

        assert_eq!(expected, computed_state);
    }

    #[test]
    fn test_mds_ntt() {
        let mut rng = StdRng::seed_from_u64(4);
        let rescue_matrix = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let rescue_ntt = Rpo256::new(SecurityLevel::Sec128, MdsMethod::Ntt);
        let state: Vec<Fp> = (0..rescue_ntt.core.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let mut matrix_state = state.clone();
        rescue_matrix.core.apply_mds(&mut matrix_state);

        let mut ntt_state = state;
        rescue_ntt.core.apply_mds(&mut ntt_state);

        assert_eq!(matrix_state, ntt_state);
    }

    #[test]
    fn test_mds_karatsuba() {
        let mut rng = StdRng::seed_from_u64(5);
        let rescue_matrix = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let rescue_karatsuba = Rpo256::new(SecurityLevel::Sec128, MdsMethod::Karatsuba);
        let state: Vec<Fp> = (0..rescue_karatsuba.core.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let mut matrix_state = state.clone();
        rescue_matrix.core.apply_mds(&mut matrix_state);

        let mut karatsuba_state = state;
        rescue_karatsuba.core.apply_mds(&mut karatsuba_state);

        assert_eq!(matrix_state, karatsuba_state);
    }

    #[test]
    fn test_add_round_constants() {
        let mut rng = StdRng::seed_from_u64(6);
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.core.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let round = 0;
        let expected_state = state
            .iter()
            .enumerate()
            .map(|(i, &x)| x + rescue.core.round_constants[round * 2 * rescue.core.m + i])
            .collect::<Vec<_>>();

        rescue.core.add_round_constants(&mut state, round, 0);

        assert_eq!(expected_state, state);
    }

    #[test]
    fn test_permutation() {
        let mut rng = StdRng::seed_from_u64(7);
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.core.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let expected_state = {
            let mut temp_state = state.clone();
            for round in 0..7 {
                rescue.core.apply_mds(&mut temp_state);
                rescue.core.add_round_constants(&mut temp_state, round, 0);
                RescueCore::apply_sbox(&mut temp_state);
                rescue.core.apply_mds(&mut temp_state);
                rescue.core.add_round_constants(&mut temp_state, round, 1);
                RescueCore::apply_inverse_sbox(&mut temp_state);
            }
            temp_state
        };

        rescue.permutation(&mut state);

        assert_eq!(expected_state, state);
    }

    #[test]
    fn test_hash_single_chunk() {
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..8).map(Fp::from).collect();
        let hash_output = rescue.hash(&input_sequence);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_hash_multiple_chunks() {
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..16).map(Fp::from).collect(); // Two chunks of size 8
        let hash_output = rescue.hash(&input_sequence);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_hash_with_padding() {
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..5).map(Fp::from).collect();
        let hash_output = rescue.hash(&input_sequence);
        assert_eq!(hash_output.len(), 4);
    }
    #[test]
    // test ported from https://github.com/0xPolygonMiden/crypto/blob/main/src/hash/rescue/rpo/tests.rs
    fn hash_padding() {
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);

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
    #[cfg(feature = "std")]
    #[test]
    fn sponge_zeroes_collision() {
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);

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
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let input_bytes = b"Rescue Prime Optimized";
        let hash_output = rescue.hash_bytes(input_bytes);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_mds_methods_consistency() {
        let rescue_matrix = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let rescue_ntt = Rpo256::new(SecurityLevel::Sec128, MdsMethod::Ntt);
        let rescue_karatsuba = Rpo256::new(SecurityLevel::Sec128, MdsMethod::Karatsuba);

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

    #[test]
    fn test_hash_vectors_128() {
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication);
        let elements: Vec<Fp> = (0..19).map(Fp::from).collect();

        EXPECTED_128.iter().enumerate().for_each(|(i, expected)| {
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
    fn test_hash_vector_160() {
        let rescue = Rpo256::new(SecurityLevel::Sec160, MdsMethod::MatrixMultiplication);
        let elements: Vec<Fp> = (0..19).map(Fp::from).collect();

        EXPECTED_160.iter().enumerate().for_each(|(i, expected)| {
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
    #[cfg(feature = "std")]
    #[test]
    fn test_hash_example_and_print() {
        let rescue = Rpo256::new(SecurityLevel::Sec128, MdsMethod::Ntt);

        let input = b"Hello there";

        let hash_result = rescue.hash_bytes(input);

        println!("Input: {input:?}");
        println!("Hash result:");
        for (i, value) in hash_result.iter().enumerate() {
            println!("  {i}: {value}");
        }

        println!("Hash as u64 values:");
        for value in hash_result.iter() {
            print!("{}, ", value.value());
        }
        println!();
        assert_eq!(hash_result.len(), 4);
    }
}
