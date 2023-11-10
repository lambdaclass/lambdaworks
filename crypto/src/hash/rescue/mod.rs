pub mod parameters;
pub mod rescue_prime;



#[cfg(test)]
mod tests {

    use lambdaworks_math::field::fields::u64_prime_field::U64FieldElement;

    use super::rescue_prime::Rescue;

    const WIDTH: usize = 12;
    const ALPHA: u64 = 5;


    const PRIME : u64 = 2147483647;

    type Mersenne31 = U64FieldElement<PRIME>;
    type RescuePrimeM31Default = Rescue<Mersenne31,WIDTH, 4, 800>;

    fn new_rescue_prime_m31_default() -> RescuePrimeM31Default {
        let num_rounds = RescuePrimeM31Default::num_rounds(6, 128, ALPHA);
        let round_constants =
            RescuePrimeM31Default::get_round_constants_rescue_prime(num_rounds, 6, 128);
        let mds = MdsMatrixMersenne31 {};
        let sbox = BasicSboxLayer::for_alpha(ALPHA);

        RescuePrimeM31Default::new(num_rounds, round_constants, mds, sbox)
    }

    const NUM_TESTS: usize = 3;

    const PERMUTATION_INPUTS: [[u64; WIDTH]; NUM_TESTS] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [
            144096679, 1638468327, 1550998769, 1713522258, 730676443, 955614588, 1970746889,
            1473251100, 1575313887, 1867935938, 364960233, 91318724,
        ],
        [
            1946786350, 648783024, 470775457, 573110744, 2049365630, 710763043, 1694076126,
            1852085316, 1518834534, 249604062, 45487116, 1543494419,
        ],
    ];

    // Generated using the rescue_XLIX_permutation function of
    // https://github.com/KULeuven-COSIC/Marvellous/blob/master/rescue_prime.sage
    const PERMUTATION_OUTPUTS: [[u64; WIDTH]; NUM_TESTS] = [
        [
            983158113, 88736227, 182376113, 380581876, 1054929865, 873254619, 1742172525,
            1018880997, 1922857524, 2128461101, 1878468735, 736900567,
        ],
        [
            504747180, 1708979401, 1023327691, 414948293, 1811202621, 427591394, 666516466,
            1900855073, 1511950466, 346735768, 708718627, 2070146754,
        ],
        [
            2043076197, 1832583290, 59074227, 991951621, 1166633601, 629305333, 1869192382,
            1355209324, 1919016607, 175801753, 279984593, 2086613859,
        ],
    ];

    #[test]
    fn test_rescue_xlix_permutation() {
        let rescue_prime = new_rescue_prime_m31_default();

        for test_run in 0..NUM_TESTS {
            let state: [Mersenne31; WIDTH] =
                PERMUTATION_INPUTS[test_run].map(Mersenne31::from_canonical_u64);

            let expected: [Mersenne31; WIDTH] =
                PERMUTATION_OUTPUTS[test_run].map(Mersenne31::from_canonical_u64);

            let actual = rescue_prime.permute(state);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_rescue_sponge() {
        let rescue_prime = new_rescue_prime_m31_default();
        let rescue_sponge = PaddingFreeSponge::<_, WIDTH, 8, 6>::new(rescue_prime);

        let input: [Mersenne31; 6] = [1, 2, 3, 4, 5, 6].map(Mersenne31::from_canonical_u64);

        let expected: [Mersenne31; 6] = [
            337439389, 568168673, 983336666, 1144682541, 1342961449, 386074361,
        ]
        .map(Mersenne31::from_canonical_u64);

        let actual = rescue_sponge.hash_iter(input);
        assert_eq!(actual, expected);
    }
}
