/// Poseidon hash for the Goldilocks field (p = 2^64 - 2^32 + 1).
///
/// This is a standalone implementation that matches Plonky2's Poseidon.
/// It does NOT use the existing `PermutationParameters` trait because that
/// trait hardcodes state_size=3 and x^3 S-box, whereas Goldilocks Poseidon
/// uses state_size=12 and x^7 S-box.
pub mod constants;

use alloc::vec::Vec;
use constants::*;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::u64_goldilocks_field::{Goldilocks64Field, GOLDILOCKS_PRIME},
    traits::IsPrimeField,
};

type FpE = FieldElement<Goldilocks64Field>;

#[derive(Clone, Default)]
pub struct PoseidonGoldilocks;

impl PoseidonGoldilocks {
    /// Reduce a 96-bit value (lo, hi) modulo the Goldilocks prime.
    ///
    /// For Goldilocks p = 2^64 - 2^32 + 1, we have 2^64 ≡ 2^32 - 1 (mod p).
    /// So (hi * 2^64 + lo) ≡ hi * (2^32 - 1) + lo (mod p).
    #[inline(always)]
    fn reduce_u96(lo: u64, hi: u32) -> FpE {
        // hi * (2^32 - 1) = hi * 0xFFFFFFFF
        let hi_times = (hi as u64) * 0xFFFF_FFFFu64;
        let (sum, carry) = lo.overflowing_add(hi_times);
        if carry {
            // sum + 2^64 ≡ sum + (2^32 - 1) (mod p)
            // = sum + 0xFFFFFFFF, which can't overflow again since sum < 2^64 and 0xFFFFFFFF < 2^32
            let val = sum.wrapping_add(0xFFFF_FFFF);
            // May still be >= p, canonicalize
            let canonical = if val >= GOLDILOCKS_PRIME {
                val - GOLDILOCKS_PRIME
            } else {
                val
            };
            FieldElement::from_raw(canonical)
        } else {
            let canonical = if sum >= GOLDILOCKS_PRIME {
                sum - GOLDILOCKS_PRIME
            } else {
                sum
            };
            FieldElement::from_raw(canonical)
        }
    }

    /// Compute one row of the MDS matrix–vector product.
    ///
    /// MDS = circ(MDS_MATRIX_CIRC) + diag(MDS_MATRIX_DIAG).
    /// Uses u128 accumulation to avoid intermediate modular reductions.
    #[inline(always)]
    fn mds_row_shf(r: usize, state: &[u64; SPONGE_WIDTH]) -> (u64, u32) {
        let mut acc = 0u128;
        for i in 0..SPONGE_WIDTH {
            acc += (state[(i + r) % SPONGE_WIDTH] as u128) * (MDS_MATRIX_CIRC[i] as u128);
        }
        acc += (state[r] as u128) * (MDS_MATRIX_DIAG[r] as u128);
        (acc as u64, (acc >> 64) as u32)
    }

    /// Apply the MDS layer to the state.
    #[inline(always)]
    fn mds_layer(state: &[FpE; SPONGE_WIDTH]) -> [FpE; SPONGE_WIDTH] {
        let mut raw = [0u64; SPONGE_WIDTH];
        for (raw_i, s) in raw.iter_mut().zip(state.iter()) {
            *raw_i = Goldilocks64Field::canonical(s.value());
        }

        let mut result = [FpE::zero(); SPONGE_WIDTH];
        for (r, result_r) in result.iter_mut().enumerate() {
            let (lo, hi) = Self::mds_row_shf(r, &raw);
            *result_r = Self::reduce_u96(lo, hi);
        }
        result
    }

    /// S-box: x^7 = x * x^2 * x^4
    #[inline(always)]
    fn sbox(x: FpE) -> FpE {
        let x2 = &x * &x;
        let x4 = &x2 * &x2;
        let x3 = &x * &x2;
        &x3 * &x4
    }

    /// Add round constants to state.
    #[inline(always)]
    fn constant_layer(state: &mut [FpE; SPONGE_WIDTH], round: usize) {
        let base = round * SPONGE_WIDTH;
        for (i, s) in state.iter_mut().enumerate() {
            *s = &*s + &FpE::from(ALL_ROUND_CONSTANTS[base + i]);
        }
    }

    /// Full round: add constants, apply S-box to all elements, MDS multiply.
    #[inline(always)]
    fn full_round(state: &mut [FpE; SPONGE_WIDTH], round: usize) {
        Self::constant_layer(state, round);
        for s in state.iter_mut() {
            *s = Self::sbox(*s);
        }
        *state = Self::mds_layer(state);
    }

    /// Partial round: add constants, apply S-box to first element only, MDS multiply.
    #[inline(always)]
    fn partial_round(state: &mut [FpE; SPONGE_WIDTH], round: usize) {
        Self::constant_layer(state, round);
        state[0] = Self::sbox(state[0]);
        *state = Self::mds_layer(state);
    }

    /// Run the Poseidon permutation on a width-12 state.
    pub fn permute(state: &mut [FpE; SPONGE_WIDTH]) {
        let mut round = 0;

        // First half of full rounds
        for _ in 0..HALF_N_FULL_ROUNDS {
            Self::full_round(state, round);
            round += 1;
        }

        // Partial rounds
        for _ in 0..N_PARTIAL_ROUNDS {
            Self::partial_round(state, round);
            round += 1;
        }

        // Second half of full rounds
        for _ in 0..HALF_N_FULL_ROUNDS {
            Self::full_round(state, round);
            round += 1;
        }
    }

    /// Hash two field elements: H(a, b).
    pub fn hash(a: &FpE, b: &FpE) -> FpE {
        let mut state = [FpE::zero(); SPONGE_WIDTH];
        state[0] = *a;
        state[1] = *b;
        Self::permute(&mut state);
        state[0]
    }

    /// Hash a single field element.
    pub fn hash_single(x: &FpE) -> FpE {
        let mut state = [FpE::zero(); SPONGE_WIDTH];
        state[0] = *x;
        Self::permute(&mut state);
        state[0]
    }

    /// Hash a variable-length slice of field elements using a sponge construction.
    pub fn hash_many(inputs: &[FpE]) -> FpE {
        let mut state = [FpE::zero(); SPONGE_WIDTH];

        // Pad input with 1 followed by zeros to fill a rate block
        let mut values: Vec<FpE> = inputs.to_vec();
        values.push(FpE::one());
        while !values.len().is_multiple_of(SPONGE_RATE) {
            values.push(FpE::zero());
        }

        // Absorb each rate-sized block
        for block in values.chunks(SPONGE_RATE) {
            for i in 0..SPONGE_RATE {
                state[i] = &state[i] + &block[i];
            }
            Self::permute(&mut state);
        }

        state[0]
    }

    /// Hash a fixed-length slice of field elements without padding.
    /// Used for Merkle leaf hashing where the input length is known.
    pub fn hash_no_pad(inputs: &[FpE]) -> FpE {
        let mut state = [FpE::zero(); SPONGE_WIDTH];

        for block in inputs.chunks(SPONGE_RATE) {
            for (i, val) in block.iter().enumerate() {
                state[i] = &state[i] + val;
            }
            Self::permute(&mut state);
        }

        state[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permute_zero_state() {
        let mut state = [FpE::zero(); SPONGE_WIDTH];
        PoseidonGoldilocks::permute(&mut state);
        assert_ne!(state[0], FpE::zero());
    }

    #[test]
    fn test_hash_deterministic() {
        let a = FpE::from(1u64);
        let b = FpE::from(2u64);
        let h1 = PoseidonGoldilocks::hash(&a, &b);
        let h2 = PoseidonGoldilocks::hash(&a, &b);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_inputs() {
        let h1 = PoseidonGoldilocks::hash(&FpE::from(1u64), &FpE::from(2u64));
        let h2 = PoseidonGoldilocks::hash(&FpE::from(2u64), &FpE::from(1u64));
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_many_vs_hash() {
        let a = FpE::from(1u64);
        let b = FpE::from(2u64);
        // hash_many pads with 1 then zeros; hash uses raw [a, b, 0...0].
        // They should NOT be equal.
        let h_pair = PoseidonGoldilocks::hash(&a, &b);
        let h_many = PoseidonGoldilocks::hash_many(&[a, b]);
        assert_ne!(h_pair, h_many);
    }

    #[test]
    fn test_sbox_is_x7() {
        let x = FpE::from(3u64);
        let expected = FpE::from(2187u64); // 3^7 = 2187
        assert_eq!(PoseidonGoldilocks::sbox(x), expected);
    }

    #[test]
    fn test_mds_layer_nonzero() {
        let mut state = [FpE::zero(); SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            state[i] = FpE::from((i + 1) as u64);
        }
        let result = PoseidonGoldilocks::mds_layer(&state);
        for r in &result {
            assert_ne!(*r, FpE::zero());
        }
    }

    #[test]
    fn test_reduce_u96() {
        // Test: reduce_u96(0, 0) = 0
        assert_eq!(PoseidonGoldilocks::reduce_u96(0, 0), FpE::zero());

        // Test: reduce_u96(1, 0) = 1
        assert_eq!(PoseidonGoldilocks::reduce_u96(1, 0), FpE::one());

        // Test: reduce_u96(p, 0) = 0 (p reduces to 0)
        assert_eq!(
            PoseidonGoldilocks::reduce_u96(GOLDILOCKS_PRIME, 0),
            FpE::zero()
        );

        // Test: reduce_u96(0, 1) = 2^64 mod p = 2^32 - 1 = 0xFFFFFFFF
        let expected = FpE::from(0xFFFF_FFFFu64);
        assert_eq!(PoseidonGoldilocks::reduce_u96(0, 1), expected);
    }

    /// Verify permutation against Plonky2's official test vectors.
    fn check_plonky2_vector(input: [u64; SPONGE_WIDTH], expected: [u64; SPONGE_WIDTH]) {
        let mut state = [FpE::zero(); SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            state[i] = FpE::from(input[i]);
        }
        PoseidonGoldilocks::permute(&mut state);
        for i in 0..SPONGE_WIDTH {
            let got = Goldilocks64Field::canonical(state[i].value());
            assert_eq!(
                got, expected[i],
                "Mismatch at index {i}: got {got:#018x}, expected {:#018x}",
                expected[i]
            );
        }
    }

    #[test]
    fn test_plonky2_vector_all_zeros() {
        check_plonky2_vector(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                0x3c18a9786cb0b359,
                0xc4055e3364a246c3,
                0x7953db0ab48808f4,
                0xc71603f33a1144ca,
                0xd7709673896996dc,
                0x46a84e87642f44ed,
                0xd032648251ee0b3c,
                0x1c687363b207df62,
                0xdf8565563e8045fe,
                0x40f5b37ff4254dae,
                0xd070f637b431067c,
                0x1792b1c4342109d7,
            ],
        );
    }

    #[test]
    fn test_plonky2_vector_sequential() {
        check_plonky2_vector(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [
                0xd64e1e3efc5b8e9e,
                0x53666633020aaa47,
                0xd40285597c6a8825,
                0x613a4f81e81231d2,
                0x414754bfebd051f0,
                0xcb1f8980294a023f,
                0x6eb2a9e4d54a9d0f,
                0x1902bc3af467e056,
                0xf045d5eafdc6021f,
                0xe4150f77caaa3be5,
                0xc9bfd01d39b50cce,
                0x5c0a27fcb0e1459b,
            ],
        );
    }

    #[test]
    fn test_plonky2_vector_all_neg_one() {
        let neg_one = GOLDILOCKS_PRIME - 1; // p - 1 = -1 mod p
        check_plonky2_vector(
            [neg_one; SPONGE_WIDTH],
            [
                0xbe0085cfc57a8357,
                0xd95af71847d05c09,
                0xcf55a13d33c1c953,
                0x95803a74f4530e82,
                0xfcd99eb30a135df1,
                0xe095905e913a3029,
                0xde0392461b42919b,
                0x7d3260e24e81d031,
                0x10d3d0465d9deaa0,
                0xa87571083dfc2a47,
                0xe18263681e9958f8,
                0xe28e96f1ae5e60d3,
            ],
        );
    }

    /// Test that the fast partial round algorithm (Plonky2) gives the same
    /// result as the standard Poseidon permutation when using the precomputed constants.
    #[test]
    fn test_fast_partial_rounds_match_standard() {
        // Plonky2 fast partial round constants
        const FP_FIRST_RC: [u64; 12] = [
            0x3cc3f892184df408,
            0xe993fd841e7e97f1,
            0xf2831d3575f0f3af,
            0xd2500e0a350994ca,
            0xc5571f35d7288633,
            0x91d89c5184109a02,
            0xf37f925d04e5667b,
            0x2d6e448371955a69,
            0x740ef19ce01398a1,
            0x694d24c0752fdf45,
            0x60936af96ee2f148,
            0xc33448feadc78f0c,
        ];
        const FP_RC: [u64; 22] = [
            0x74cb2e819ae421ab,
            0xd2559d2370e7f663,
            0x62bf78acf843d17c,
            0xd5ab7b67e14d1fb4,
            0xb9fe2ae6e0969bdc,
            0xe33fdf79f92a10e8,
            0x0ea2bb4c2b25989b,
            0xca9121fbf9d38f06,
            0xbdd9b0aa81f58fa4,
            0x83079fa4ecf20d7e,
            0x650b838edfcc4ad3,
            0x77180c88583c76ac,
            0xaf8c20753143a180,
            0xb8ccfe9989a39175,
            0x954a1729f60cc9c5,
            0xdeb5b550c4dca53b,
            0xf01bb0b00f77011e,
            0xa1ebb404b676afd9,
            0x860b6e1597a0173e,
            0x308bb65a036acbce,
            0x1aca78f31c97c876,
            0x0,
        ];
        #[rustfmt::skip]
        const FP_VS: [[u64; 11]; 22] = [
            [0x94877900674181c3, 0xc6c67cc37a2a2bbd, 0xd667c2055387940f, 0x0ba63a63e94b5ff0,
             0x99460cc41b8f079f, 0x7ff02375ed524bb3, 0xea0870b47a8caf0e, 0xabcad82633b7bc9d,
             0x3b8d135261052241, 0xfb4515f5e5b0d539, 0x3ee8011c2b37f77c],
            [0x0adef3740e71c726, 0xa37bf67c6f986559, 0xc6b16f7ed4fa1b00, 0x6a065da88d8bfc3c,
             0x4cabc0916844b46f, 0x407faac0f02e78d1, 0x07a786d9cf0852cf, 0x42433fb6949a629a,
             0x891682a147ce43b0, 0x26cfd58e7b003b55, 0x2bbf0ed7b657acb3],
            [0x481ac7746b159c67, 0xe367de32f108e278, 0x73f260087ad28bec, 0x5cfc82216bc1bdca,
             0xcaccc870a2663a0e, 0xdb69cd7b4298c45d, 0x7bc9e0c57243e62d, 0x3cc51c5d368693ae,
             0x366b4e8cc068895b, 0x2bd18715cdabbca4, 0xa752061c4f33b8cf],
            [0xb22d2432b72d5098, 0x9e18a487f44d2fe4, 0x4b39e14ce22abd3c, 0x9e77fde2eb315e0d,
             0xca5e0385fe67014d, 0x0c2cb99bf1b6bddb, 0x99ec1cd2a4460bfe, 0x8577a815a2ff843f,
             0x7d80a6b4fd6518a5, 0xeb6c67123eab62cb, 0x8f7851650eca21a5],
            [0x11ba9a1b81718c2a, 0x9f7d798a3323410c, 0xa821855c8c1cf5e5, 0x535e8d6fac0031b2,
             0x404e7c751b634320, 0xa729353f6e55d354, 0x4db97d92e58bb831, 0xb53926c27897bf7d,
             0x965040d52fe115c5, 0x9565fa41ebd31fd7, 0xaae4438c877ea8f4],
            [0x37f4e36af6073c6e, 0x4edc0918210800e9, 0xc44998e99eae4188, 0x9f4310d05d068338,
             0x9ec7fe4350680f29, 0xc5b2c1fdc0b50874, 0xa01920c5ef8b2ebe, 0x59fa6f8bd91d58ba,
             0x8bfc9eb89b515a82, 0xbe86a7a2555ae775, 0xcbb8bbaa3810babf],
            [0x577f9a9e7ee3f9c2, 0x88c522b949ace7b1, 0x82f07007c8b72106, 0x8283d37c6675b50e,
             0x98b074d9bbac1123, 0x75c56fb7758317c1, 0xfed24e206052bc72, 0x26d7c3d1bc07dae5,
             0xf88c5e441e28dbb4, 0x4fe27f9f96615270, 0x514d4ba49c2b14fe],
            [0xf02a3ac068ee110b, 0x0a3630dafb8ae2d7, 0xce0dc874eaf9b55c, 0x9a95f6cff5b55c7e,
             0x626d76abfed00c7b, 0xa0c1cf1251c204ad, 0xdaebd3006321052c, 0x3d4bd48b625a8065,
             0x7f1e584e071f6ed2, 0x720574f0501caed3, 0xe3260ba93d23540a],
            [0xab1cbd41d8c1e335, 0x9322ed4c0bc2df01, 0x51c3c0983d4284e5, 0x94178e291145c231,
             0xfd0f1a973d6b2085, 0xd427ad96e2b39719, 0x8a52437fecaac06b, 0xdc20ee4b8c4c9a80,
             0xa2c98e9549da2100, 0x1603fe12613db5b6, 0x0e174929433c5505],
            [0x3d4eab2b8ef5f796, 0xcfff421583896e22, 0x4143cb32d39ac3d9, 0x22365051b78a5b65,
             0x6f7fd010d027c9b6, 0xd9dd36fba77522ab, 0xa44cf1cb33e37165, 0x3fc83d3038c86417,
             0xc4588d418e88d270, 0xce1320f10ab80fe2, 0xdb5eadbbec18de5d],
            [0x1183dfce7c454afd, 0x21cea4aa3d3ed949, 0x0fce6f70303f2304, 0x19557d34b55551be,
             0x4c56f689afc5bbc9, 0xa1e920844334f944, 0xbad66d423d2ec861, 0xf318c785dc9e0479,
             0x99e2032e765ddd81, 0x400ccc9906d66f45, 0xe1197454db2e0dd9],
            [0x84d1ecc4d53d2ff1, 0xd8af8b9ceb4e11b6, 0x335856bb527b52f4, 0xc756f17fb59be595,
             0xc0654e4ea5553a78, 0x9e9a46b61f2ea942, 0x14fc8b5b3b809127, 0xd7009f0f103be413,
             0x3e0ee7b7a9fb4601, 0xa74e888922085ed7, 0xe80a7cde3d4ac526],
            [0x238aa6daa612186d, 0x9137a5c630bad4b4, 0xc7db3817870c5eda, 0x217e4f04e5718dc9,
             0xcae814e2817bd99d, 0xe3292e7ab770a8ba, 0x7bb36ef70b6b9482, 0x3c7835fb85bca2d3,
             0xfe2cdf8ee3c25e86, 0x61b3915ad7274b20, 0xeab75ca7c918e4ef],
            [0xd6e15ffc055e154e, 0xec67881f381a32bf, 0xfbb1196092bf409c, 0xdc9d2e07830ba226,
             0x0698ef3245ff7988, 0x194fae2974f8b576, 0x7a5d9bea6ca4910e, 0x7aebfea95ccdd1c9,
             0xf9bd38a67d5f0e86, 0xfa65539de65492d8, 0xf0dfcbe7653ff787],
            [0x0bd87ad390420258, 0x0ad8617bca9e33c8, 0x0c00ad377a1e2666, 0x0ac6fc58b3f0518f,
             0x0c0cc8a892cc4173, 0x0c210accb117bc21, 0x0b73630dbb46ca18, 0x0c8be4920cbd4a54,
             0x0bfe877a21be1690, 0x0ae790559b0ded81, 0x0bf50db2f8d6ce31],
            [0x000cf29427ff7c58, 0x000bd9b3cf49eec8, 0x000d1dc8aa81fb26, 0x000bc792d5c394ef,
             0x000d2ae0b2266453, 0x000d413f12c496c1, 0x000c84128cfed618, 0x000db5ebd48fc0d4,
             0x000d1b77326dcb90, 0x000beb0ccc145421, 0x000d10e5b22b11d1],
            [0x00000e24c99adad8, 0x00000cf389ed4bc8, 0x00000e580cbf6966, 0x00000cde5fd7e04f,
             0x00000e63628041b3, 0x00000e7e81a87361, 0x00000dabe78f6d98, 0x00000efb14cac554,
             0x00000e5574743b10, 0x00000d05709f42c1, 0x00000e4690c96af1],
            [0x0000000f7157bc98, 0x0000000e3006d948, 0x0000000fa65811e6, 0x0000000e0d127e2f,
             0x0000000fc18bfe53, 0x0000000fd002d901, 0x0000000eed6461d8, 0x0000001068562754,
             0x0000000fa0236f50, 0x0000000e3af13ee1, 0x0000000fa460f6d1],
            [0x0000000011131738, 0x000000000f56d588, 0x0000000011050f86, 0x000000000f848f4f,
             0x00000000111527d3, 0x00000000114369a1, 0x00000000106f2f38, 0x0000000011e2ca94,
             0x00000000110a29f0, 0x000000000fa9f5c1, 0x0000000010f625d1],
            [0x000000000011f718, 0x000000000010b6c8, 0x0000000000134a96, 0x000000000010cf7f,
             0x0000000000124d03, 0x000000000013f8a1, 0x0000000000117c58, 0x0000000000132c94,
             0x0000000000134fc0, 0x000000000010a091, 0x0000000000128961],
            [0x0000000000001300, 0x0000000000001750, 0x000000000000114e, 0x000000000000131f,
             0x000000000000167b, 0x0000000000001371, 0x0000000000001230, 0x000000000000182c,
             0x0000000000001368, 0x0000000000000f31, 0x00000000000015c9],
            [0x0000000000000014, 0x0000000000000022, 0x0000000000000012, 0x0000000000000027,
             0x000000000000000d, 0x000000000000000d, 0x000000000000001c, 0x0000000000000002,
             0x0000000000000010, 0x0000000000000029, 0x000000000000000f],
        ];
        #[rustfmt::skip]
        const FP_W_HATS: [[u64; 11]; 22] = [
            [0x3d999c961b7c63b0, 0x814e82efcd172529, 0x2421e5d236704588, 0x887af7d4dd482328,
             0xa5e9c291f6119b27, 0xbdc52b2676a4b4aa, 0x64832009d29bcf57, 0x09c4155174a552cc,
             0x463f9ee03d290810, 0xc810936e64982542, 0x043b1c289f7bc3ac],
            [0x673655aae8be5a8b, 0xd510fe714f39fa10, 0x2c68a099b51c9e73, 0xa667bfa9aa96999d,
             0x4d67e72f063e2108, 0xf84dde3e6acda179, 0x40f9cc8c08f80981, 0x5ead032050097142,
             0x6591b02092d671bb, 0x00e18c71963dd1b7, 0x8a21bcd24a14218a],
            [0x202800f4addbdc87, 0xe4b5bdb1cc3504ff, 0xbe32b32a825596e7, 0x8e0f68c5dc223b9a,
             0x58022d9e1c256ce3, 0x584d29227aa073ac, 0x8b9352ad04bef9e7, 0xaead42a3f445ecbf,
             0x3c667a1d833a3cca, 0xda6f61838efa1ffe, 0xe8f749470bd7c446],
            [0xc5b85bab9e5b3869, 0x45245258aec51cf7, 0x16e6b8e68b931830, 0xe2ae0f051418112c,
             0x0470e26a0093a65b, 0x6bef71973a8146ed, 0x119265be51812daf, 0xb0be7356254bea2e,
             0x8584defff7589bd7, 0x3c5fe4aeb1fb52ba, 0x9e7cd88acf543a5e],
            [0x179be4bba87f0a8c, 0xacf63d95d8887355, 0x6696670196b0074f, 0xd99ddf1fe75085f9,
             0xc2597881fef0283b, 0xcf48395ee6c54f14, 0x15226a8e4cd8d3b6, 0xc053297389af5d3b,
             0x2c08893f0d1580e2, 0x0ed3cbcff6fcc5ba, 0xc82f510ecf81f6d0],
            [0x94b06183acb715cc, 0x500392ed0d431137, 0x861cc95ad5c86323, 0x05830a443f86c4ac,
             0x3b68225874a20a7c, 0x10b3309838e236fb, 0x9b77fc8bcd559e2c, 0xbdecf5e0cb9cb213,
             0x30276f1221ace5fa, 0x7935dd342764a144, 0xeac6db520bb03708],
            [0x7186a80551025f8f, 0x622247557e9b5371, 0xc4cbe326d1ad9742, 0x55f1523ac6a23ea2,
             0xa13dfe77a3d52f53, 0xe30750b6301c0452, 0x08bd488070a3a32b, 0xcd800caef5b72ae3,
             0x83329c90f04233ce, 0xb5b99e6664a0a3ee, 0x6b0731849e200a7f],
            [0xec3fabc192b01799, 0x382b38cee8ee5375, 0x3bfb6c3f0e616572, 0x514abd0cf6c7bc86,
             0x47521b1361dcc546, 0x178093843f863d14, 0xad1003c5d28918e7, 0x738450e42495bc81,
             0xaf947c59af5e4047, 0x4653fb0685084ef2, 0x057fde2062ae35bf],
            [0xe376678d843ce55e, 0x66f3860d7514e7fc, 0x7817f3dfff8b4ffa, 0x3929624a9def725b,
             0x0126ca37f215a80a, 0xfce2f5d02762a303, 0x1bc927375febbad7, 0x85b481e5243f60bf,
             0x2d3c5f42a39c91a0, 0x0811719919351ae8, 0xf669de0add993131],
            [0x7de38bae084da92d, 0x5b848442237e8a9b, 0xf6c705da84d57310, 0x31e6a4bdb6a49017,
             0x889489706e5c5c0f, 0x0e4a205459692a1b, 0xbac3fa75ee26f299, 0x5f5894f4057d755e,
             0xb0dc3ecd724bb076, 0x5e34d8554a6452ba, 0x04f78fd8c1fdcc5f],
            [0x4dd19c38779512ea, 0xdb79ba02704620e9, 0x92a29a3675a5d2be, 0xd5177029fe495166,
             0xd32b3298a13330c1, 0x251c4a3eb2c5f8fd, 0xe1c48b26e0d98825, 0x3301d3362a4ffccb,
             0x09bb6c88de8cd178, 0xdc05b676564f538a, 0x60192d883e473fee],
            [0x16b9774801ac44a0, 0x3cb8411e786d3c8e, 0xa86e9cf505072491, 0x0178928152e109ae,
             0x5317b905a6e1ab7b, 0xda20b3be7f53d59f, 0xcb97dedecebee9ad, 0x4bd545218c59f58d,
             0x77dc8d856c05a44a, 0x87948589e4f243fd, 0x7e5217af969952c2],
            [0xbc58987d06a84e4d, 0x0b5d420244c9cae3, 0xa3c4711b938c02c0, 0x3aace640a3e03990,
             0x865a0f3249aacd8a, 0x8d00b2a7dbed06c7, 0x6eacb905beb7e2f8, 0x045322b216ec3ec7,
             0xeb9de00d594828e6, 0x088c5f20df9e5c26, 0xf555f4112b19781f],
            [0xa8cedbff1813d3a7, 0x50dcaee0fd27d164, 0xf1cb02417e23bd82, 0xfaf322786e2abe8b,
             0x937a4315beb5d9b6, 0x1b18992921a11d85, 0x7d66c4368b3c497b, 0x0e7946317a6b4e99,
             0xbe4430134182978b, 0x3771e82493ab262d, 0xa671690d8095ce82],
            [0xb035585f6e929d9d, 0xba1579c7e219b954, 0xcb201cf846db4ba3, 0x287bf9177372cf45,
             0xa350e4f61147d0a6, 0xd5d0ecfb50bcff99, 0x2e166aa6c776ed21, 0xe1e66c991990e282,
             0x662b329b01e7bb38, 0x8aa674b36144d9a9, 0xcbabf78f97f95e65],
            [0xeec24b15a06b53fe, 0xc8a7aa07c5633533, 0xefe9c6fa4311ad51, 0xb9173f13977109a1,
             0x69ce43c9cc94aedc, 0xecf623c9cd118815, 0x28625def198c33c7, 0xccfc5f7de5c3636a,
             0xf5e6c40f1621c299, 0xcec0e58c34cb64b1, 0xa868ea113387939f],
            [0xd8dddbdc5ce4ef45, 0xacfc51de8131458c, 0x146bb3c0fe499ac0, 0x9e65309f15943903,
             0x80d0ad980773aa70, 0xf97817d4ddbf0607, 0xe4626620a75ba276, 0x0dfdc7fd6fc74f66,
             0xf464864ad6f2bb93, 0x02d55e52a5d44414, 0xdd8de62487c40925],
            [0xc15acf44759545a3, 0xcbfdcf39869719d4, 0x33f62042e2f80225, 0x2599c5ead81d8fa3,
             0x0b306cb6c1d7c8d0, 0x658c80d3df3729b1, 0xe8d1b2b21b41429c, 0xa1b67f09d4b3ccb8,
             0x0e1adf8b84437180, 0x0d593a5e584af47b, 0xa023d94c56e151c7],
            [0x49026cc3a4afc5a6, 0xe06dff00ab25b91b, 0x0ab38c561e8850ff, 0x92c3c8275e105eeb,
             0xb65256e546889bd0, 0x3c0468236ea142f6, 0xee61766b889e18f2, 0xa206f41b12c30415,
             0x02fe9d756c9f12d1, 0xe9633210630cbf12, 0x1ffea9fe85a0b0b1],
            [0x81d1ae8cc50240f3, 0xf4c77a079a4607d7, 0xed446b2315e3efc1, 0x0b0a6b70915178c3,
             0xb11ff3e089f15d9a, 0x1d4dba0b7ae9cc18, 0x65d74e2f43b48d05, 0xa2df8c6b8ae0804a,
             0xa4e6f0a8c33348a6, 0xc0a26efc7be5669b, 0xa6b6582c547d0d60],
            [0x84afc741f1c13213, 0x2f8f43734fc906f3, 0xde682d72da0a02d9, 0x0bb005236adb9ef2,
             0x5bdf35c10a8b5624, 0x0739a8a343950010, 0x52f515f44785cfbc, 0xcbaf4e5d82856c60,
             0xac9ea09074e3e150, 0x8f0fa011a2035fb0, 0x1a37905d8450904a],
            [0x3abeb80def61cc85, 0x9d19c9dd4eac4133, 0x075a652d9641a985, 0x9daf69ae1b67e667,
             0x364f71da77920a18, 0x50bd769f745c95b1, 0xf223d1180dbbf3fc, 0x2f885e584e04aa99,
             0xb69a0fa70aea684a, 0x09584acaa6e062a0, 0x0bc051640145b19b],
        ];
        #[rustfmt::skip]
        const FP_INIT_MAT: [[u64; 11]; 11] = [
            [0x80772dc2645b280b, 0xdc927721da922cf8, 0xc1978156516879ad, 0x90e80c591f48b603,
             0x3a2432625475e3ae, 0x00a2d4321cca94fe, 0x77736f524010c932, 0x904d3f2804a36c54,
             0xbf9b39e28a16f354, 0x3a1ded54a6cd058b, 0x42392870da5737cf],
            [0xe796d293a47a64cb, 0xb124c33152a2421a, 0x0ee5dc0ce131268a, 0xa9032a52f930fae6,
             0x7e33ca8c814280de, 0xad11180f69a8c29e, 0xc75ac6d5b5a10ff3, 0xf0674a8dc5a387ec,
             0xb36d43120eaa5e2b, 0x6f232aab4b533a25, 0x3a1ded54a6cd058b],
            [0xdcedab70f40718ba, 0x14a4a64da0b2668f, 0x4715b8e5ab34653b, 0x1e8916a99c93a88e,
             0xbba4b5d86b9a3b2c, 0xe76649f9bd5d5c2e, 0xaf8e2518a1ece54d, 0xdcda1344cdca873f,
             0xcd080204256088e5, 0xb36d43120eaa5e2b, 0xbf9b39e28a16f354],
            [0xf4a437f2888ae909, 0xc537d44dc2875403, 0x7f68007619fd8ba9, 0xa4911db6a32612da,
             0x2f7e9aade3fdaec1, 0xe7ffd578da4ea43d, 0x43a608e7afa6b5c2, 0xca46546aa99e1575,
             0xdcda1344cdca873f, 0xf0674a8dc5a387ec, 0x904d3f2804a36c54],
            [0xf97abba0dffb6c50, 0x5e40f0c9bb82aab5, 0x5996a80497e24a6b, 0x07084430a7307c9a,
             0xad2f570a5b8545aa, 0xab7f81fef4274770, 0xcb81f535cf98c9e9, 0x43a608e7afa6b5c2,
             0xaf8e2518a1ece54d, 0xc75ac6d5b5a10ff3, 0x77736f524010c932],
            [0x7f8e41e0b0a6cdff, 0x4b1ba8d40afca97d, 0x623708f28fca70e8, 0xbf150dc4914d380f,
             0xc26a083554767106, 0x753b8b1126665c22, 0xab7f81fef4274770, 0xe7ffd578da4ea43d,
             0xe76649f9bd5d5c2e, 0xad11180f69a8c29e, 0x00a2d4321cca94fe],
            [0x726af914971c1374, 0x1d7f8a2cce1a9d00, 0x18737784700c75cd, 0x7fb45d605dd82838,
             0x862361aeab0f9b6e, 0xc26a083554767106, 0xad2f570a5b8545aa, 0x2f7e9aade3fdaec1,
             0xbba4b5d86b9a3b2c, 0x7e33ca8c814280de, 0x3a2432625475e3ae],
            [0x64dd936da878404d, 0x4db9a2ead2bd7262, 0xbe2e19f6d07f1a83, 0x02290fe23c20351a,
             0x7fb45d605dd82838, 0xbf150dc4914d380f, 0x07084430a7307c9a, 0xa4911db6a32612da,
             0x1e8916a99c93a88e, 0xa9032a52f930fae6, 0x90e80c591f48b603],
            [0x85418a9fef8a9890, 0xd8a2eb7ef5e707ad, 0xbfe85ababed2d882, 0xbe2e19f6d07f1a83,
             0x18737784700c75cd, 0x623708f28fca70e8, 0x5996a80497e24a6b, 0x7f68007619fd8ba9,
             0x4715b8e5ab34653b, 0x0ee5dc0ce131268a, 0xc1978156516879ad],
            [0x156048ee7a738154, 0x91f7562377e81df5, 0xd8a2eb7ef5e707ad, 0x4db9a2ead2bd7262,
             0x1d7f8a2cce1a9d00, 0x4b1ba8d40afca97d, 0x5e40f0c9bb82aab5, 0xc537d44dc2875403,
             0x14a4a64da0b2668f, 0xb124c33152a2421a, 0xdc927721da922cf8],
            [0xd841e8ef9dde8ba0, 0x156048ee7a738154, 0x85418a9fef8a9890, 0x64dd936da878404d,
             0x726af914971c1374, 0x7f8e41e0b0a6cdff, 0xf97abba0dffb6c50, 0xf4a437f2888ae909,
             0xdcedab70f40718ba, 0xe796d293a47a64cb, 0x80772dc2645b280b],
        ];

        // Fast permutation using precomputed constants
        fn permute_fast(state: &mut [FpE; SPONGE_WIDTH]) {
            // Phase 1: First 4 full rounds
            for round in 0..HALF_N_FULL_ROUNDS {
                PoseidonGoldilocks::full_round(state, round);
            }

            // Phase 2a: Add first partial round constants
            for i in 0..SPONGE_WIDTH {
                state[i] = &state[i] + &FpE::from(FP_FIRST_RC[i]);
            }

            // Phase 2b: Apply initial matrix to state[1..11]
            // Plonky2: result[c] = sum_r state[r] * INIT_MAT[r-1][c-1]
            let mut tmp = [FpE::zero(); SPONGE_WIDTH];
            tmp[0] = state[0];
            for c in 0..11 {
                let mut sum = FpE::zero();
                for r in 0..11 {
                    sum = &sum + &(&state[r + 1] * &FpE::from(FP_INIT_MAT[r][c]));
                }
                tmp[c + 1] = sum;
            }
            *state = tmp;

            // Phase 2c: 22 partial rounds with fast MDS
            for i in 0..N_PARTIAL_ROUNDS {
                state[0] = PoseidonGoldilocks::sbox(state[0]);
                state[0] = &state[0] + &FpE::from(FP_RC[i]);

                let s0 = state[0];

                // d = mds0to0 * s0 + dot(w_hat, state[1..11])
                let mut d = &s0 * &FpE::from(25u64);
                for j in 0..11 {
                    d = &d + &(&state[j + 1] * &FpE::from(FP_W_HATS[i][j]));
                }

                // state[j+1] += s0 * vs[j]
                for j in 0..11 {
                    state[j + 1] = &state[j + 1] + &(&s0 * &FpE::from(FP_VS[i][j]));
                }

                state[0] = d;
            }

            // Phase 3: Last 4 full rounds
            for round in (HALF_N_FULL_ROUNDS + N_PARTIAL_ROUNDS)..N_ROUNDS {
                PoseidonGoldilocks::full_round(state, round);
            }
        }

        // Test with multiple inputs
        let test_inputs: [[u64; 12]; 3] = [
            [0; 12],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [
                0x8ccbbbea4fe5d2b7,
                0xc2af59ee9ec49970,
                0x90f7e1a9e658446a,
                0xdcc0630a3ab8b1b8,
                0x7ff8256bca20588c,
                0x5d99a7ca0c44ecfb,
                0x48452b17a70fbee3,
                0xeb09d654690b6c88,
                0x4a55d3a39c676a88,
                0xc0407a38d2285139,
                0xa234bac9356386d1,
                0xe1633f2bad98a52f,
            ],
        ];

        for (idx, input) in test_inputs.iter().enumerate() {
            let mut standard = [FpE::zero(); SPONGE_WIDTH];
            let mut fast = [FpE::zero(); SPONGE_WIDTH];
            for i in 0..SPONGE_WIDTH {
                standard[i] = FpE::from(input[i]);
                fast[i] = FpE::from(input[i]);
            }
            PoseidonGoldilocks::permute(&mut standard);
            permute_fast(&mut fast);
            for i in 0..SPONGE_WIDTH {
                assert_eq!(
                    standard[i], fast[i],
                    "Mismatch at test {idx} element {i}: standard={:?}, fast={:?}",
                    standard[i], fast[i]
                );
            }
        }
    }

    #[test]
    fn test_plonky2_vector_random() {
        check_plonky2_vector(
            [
                0x8ccbbbea4fe5d2b7,
                0xc2af59ee9ec49970,
                0x90f7e1a9e658446a,
                0xdcc0630a3ab8b1b8,
                0x7ff8256bca20588c,
                0x5d99a7ca0c44ecfb,
                0x48452b17a70fbee3,
                0xeb09d654690b6c88,
                0x4a55d3a39c676a88,
                0xc0407a38d2285139,
                0xa234bac9356386d1,
                0xe1633f2bad98a52f,
            ],
            [
                0xa89280105650c4ec,
                0xab542d53860d12ed,
                0x5704148e9ccab94f,
                0xd3a826d4b62da9f5,
                0x8a7a6ca87892574f,
                0xc7017e1cad1a674e,
                0x1f06668922318e34,
                0xa3b203bc8102676f,
                0xfcc781b0ce382bf2,
                0x934c69ff3ed14ba5,
                0x504688a5996e8f13,
                0x401f3f2ed524a2ba,
            ],
        );
    }
}
