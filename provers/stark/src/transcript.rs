use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        traits::{IsFFTField, IsField, IsPrimeField},
    },
    traits::ByteConversion,
    unsigned_integer::element::U256,
};
use sha3::{Digest, Keccak256};

pub trait IsStarkTranscript<F: IsField> {
    fn append_field_element(&mut self, element: &FieldElement<F>);
    fn append_bytes(&mut self, new_bytes: &[u8]);
    fn state(&self) -> [u8; 32];
    fn sample_field_element(&mut self) -> FieldElement<F>;
    fn sample_u64(&mut self, upper_bound: u64) -> u64;
}

fn keccak_hash(data: &[u8]) -> Keccak256 {
    let mut hasher = Keccak256::new();
    hasher.update(data);
    hasher
}

const MODULUS_MAX_MULTIPLE: U256 =
    U256::from_hex_unchecked("f80000000000020f00000000000000000000000000000000000000000000001f");
const R_INV: U256 =
    U256::from_hex_unchecked("0x40000000000001100000000000012100000000000000000000000000000000");

pub struct StoneProverTranscript {
    hash: Keccak256,
    seed_increment: U256,
    counter: usize,
    spare_bytes: Vec<u8>,
}

impl StoneProverTranscript {
    pub fn new(public_input_data: &[u8]) -> Self {
        let hash = keccak_hash(public_input_data);
        StoneProverTranscript {
            hash,
            seed_increment: U256::from_hex_unchecked("1"),
            counter: 0,
            spare_bytes: vec![],
        }
    }

    pub fn sample_block(&mut self, used_bytes: usize) -> Vec<u8> {
        let mut first_part: Vec<u8> = self.hash.clone().finalize().to_vec();
        let mut counter_bytes: Vec<u8> = vec![0; 24]
            .into_iter()
            .chain(self.counter.to_be_bytes().to_vec())
            .collect();
        self.counter += 1;
        first_part.append(&mut counter_bytes);
        let block = keccak_hash(&first_part).finalize().to_vec();
        self.spare_bytes.extend(&block[used_bytes..]);
        block[..used_bytes].to_vec()
    }

    pub fn sample(&mut self, num_bytes: usize) -> Vec<u8> {
        let num_blocks = num_bytes / 32;
        let mut result: Vec<u8> = Vec::new();

        for _ in 0..num_blocks {
            let mut block = self.sample_block(32);
            result.append(&mut block);
        }

        let rest = num_bytes % 32;
        if rest <= self.spare_bytes.len() {
            result.append(&mut self.spare_bytes[..rest].to_vec());
            self.spare_bytes.drain(..rest);
        } else {
            let mut block = self.sample_block(rest);
            result.append(&mut block);
        }
        result
    }

    pub fn sample_big_int(&mut self) -> U256 {
        U256::from_bytes_be(&self.sample(32)).unwrap()
    }
}

impl IsStarkTranscript<Stark252PrimeField> for StoneProverTranscript {
    fn append_field_element(&mut self, element: &FieldElement<Stark252PrimeField>) {
        let limbs = element.value().limbs;
        let mut bytes: [u8; 32] = [0; 32];

        for i in (0..4).rev() {
            let limb_bytes = limbs[i].to_be_bytes();
            for j in 0..8 {
                bytes[i * 8 + j] = limb_bytes[j]
            }
        }
        self.append_bytes(&bytes);
    }

    fn append_bytes(&mut self, new_bytes: &[u8]) {
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&self.hash.clone().finalize_reset());
        result_hash.reverse();

        let digest = U256::from_bytes_be(&self.hash.clone().finalize()).unwrap();
        let new_seed = (digest + self.seed_increment).to_bytes_be();
        self.hash = keccak_hash(&[&new_seed, new_bytes].concat());
        self.counter = 0;
        self.spare_bytes.clear();
    }

    fn state(&self) -> [u8; 32] {
        let mut state = [0u8; 32];
        state.copy_from_slice(&self.hash.clone().finalize());
        state
    }

    fn sample_field_element(&mut self) -> FieldElement<Stark252PrimeField> {
        let mut result = self.sample_big_int();
        while result >= MODULUS_MAX_MULTIPLE {
            result = self.sample_big_int();
        }
        FieldElement::new(result) * FieldElement::new(R_INV)
    }

    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        // assert!(upper_bound < (1 << 12));
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.sample(8));
        let u64_val: u64 = u64::from_be_bytes(bytes);
        u64_val % upper_bound
    }
}

pub fn sample_z_ood<F: IsPrimeField>(
    lde_roots_of_unity_coset: &[FieldElement<F>],
    trace_roots_of_unity: &[FieldElement<F>],
    transcript: &mut impl IsStarkTranscript<F>,
) -> FieldElement<F>
where
    FieldElement<F>: ByteConversion,
{
    loop {
        let value: FieldElement<F> = transcript.sample_field_element();
        if !lde_roots_of_unity_coset.iter().any(|x| x == &value)
            && !trace_roots_of_unity.iter().any(|x| x == &value)
        {
            return value;
        }
    }
}

pub fn batch_sample_challenges<F: IsFFTField>(
    size: usize,
    transcript: &mut impl IsStarkTranscript<F>,
) -> Vec<FieldElement<F>>
where
    FieldElement<F>: ByteConversion,
{
    (0..size)
        .map(|_| transcript.sample_field_element())
        .collect()
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };

    use crate::transcript::{IsStarkTranscript, StoneProverTranscript};

    // #[test]
    // fn test_stark_prime_field_random_to_field_32() {
    //     #[rustfmt::skip]
    //     let mut randomness: [u8; 32] = [
    //         248, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 32,
    //     ];
    //
    //     type FE = FieldElement<Stark252PrimeField>;
    //     let field_element: FE = randomness_to_field(&mut randomness);
    //     let expected_fe = FE::from(32u64);
    //     assert_eq!(field_element, expected_fe)
    // }
    //
    // #[test]
    // fn test_stark_prime_field_random_to_fiel_repeated_f_and_zero() {
    //     #[rustfmt::skip]
    //     let mut randomness: [u8; 32] = [
    //         255, 0, 255, 0, 255, 0, 255, 0,
    //         255, 0, 255, 0, 255, 0, 255, 0,
    //         255, 0, 255, 0, 255, 0, 255, 0,
    //         255, 0, 255, 0, 255, 0, 255, 0,
    //     ];
    //
    //     type FE = FieldElement<Stark252PrimeField>;
    //
    //     // 251 bits should be used (252 of StarkField - 1) to avoid duplicates
    //     // This leaves a 7
    //     let expected_fe = FE::from_hex_unchecked(
    //         "\
    //         0700FF00FF00FF00\
    //         FF00FF00FF00FF00\
    //         FF00FF00FF00FF00\
    //         FF00FF00FF00FF00",
    //     );
    //
    //     let field_element: FE = randomness_to_field(&mut randomness);
    //
    //     assert_eq!(field_element, expected_fe)
    // }
    //
    // #[test]
    // fn test_241_bit_random_to_field() {
    //     #[derive(Clone, Debug)]
    //     pub struct TestModulus;
    //     impl IsModulus<U256> for TestModulus {
    //         const MODULUS: U256 = U256::from_hex_unchecked(
    //             "\
    //             0001000000000011\
    //             0000000000000000\
    //             0000000000000000\
    //             0000000000000001",
    //         );
    //     }
    //
    //     pub type TestField = U256PrimeField<TestModulus>;
    //
    //     #[rustfmt::skip]
    //     let mut randomness: [u8; 32] = [
    //         255, 255, 255, 1, 2, 3, 4, 5,
    //         6, 7, 8, 1, 2, 3, 4, 5,
    //         6, 7, 8, 1, 2, 3, 4, 5,
    //         6, 7, 8, 1, 2, 3, 4, 5,
    //     ];
    //
    //     type FE = FieldElement<TestField>;
    //
    //     let expected_fe = FE::from_hex_unchecked(
    //         "\
    //         0000FF0102030405\
    //         0607080102030405\
    //         0607080102030405\
    //         0607080102030405",
    //     );
    //
    //     let field_element: FE = randomness_to_field(&mut randomness);
    //
    //     assert_eq!(field_element, expected_fe);
    // }
    //
    // #[test]
    // fn test_249_bit_random_to_field() {
    //     #[derive(Clone, Debug)]
    //     pub struct TestModulus;
    //     impl IsModulus<U256> for TestModulus {
    //         const MODULUS: U256 = U256::from_hex_unchecked(
    //             "\
    //             0200000000000011\
    //             0000000000000000\
    //             0000000000000000\
    //             0000000000000001",
    //         );
    //     }
    //
    //     pub type TestField = U256PrimeField<TestModulus>;
    //
    //     #[rustfmt::skip]
    //     let mut randomness: [u8; 32] = [
    //         255, 0, 255, 0, 255, 0, 255, 0,
    //         255, 0, 255, 0, 255, 0, 255, 0,
    //         255, 0, 255, 0, 255, 0, 255, 0,
    //         255, 0, 255, 0, 255, 0, 255, 0,
    //     ];
    //
    //     let expected_fe = FE::from_hex_unchecked(
    //         "\
    //             0100FF00FF00FF00\
    //             FF00FF00FF00FF00\
    //             FF00FF00FF00FF00\
    //             FF00FF00FF00FF00",
    //     );
    //
    //     type FE = FieldElement<TestField>;
    //
    //     let field_element: FE = randomness_to_field(&mut randomness);
    //
    //     assert_eq!(field_element, expected_fe)
    // }

    use std::num::ParseIntError;

    type FE = FieldElement<Stark252PrimeField>;

    pub fn decode_hex(s: &str) -> Result<Vec<u8>, ParseIntError> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect()
    }

    pub fn send_field_element(s: &str) -> Vec<u8> {
        // Taken from serialize_be method, but reverses the limbs for
        // compatibility with the stone prover.
        let a = FE::from_hex_unchecked(s);
        let limbs = a.value().limbs;
        let mut bytes: [u8; 32] = [0; 32];

        for i in (0..4).rev() {
            let limb_bytes = limbs[i].to_be_bytes();
            for j in 0..8 {
                bytes[i * 8 + j] = limb_bytes[j]
            }
        }
        bytes.to_vec()
    }

    #[test]
    fn sample_bytes_from_stone_prover_channel() {
        let mut transcript = StoneProverTranscript::new(&[0x01, 0x02, 0x03]);
        transcript.append_bytes(&[0x04, 0x05, 0x06]);
        assert_eq!(
            transcript.sample(32),
            vec![
                0x8a, 0x3a, 0x67, 0xd1, 0x25, 0xa5, 0xa5, 0xea, 0x57, 0xc3, 0xfb, 0xe2, 0xc2, 0x55,
                0xb6, 0x0d, 0x0c, 0x89, 0x13, 0xa6, 0x27, 0x13, 0xe0, 0x99, 0xb3, 0x77, 0xc6, 0xc2,
                0x9a, 0x21, 0x85, 0x97
            ]
        );
        assert_eq!(
            transcript.sample(64),
            vec![
                0x56, 0xde, 0x56, 0x2a, 0xfd, 0x98, 0x19, 0xb9, 0xaa, 0xa0, 0x1b, 0x16, 0xf4, 0xeb,
                0x33, 0x71, 0xd5, 0xd8, 0x0f, 0x35, 0x29, 0xd8, 0xc1, 0x7a, 0x4b, 0xf4, 0x10, 0xe3,
                0x19, 0xb7, 0x64, 0x4a, 0xd2, 0x1c, 0xff, 0x14, 0x3d, 0xfd, 0xca, 0x32, 0x2c, 0x59,
                0xa3, 0x47, 0x5d, 0xd0, 0x34, 0xdf, 0x6d, 0xa7, 0x0c, 0xf5, 0xd2, 0x6a, 0xdd, 0x65,
                0xe0, 0x6d, 0x1e, 0x4f, 0xc7, 0x39, 0x52, 0x32
            ]
        );
        assert_eq!(
            transcript.sample(48),
            vec![
                0xe4, 0xb6, 0x3c, 0xfc, 0x03, 0xc9, 0x82, 0x8b, 0x63, 0x53, 0xb9, 0xad, 0x73, 0x6d,
                0x23, 0x88, 0x4c, 0x07, 0xb4, 0x9d, 0xf1, 0x1d, 0xef, 0xb9, 0x53, 0xfa, 0x02, 0xb5,
                0x3c, 0x43, 0xcf, 0xa3, 0x30, 0x5a, 0x02, 0x7e, 0xa6, 0x5e, 0x3c, 0x86, 0x3d, 0xdb,
                0x48, 0xea, 0x73, 0xbf, 0xdf, 0xab
            ]
        );
        assert_eq!(
            transcript.sample(32),
            vec![
                0x82, 0xe1, 0xd4, 0xf8, 0xf0, 0x61, 0xa4, 0x17, 0x4b, 0xed, 0x58, 0x4e, 0xb5, 0x73,
                0x26, 0xb7, 0x63, 0x10, 0x37, 0x97, 0xbe, 0x0b, 0x57, 0xaf, 0x74, 0xfe, 0x33, 0x19,
                0xbd, 0xe5, 0x53, 0x21,
            ]
        );
        assert_eq!(
            transcript.sample(16),
            vec![
                0xb0, 0xc6, 0x7a, 0x04, 0x19, 0x0a, 0x25, 0x72, 0xa8, 0x2e, 0xfa, 0x97, 0x92, 0x44,
                0x73, 0xe9
            ]
        );
        assert_eq!(
            transcript.sample(8),
            vec![0xbd, 0x41, 0x28, 0xdd, 0x3a, 0xbc, 0x66, 0x18]
        );
        assert_eq!(
            transcript.sample(32),
            vec![
                0xcb, 0x66, 0xc9, 0x72, 0x39, 0x85, 0xe8, 0x7c, 0x30, 0xe1, 0xc7, 0x1d, 0x2f, 0x83,
                0x4a, 0xcd, 0x33, 0x85, 0xfb, 0xd5, 0x40, 0x69, 0x22, 0x6e, 0xc0, 0xf1, 0x8c, 0x40,
                0x26, 0x2f, 0x5f, 0x7c,
            ]
        );
        transcript.append_bytes(&[0x03, 0x02]);
        assert_eq!(
            transcript.sample(32),
            vec![
                0x69, 0x63, 0x72, 0x01, 0x84, 0x8b, 0x22, 0x82, 0xa6, 0x14, 0x6d, 0x47, 0xbb, 0xa9,
                0xa3, 0xc8, 0xdc, 0x1b, 0x8e, 0x2e, 0x2e, 0x21, 0x87, 0x77, 0xac, 0xe0, 0x3e, 0xce,
                0x6e, 0xa7, 0x9e, 0xb0,
            ]
        );
    }

    #[test]
    fn sample_numbers_and_field_elements_from_stone_prover_channel() {
        let mut transcript = StoneProverTranscript::new(&[0x01, 0x02]);
        transcript.append_bytes(&[0x01, 0x02]);
        assert_eq!(transcript.sample(4), vec![0x06, 0xe5, 0x36, 0xf5]);
        assert_eq!(transcript.sample_u64(16), 5);
    }

    #[test]
    fn fibonacci_transcript_replicate() {
        let mut transcript = StoneProverTranscript::new(&[0xca, 0xfe, 0xca, 0xfe]);
        // Send hash of trace commitment
        transcript.append_bytes(
            &decode_hex("0eb9dcc0fb1854572a01236753ce05139d392aa3aeafe72abff150fe21175594")
                .unwrap(),
        );
        // Sample challenge to collapse the constraints for the composition polynomial
        assert_eq!(
            transcript.sample_field_element(),
            FE::from_hex_unchecked(
                "86105fff7b04ed4068ecccb8dbf1ed223bd45cd26c3532d6c80a818dbd4fa7"
            )
        );
        // Send hash of composition poly commitment H(z)
        transcript.append_bytes(
            &decode_hex("7cdd8d5fe3bd62254a417e2e260e0fed4fccdb6c9005e828446f645879394f38")
                .unwrap(),
        );
        // Sample challenge Z to compute t_j(z), H(z)
        assert_eq!(
            transcript.sample_field_element(),
            FE::from_hex_unchecked(
                "317629e783794b52cd27ac3a5e418c057fec9dd42f2b537cdb3f24c95b3e550"
            )
        );
        // Append t_j(z), H(z)
        transcript.append_field_element(&FE::from_hex_unchecked(
            "70d8181785336cc7e0a0a1078a79ee6541ca0803ed3ff716de5a13c41684037",
        ));
        transcript.append_field_element(&FE::from_hex_unchecked(
            "29808fc8b7480a69295e4b61600480ae574ca55f8d118100940501b789c1630",
        ));
        transcript.append_field_element(&FE::from_hex_unchecked(
            "7d8110f21d1543324cc5e472ab82037eaad785707f8cae3d64c5b9034f0abd2",
        ));
        transcript.append_field_element(&FE::from_hex_unchecked(
            "1b58470130218c122f71399bf1e04cf75a6e8556c4751629d5ce8c02cc4e62d",
        ));
        transcript.append_field_element(&FE::from_hex_unchecked(
            "1c0b7c2275e36d62dfb48c791be122169dcc00c616c63f8efb2c2a504687e85",
        ));
        // Sample challenge Gamma to collapse the terms of the deep composition polynomial (batch open).
        // Powers of this challenge are used if more than two terms.
        assert_eq!(
            transcript.sample_field_element(),
            FE::from_hex_unchecked(
                "a0c79c1c77ded19520873d9c2440451974d23302e451d13e8124cf82fc15dd"
            )
        );
        // FRI: Sample challenge Zeta to split the polynomial in half
        assert_eq!(
            transcript.sample_field_element(),
            FE::from_hex_unchecked(
                "5c6b5a66c9fda19f583f0b10edbaade98d0e458288e62c2fa40e3da2b293cef"
            )
        );
        // FRI: Send hash of commitment at Layer 1
        transcript.append_bytes(
            &decode_hex("49c5672520e20eccc72aa28d6fa0d7ef446f1ede38d7c64fbb95d0f34a281803")
                .unwrap(),
        );
        // FRI: Sample challenge to split the polynomial in half
        assert_eq!(
            transcript.sample_field_element(),
            FE::from_hex_unchecked(
                "4243ca9a618e2127590af8e1b38c63a156863fe95e4211cc1ade9b50667bbfa"
            )
        );
        // Send field element at final layer of FRI
        transcript.append_field_element(&FE::from_hex_unchecked(
            "702ddae5809ad82a82556eed2d202202d770962b7d4d82581e183df3efa2da6",
        ));
        // Send proof of work
        transcript.append_bytes(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x30, 0x4d]); // Eight bytes
                                                                                    // Sample query indices
        assert_eq!(transcript.sample_u64(8), 0);

        transcript.append_field_element(&FE::from_hex_unchecked(
            "643e5520c60d06219b27b34da0856a2c23153efe9da75c6036f362c8f19615e",
        ));
        transcript.append_field_element(&FE::from_hex_unchecked(
            "165d7fb12913882268bb8cf470c81f42349fde7dec7b0a90526d142d6a61205",
        ));
        transcript.append_field_element(&FE::from_hex_unchecked(
            "1bc1aadf39f2faee64d84cb25f7a95d3dceac1016258a39fc90c9d370e69ea2",
        ));
        transcript.append_field_element(&FE::from_hex_unchecked(
            "69a2804ed6ec78ed9744730b8f37e0bdcb6021821384f56fad92ebd2959edf4",
        ));

        transcript.append_bytes(
            &decode_hex("0160a780da72e50c596b9b6712bd040475d30777a4fef2c9f9be3a7fbaa98072")
                .unwrap(),
        );
        transcript.append_bytes(
            &decode_hex("993b044db22444c0c0ebf1095b9a51faeb001c9b4dea36abe905f7162620dbbd")
                .unwrap(),
        );
        transcript.append_bytes(
            &decode_hex("5017abeca33fa82576b5c5c2c61792693b48c9d4414a407eef66b6029dae07ea")
                .unwrap(),
        );

        transcript.append_field_element(&FE::from_hex_unchecked(
            "483069de80bf48a1b5ca2f55bdeb9ec3ed1b7bf9c794c3c8832f14928124cbb",
        ));
        transcript.append_field_element(&FE::from_hex_unchecked(
            "1cf5d5ed8348c3dee617bceff2d59cb14099d2978b1f7f928027dbbded1d66f",
        ));

        transcript.append_bytes(
            &decode_hex("6a23307160a636ea45c08f6b56e7585a850b5e14170a6c63f4d166a2220a7c2f")
                .unwrap(),
        );
        transcript.append_bytes(
            &decode_hex("7950888c0355c204a1e83ecbee77a0a6a89f93d41cc2be6b39ddd1e727cc9650")
                .unwrap(),
        );
        transcript.append_bytes(
            &decode_hex("58befe2c5de74cc5a002aa82ea219c5b242e761b45fd266eb95521e9f53f44eb")
                .unwrap(),
        );

        transcript.append_field_element(&FE::from_hex_unchecked(
            "724fcd17f8649ed5e180d4e98ba7e8900c8da2643f5ed548773b145230cf12d",
        ));

        transcript.append_bytes(
            &decode_hex("f1f135fc9228ae46afe83d108b256dda8a6ad63e05d630be1f8b461bf2dccf3d")
                .unwrap(),
        );
        transcript.append_bytes(
            &decode_hex("3fdabd3f5fae2bf405d423417141678f4b9afa5666b00790baac61116c5ea8af")
                .unwrap(),
        );
    }
}
