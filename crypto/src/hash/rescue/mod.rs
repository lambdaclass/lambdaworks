use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;
//use num_integer::binomial;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;

mod util;
mod mds;

const PRIME : u64 = (1 << 31) - 1; // mersenne 31 (2^31 - 1)

pub struct RescueM31 {
    pub width: usize,
    pub capacity: usize,
    pub rate: usize,
}

impl RescueM31 {
    pub fn new(width: usize, capacity: usize) -> Self {
        let rate = width - capacity;
        Self {width, capacity, rate}
    }

    pub fn hash(&self, mut input_sequence: Vec<FieldElement<Mersenne31Field>>) -> Vec<FieldElement<Mersenne31Field>> {
        let rate = self.rate;
        let fp0 = FieldElement::new(Mersenne31Field::from_u64(0));
        let fp1 = FieldElement::new(Mersenne31Field::from_u64(1));
    
        // Padding
        if input_sequence.len()%rate != 0 {
            input_sequence.push(fp1);
        }
        while input_sequence.len() % rate != 0 {
            input_sequence.push(fp0.clone());
        }
    
        // Check if the padded input sequence length matches 'self.width'
        assert_eq!(input_sequence.len(), self.width);
    
        self.permutation(&mut input_sequence);
    
        input_sequence
    }

    pub fn get_num_rounds(&self,sec_level: usize, alpha: u64) -> usize {
        let m = self.width;
        let r = self.rate;
        let dcon = |n: usize| {
            (0.5 * ((alpha - 1) * m as u64 * (n as u64 - 1)) as f64 + 2.0).floor() as usize
        };
        let v = |n: usize| m * (n - 1) + r;
        let target: UnsignedInteger<4> = UnsignedInteger::from_u64(1 as u64) << sec_level;
        let is_sufficient = |l1: &usize| {
            let n = UnsignedInteger::from_u64((v(*l1) + dcon(*l1)) as u64);
            let k = UnsignedInteger::from_u64((v(*l1)) as u64);
            let bin = util::binomial(n, k);
            &bin * &bin > target
        };
        let l1 = (1..25).find(is_sufficient).unwrap();
        (l1.max(5) as f32 * 1.5).ceil() as usize
    }

    fn get_round_constants(
        &self,
        num_rounds: usize,
        sec_level: usize,
    ) -> Vec<FieldElement<Mersenne31Field>> {
        let m = self.width;
        let num_constants = 2 * m * num_rounds;
        let bytes_per_constant = ((64 - PRIME.leading_zeros() as usize) + 7) / 8 + 1;
        let num_bytes = bytes_per_constant * num_constants;
        let seed_string = format!(
            "Rescue-XLIX({},{},{},{})",
            PRIME,
            m,
            self.capacity,
            sec_level,
        );

        let byte_string = util::shake256_hash(seed_string.as_bytes(), num_bytes);

        byte_string
            .chunks(bytes_per_constant)
            .map(|chunk| {
                let integer = chunk
                    .iter()
                    .rev()
                    .fold(0, |acc, &byte| (acc << 8) + byte as u64);
                FieldElement::new(Mersenne31Field::from_u64(integer % PRIME))
            })
            .collect()
    }


    pub fn permutation(&self, state: &mut Vec<FieldElement<Mersenne31Field>>) {
    
        let (alpha, alpha_inv) = util::get_alphas(PRIME); 

        let m = self.width;
        let n = self.get_num_rounds(128, alpha);

        let round_constants: Vec<FieldElement<Mersenne31Field>> = self.get_round_constants(n, 128);

        for round in 0..n {
            util::sbox(state, alpha);
            
            mds::apply_mds(state, m);
            
            util::add_round_constants(state, round_constants[2*round*m..].to_vec());
            
            util::sbox_inv(state, alpha_inv);

            mds::apply_mds(state, m);
            
            util::add_round_constants(state, round_constants[2*round*m + m..].to_vec());

        }

    }

}


// tests taken from plonky3

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rescue_hash1() {
        // Create a Rescue instance
        let rescue = RescueM31::new(12, 6);
    
        // Input state
        let state = vec![
            FieldElement::<Mersenne31Field>::from(144096679),
            FieldElement::<Mersenne31Field>::from(1638468327),
            FieldElement::<Mersenne31Field>::from(1550998769),
            FieldElement::<Mersenne31Field>::from(1713522258),
            FieldElement::<Mersenne31Field>::from(730676443),
            FieldElement::<Mersenne31Field>::from(955614588),
            FieldElement::<Mersenne31Field>::from(1970746889),
            FieldElement::<Mersenne31Field>::from(1473251100),
            FieldElement::<Mersenne31Field>::from(1575313887),
            FieldElement::<Mersenne31Field>::from(1867935938),
            FieldElement::<Mersenne31Field>::from(364960233),
            FieldElement::<Mersenne31Field>::from(91318724),
        ];
    
        // Expected output
        let expected_output = vec![
            FieldElement::<Mersenne31Field>::from(504747180),
            FieldElement::<Mersenne31Field>::from(1708979401),
            FieldElement::<Mersenne31Field>::from(1023327691),
            FieldElement::<Mersenne31Field>::from(414948293),
            FieldElement::<Mersenne31Field>::from(1811202621),
            FieldElement::<Mersenne31Field>::from(427591394),
            FieldElement::<Mersenne31Field>::from(666516466),
            FieldElement::<Mersenne31Field>::from(1900855073),
            FieldElement::<Mersenne31Field>::from(1511950466),
            FieldElement::<Mersenne31Field>::from(346735768),
            FieldElement::<Mersenne31Field>::from(708718627),
            FieldElement::<Mersenne31Field>::from(2070146754),
        ];
    
        
        let result = rescue.hash(state);
    
        assert_eq!(result, expected_output);
    }

    #[test]
    fn test_rescue_hash2() {
        // Create a Rescue instance
        let rescue = RescueM31::new(12, 6);
    
        // Input state
        let state = vec![
            FieldElement::<Mersenne31Field>::from(0),
            FieldElement::<Mersenne31Field>::from(1),
            FieldElement::<Mersenne31Field>::from(2),
            FieldElement::<Mersenne31Field>::from(3),
            FieldElement::<Mersenne31Field>::from(4),
            FieldElement::<Mersenne31Field>::from(5),
            FieldElement::<Mersenne31Field>::from(6),
            FieldElement::<Mersenne31Field>::from(7),
            FieldElement::<Mersenne31Field>::from(8),
            FieldElement::<Mersenne31Field>::from(9),
            FieldElement::<Mersenne31Field>::from(10),
            FieldElement::<Mersenne31Field>::from(11),
        ];
    
        // Expected output
        let expected_output = vec![
            FieldElement::<Mersenne31Field>::from(983158113),
            FieldElement::<Mersenne31Field>::from(88736227),
            FieldElement::<Mersenne31Field>::from(182376113),
            FieldElement::<Mersenne31Field>::from(380581876),
            FieldElement::<Mersenne31Field>::from(1054929865),
            FieldElement::<Mersenne31Field>::from(873254619),
            FieldElement::<Mersenne31Field>::from(1742172525),
            FieldElement::<Mersenne31Field>::from(1018880997),
            FieldElement::<Mersenne31Field>::from(1922857524),
            FieldElement::<Mersenne31Field>::from(2128461101),
            FieldElement::<Mersenne31Field>::from(1878468735),
            FieldElement::<Mersenne31Field>::from(736900567),
        
        ];
    
        
        let result = rescue.hash( state);
        assert_eq!(result, expected_output);

    }
    
    

    #[test]
    fn test_num_rounds() {
        let rescue = RescueM31::new(12, 6);

        let ans = rescue.get_num_rounds(128, 5);
        let expected: usize = 8;

        assert_eq!(ans, expected);
    }

}

