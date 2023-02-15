use super::traits::IsCryptoHash;
use lambdaworks_math::{
    cyclic_group::IsCyclicGroup,
    elliptic_curve::{
        self,
        curves::bls12_381::{curve::BLS12381Curve, field_extension::BLS12381PrimeField},
    },
    field::{self, traits::IsField},
    traits::ByteConversion,
};

type FE = field::element::FieldElement<BLS12381PrimeField>;
type Point = elliptic_curve::element::EllipticCurveElement<BLS12381Curve>;

const WINDOW_SIZE: usize = 4;
const NUM_WINDOWS: usize = 96;

// TODO: this function should be replaced with the trait method once it is implemented.
pub fn random_field_element(rng: &mut rand::rngs::ThreadRng) -> FE 
{
    FE::from(rand::Rng::gen::<u64>(rng))
}

pub fn random_point(rng: &mut rand::rngs::ThreadRng) -> Point {
    Point::generator().operate_with_self(rand::Rng::gen::<u128>(rng))
}

pub fn create_generators(rng: &mut rand::rngs::ThreadRng) -> Vec<Vec<FE>> 
{
    (0..NUM_WINDOWS)
        .into_iter()
        .map(|_| generator_powers(WINDOW_SIZE, rng))
        .collect()
}

pub fn generator_powers(num_powers: usize, rng: &mut rand::rngs::ThreadRng) -> Vec<FE>
{
    let base = random_field_element(rng);
    (0..num_powers)
        .into_iter()
        .map(|exponent| base.pow(exponent))
        .collect()
}

pub struct Pedersen<E>
where
    E: elliptic_curve::traits::IsEllipticCurve
{
    parameters: Vec<Vec<<E::BaseField as IsField>::BaseType>>,
}

impl IsCryptoHash<BLS12381PrimeField> for Pedersen<BLS12381Curve>
{
    fn hash_one(&self, input: FE) -> FE {
        // Compute sum of h_i^{m_i} for all i.
        let bits = to_bits(input);
        let generators = create_generators(&mut rand::thread_rng());
        bits
            .chunks(WINDOW_SIZE)
            .zip(&generators)
            .map(|(bits, generator_powers)| {
                let mut encoded = FE::zero();
                for (bit, base) in bits.iter().zip(generator_powers.iter()) {
                    if *bit {
                        encoded += base.clone();
                    }
                }
                encoded
            })
            // This last step is the same as doing .sum() but std::iter::Sum is 
            // not implemented for FieldElement yet.
            .fold(FE::zero(), |acc, x| acc + x)
    }

    fn hash_two(
        &self,
        left: FE,
        right: FE,
    ) -> FE {
        let left_input_bytes = left.value().to_bytes_be().unwrap();
        let right_input_bytes = right.value().to_bytes_be().unwrap();
        let mut buffer = vec![0u8; (HALF_INPUT_SIZE_BITS + HALF_INPUT_SIZE_BITS) / 8];

        buffer
            .iter_mut()
            .zip(left_input_bytes.iter().chain(right_input_bytes.iter()))
            .for_each(|(b, l_b)| *b = *l_b);

        let base_type_value = U384::from_bytes_be(&buffer).unwrap();
        let new_input_value = BLS12381PrimeField::from_base_type(base_type_value);
        let new_input = FE::from(&new_input_value);

        self.hash_one(new_input)
    }
}

fn to_bits(felt: FE) -> Vec<bool> 
{
    let felt_bytes = felt.value().to_bytes_be();
    let mut bits = Vec::with_capacity(felt_bytes.len() * 8);
    for byte in felt_bytes {
        for i in 0..8 {
            bits.push(byte & (1 << i) != 0);
        }
    }
    bits
}
