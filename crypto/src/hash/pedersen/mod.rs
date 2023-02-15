use super::traits::IsCryptoHash;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        self,
        short_weierstrass::{
            curves::bls12_381::{curve::BLS12381Curve, field_extension::BLS12381PrimeField},
            element::ProjectivePoint,
        },
        traits::IsEllipticCurve,
    },
    field::{self, element::FieldElement, traits::IsField},
    traits::ByteConversion,
    unsigned_integer::element::U384,
};

type FE = field::element::FieldElement<BLS12381PrimeField>;
type Point = ProjectivePoint<BLS12381Curve>;

const WINDOW_SIZE: usize = 4;
const NUM_WINDOWS: usize = 96;
const INPUT_SIZE_IN_BITS: usize = WINDOW_SIZE * NUM_WINDOWS;
const HALF_INPUT_SIZE_BITS: usize = INPUT_SIZE_IN_BITS / 2;

pub fn random_point(rng: &mut rand::rngs::ThreadRng) -> Point {
    BLS12381Curve::generator().operate_with_self(rand::Rng::gen::<u128>(rng))
}

// TODO: this function should be replaced with the trait method once it is implemented.
pub fn random_field_element<F>(rng: &mut rand::rngs::ThreadRng) -> FieldElement<F>
where
    F: field::traits::IsField,
{
    FieldElement::<F>::from(rand::Rng::gen::<u64>(rng))
}

pub struct Pedersen<E>
where
    E: elliptic_curve::short_weierstrass::traits::IsShortWeierstrass,
{
    parameters: Vec<Vec<FieldElement<E::BaseField>>>,
}

impl<E> Pedersen<E>
where
    E: elliptic_curve::short_weierstrass::traits::IsShortWeierstrass,
{
    fn create_generators(rng: &mut rand::rngs::ThreadRng) -> Vec<Vec<FieldElement<E::BaseField>>> {
        (0..NUM_WINDOWS)
            .into_iter()
            .map(|_| Self::generator_powers(WINDOW_SIZE, rng))
            .collect()
    }

    fn generator_powers(
        num_powers: usize,
        rng: &mut rand::rngs::ThreadRng,
    ) -> Vec<FieldElement<E::BaseField>> {
        let base = random_field_element::<E::BaseField>(rng);
        (0..num_powers)
            .into_iter()
            .map(|exponent| base.pow(exponent))
            .collect()
    }
}

impl IsCryptoHash<BLS12381PrimeField> for Pedersen<BLS12381Curve> {
    fn new() -> Self {
        Self {
            parameters: Self::create_generators(&mut rand::thread_rng()),
        }
    }

    fn hash_one(&self, input: FE) -> FE {
        // Compute sum of h_i^{m_i} for all i.
        let bits = to_bits(input);
        bits.chunks(WINDOW_SIZE)
            .zip(&self.parameters)
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

    fn hash_two(&self, left: FE, right: FE) -> FE {
        let left_input_bytes = left.value().to_bytes_be();
        let right_input_bytes = right.value().to_bytes_be();
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

fn to_bits(felt: FE) -> Vec<bool> {
    let felt_bytes = felt.value().to_bytes_be();
    let mut bits = Vec::with_capacity(felt_bytes.len() * 8);
    for byte in felt_bytes {
        for i in 0..8 {
            bits.push(byte & (1 << i) != 0);
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use crate::hash::{pedersen::{FE, Pedersen}, traits::IsCryptoHash};

    #[test]
    fn test_pedersen_hash() {
        let in1 = FE::new_base("03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb");
        let in2 = FE::new_base("0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a");
        let expected_hash = FE::new_base("030e480bed5fe53fa909cc0f8c4d99b8f9f2c016be4c41e13a4848797979c662");

        let hasher = Pedersen::new();

        assert_eq!(hasher.hash_two(in1, in2), expected_hash);
    }
}
