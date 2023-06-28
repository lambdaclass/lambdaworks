use crate::merkle_tree::traits::IsMerkleTreeBackend;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
};
use sha3::{
    digest::{generic_array::GenericArray, OutputSizeUser},
    Digest,
};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Tree512Bits<F, D: Digest> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest> Default for Tree512Bits<F, D> {
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest> IsMerkleTreeBackend for Tree512Bits<F, D>
where
    F: IsField,
    FieldElement<F>: ByteConversion,
    [u8; 64]: From<GenericArray<u8, <D as OutputSizeUser>::OutputSize>>,
{
    type Node = [u8; 64];
    type Data = FieldElement<F>;

    fn hash_data(&self, input: &FieldElement<F>) -> [u8; 64] {
        let mut hasher = D::new();
        hasher.update(input.to_bytes_be());
        hasher.finalize().into()
    }

    fn hash_new_parent(&self, left: &[u8; 64], right: &[u8; 64]) -> [u8; 64] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };
    use sha3::{Keccak512, Sha3_512};

    use crate::merkle_tree::{backends::hash_512_bits::Tree512Bits, merkle::MerkleTree};

    type F = Stark252PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn hash_data_field_element_backend_works_with_keccak512() {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree = MerkleTree::<Tree512Bits<F, Keccak512>>::build(&values);
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<Tree512Bits<F, Keccak512>>(&merkle_tree.root, 0, &values[0]));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha3() {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree = MerkleTree::<Tree512Bits<F, Sha3_512>>::build(&values);
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<Tree512Bits<F, Sha3_512>>(&merkle_tree.root, 0, &values[0]));
    }
}
