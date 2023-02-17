use crate::hash::traits::IsCryptoHash;
use lambdaworks_math::{
    errors::ByteConversionError,
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
};

pub struct Proof<F: IsField, H: IsCryptoHash<F>> {
    pub value: FieldElement<F>,
    pub merkle_path: Vec<(FieldElement<F>, bool)>,
    pub hasher: H,
}

impl<F, H> ByteConversion for Proof<F, H>
where
    F: IsField,
    H: IsCryptoHash<F>,
    FieldElement<F>: ByteConversion,
{
    /// Returns the byte representation of the element in big-endian order.
    fn to_bytes_be(&self) -> Vec<u8> {
        let mut buffer: Vec<u8> = Vec::new();

        for val in self.value.to_bytes_be().iter() {
            buffer.push(*val);
        }

        for (value, is_left) in self.merkle_path.iter() {
            for val in value.to_bytes_be().iter() {
                buffer.push(*val);
            }

            if *is_left {
                buffer.push(1);
            } else {
                buffer.push(0);
            }
        }

        buffer.to_vec()
    }

    /// Returns the byte representation of the element in little-endian order.
    fn to_bytes_le(&self) -> Vec<u8> {
        let mut buffer: Vec<u8> = Vec::new();

        for val in self.value.to_bytes_le().iter() {
            buffer.push(*val);
        }

        for (value, is_left) in self.merkle_path.iter() {
            for val in value.to_bytes_le().iter() {
                buffer.push(*val);
            }

            if *is_left {
                buffer.push(1);
            } else {
                buffer.push(0);
            }
        }

        buffer.to_vec()
    }

    /// Returns the element from its byte representation in big-endian order.
    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let mut merkle_path = Vec::new();

        for elem in bytes[8..].chunks(9) {
            let field = FieldElement::from_bytes_be(&elem[..elem.len() - 1])?;
            merkle_path.push((field, elem[elem.len() - 1] == 1));
        }

        Ok(Proof {
            value: FieldElement::from_bytes_be(&bytes[..8])?,
            merkle_path,
            hasher: H::new(),
        })
    }

    /// Returns the element from its byte representation in little-endian order.
    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let mut merkle_path = Vec::new();

        for elem in bytes[8..].chunks(9) {
            let field = FieldElement::from_bytes_le(&elem[..elem.len() - 1])?;
            merkle_path.push((field, elem[elem.len() - 1] == 1));
        }

        Ok(Proof {
            value: FieldElement::from_bytes_le(&bytes[..8])?,
            merkle_path,
            hasher: H::new(),
        })
    }
}

#[cfg(test)]
mod tests {

    use lambdaworks_math::traits::ByteConversion;

    use crate::merkle_tree::{proof::Proof, DefaultHasher, U64Proof, U64FE};

    #[test]
    fn serialize_proof_and_deserialize_using_be_it_get_a_consistent_proof() {
        let merkle_path = [
            (U64FE::new(2), true),
            (U64FE::new(1), false),
            (U64FE::new(1), false),
        ]
        .to_vec();
        let original_proof = U64Proof {
            hasher: DefaultHasher,
            merkle_path,
            value: U64FE::new(1),
        };
        let serialize_proof = original_proof.to_bytes_be();
        let proof: U64Proof = Proof::from_bytes_be(&serialize_proof).unwrap();

        assert_eq!(original_proof.value, proof.value);

        for ((o_node, o_is_left), (node, is_left)) in
            original_proof.merkle_path.iter().zip(proof.merkle_path)
        {
            assert_eq!(*o_node, node);
            assert_eq!(*o_is_left, is_left);
        }
    }

    #[test]
    fn serialize_proof_and_deserialize_using_le_it_get_a_consistent_proof() {
        let merkle_path = [
            (U64FE::new(2), true),
            (U64FE::new(1), false),
            (U64FE::new(1), false),
        ]
        .to_vec();
        let original_proof = U64Proof {
            hasher: DefaultHasher,
            merkle_path,
            value: U64FE::new(1),
        };
        let serialize_proof = original_proof.to_bytes_le();
        let proof: U64Proof = Proof::from_bytes_le(&serialize_proof).unwrap();

        assert_eq!(original_proof.value, proof.value);

        for ((o_node, o_is_left), (node, is_left)) in
            original_proof.merkle_path.iter().zip(proof.merkle_path)
        {
            assert_eq!(*o_node, node);
            assert_eq!(*o_is_left, is_left);
        }
    }
}
